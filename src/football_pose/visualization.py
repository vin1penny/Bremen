from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from football_pose.archive import read_archive
from football_pose.artifacts import iter_artifact
from football_pose.configuration import VideoOutputConfig
from football_pose.domain import FramePacket
from football_pose.video import VideoSource


COCO_SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


@dataclass(frozen=True, slots=True)
class VideoRenderResult:
    path: Path
    frames: int
    codec: str
    wall_seconds: float


def _next_artifact_group(
    groups: Iterator[tuple[int, Iterator[FramePacket]]],
) -> tuple[int, list[FramePacket]] | None:
    try:
        frame_index, packets = next(groups)
    except StopIteration:
        return None
    return frame_index, list(packets)


def _reconstruct_processed_frame(
    source_packet: FramePacket, processed_packets: list[FramePacket]
) -> np.ndarray:
    """Composite processed packets back into original-video coordinates."""

    if not processed_packets:
        return source_packet.image.copy()
    if (
        len(processed_packets) == 1
        and processed_packets[0].image.shape == source_packet.image.shape
        and np.allclose(processed_packets[0].to_source, np.eye(3))
    ):
        return processed_packets[0].image.copy()

    height, width = source_packet.source_height, source_packet.source_width
    pixel_sum = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    for packet in processed_packets:
        transform = np.asarray(packet.to_source, dtype=np.float64)
        warped = cv2.warpPerspective(
            packet.image,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask = cv2.warpPerspective(
            np.ones((packet.height, packet.width), dtype=np.uint8),
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        selected = mask > 0
        pixel_sum[selected] += warped[selected]
        weights[selected] += 1.0

    output = source_packet.image.copy()
    covered = weights > 0
    output[covered] = np.clip(
        pixel_sum[covered] / weights[covered, np.newaxis], 0, 255
    ).astype(np.uint8)
    return output


def _person_color(person_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(person_id.encode("utf-8")).digest()
    return tuple(80 + int(value) % 176 for value in digest[:3])


def _point(keypoint: dict[str, Any]) -> tuple[int, int]:
    return int(round(float(keypoint["x"]))), int(round(float(keypoint["y"])))


def _bbox_for_prediction(
    prediction: dict[str, Any], threshold: float
) -> tuple[int, int, int, int] | None:
    bbox = prediction.get("source_bbox")
    if bbox is not None:
        return tuple(int(round(float(value))) for value in bbox)  # type: ignore[return-value]
    visible = [
        keypoint
        for keypoint in prediction["keypoints"]
        if float(keypoint["confidence"]) >= threshold
    ]
    if not visible:
        return None
    xs = [float(keypoint["x"]) for keypoint in visible]
    ys = [float(keypoint["y"]) for keypoint in visible]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _draw_regions(image: np.ndarray, packets: list[FramePacket], thickness: int) -> None:
    if len(packets) == 1 and packets[0].crop_id is None:
        return
    for packet in packets:
        bounds = packet.metadata.get("tile", {}).get("source_bounds")
        if bounds is None:
            bounds = packet.source_bbox
        if bounds is None:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in bounds)
        cv2.rectangle(image, (x1, y1), (x2, y2), (160, 160, 160), thickness)


def _draw_pitch_region(
    image: np.ndarray,
    pitch_bbox: tuple[float, float, float, float] | None,
    thickness: int,
) -> None:
    if pitch_bbox is None:
        return
    x1, y1, x2, y2 = (int(round(value)) for value in pitch_bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 180, 0), thickness)


def _draw_predictions(
    image: np.ndarray,
    predictions: list[dict[str, Any]],
    *,
    model_id: str,
    frame_index: int,
    settings: VideoOutputConfig,
    classifications: dict[tuple[int, str, str], str],
) -> None:
    thickness = max(1, round(image.shape[1] / 960))
    radius = max(2, thickness + 1)
    threshold = settings.keypoint_confidence
    counts = {"inside": 0, "outside": 0, "duplicate": 0, "unavailable": 0}
    for prediction in predictions:
        key = (
            frame_index,
            str(prediction["source_id"]),
            str(prediction["person_id"]),
        )
        classification = classifications.get(key)
        if classification is not None:
            counts[classification] = counts.get(classification, 0) + 1
        color = {
            "inside": (60, 220, 60),
            "outside": (40, 40, 230),
            "duplicate": (150, 150, 150),
            "unavailable": (0, 210, 255),
        }.get(classification, _person_color(str(prediction["person_id"])))
        keypoints = prediction["keypoints"]
        for start, end in COCO_SKELETON:
            first, second = keypoints[start], keypoints[end]
            if (
                float(first["confidence"]) >= threshold
                and float(second["confidence"]) >= threshold
            ):
                cv2.line(image, _point(first), _point(second), color, thickness, cv2.LINE_AA)
        for keypoint in keypoints:
            if float(keypoint["confidence"]) >= threshold:
                cv2.circle(image, _point(keypoint), radius, color, -1, cv2.LINE_AA)
        if settings.draw_bboxes:
            bbox = _bbox_for_prediction(prediction, threshold)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if classifications:
        label = (
            f"{model_id} | frame {frame_index} | raw {len(predictions)} | "
            f"pitch {counts['inside']} | outside {counts['outside']} | "
            f"duplicates {counts['duplicate']} | unknown {counts['unavailable']}"
        )
    else:
        label = f"{model_id} | frame {frame_index} | poses {len(predictions)}"
    font_scale = max(0.5, image.shape[1] / 1920)
    text_thickness = max(1, thickness)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
    )
    cv2.rectangle(
        image,
        (8, 8),
        (16 + text_width, 16 + text_height + baseline),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        label,
        (12, 12 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def render_annotated_video(
    *,
    source_path: str | Path,
    artifact_path: str | Path,
    prediction_parquet: str | Path,
    output_path: str | Path,
    model_id: str,
    settings: VideoOutputConfig,
    decision_parquet: str | Path | None = None,
    pitch_geometry_json: str | Path | None = None,
) -> VideoRenderResult:
    """Render processed frames plus raw model predictions to an MP4 file."""

    start = time.perf_counter()
    source = VideoSource(source_path)
    info = source.probe()
    rate = info.average_rate or Fraction(30, 1)
    predictions_by_frame: dict[int, list[dict[str, Any]]] = {}
    for prediction in read_archive(prediction_parquet):
        predictions_by_frame.setdefault(int(prediction["frame_index"]), []).append(prediction)
    classifications: dict[tuple[int, str, str], str] = {}
    if decision_parquet is not None:
        for decision in pq.read_table(decision_parquet).to_pylist():
            classifications[
                (
                    int(decision["frame_index"]),
                    str(decision["source_id"]),
                    str(decision["person_id"]),
                )
            ] = str(decision["classification"])
    pitch_bboxes: dict[int, tuple[float, float, float, float]] = {}
    if pitch_geometry_json is not None:
        geometry_payload = json.loads(
            Path(pitch_geometry_json).read_text(encoding="utf-8")
        )
        for geometry_frame in geometry_payload.get("frames", []):
            bbox = geometry_frame.get("pitch_bbox")
            if bbox is not None:
                pitch_bboxes[int(geometry_frame["frame_index"])] = tuple(
                    float(value) for value in bbox
                )  # type: ignore[assignment]

    artifact_groups = iter(
        groupby(iter_artifact(artifact_path), key=lambda packet: packet.frame_index)
    )
    current_group = _next_artifact_group(artifact_groups)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{uuid.uuid4().hex}{destination.suffix}"
    )
    frame_count = 0
    try:
        with av.open(str(temporary), mode="w") as container:
            stream = container.add_stream(settings.codec, rate=rate)
            stream.width = info.width
            stream.height = info.height
            stream.pix_fmt = "yuv420p"
            if settings.codec in {"h264", "libx264"}:
                stream.options = {"crf": "20", "preset": "medium"}
            for source_packet in source.frames():
                while current_group is not None and current_group[0] < source_packet.frame_index:
                    current_group = _next_artifact_group(artifact_groups)
                processed_packets: list[FramePacket] = []
                if current_group is not None and current_group[0] == source_packet.frame_index:
                    processed_packets = current_group[1]
                    current_group = _next_artifact_group(artifact_groups)
                image = _reconstruct_processed_frame(source_packet, processed_packets)
                if settings.draw_regions:
                    _draw_regions(image, processed_packets, max(1, round(info.width / 960)))
                _draw_pitch_region(
                    image,
                    pitch_bboxes.get(source_packet.frame_index),
                    max(1, round(info.width / 960)),
                )
                _draw_predictions(
                    image,
                    predictions_by_frame.get(source_packet.frame_index, []),
                    model_id=model_id,
                    frame_index=source_packet.frame_index,
                    settings=settings,
                    classifications=classifications,
                )
                frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                frame.pts = frame_count
                frame.time_base = Fraction(rate.denominator, rate.numerator)
                for packet in stream.encode(frame):
                    container.mux(packet)
                frame_count += 1
            for packet in stream.encode():
                container.mux(packet)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return VideoRenderResult(
        path=destination,
        frames=frame_count,
        codec=settings.codec,
        wall_seconds=time.perf_counter() - start,
    )
