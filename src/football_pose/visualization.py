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
from football_pose.artifacts import iter_artifact, sha256_file, stable_hash
from football_pose.configuration import VideoOutputConfig
from football_pose.domain import FramePacket
from football_pose.pitch_filter import PITCH_LENGTH_CM, PITCH_WIDTH_CM
from football_pose.video import VideoSource
from football_pose.jobs import atomic_write_json


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
    pitch_polygon: list[list[float]] | None,
    thickness: int,
) -> None:
    if pitch_polygon is None:
        return
    polygon = np.rint(pitch_polygon).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [polygon], True, (255, 180, 0), thickness)


def _draw_pitch_keypoints(image: np.ndarray, points: list[dict], thickness: int) -> None:
    for point in points:
        # Retain all raw predictions in the geometry archive, display reliable ones.
        if point["confidence"] < 0.5:
            continue
        x, y = int(round(point["x"])), int(round(point["y"]))
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            continue
        cv2.circle(image, (x, y), 3 * thickness, (255, 0, 255), -1)
        cv2.putText(image, f'P{point["id"]} {point["confidence"]:.2f}',
                    (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * thickness, (255, 0, 255), thickness)


def _radar_canvas() -> tuple[np.ndarray, float, tuple[int, int]]:
    """Fixed, uniformly scaled Weserstadion template in metre coordinates."""
    scale = 4.0
    origin = (24, 42)
    length, width = PITCH_LENGTH_CM / 100., PITCH_WIDTH_CM / 100.
    panel = np.full((int(width * scale) + 82, int(length * scale) + 48, 3),
                    (28, 60, 32), dtype=np.uint8)
    white = (220, 235, 220)

    def point(x: float, y: float) -> tuple[int, int]:
        return round(origin[0] + x * scale), round(origin[1] + y * scale)

    def rectangle(x1, y1, x2, y2):
        cv2.rectangle(panel, point(x1, y1), point(x2, y2), white, 1, cv2.LINE_AA)

    rectangle(0, 0, length, width)
    cv2.line(panel, point(length / 2, 0), point(length / 2, width), white, 1, cv2.LINE_AA)
    cv2.circle(panel, point(length / 2, width / 2), round(9.15 * scale), white, 1, cv2.LINE_AA)
    cv2.circle(panel, point(length / 2, width / 2), 2, white, -1)
    for left in (True, False):
        start, end = (0., 16.5) if left else (length - 16.5, length)
        rectangle(start, (width - 40.32) / 2, end, (width + 40.32) / 2)
        start, end = (0., 5.5) if left else (length - 5.5, length)
        rectangle(start, (width - 18.32) / 2, end, (width + 18.32) / 2)
        start, end = (-2., 0.) if left else (length, length + 2.)
        rectangle(start, (width - 7.32) / 2, end, (width + 7.32) / 2)
        cv2.circle(panel, point(11. if left else length - 11., width / 2), 2, white, -1)
    cv2.putText(panel, "105 x 68 m | P0 top-left | x right, y down", (14, panel.shape[0] - 13),
                cv2.FONT_HERSHEY_SIMPLEX, .36, white, 1, cv2.LINE_AA)
    return panel, scale, origin


def _draw_radar(
    image: np.ndarray, locations: list[dict[str, Any]], *,
    geometry_source: str, template: tuple[np.ndarray, float, tuple[int, int]],
) -> None:
    """Bottom-centred live ground positions, not skeletons or inferred team labels."""
    height, width = image.shape[:2]
    if width < 320 or height < 200:
        return  # Tiny diagnostic fixtures cannot accommodate a legible inset.
    base, scale, origin = template
    panel = base.copy()
    available = geometry_source in {"observed", "fallback"}
    plotted = 0
    if available:
        for row in locations:
            if not row["kept"] or row["projection_status"] != "projected":
                continue
            x, y = row["pitch_x_m"], row["pitch_y_m"]
            if x is None or y is None or not np.isfinite([x, y]).all():
                continue
            point = round(origin[0] + x * scale), round(origin[1] + y * scale)
            # Preserve the filter's small pitch margin; never clamp to a boundary.
            if not (4 <= point[0] < panel.shape[1] - 4 and
                    origin[1] - 8 <= point[1] < panel.shape[0] - 30):
                continue
            cv2.circle(panel, point, 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(panel, point, 4, (60, 220, 60), -1, cv2.LINE_AA)
            plotted += 1
    title = (f"ON-PITCH {plotted}" if available else "PITCH FIT UNAVAILABLE")
    if geometry_source == "fallback":
        title += " | FALLBACK FIT"
    cv2.putText(panel, title, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, .54,
                (235, 245, 235) if available else (0, 210, 255), 1, cv2.LINE_AA)
    margin = max(6, round(width / 160))
    factor = min(width * .28 / panel.shape[1], height * .32 / panel.shape[0])
    panel_width, panel_height = round(panel.shape[1] * factor), round(panel.shape[0] * factor)
    panel = cv2.resize(panel, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    x1, y1 = (width - panel_width) // 2, height - panel_height - margin
    image[y1:y1 + panel_height, x1:x1 + panel_width] = panel


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
    player_locations_parquet: str | Path | None = None,
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
    pitch_polygons: dict[int, list[list[float]]] = {}
    pitch_keypoints: dict[int, list[dict]] = {}
    geometry_sources: dict[int, str] = {}
    if pitch_geometry_json is not None:
        geometry_payload = json.loads(
            Path(pitch_geometry_json).read_text(encoding="utf-8")
        )
        for geometry_frame in geometry_payload.get("frames", []):
            geometry_sources[int(geometry_frame["frame_index"])] = (
                geometry_frame["source"] if geometry_frame.get("homography") is not None else "unavailable"
            )
            pitch_keypoints[int(geometry_frame["frame_index"])] = geometry_frame.get("predicted_keypoints", [])
            polygon = geometry_frame.get("pitch_polygon")
            if polygon is not None:
                pitch_polygons[int(geometry_frame["frame_index"])] = polygon

    radar_template = None
    locations_by_frame: dict[int, list[dict[str, Any]]] = {}
    if settings.draw_radar and pitch_geometry_json is not None:
        if player_locations_parquet is None:
            raise ValueError("Radar needs player-locations.parquet; export locations before rendering")
        for row in pq.read_table(player_locations_parquet, columns=[
            "frame_index", "pitch_x_m", "pitch_y_m", "kept", "projection_status",
        ]).to_pylist():
            locations_by_frame.setdefault(int(row["frame_index"]), []).append(row)
        radar_template = _radar_canvas()

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
                    pitch_polygons.get(source_packet.frame_index),
                    max(1, round(info.width / 960)),
                )
                _draw_pitch_keypoints(image, pitch_keypoints.get(source_packet.frame_index, []),
                                      max(1, round(info.width / 960)))
                _draw_predictions(
                    image,
                    predictions_by_frame.get(source_packet.frame_index, []),
                    model_id=model_id,
                    frame_index=source_packet.frame_index,
                    settings=settings,
                    classifications=classifications,
                )
                if radar_template is not None:
                    _draw_radar(
                        image, locations_by_frame.get(source_packet.frame_index, []),
                        geometry_source=geometry_sources.get(source_packet.frame_index, "unavailable"),
                        template=radar_template,
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


def render_report_videos(report_path: Path, output_root: Path) -> Path:
    """CPU-only re-render of saved summaries/screenings; preserve the source reports/videos."""
    from football_pose.configuration import PitchFilterConfig
    from football_pose.pitch_filter import PitchGeometryManifest, PreparedPitchGeometry
    from football_pose.pitch_locations import export_pitch_locations

    report = json.loads(report_path.read_text())
    conditions = (report["conditions"] if "conditions" in report else
                  [{"method": report.get("configuration", {}).get("name"), "summary": report}])
    rendered, skipped = [], []
    for condition in conditions:
        summary = condition.get("summary")
        if not summary:
            skipped.append({"condition": condition.get("method"), "reason": "no saved summary"})
            continue
        config = summary["configuration"]
        settings = VideoOutputConfig.model_validate(config.get("video_output", {}))
        geometry_path = summary.get("pitch_filter", {}).get("geometry")
        geometry = None
        if geometry_path is not None:
            path = Path(geometry_path)
            manifest = PitchGeometryManifest.model_validate_json(path.read_text())
            geometry = PreparedPitchGeometry(manifest.geometry_id, path, manifest, True)
        for job in summary.get("jobs", []):
            context = {"condition": condition.get("method"), "model_id": job["model_id"], "job_id": job["job_id"]}
            if job["status"] != "COMPLETE":
                skipped.append({**context, "reason": "job not COMPLETE"})
                continue
            outputs = job["outputs"]
            locations = None
            if geometry is not None and settings.draw_radar:
                locations = export_pitch_locations(
                    prediction_parquet=outputs["parquet"], decisions_parquet=outputs["pitch_decisions"],
                    geometry=geometry, config=PitchFilterConfig.model_validate(config["pitch_filter"]),
                    output_root=output_root / "locations",
                )
            identifier = stable_hash({
                "prediction_sha256": sha256_file(outputs["parquet"]),
                "decision_sha256": sha256_file(outputs["pitch_decisions"]) if outputs.get("pitch_decisions") else None,
                "geometry_sha256": sha256_file(geometry.path) if geometry else None,
                "locations_sha256": sha256_file(locations.player_parquet) if locations else None,
                "artifact_manifest_sha256": sha256_file(Path(summary["artifact"]) / "artifact.json"),
                "source_sha256": sha256_file(config["input"]),
                "settings": settings.model_dump(mode="json"), "model_id": job["model_id"],
                "renderer_sha256": sha256_file(Path(__file__)),
            })[:16]
            video_path = output_root / identifier / "video.mp4"
            if not video_path.is_file():
                render_annotated_video(
                    source_path=config["input"], artifact_path=summary["artifact"],
                    prediction_parquet=outputs["parquet"], output_path=video_path,
                    model_id=job["model_id"], settings=settings,
                    decision_parquet=outputs.get("pitch_decisions"),
                    pitch_geometry_json=geometry.path if geometry else None,
                    player_locations_parquet=locations.player_parquet if locations else None,
                )
            rendered.append({**context, "video": str(video_path)})
    if not rendered:
        raise ValueError("No completed jobs to render")
    index = output_root / f"videos-{stable_hash(str(report_path.resolve()))[:12]}.json"
    atomic_write_json(index, {"source_report": str(report_path), "videos": rendered, "skipped": skipped})
    return index
