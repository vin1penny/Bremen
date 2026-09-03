from __future__ import annotations

import gc
import json
import os
import shutil
import time
import uuid
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator

from football_pose.archive import PREDICTION_SCHEMA, read_archive
from football_pose.artifacts import sha256_file, stable_hash
from football_pose.configuration import PitchFilterConfig
from football_pose.jobs import atomic_write_json
from football_pose.video import VideoSource


PITCH_LENGTH_CM = 12_000.0
PITCH_WIDTH_CM = 7_000.0
PITCH_VERTICES = np.asarray(
    [
        (0, 0),
        (0, 1450),
        (0, 2584),
        (0, 4416),
        (0, 5550),
        (0, 7000),
        (550, 2584),
        (550, 4416),
        (1100, 3500),
        (2015, 1450),
        (2015, 2584),
        (2015, 4416),
        (2015, 5550),
        (6000, 0),
        (6000, 2585),
        (6000, 4415),
        (6000, 7000),
        (9985, 1450),
        (9985, 2584),
        (9985, 4416),
        (9985, 5550),
        (10900, 3500),
        (11450, 2584),
        (11450, 4416),
        (12000, 0),
        (12000, 1450),
        (12000, 2584),
        (12000, 4416),
        (12000, 5550),
        (12000, 7000),
        (5085, 3500),
        (6915, 3500),
    ],
    dtype=np.float64,
)


class PitchGeometryFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    homography: list[float] | None = None
    pitch_bbox: tuple[float, float, float, float] | None = None
    source: Literal["observed", "fallback", "bbox_only", "unavailable"]
    landmarks: int = Field(ge=0)
    inliers: int = Field(ge=0)
    median_reprojection_cm: float | None = Field(default=None, ge=0.0)

    @field_validator("homography")
    @classmethod
    def valid_homography(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and (len(value) != 9 or not np.isfinite(value).all()):
            raise ValueError("homography must contain nine finite values")
        return value

    @field_validator("pitch_bbox")
    @classmethod
    def valid_pitch_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        x1, y1, x2, y2 = value
        if not np.isfinite(value).all() or x2 < x1 or y2 < y1:
            raise ValueError("pitch_bbox must contain finite, ordered coordinates")
        return tuple(float(item) for item in value)


class PitchGeometryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "football-pose-pitch-geometry-v1"
    geometry_id: str
    source_sha256: str
    checkpoint_sha256: str
    settings: dict[str, Any]
    frame_count: int = Field(ge=0)
    observed_frames: int = Field(ge=0)
    bbox_frames: int = Field(ge=0)
    usable_frames: int = Field(ge=0)
    processing_seconds: float = Field(ge=0.0)
    frames: list[PitchGeometryFrame]


@dataclass(frozen=True, slots=True)
class PreparedPitchGeometry:
    identifier: str
    path: Path
    manifest: PitchGeometryManifest
    cache_hit: bool

    def homographies(self) -> dict[int, np.ndarray]:
        return {
            frame.frame_index: np.asarray(frame.homography, dtype=np.float64).reshape(3, 3)
            for frame in self.manifest.frames
            if frame.homography is not None
        }


@dataclass(frozen=True, slots=True)
class PitchPostprocessResult:
    directory: Path
    on_pitch_parquet: Path
    decisions_parquet: Path
    manifest_path: Path
    raw_records: int
    deduplicated_records: int
    duplicate_records: int
    on_pitch_records: int
    outside_pitch_records: int
    unclassified_records: int
    wall_seconds: float


DECISION_SCHEMA = pa.schema(
    [
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("person_id", pa.string(), nullable=False),
        pa.field("anchor_x", pa.float64(), nullable=False),
        pa.field("anchor_y", pa.float64(), nullable=False),
        pa.field("anchor_method", pa.string(), nullable=False),
        pa.field("pitch_x_cm", pa.float64(), nullable=True),
        pa.field("pitch_y_cm", pa.float64(), nullable=True),
        pa.field("classification", pa.string(), nullable=False),
        pa.field("filter_method", pa.string(), nullable=False),
        pa.field("kept", pa.bool_(), nullable=False),
        pa.field("pose_score", pa.float64(), nullable=False),
    ]
)


def _geometry_settings(config: PitchFilterConfig) -> dict[str, Any]:
    return {
        "image_size": config.image_size,
        "detection_confidence": config.detection_confidence,
        "landmark_confidence": config.landmark_confidence,
        "minimum_landmarks": config.minimum_landmarks,
        "ransac_threshold_cm": config.ransac_threshold_cm,
        "minimum_inlier_ratio": config.minimum_inlier_ratio,
        "maximum_median_reprojection_cm": config.maximum_median_reprojection_cm,
        "max_fallback_frames": config.max_fallback_frames,
    }


def _fit_homography(
    result: Any, config: PitchFilterConfig
) -> tuple[np.ndarray | None, int, int, float | None]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return None, 0, 0, None
    xy = keypoints.xy.detach().cpu().numpy()
    if xy.ndim != 3 or xy.shape[0] == 0 or xy.shape[1] != len(PITCH_VERTICES):
        return None, 0, 0, None
    confidence_value = getattr(keypoints, "conf", None)
    if confidence_value is None:
        data = keypoints.data.detach().cpu().numpy()
        confidence = data[..., 2] if data.shape[-1] >= 3 else np.ones(xy.shape[:2])
    else:
        confidence = confidence_value.detach().cpu().numpy()
    candidate_counts = np.sum(confidence >= config.landmark_confidence, axis=1)
    candidate_index = int(np.argmax(candidate_counts))
    selected = confidence[candidate_index] >= config.landmark_confidence
    landmark_count = int(selected.sum())
    if landmark_count < config.minimum_landmarks:
        return None, landmark_count, 0, None
    image_points = xy[candidate_index][selected].astype(np.float64)
    pitch_points = PITCH_VERTICES[selected]
    try:
        homography, mask = cv2.findHomography(
            image_points,
            pitch_points,
            cv2.RANSAC,
            config.ransac_threshold_cm,
        )
    except cv2.error:
        return None, landmark_count, 0, None
    inliers = int(mask.sum()) if mask is not None else 0
    median_reprojection: float | None = None
    if homography is not None:
        projected = cv2.perspectiveTransform(
            image_points.reshape(1, -1, 2).astype(np.float32),
            homography.astype(np.float32),
        )[0]
        errors = np.linalg.norm(projected - pitch_points, axis=1)
        median_reprojection = float(np.median(errors))
    inlier_ratio = inliers / landmark_count
    if (
        homography is None
        or inliers < config.minimum_landmarks
        or inlier_ratio < config.minimum_inlier_ratio
        or median_reprojection is None
        or median_reprojection > config.maximum_median_reprojection_cm
        or not np.isfinite(homography).all()
        or abs(float(np.linalg.det(homography))) < 1e-12
        or abs(float(homography[2, 2])) < 1e-12
    ):
        return None, landmark_count, inliers, median_reprojection
    homography = homography / homography[2, 2]
    return homography.astype(np.float64), landmark_count, inliers, median_reprojection


def _pitch_bbox(result: Any) -> tuple[float, float, float, float] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    coordinates = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    candidate_index = int(np.argmax(confidences))
    bbox = tuple(float(value) for value in coordinates[candidate_index])
    if len(bbox) != 4 or not np.isfinite(bbox).all():
        return None
    return bbox  # type: ignore[return-value]


def _fill_short_gaps(
    frames: list[PitchGeometryFrame], max_fallback_frames: int
) -> list[PitchGeometryFrame]:
    if max_fallback_frames == 0:
        return frames
    valid_indices = [frame.frame_index for frame in frames if frame.homography is not None]
    if not valid_indices:
        return frames
    by_index = {frame.frame_index: frame for frame in frames}
    output: list[PitchGeometryFrame] = []
    for frame in frames:
        if frame.homography is not None:
            output.append(frame)
            continue
        position = bisect_left(valid_indices, frame.frame_index)
        candidates: list[int] = []
        if position > 0:
            candidates.append(valid_indices[position - 1])
        if position < len(valid_indices):
            candidates.append(valid_indices[position])
        nearest = min(candidates, key=lambda value: abs(value - frame.frame_index))
        if abs(nearest - frame.frame_index) <= max_fallback_frames:
            source = by_index[nearest]
            output.append(
                PitchGeometryFrame(
                    frame_index=frame.frame_index,
                    homography=source.homography,
                    pitch_bbox=frame.pitch_bbox or source.pitch_bbox,
                    source="fallback",
                    landmarks=frame.landmarks,
                    inliers=frame.inliers,
                    median_reprojection_cm=frame.median_reprojection_cm,
                )
            )
        else:
            output.append(frame)
    return output


def _release_pitch_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class PitchGeometryStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def prepare(
        self,
        *,
        source_path: str | Path,
        source_sha256: str,
        config: PitchFilterConfig,
    ) -> PreparedPitchGeometry:
        if config.checkpoint is None:
            raise ValueError("pitch-filter checkpoint is required")
        checkpoint_sha256 = sha256_file(config.checkpoint)
        settings = _geometry_settings(config)
        identifier = stable_hash(
            {
                "source_sha256": source_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "settings": settings,
                "implementation_sha256": sha256_file(Path(__file__)),
            }
        )
        target = self.root / identifier
        manifest_path = target / "geometry.json"
        if manifest_path.is_file():
            manifest = PitchGeometryManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            return PreparedPitchGeometry(identifier, manifest_path, manifest, True)

        lock_path = self.root / f".{identifier}.lock"
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(
                f"pitch geometry is currently being written: {identifier}"
            ) from error
        os.close(lock_fd)
        temporary = self.root / f".{identifier}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        start = time.perf_counter()
        model: Any | None = None
        try:
            from ultralytics import YOLO

            model = YOLO(str(config.checkpoint))
            frames: list[PitchGeometryFrame] = []
            packets: list[Any] = []

            def process_batch() -> None:
                if not packets:
                    return
                results = model.predict(
                    [packet.image for packet in packets],
                    imgsz=config.image_size,
                    conf=config.detection_confidence,
                    device=config.device,
                    verbose=False,
                )
                if len(results) != len(packets):
                    raise RuntimeError("pitch model returned an unexpected result count")
                for packet, result in zip(packets, results, strict=True):
                    homography, landmarks, inliers, median_reprojection = _fit_homography(
                        result, config
                    )
                    pitch_bbox = _pitch_bbox(result)
                    frames.append(
                        PitchGeometryFrame(
                            frame_index=packet.frame_index,
                            homography=(
                                homography.reshape(-1).tolist()
                                if homography is not None
                                else None
                            ),
                            pitch_bbox=pitch_bbox,
                            source=(
                                "observed"
                                if homography is not None
                                else "bbox_only"
                                if pitch_bbox is not None
                                else "unavailable"
                            ),
                            landmarks=landmarks,
                            inliers=inliers,
                            median_reprojection_cm=median_reprojection,
                        )
                    )
                packets.clear()

            for packet in VideoSource(source_path).frames():
                packets.append(packet)
                if len(packets) >= config.batch_size:
                    process_batch()
            process_batch()
            frames = _fill_short_gaps(frames, config.max_fallback_frames)
            manifest = PitchGeometryManifest(
                geometry_id=identifier,
                source_sha256=source_sha256,
                checkpoint_sha256=checkpoint_sha256,
                settings=settings,
                frame_count=len(frames),
                observed_frames=sum(frame.source == "observed" for frame in frames),
                bbox_frames=sum(frame.pitch_bbox is not None for frame in frames),
                usable_frames=sum(
                    frame.homography is not None or frame.pitch_bbox is not None
                    for frame in frames
                ),
                processing_seconds=time.perf_counter() - start,
                frames=frames,
            )
            temporary_manifest = temporary / "geometry.json"
            temporary_manifest.write_text(manifest.model_dump_json(), encoding="utf-8")
            if target.exists():
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, target)
            return PreparedPitchGeometry(
                identifier,
                target / "geometry.json",
                PitchGeometryManifest.model_validate_json(
                    (target / "geometry.json").read_text(encoding="utf-8")
                ),
                False,
            )
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        finally:
            if model is not None:
                _release_pitch_model(model)
            lock_path.unlink(missing_ok=True)


def _pose_score(record: dict[str, Any]) -> float:
    confidences = sorted(
        (float(keypoint["confidence"]) for keypoint in record["keypoints"]), reverse=True
    )
    strongest = confidences[:5]
    return float(sum(strongest) / len(strongest)) if strongest else 0.0


def _pose_bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = record.get("source_bbox")
    if bbox is not None:
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]
    visible = [
        keypoint for keypoint in record["keypoints"] if float(keypoint["confidence"]) > 0
    ]
    if not visible:
        return None
    xs = [float(keypoint["x"]) for keypoint in visible]
    ys = [float(keypoint["y"]) for keypoint in visible]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _cross_region_duplicates(
    records: list[dict[str, Any]], threshold: float
) -> set[int]:
    duplicates: set[int] = set()
    selected: list[int] = []
    ordered = sorted(range(len(records)), key=lambda index: (-_pose_score(records[index]), index))
    bboxes = [_pose_bbox(record) for record in records]
    for index in ordered:
        bbox = bboxes[index]
        is_duplicate = False
        if bbox is not None:
            for kept_index in selected:
                if records[index]["source_id"] == records[kept_index]["source_id"]:
                    continue
                kept_bbox = bboxes[kept_index]
                if kept_bbox is not None and _iou(bbox, kept_bbox) >= threshold:
                    is_duplicate = True
                    break
        if is_duplicate:
            duplicates.add(index)
        else:
            selected.append(index)
    return duplicates


def _ground_anchor(
    record: dict[str, Any], ankle_confidence: float
) -> tuple[float, float, str]:
    ankles = [
        keypoint
        for keypoint in (record["keypoints"][15], record["keypoints"][16])
        if float(keypoint["confidence"]) >= ankle_confidence
    ]
    if ankles:
        return (
            float(sum(float(keypoint["x"]) for keypoint in ankles) / len(ankles)),
            float(sum(float(keypoint["y"]) for keypoint in ankles) / len(ankles)),
            "ankles" if len(ankles) == 2 else "ankle",
        )
    bbox = _pose_bbox(record)
    if bbox is not None:
        return (bbox[0] + bbox[2]) / 2, bbox[3], "bbox_bottom_center"
    keypoints = record["keypoints"]
    return (
        float(sum(float(keypoint["x"]) for keypoint in keypoints) / len(keypoints)),
        float(sum(float(keypoint["y"]) for keypoint in keypoints) / len(keypoints)),
        "keypoint_centroid",
    )


def _pitch_point(homography: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    transformed = homography @ np.asarray([x, y, 1.0], dtype=np.float64)
    if abs(float(transformed[2])) < 1e-12:
        return None
    point = transformed[:2] / transformed[2]
    if not np.isfinite(point).all():
        return None
    return float(point[0]), float(point[1])


def _write_parquet(rows: list[dict[str, Any]], schema: pa.Schema, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def postprocess_pitch_predictions(
    *,
    prediction_parquet: str | Path,
    output_root: str | Path,
    geometry: PreparedPitchGeometry,
    config: PitchFilterConfig,
) -> PitchPostprocessResult:
    start = time.perf_counter()
    settings_id = stable_hash(
        {
            "geometry_id": geometry.identifier,
            "pitch_margin_cm": config.pitch_margin_cm,
            "pitch_bbox_margin_px": config.pitch_bbox_margin_px,
            "ankle_confidence": config.ankle_confidence,
            "deduplication_iou": config.deduplication_iou,
        }
    )[:16]
    directory = Path(output_root) / settings_id
    on_pitch_path = directory / "predictions-on-pitch.parquet"
    decisions_path = directory / "pitch-decisions.parquet"
    manifest_path = directory / "manifest.json"
    if on_pitch_path.is_file() and decisions_path.is_file() and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts = payload["counts"]
        return PitchPostprocessResult(
            directory=directory,
            on_pitch_parquet=on_pitch_path,
            decisions_parquet=decisions_path,
            manifest_path=manifest_path,
            raw_records=int(counts["raw"]),
            deduplicated_records=int(counts["deduplicated"]),
            duplicate_records=int(counts["duplicates"]),
            on_pitch_records=int(counts["on_pitch"]),
            outside_pitch_records=int(counts["outside_pitch"]),
            unclassified_records=int(counts["unclassified"]),
            wall_seconds=0.0,
        )

    records = read_archive(prediction_parquet)
    geometry_frames = {
        frame.frame_index: frame for frame in geometry.manifest.frames
    }
    frame_groups: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        frame_groups.setdefault(int(record["frame_index"]), []).append(record)

    decisions: list[dict[str, Any]] = []
    on_pitch: list[dict[str, Any]] = []
    duplicate_count = 0
    outside_count = 0
    unclassified_count = 0
    for frame_index in sorted(frame_groups):
        frame_records = frame_groups[frame_index]
        duplicates = _cross_region_duplicates(frame_records, config.deduplication_iou)
        geometry_frame = geometry_frames.get(frame_index)
        homography = (
            np.asarray(geometry_frame.homography, dtype=np.float64).reshape(3, 3)
            if geometry_frame is not None and geometry_frame.homography is not None
            else None
        )
        pitch_bbox = geometry_frame.pitch_bbox if geometry_frame is not None else None
        for index, record in enumerate(frame_records):
            anchor_x, anchor_y, anchor_method = _ground_anchor(
                record, config.ankle_confidence
            )
            score = _pose_score(record)
            pitch_x: float | None = None
            pitch_y: float | None = None
            if index in duplicates:
                classification = "duplicate"
                filter_method = "cross_region_iou"
                duplicate_count += 1
            elif geometry_frame is None:
                classification = "unavailable"
                filter_method = "unavailable"
                unclassified_count += 1
            else:
                bbox_inside = True
                if pitch_bbox is not None:
                    margin_px = config.pitch_bbox_margin_px
                    bbox_inside = (
                        pitch_bbox[0] - margin_px <= anchor_x <= pitch_bbox[2] + margin_px
                        and pitch_bbox[1] - margin_px
                        <= anchor_y
                        <= pitch_bbox[3] + margin_px
                    )
                if pitch_bbox is not None and not bbox_inside:
                    classification = "outside"
                    filter_method = "pitch_bbox"
                    outside_count += 1
                elif homography is not None:
                    point = _pitch_point(homography, anchor_x, anchor_y)
                    if point is None:
                        if pitch_bbox is not None:
                            classification = "inside"
                            filter_method = "pitch_bbox_fallback"
                            on_pitch.append(record)
                        else:
                            classification = "unavailable"
                            filter_method = "unavailable"
                            unclassified_count += 1
                    else:
                        pitch_x, pitch_y = point
                        margin = config.pitch_margin_cm
                        inside = -margin <= pitch_y <= PITCH_WIDTH_CM + margin
                        classification = "inside" if inside else "outside"
                        filter_method = "touchline_homography"
                        if inside:
                            on_pitch.append(record)
                        else:
                            outside_count += 1
                elif pitch_bbox is not None:
                    classification = "inside"
                    filter_method = "pitch_bbox"
                    on_pitch.append(record)
                else:
                    classification = "unavailable"
                    filter_method = "unavailable"
                    unclassified_count += 1
            decisions.append(
                {
                    "frame_index": frame_index,
                    "source_id": str(record["source_id"]),
                    "person_id": str(record["person_id"]),
                    "anchor_x": anchor_x,
                    "anchor_y": anchor_y,
                    "anchor_method": anchor_method,
                    "pitch_x_cm": pitch_x,
                    "pitch_y_cm": pitch_y,
                    "classification": classification,
                    "filter_method": filter_method,
                    "kept": classification == "inside",
                    "pose_score": score,
                }
            )

    directory.mkdir(parents=True, exist_ok=True)
    _write_parquet(on_pitch, PREDICTION_SCHEMA, on_pitch_path)
    _write_parquet(decisions, DECISION_SCHEMA, decisions_path)
    counts = {
        "raw": len(records),
        "deduplicated": len(records) - duplicate_count,
        "duplicates": duplicate_count,
        "on_pitch": len(on_pitch),
        "outside_pitch": outside_count,
        "unclassified": unclassified_count,
    }
    atomic_write_json(
        manifest_path,
        {
            "schema": "football-pose-pitch-filter-v1",
            "geometry": str(geometry.path),
            "settings": {
                "pitch_margin_cm": config.pitch_margin_cm,
                "pitch_bbox_margin_px": config.pitch_bbox_margin_px,
                "ankle_confidence": config.ankle_confidence,
                "deduplication_iou": config.deduplication_iou,
            },
            "counts": counts,
            "raw_prediction_parquet": str(Path(prediction_parquet)),
            "on_pitch_prediction_parquet": str(on_pitch_path),
            "decision_parquet": str(decisions_path),
        },
    )
    return PitchPostprocessResult(
        directory=directory,
        on_pitch_parquet=on_pitch_path,
        decisions_parquet=decisions_path,
        manifest_path=manifest_path,
        raw_records=len(records),
        deduplicated_records=len(records) - duplicate_count,
        duplicate_records=duplicate_count,
        on_pitch_records=len(on_pitch),
        outside_pitch_records=outside_count,
        unclassified_records=unclassified_count,
        wall_seconds=time.perf_counter() - start,
    )
