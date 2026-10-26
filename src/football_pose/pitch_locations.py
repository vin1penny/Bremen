"""Auditable image-to-pitch locations; body-joint outputs are plane projections, not 3D."""
from __future__ import annotations

import csv
import json
import os
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from football_pose.archive import read_archive
from football_pose.artifacts import sha256_file, stable_hash
from football_pose.configuration import PitchFilterConfig
from football_pose.jobs import atomic_write_json
from football_pose.model_mapping import COCO_KEYPOINT_NAMES
from football_pose.pitch_filter import (
    PITCH_LENGTH_CM, PITCH_WIDTH_CM, PitchGeometryFrame, PitchGeometryManifest, PreparedPitchGeometry,
    _pitch_point, _write_parquet,
)

RECORD_FIELDS = (
    "experiment_id", "pipeline_id", "model_id", "source_video_id", "source_id", "person_id",
    "frame_index", "timestamp_seconds", "crop_id", "track_id",
)
COMMON_FIELDS = [
    pa.field("prediction_index", pa.int64(), nullable=False),
    *[pa.field(name, pa.string(), nullable=False) for name in (
        "experiment_id", "pipeline_id", "model_id", "source_video_id", "source_id", "person_id",
    )],
    pa.field("frame_index", pa.int64(), nullable=False),
    pa.field("timestamp_seconds", pa.float64(), nullable=False),
    pa.field("crop_id", pa.string()),
    pa.field("track_id", pa.string()),
    pa.field("classification", pa.string(), nullable=False),
    pa.field("kept", pa.bool_(), nullable=False),
    pa.field("geometry_source", pa.string(), nullable=False),
    pa.field("landmarks", pa.int64(), nullable=False),
    pa.field("inliers", pa.int64(), nullable=False),
    pa.field("median_reprojection_cm", pa.float64()),
]
COORDINATE_FIELDS = [
    pa.field("image_x_px", pa.float64(), nullable=False),
    pa.field("image_y_px", pa.float64(), nullable=False),
    pa.field("pitch_x_m", pa.float64()),
    pa.field("pitch_y_m", pa.float64()),
    pa.field("projection_status", pa.string(), nullable=False),
]
PLAYER_LOCATION_SCHEMA = pa.schema([
    *COMMON_FIELDS, *COORDINATE_FIELDS,
    pa.field("anchor_method", pa.string(), nullable=False),
    pa.field("pose_score", pa.float64(), nullable=False),
])
KEYPOINT_PROJECTION_SCHEMA = pa.schema([
    *COMMON_FIELDS,
    pa.field("keypoint_index", pa.int64(), nullable=False),
    pa.field("keypoint_name", pa.string(), nullable=False),
    pa.field("confidence", pa.float64(), nullable=False),
    pa.field("visibility", pa.float64()),
    *COORDINATE_FIELDS,
], metadata={b"meaning": b"image ray / pitch plane intersection; NOT a 3D joint position"})


@dataclass(frozen=True)
class PitchLocationResult:
    player_parquet: Path
    player_csv: Path
    keypoint_parquet: Path
    manifest: Path
    player_rows: int
    keypoint_rows: int
    cache_hit: bool
    wall_seconds: float

    def outputs(self) -> dict[str, str]:
        return {
            "player_locations": str(self.player_parquet),
            "player_locations_csv": str(self.player_csv),
            "keypoint_pitch_projections": str(self.keypoint_parquet),
            "pitch_locations_manifest": str(self.manifest),
        }


def _identity(row: dict[str, Any]) -> tuple[int, str, str]:
    return int(row["frame_index"]), row["source_id"], row["person_id"]


def _project(h: np.ndarray | None, x: float, y: float) -> dict[str, Any]:
    point = _pitch_point(h, x, y) if h is not None else None
    return {
        "image_x_px": x, "image_y_px": y,
        "pitch_x_m": point[0] / 100.0 if point is not None else None,
        "pitch_y_m": point[1] / 100.0 if point is not None else None,
        "projection_status": (
            "no_geometry" if h is None else "projected" if point is not None else "invalid_projection"
        ),
    }


def _write_joint_batches(rows: Iterable[dict[str, Any]], path: Path) -> tuple[int, int]:
    """Bound the extra memory for the 17x larger joint table; publish atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    count = projected = 0
    pending = []
    try:
        with pq.ParquetWriter(temporary, KEYPOINT_PROJECTION_SCHEMA, compression="zstd") as writer:
            for row in rows:
                pending.append(row)
                count += 1
                projected += row["projection_status"] == "projected"
                if len(pending) >= 8192:
                    writer.write_table(pa.Table.from_pylist(pending, schema=KEYPOINT_PROJECTION_SCHEMA))
                    pending.clear()
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=KEYPOINT_PROJECTION_SCHEMA))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return count, projected


def _location_rows(
    records: list[dict[str, Any]],
    decision_lookup: dict[tuple[int, str, str], dict[str, Any]],
    frames: dict[int, PitchGeometryFrame],
    threshold: float,
    players: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for prediction_index, record in enumerate(records):
        decision = decision_lookup[_identity(record)]
        frame = frames.get(int(record["frame_index"]))
        h = (np.asarray(frame.homography, dtype=np.float64).reshape(3, 3)
             if frame is not None and frame.homography is not None else None)
        common = {
            "prediction_index": prediction_index,
            **{name: record.get(name) for name in RECORD_FIELDS},
            "classification": decision["classification"], "kept": decision["kept"],
            "geometry_source": frame.source if frame is not None else "unavailable",
            "landmarks": frame.landmarks if frame is not None else 0,
            "inliers": frame.inliers if frame is not None else 0,
            "median_reprojection_cm": frame.median_reprojection_cm if frame is not None else None,
        }
        players.append({
            **common, **_project(h, decision["anchor_x"], decision["anchor_y"]),
            "anchor_method": decision["anchor_method"], "pose_score": decision["pose_score"],
        })
        for joint_index, keypoint in enumerate(record["keypoints"]):
            projection = _project(h, keypoint["x"], keypoint["y"])
            if keypoint["confidence"] <= 0 or keypoint["confidence"] < threshold:
                projection.update(pitch_x_m=None, pitch_y_m=None, projection_status="low_confidence")
            yield {
                **common, **projection, "keypoint_index": joint_index,
                "keypoint_name": COCO_KEYPOINT_NAMES[joint_index],
                "confidence": keypoint["confidence"], "visibility": keypoint.get("visibility"),
            }


def export_pitch_locations(
    *, prediction_parquet: str | Path, decisions_parquet: str | Path,
    geometry: PreparedPitchGeometry, config: PitchFilterConfig, output_root: str | Path,
) -> PitchLocationResult:
    """Use canonical original-frame pixels and the matching frame's accepted homography.

    Preserve every raw pose and all 17 joints. Use `kept` for the deduplicated on-pitch
    subset. Never clamp coordinates, fabricate missing fits or infer tracking IDs.
    """
    started = time.perf_counter()
    provenance = {
        "schema": "football-pose-pitch-locations-v1",
        "prediction_sha256": sha256_file(prediction_parquet),
        "decisions_sha256": sha256_file(decisions_parquet),
        "geometry_sha256": sha256_file(geometry.path),
        "geometry_id": geometry.identifier,
        "implementation_sha256": sha256_file(Path(__file__)),
        "projection_implementation_sha256": sha256_file(Path(__file__).with_name("pitch_filter.py")),
        "keypoint_confidence": config.location_keypoint_confidence,
        "pitch_size_m": [PITCH_LENGTH_CM / 100.0, PITCH_WIDTH_CM / 100.0],
    }
    directory = Path(output_root) / stable_hash(provenance)[:16]
    player_path = directory / "player-locations.parquet"
    csv_path = directory / "player-locations.csv"
    joint_path = directory / "keypoint-pitch-projections.parquet"
    manifest_path = directory / "manifest.json"
    if all(path.is_file() for path in (player_path, csv_path, joint_path, manifest_path)):
        manifest = json.loads(manifest_path.read_text())
        return PitchLocationResult(player_path, csv_path, joint_path, manifest_path,
                                   manifest["player_rows"], manifest["keypoint_rows"], True,
                                   time.perf_counter() - started)

    records = read_archive(prediction_parquet)
    decisions = pq.read_table(decisions_parquet).to_pylist()
    decision_lookup = {_identity(row): row for row in decisions}
    record_keys = [_identity(row) for row in records]
    if (len(decision_lookup) != len(decisions) or len(set(record_keys)) != len(records)
            or set(record_keys) != set(decision_lookup)):
        raise ValueError("Pitch decisions must match raw predictions one-to-one by frame/source/person ID")
    frames = {frame.frame_index: frame for frame in geometry.manifest.frames}
    players: list[dict[str, Any]] = []
    joint_count, projected_joint_count = _write_joint_batches(
        _location_rows(records, decision_lookup, frames, config.location_keypoint_confidence, players),
        joint_path,
    )
    _write_parquet(players, PLAYER_LOCATION_SCHEMA, player_path)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PLAYER_LOCATION_SCHEMA.names)
            writer.writeheader()
            writer.writerows(players)
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_write_json(manifest_path, {
        **provenance,
        "prediction_parquet": str(prediction_parquet),
        "decisions_parquet": str(decisions_parquet), "geometry": str(geometry.path),
        "pitch_checkpoint_sha256": geometry.manifest.checkpoint_sha256,
        "player_rows": len(players), "keypoint_rows": joint_count,
        "projected_players": sum(row["projection_status"] == "projected" for row in players),
        "projected_keypoints": projected_joint_count,
        "coordinates": {
            "venue": "Weserstadion, Bremen only", "units": "metres",
            "origin": "P0 corner; x towards P24 (length), y towards P5 (width)",
            "image_coordinates": "original frame pixels, x right, y down; NOT 640-model pixels",
            "player_location": "estimated ground anchor from pitch decisions; ankles then bbox fallback",
            "keypoint_projection": "intersection with the pitch plane; NOT true 3D joint or vertical ground projection",
            "tracking": "person_id is a detection ID; track_id is preserved, never generated",
            "orientation": "fixed landmark template, not team attack direction or a verified DFL coordinate convention",
            "missing": "null for missing/invalid geometry or low-confidence joints; never clamped to pitch",
        },
        "outputs": {"player_locations": str(player_path), "player_locations_csv": str(csv_path),
                    "keypoint_pitch_projections": str(joint_path)},
    })
    return PitchLocationResult(player_path, csv_path, joint_path, manifest_path,
                               len(players), joint_count, False, time.perf_counter() - started)


def export_report_locations(report_path: Path, output_root: Path) -> Path:
    """Backfill a summary.json or screening.json on CPU without touching old reports."""
    report = json.loads(report_path.read_text())
    conditions = (report["conditions"] if "conditions" in report else
                  [{"method": report.get("configuration", {}).get("name"), "summary": report}])
    jobs = []
    skipped_conditions = []
    for condition in conditions:
        if not condition.get("summary"):
            skipped_conditions.append({"condition": condition.get("method"), "reason": "no saved summary"})
            continue
        summary = condition["summary"]
        context = {"condition": condition.get("method"), "experiment_id": summary.get("experiment_id")}
        for job in summary.get("jobs", []):
            outputs = job.get("outputs", {})
            if job.get("status") != "COMPLETE":
                jobs.append({**context, "job_id": job["job_id"], "model_id": job["model_id"],
                             "status": "skipped", "reason": "job not COMPLETE"})
                continue
            if not summary.get("pitch_filter", {}).get("geometry") or not outputs.get("pitch_decisions"):
                raise ValueError("This report has no saved pitch geometry/decisions; run with pitch filtering first")
            geometry_path = Path(summary["pitch_filter"]["geometry"])
            manifest = PitchGeometryManifest.model_validate_json(geometry_path.read_text())
            geometry = PreparedPitchGeometry(manifest.geometry_id, geometry_path, manifest, True)
            config = PitchFilterConfig.model_validate(summary["configuration"]["pitch_filter"])
            result = export_pitch_locations(
                prediction_parquet=outputs["parquet"], decisions_parquet=outputs["pitch_decisions"],
                geometry=geometry, config=config, output_root=output_root,
            )
            jobs.append({**context, "job_id": job["job_id"], "model_id": job["model_id"],
                         "status": "COMPLETE", "outputs": result.outputs()})
    if not any(job["status"] == "COMPLETE" for job in jobs):
        raise ValueError("No completed jobs with saved predictions to export")
    index = output_root / f"locations-{stable_hash(str(report_path.resolve()))[:12]}.json"
    atomic_write_json(index, {"source_report": str(report_path), "jobs": jobs,
                             "skipped_conditions": skipped_conditions})
    return index
