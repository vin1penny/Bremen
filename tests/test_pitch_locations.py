from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from football_pose.archive import PREDICTION_SCHEMA
from football_pose.cli import app
from football_pose.configuration import ExperimentConfig, PitchFilterConfig
from football_pose.experiments import ExperimentRunner
from football_pose.domain import FramePacket
from football_pose.model_mapping import canonical_keypoints
from football_pose.pitch_filter import (
    PitchGeometryFrame, PitchGeometryManifest, PitchGeometryStore,
    PreparedPitchGeometry, postprocess_pitch_predictions,
)
from football_pose.pitch_locations import export_pitch_locations


def _geometry(tmp_path: Path) -> PreparedPitchGeometry:
    frames = [
        PitchGeometryFrame(
            frame_index=index, homography=h, source="observed" if h is not None else "unavailable",
            landmarks=8 if h is not None else 2, inliers=8 if h is not None else 0,
            median_reprojection_cm=1.0 if h is not None else None,
        )
        for index, h in enumerate([
            [100., 0., 0., 0., 100., 0., 0., 0., 1.],
            [100., 0., -1000., 0., 100., -2000., 0., 0., 1.],
            None,
            [100., 0., 0., 0., 100., 0., 1., 0., -20.],
        ])
    ]
    manifest = PitchGeometryManifest(
        geometry_id="geometry", source_sha256="video", checkpoint_sha256="checkpoint",
        settings={}, frame_count=4, observed_frames=3, bbox_frames=0,
        usable_frames=3, processing_seconds=.1, frames=frames,
    )
    path = tmp_path / "geometry.json"
    path.write_text(manifest.model_dump_json())
    return PreparedPitchGeometry("geometry", path, manifest, True)


def _row(frame: int, source: str = "frame", x: float = 20., y: float = 30.) -> dict:
    return {
        "experiment_id": "experiment", "pipeline_id": "pipeline", "model_id": "model",
        "source_video_id": "video", "source_id": source, "frame_index": frame,
        "timestamp_seconds": frame / 25., "crop_id": source, "track_id": None,
        "person_id": "person", "source_bbox": [x - 5, y - 10, x + 5, y],
        "keypoints": [{"x": x, "y": y, "confidence": .9, "visibility": None} for _ in range(17)],
        "coordinate_space": "original_frame", "inference_time_ms": 1.,
    }


def _prepare(tmp_path: Path, rows: list[dict]):
    raw = tmp_path / "raw.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA), raw)
    geometry = _geometry(tmp_path)
    config = PitchFilterConfig(enabled=True, checkpoint=tmp_path / "unused.pt")
    filtered = postprocess_pitch_predictions(
        prediction_parquet=raw, geometry=geometry, config=config, output_root=tmp_path / "filter",
    )
    return raw, geometry, config, filtered


def test_locations_use_source_pixels_and_matching_camera_frame(tmp_path):
    # Deliberately reverse frame order; decisions are sorted, archive is not.
    rows = [_row(1, x=30., y=50.), _row(0), _row(2), _row(3)]
    rows[1]["keypoints"][0]["confidence"] = .1
    raw, geometry, config, filtered = _prepare(tmp_path, rows)
    result = export_pitch_locations(
        prediction_parquet=raw, decisions_parquet=filtered.decisions_parquet,
        geometry=geometry, config=config, output_root=tmp_path / "locations",
    )
    players = pq.read_table(result.player_parquet).to_pylist()
    joints = pq.read_table(result.keypoint_parquet).to_pylist()
    assert result.player_rows == 4 and result.keypoint_rows == 68
    assert [row["frame_index"] for row in players] == [1, 0, 2, 3]
    for player in players[:2]:
        assert (player["pitch_x_m"], player["pitch_y_m"]) == (20., 30.)
        assert player["anchor_method"] == "ankles"
        assert player["kept"] and player["track_id"] is None
    assert players[0]["image_x_px"] == 30.
    assert players[0]["timestamp_seconds"] == .04
    assert players[2]["projection_status"] == "no_geometry"
    assert players[2]["pitch_x_m"] is None
    assert players[3]["projection_status"] == "invalid_projection"
    assert players[3]["pitch_y_m"] is None
    nose = joints[17]
    assert nose["keypoint_name"] == "nose" and nose["keypoint_index"] == 0
    assert nose["confidence"] == .1 and nose["projection_status"] == "low_confidence"
    assert nose["pitch_x_m"] is None
    assert joints[18]["pitch_x_m"] == 20.
    with result.player_csv.open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 4 and csv_rows[0]["pitch_x_m"] == "20.0"
    manifest = json.loads(result.manifest.read_text())
    assert manifest["pitch_size_m"] == [105., 68.]
    assert "NOT true 3D" in manifest["coordinates"]["keypoint_projection"]
    assert "P0" in manifest["coordinates"]["origin"]
    assert not result.cache_hit
    cached = export_pitch_locations(
        prediction_parquet=raw, decisions_parquet=filtered.decisions_parquet,
        geometry=geometry, config=config, output_root=tmp_path / "locations",
    )
    assert cached.cache_hit and cached.player_parquet == result.player_parquet
    config.location_keypoint_confidence = .05
    updated = export_pitch_locations(
        prediction_parquet=raw, decisions_parquet=filtered.decisions_parquet,
        geometry=geometry, config=config, output_root=tmp_path / "locations",
    )
    assert updated.player_parquet != result.player_parquet
    assert pq.read_table(updated.keypoint_parquet).to_pylist()[17]["projection_status"] == "projected"


def test_outside_and_duplicate_locations_retained_without_clamping(tmp_path):
    rows = [_row(0, "tile-a"), _row(0, "tile-b"), _row(0, "outside", x=120., y=80.)]
    # No confident ankles: existing filter uses bbox bottom centre (upstream radar anchor).
    for row in rows:
        for joint in row["keypoints"][15:]:
            joint["confidence"] = 0.
    raw, geometry, config, filtered = _prepare(tmp_path, rows)
    result = export_pitch_locations(
        prediction_parquet=raw, decisions_parquet=filtered.decisions_parquet,
        geometry=geometry, config=config, output_root=tmp_path / "locations",
    )
    players = pq.read_table(result.player_parquet).to_pylist()
    assert [p["classification"] for p in players] == ["inside", "duplicate", "outside"]
    assert [p["kept"] for p in players] == [True, False, False]
    assert players[1]["pitch_x_m"] == 20.  # Diagnostic duplicate, not counted twice.
    assert players[2]["pitch_x_m"] == 120. and players[2]["pitch_y_m"] == 80.
    assert all(p["anchor_method"] == "bbox_bottom_center" for p in players)


def test_crop_and_resize_are_reversed_before_projective_pitch_mapping(tmp_path):
    packet = FramePacket(
        image=np.zeros((32, 32, 3), dtype=np.uint8), frame_index=0, timestamp_seconds=0.,
        source_width=1920, source_height=1080,
        to_source=np.array([[2., 0., 100.], [0., 2., 50.], [0., 0., 1.]]),
    )
    keypoints = canonical_keypoints(packet, np.tile([10., 20.], (17, 1)), np.ones(17))
    assert (keypoints[0].x, keypoints[0].y) == (120., 90.)
    record = _row(0, x=120., y=90.)
    record["keypoints"] = [point.model_dump() for point in keypoints]
    raw = tmp_path / "raw.parquet"
    pq.write_table(pa.Table.from_pylist([record], schema=PREDICTION_SCHEMA), raw)
    h = np.array([[50., 3., 100.], [2., 40., 200.], [.001, .002, 1.]])
    geometry = _geometry(tmp_path)
    geometry.manifest.frames[0].homography = h.reshape(-1).tolist()
    geometry.path.write_text(geometry.manifest.model_dump_json())
    config = PitchFilterConfig(enabled=True, checkpoint=tmp_path / "unused.pt")
    filtered = postprocess_pitch_predictions(
        prediction_parquet=raw, geometry=geometry, config=config, output_root=tmp_path / "filter",
    )
    result = export_pitch_locations(
        prediction_parquet=raw, decisions_parquet=filtered.decisions_parquet,
        geometry=geometry, config=config, output_root=tmp_path / "locations",
    )
    projected = pq.read_table(result.keypoint_parquet).to_pylist()[0]
    expected_m = cv2.perspectiveTransform(np.array([[[120., 90.]]]), h)[0, 0] / 100.
    assert [projected["pitch_x_m"], projected["pitch_y_m"]] == pytest.approx(expected_m)
    assert projected["image_x_px"] == 120. and projected["image_y_px"] == 90.


def test_export_rejects_unmatched_decisions_and_handles_empty_predictions(tmp_path):
    raw, geometry, config, filtered = _prepare(tmp_path, [_row(0)])
    pq.write_table(pq.read_table(filtered.decisions_parquet).slice(0, 0), filtered.decisions_parquet)
    with pytest.raises(ValueError, match="one-to-one"):
        export_pitch_locations(
            prediction_parquet=raw, decisions_parquet=filtered.decisions_parquet,
            geometry=geometry, config=config, output_root=tmp_path / "locations",
        )
    empty = tmp_path / "empty"
    empty.mkdir()
    raw, geometry, config, filtered = _prepare(empty, [])
    result = export_pitch_locations(
        prediction_parquet=raw, decisions_parquet=filtered.decisions_parquet,
        geometry=geometry, config=config, output_root=tmp_path / "locations",
    )
    assert pq.read_table(result.player_parquet).num_rows == 0
    assert pq.read_table(result.keypoint_parquet).num_rows == 0
    assert result.player_csv.read_text().startswith("prediction_index,")


def test_summary_and_screening_backfill_require_no_weights_or_gpu(tmp_path):
    raw, geometry, config, filtered = _prepare(tmp_path, [_row(0)])
    summary = {
        "configuration": {"pitch_filter": config.model_dump(mode="json")},
        "pitch_filter": {"geometry": str(geometry.path)},
        "jobs": [{"job_id": "job", "model_id": "model", "status": "COMPLETE",
                  "outputs": {"parquet": str(raw), "pitch_decisions": str(filtered.decisions_parquet)}}],
    }
    for name, report in [("summary", summary), ("screening", {"conditions": [{"summary": summary}]})]:
        path = tmp_path / f"{name}.json"
        original = json.dumps(report)
        path.write_text(original)
        result = CliRunner().invoke(app, ["export-pitch-locations", str(path),
                                         "--output-root", str(tmp_path / name)])
        assert result.exit_code == 0, result.output
        assert "Location exports:" in result.output
        assert path.read_text() == original
        index = json.loads(next((tmp_path / name).glob("locations-*.json")).read_text())
        assert Path(index["jobs"][0]["outputs"]["player_locations"]).is_file()


def test_pipeline_adds_location_paths_to_summary(tmp_path, tiny_video, monkeypatch):
    geometry = _geometry(tmp_path)
    monkeypatch.setattr(PitchGeometryStore, "prepare", lambda *args, **kwargs: geometry)
    repository = Path(__file__).resolve().parents[1]
    config = ExperimentConfig.model_validate({
        "name": "locations", "input": tiny_video, "output_dir": tmp_path / "output",
        "cache": {"root": tmp_path / "cache", "format": "png_shards"},
        "pitch_filter": {"enabled": True, "checkpoint": str(tmp_path / "unused.pt")},
        "models": [{"id": "mock", "command": ["{python}", str(repository / "runners/mock/run.py")]}],
    })
    report = ExperimentRunner(config).run()
    assert report["success"], report["jobs"]
    outputs = report["jobs"][0]["outputs"]
    assert pq.read_table(outputs["player_locations"]).num_rows == 6
    assert pq.read_table(outputs["keypoint_pitch_projections"]).num_rows == 6 * 17
    assert "pitch_locations_seconds" in report["jobs"][0]["timings"]
    resumed = ExperimentRunner(config).run()
    assert resumed["jobs"][0]["attempts"] == 1
    assert resumed["jobs"][0]["outputs"]["player_locations"] == outputs["player_locations"]
