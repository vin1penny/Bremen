from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from football_pose.archive import PREDICTION_SCHEMA, read_archive
from football_pose.configuration import PitchFilterConfig
from football_pose.pitch_filter import (
    PitchGeometryFrame,
    PitchGeometryManifest,
    PreparedPitchGeometry,
    _fill_short_gaps,
    postprocess_pitch_predictions,
)


def _prediction(
    *, source_id: str, person_id: str, bbox: tuple[float, float, float, float]
) -> dict[str, object]:
    return {
        "experiment_id": "experiment",
        "pipeline_id": "pipeline",
        "model_id": "model",
        "source_video_id": "video",
        "source_id": source_id,
        "frame_index": 0,
        "timestamp_seconds": 0.0,
        "crop_id": source_id,
        "track_id": None,
        "person_id": person_id,
        "source_bbox": list(bbox),
        "keypoints": [
            {"x": 0.0, "y": 0.0, "confidence": 0.0, "visibility": None}
            for _ in range(17)
        ],
        "coordinate_space": "original_frame",
        "inference_time_ms": 1.0,
    }


def _geometry(tmp_path: Path) -> PreparedPitchGeometry:
    manifest = PitchGeometryManifest(
        geometry_id="geometry",
        source_sha256="source",
        checkpoint_sha256="checkpoint",
        settings={},
        frame_count=1,
        observed_frames=1,
        bbox_frames=1,
        usable_frames=1,
        processing_seconds=0.1,
        frames=[
            PitchGeometryFrame(
                frame_index=0,
                homography=[100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1.0],
                pitch_bbox=(0.0, 0.0, 100.0, 100.0),
                source="observed",
                landmarks=8,
                inliers=8,
                median_reprojection_cm=10.0,
            )
        ],
    )
    path = tmp_path / "geometry.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    return PreparedPitchGeometry("geometry", path, manifest, True)


def test_pitch_filter_deduplicates_tiles_and_rejects_off_pitch_people(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.parquet"
    rows = [
        _prediction(source_id="tile-a", person_id="person-a", bbox=(10, 10, 30, 20)),
        _prediction(source_id="tile-b", person_id="person-b", bbox=(10, 10, 30, 20)),
        _prediction(source_id="tile-a", person_id="crowd", bbox=(20, 60, 40, 80)),
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA), raw_path)
    config = PitchFilterConfig(
        enabled=True,
        checkpoint=tmp_path / "unused.pt",
        deduplication_iou=0.5,
    )

    result = postprocess_pitch_predictions(
        prediction_parquet=raw_path,
        output_root=tmp_path / "postprocess",
        geometry=_geometry(tmp_path),
        config=config,
    )

    assert result.raw_records == 3
    assert result.deduplicated_records == 2
    assert result.duplicate_records == 1
    assert result.on_pitch_records == 1
    assert result.outside_pitch_records == 1
    assert result.unclassified_records == 0
    assert len(read_archive(result.on_pitch_parquet)) == 1
    classifications = {
        row["classification"] for row in pq.read_table(result.decisions_parquet).to_pylist()
    }
    assert classifications == {"inside", "outside", "duplicate"}


def test_pitch_bbox_filters_crowd_when_homography_is_unavailable(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.parquet"
    rows = [
        _prediction(source_id="frame", person_id="player", bbox=(20, 30, 30, 60)),
        _prediction(source_id="frame", person_id="crowd", bbox=(20, 0, 30, 10)),
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA), raw_path)
    manifest = PitchGeometryManifest(
        geometry_id="geometry-bbox",
        source_sha256="source",
        checkpoint_sha256="checkpoint",
        settings={},
        frame_count=1,
        observed_frames=0,
        bbox_frames=1,
        usable_frames=1,
        processing_seconds=0.1,
        frames=[
            PitchGeometryFrame(
                frame_index=0,
                pitch_bbox=(0.0, 20.0, 100.0, 100.0),
                source="bbox_only",
                landmarks=2,
                inliers=0,
            )
        ],
    )
    geometry_path = tmp_path / "geometry-bbox.json"
    geometry_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    geometry = PreparedPitchGeometry(
        "geometry-bbox", geometry_path, manifest, True
    )

    result = postprocess_pitch_predictions(
        prediction_parquet=raw_path,
        output_root=tmp_path / "postprocess-bbox",
        geometry=geometry,
        config=PitchFilterConfig(
            enabled=True,
            checkpoint=tmp_path / "unused.pt",
            pitch_bbox_margin_px=0.0,
        ),
    )

    assert result.on_pitch_records == 1
    assert result.outside_pitch_records == 1
    decisions = pq.read_table(result.decisions_parquet).to_pylist()
    assert {row["filter_method"] for row in decisions} == {"pitch_bbox"}


def test_pitch_geometry_only_fills_short_missing_intervals() -> None:
    identity = np.eye(3).reshape(-1).tolist()
    frames = [
        PitchGeometryFrame(
            frame_index=0,
            homography=identity,
            source="observed",
            landmarks=8,
            inliers=8,
            median_reprojection_cm=10.0,
        ),
        *[
            PitchGeometryFrame(
                frame_index=index,
                homography=None,
                source="unavailable",
                landmarks=2,
                inliers=0,
                median_reprojection_cm=None,
            )
            for index in range(1, 4)
        ],
    ]

    filled = _fill_short_gaps(frames, max_fallback_frames=2)

    assert filled[1].source == "fallback"
    assert filled[2].source == "fallback"
    assert filled[3].source == "unavailable"
