from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from football_pose.domain import PredictionRecord
from football_pose.jobs import atomic_write_json


KEYPOINT_SCHEMA = pa.struct(
    [
        pa.field("x", pa.float64(), nullable=False),
        pa.field("y", pa.float64(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("visibility", pa.float64(), nullable=True),
    ]
)

PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("experiment_id", pa.string(), nullable=False),
        pa.field("pipeline_id", pa.string(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("source_video_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("timestamp_seconds", pa.float64(), nullable=False),
        pa.field("crop_id", pa.string(), nullable=True),
        pa.field("track_id", pa.string(), nullable=True),
        pa.field("person_id", pa.string(), nullable=False),
        # Pydantic enforces four values. Arrow's nullable fixed-size-list reader
        # rejects null values in some releases, so keep the storage list variable.
        pa.field("source_bbox", pa.list_(pa.float64()), nullable=True),
        pa.field("keypoints", pa.list_(KEYPOINT_SCHEMA, 17), nullable=False),
        pa.field("coordinate_space", pa.string(), nullable=False),
        pa.field("inference_time_ms", pa.float64(), nullable=False),
    ]
)


def validate_jsonl(path: str | Path) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(PredictionRecord.model_validate_json(line))
            except Exception as error:
                raise ValueError(f"invalid prediction JSONL at line {line_number}") from error
    return records


def write_archive(
    jsonl_path: str | Path,
    parquet_path: str | Path,
    manifest_path: str | Path,
    provenance: dict[str, Any],
) -> int:
    records = validate_jsonl(jsonl_path)
    rows = [record.model_dump(mode="json") for record in records]
    table = pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA)
    output = Path(parquet_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(output)
    manifest = {
        "schema": "football-pose-coco17-v1",
        "record_count": len(records),
        "prediction_jsonl": str(Path(jsonl_path)),
        "prediction_parquet": str(output),
        "provenance": provenance,
    }
    atomic_write_json(Path(manifest_path), manifest)
    return len(records)


def read_archive(path: str | Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()
