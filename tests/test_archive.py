from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from football_pose.archive import write_archive


def test_empty_archive_still_has_canonical_schema(tmp_path: Path) -> None:
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("", encoding="utf-8")
    parquet = tmp_path / "predictions.parquet"
    count = write_archive(jsonl, parquet, tmp_path / "manifest.json", {})
    schema = pq.read_schema(parquet)
    assert count == 0
    assert schema.field("crop_id").type == pa.string()
    assert schema.field("source_bbox").type == pa.list_(pa.float64())
    assert schema.field("keypoints").type.list_size == 17
