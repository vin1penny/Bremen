from __future__ import annotations

import json
from pathlib import Path

import pytest

from football_pose.overview import collect_results, render_overview, write_overview


def _write_summary(
    root: Path,
    name: str,
    *,
    processors: list[dict[str, object]],
    jobs: list[dict[str, object]],
) -> None:
    path = root / name / "experiments" / "experiment" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "configuration": {
                    "input": "/videos/sample.mov",
                    "processors": processors,
                },
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )


def _job(
    model_id: str,
    records: int,
    updated: float = 1.0,
    on_pitch: int | None = None,
) -> dict[str, object]:
    outputs = {"records": str(records)}
    if on_pitch is not None:
        outputs["on_pitch_records"] = str(on_pitch)
    return {
        "model_id": model_id,
        "status": "COMPLETE",
        "updated_at_unix": updated,
        "outputs": outputs,
    }


def test_overview_combines_models_and_calculates_baseline_delta(tmp_path: Path) -> None:
    _write_summary(tmp_path, "baseline-yolo", processors=[], jobs=[_job("yolo-pose", 0)])
    _write_summary(
        tmp_path,
        "baseline-openpose",
        processors=[],
        jobs=[_job("openpose-body25", 100)],
    )
    _write_summary(
        tmp_path,
        "clahe",
        processors=[{"type": "clahe", "params": {"clip_limit": 2.0}}],
        jobs=[_job("yolo-pose", 0), _job("openpose-body25", 140)],
    )

    grouped = collect_results(tmp_path)
    markdown = render_overview(grouped, source_root=tmp_path)

    assert (
        "| Baseline: full frame, unchanged | 0 | reference | 100 | reference |"
        in markdown
    )
    assert "| CLAHE | 0 | n/a | 140 | +40.0% |" in markdown
    assert "no manual decisions are included" in markdown


def test_overview_uses_latest_job_and_filters_models(tmp_path: Path) -> None:
    _write_summary(
        tmp_path,
        "first",
        processors=[{"type": "gamma", "params": {"gamma": 1.2}}],
        jobs=[_job("openpose-body25", 10, updated=1), _job("yolo-pose", 5)],
    )
    _write_summary(
        tmp_path,
        "second",
        processors=[{"type": "gamma", "params": {"gamma": 1.2}}],
        jobs=[_job("openpose-body25", 20, updated=2)],
    )

    grouped = collect_results(tmp_path, selected_models={"openpose-body25"})
    markdown = render_overview(grouped, source_root=tmp_path)

    assert "OpenPose records" in markdown
    assert "YOLO records" not in markdown
    assert "| Gamma 1.2 | 20 | n/a |" in markdown


def test_write_overview_creates_default_folder(tmp_path: Path) -> None:
    _write_summary(tmp_path, "baseline", processors=[], jobs=[_job("openpose-body25", 4)])

    output = write_overview(tmp_path)

    assert output == tmp_path / "results-overview" / "records.md"
    assert output.is_file()


def test_overview_reports_raw_and_on_pitch_counts(tmp_path: Path) -> None:
    _write_summary(
        tmp_path,
        "baseline",
        processors=[],
        jobs=[_job("openpose-body25", 120, on_pitch=100)],
    )
    _write_summary(
        tmp_path,
        "tiled",
        processors=[{"type": "tile", "params": {"rows": 2, "columns": 2}}],
        jobs=[_job("openpose-body25", 300, on_pitch=110)],
    )

    markdown = render_overview(collect_results(tmp_path), source_root=tmp_path)

    assert "OpenPose on-pitch | OpenPose raw | OpenPose vs baseline" in markdown
    assert "| Baseline: full frame, unchanged | 100 | 120 | reference |" in markdown
    assert "| 2 x 2 tiling (? overlap) | 110 | 300 | +10.0% |" in markdown


def test_collect_results_rejects_missing_summaries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no summary.json files"):
        collect_results(tmp_path)
