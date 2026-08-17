from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from football_pose.archive import read_archive
from football_pose.configuration import ExperimentConfig
from football_pose.experiments import ExperimentRunner
from football_pose.runners import ExternalModelRunner, RunnerExecutionError


REPOSITORY = Path(__file__).resolve().parents[1]


def test_runner_rejects_gpu_outside_shared_reservation(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    config = ExperimentConfig.model_validate(
        {
            "name": "reservation",
            "input": "unused.mp4",
            "models": [
                {"id": "model", "command": ["runner"], "devices": [2, 4]}
            ],
        }
    )
    runner = ExternalModelRunner(config.models[0])
    with pytest.raises(RunnerExecutionError, match="outside"):
        runner._validate_gpu_reservation()


def test_end_to_end_mock_runner_is_resumable_and_multi_sharded(
    tmp_path: Path, tiny_video: Path
) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "e2e",
            "input": tiny_video,
            "output_dir": tmp_path / "output",
            "processors": [
                {"type": "resize", "params": {"width": 32}},
                {"type": "gamma", "params": {"gamma": 1.1}},
            ],
            "cache": {"root": tmp_path / "cache", "format": "ffv1"},
            "models": [
                {
                    "id": "mock",
                    "command": ["{python}", str(REPOSITORY / "runners/mock/run.py")],
                    "batch_size": 4,
                    "devices": [0, 1],
                }
            ],
        }
    )
    first = ExperimentRunner(config).run()
    assert first["success"]
    assert not first["cache_hit"]
    job = first["jobs"][0]
    assert job["status"] == "COMPLETE"
    assert job["attempts"] == 1
    assert len(read_archive(job["outputs"]["parquet"])) == 6
    schema = pq.read_schema(job["outputs"]["parquet"])
    assert schema.field("crop_id").type == pa.string()
    assert pa.types.is_fixed_size_list(schema.field("keypoints").type)
    summary_path = (
        config.output_dir / "experiments" / first["experiment_id"] / "summary.json"
    )
    assert summary_path.is_file()

    second = ExperimentRunner(config).run()
    assert second["success"]
    assert second["cache_hit"]
    assert second["preprocessing_mode"] == "warm_cache"
    assert second["preprocessing_stage_timings"] == {}
    assert second["jobs"][0]["attempts"] == 1


def test_oom_restarts_all_shards_with_smaller_batch(tmp_path: Path, tiny_video: Path) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "oom-retry",
            "input": tiny_video,
            "output_dir": tmp_path / "output",
            "cache": {"root": tmp_path / "cache", "format": "png_shards"},
            "models": [
                {
                    "id": "oom-fixture",
                    "command": [
                        sys.executable,
                        str(REPOSITORY / "tests/fixtures/fake_oom_runner.py"),
                    ],
                    "batch_size": 4,
                    "min_batch_size": 1,
                    "devices": [0, 1],
                }
            ],
        }
    )
    result = ExperimentRunner(config).run()
    assert result["success"]
    job = result["jobs"][0]
    assert job["batch_size"] == 2
    assert Path(job["outputs"]["parquet"]).is_file()
