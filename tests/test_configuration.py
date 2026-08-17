from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from football_pose.configuration import ExperimentConfig, load_config


def test_config_rejects_duplicate_model_ids() -> None:
    with pytest.raises(ValidationError, match="model ids must be unique"):
        ExperimentConfig.model_validate(
            {
                "name": "bad",
                "input": "video.mp4",
                "models": [
                    {"id": "same", "command": ["one"]},
                    {"id": "same", "command": ["two"]},
                ],
            }
        )


def test_paths_are_resolved_relative_to_yaml(tmp_path: Path, tiny_video: Path) -> None:
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"weights")
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: paths",
                f"input: {tiny_video.name}",
                "output_dir: output",
                "cache:",
                "  root: cache",
                "processors:",
                "  - type: super_resolution",
                "    params:",
                "      checkpoint: weights.bin",
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_config(config_path)
    assert loaded.input == tiny_video.resolve()
    assert loaded.output_dir == (tmp_path / "output").resolve()
    assert loaded.cache.root == (tmp_path / "cache").resolve()
    assert loaded.processors[0].params["checkpoint"] == str(checkpoint.resolve())
