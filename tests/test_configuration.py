from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from football_pose.configuration import ExperimentConfig, load_config


REPOSITORY = Path(__file__).resolve().parents[1]


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


def test_lyra_yolo_smoke_config_is_scoped_to_one_model_and_gpu() -> None:
    loaded = load_config(
        REPOSITORY / "configs/lyra-yolo-one-gpu.yaml", check_paths=False
    )

    assert [processor.type for processor in loaded.processors] == ["resize", "clahe"]
    assert [model.id for model in loaded.models] == ["yolo-pose"]
    assert loaded.models[0].devices == [0]
    assert loaded.models[0].command[-1] == "vincent/football-pose-yolo:dev"


def test_full_frame_and_tiled_yolo_configs_hold_model_settings_constant() -> None:
    full_frame = load_config(
        REPOSITORY / "configs/lyra-yolo-full-frame.yaml", check_paths=False
    )
    tiled = load_config(
        REPOSITORY / "configs/lyra-yolo-tiled.yaml", check_paths=False
    )

    assert full_frame.processors == []
    assert [processor.type for processor in tiled.processors] == ["tile"]
    assert tiled.processors[0].params == {
        "rows": 2,
        "columns": 2,
        "overlap_ratio": 0.1,
    }
    assert full_frame.models == tiled.models
