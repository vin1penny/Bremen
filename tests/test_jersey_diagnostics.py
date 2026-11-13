from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from football_pose.artifacts import iter_artifact
from football_pose.cli import app
from football_pose.configuration import ExperimentConfig, JerseyDiagnosticsConfig, ModelSpec, ProcessorSpec
from football_pose.experiments import ExperimentRunner
from football_pose.preprocessing.jersey import JerseyColorEnhancementProcessor
from football_pose.screening import expand_screening
from football_pose.video import VideoSource


REPOSITORY = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, tiny_video: Path, **params) -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "name": "jersey-diagnostics", "input": tiny_video,
        "output_dir": tmp_path / "results", "cache": {"root": tmp_path / "cache"},
        "metadata": {"jersey_colors": {"home": ["#00B400"]}},
        "processors": [{"type": "jersey_color_enhance", "params": params}],
        "jersey_diagnostics": {"enabled": True, "frame_indices": [0, 2, 99]},
    })


def test_warm_cache_diagnostics_match_inference_and_can_be_regenerated(tmp_path, tiny_video):
    config = _config(tmp_path, tiny_video)
    config.jersey_diagnostics.enabled = False
    runner = ExperimentRunner(config)
    original_run = runner.prepare()
    model = ModelSpec(id="mock", command=["unused"])
    original_job_id = runner._job_id(original_run, model)
    config.jersey_diagnostics.enabled = True
    runner = ExperimentRunner(config)
    prepared = runner.prepare()
    assert prepared.cache_hit and prepared.pipeline_id == original_run.pipeline_id
    assert runner._job_id(prepared, model) == original_job_id

    result = runner.prepare_diagnostics(prepared)
    assert result["sample_count"] == 2 and result["missing_frame_indices"] == [99]
    assert not result["cache_hit"]
    manifest_path = Path(result["manifest"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["effective_parameters"]["strength"] == 0.35
    assert manifest["effective_parameters"]["tile_grid"] == [8, 8]
    originals = list(VideoSource(tiny_video).frames())
    cached = list(iter_artifact(prepared.artifact_path))
    processor = JerseyColorEnhancementProcessor(colors=["#00B400"])
    for sample in manifest["samples"]:
        index = sample["frame_index"]
        files = {key: manifest_path.parent / value for key, value in sample["files"].items()}
        np.testing.assert_array_equal(cv2.imread(str(files["input"])), originals[index].image)
        np.testing.assert_array_equal(cv2.imread(str(files["enhanced"])), cached[index].image)
        _, mask = processor.inspect(originals[index])
        np.testing.assert_array_equal(
            cv2.imread(str(files["mask"]), cv2.IMREAD_GRAYSCALE),
            np.rint(mask * 255).astype(np.uint8),
        )
        assert sample["selected_pixel_fraction"] == pytest.approx(float(np.mean(mask > 0)))
        assert sample["mean_blend_weight"] == pytest.approx(float(mask.mean()) * 0.35)
        assert sample["timestamp_seconds"] == originals[index].timestamp_seconds
        assert files["comparison"].is_file()

    assert runner.prepare_diagnostics(prepared)["cache_hit"]
    missing_file = manifest_path.parent / manifest["samples"][0]["files"]["mask"]
    missing_file.unlink()
    assert not runner.prepare_diagnostics(prepared)["cache_hit"]
    assert missing_file.is_file()
    config.jersey_diagnostics.frame_indices = [1]
    assert runner.prepare().cache_hit
    changed = runner.prepare_diagnostics(prepared)
    assert changed["manifest"] != result["manifest"]
    assert changed["sample_count"] == 1


@pytest.mark.parametrize("params", [{"strength": 0}, {"colors": ["#FF00FF"]}])
def test_identity_cases_have_honest_masks_and_unchanged_outputs(tmp_path, tiny_video, params):
    config = _config(tmp_path, tiny_video, **params)
    runner = ExperimentRunner(config)
    result = runner.prepare_diagnostics(runner.prepare())
    path = Path(result["manifest"])
    for sample in json.loads(path.read_text())["samples"]:
        before = cv2.imread(str(path.parent / sample["files"]["input"]))
        after = cv2.imread(str(path.parent / sample["files"]["enhanced"]))
        np.testing.assert_array_equal(before, after)
        if "strength" in params:
            assert sample["mean_selection_weight"] > 0
            assert sample["mean_blend_weight"] == 0
        else:
            assert sample["selected_pixel_fraction"] == 0


def test_summary_and_prepare_only_cli_expose_diagnostics(tmp_path, tiny_video):
    config = _config(tmp_path, tiny_video)
    config.models = []
    summary = ExperimentRunner(config).run()
    assert summary["success"]
    assert summary["jersey_diagnostics"]["sample_count"] == 2
    assert summary["jersey_diagnostics"]["wall_seconds"] >= 0
    assert set(summary["preprocessing_stage_timings"]) == {"00-jersey_color_enhance"}
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    result = CliRunner().invoke(app, ["run", str(config_path), "--prepare-only"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["cache_hit"] and payload["jersey_diagnostics"]["cache_hit"]


def test_diagnostics_are_opt_in_skip_baseline_and_reject_misleading_chain(tmp_path, tiny_video):
    config = _config(tmp_path, tiny_video)
    config.processors = []
    runner = ExperimentRunner(config)
    assert runner.prepare_diagnostics(runner.prepare())["status"] == "not_applicable"
    config.processors = [ProcessorSpec(type="gamma"), ProcessorSpec(type="jersey_color_enhance")]
    with pytest.raises(ValueError, match="standalone"):
        config.resolved_processors()
    config.jersey_diagnostics.enabled = False
    assert len(config.resolved_processors()) == 2
    assert JerseyDiagnosticsConfig().enabled is False


@pytest.mark.parametrize("indices", [[], [-1], [True], [1.5], list(range(17))])
def test_preview_selection_is_validated_and_bounded(indices):
    with pytest.raises(ValueError):
        JerseyDiagnosticsConfig(frame_indices=indices)


def test_example_suite_shares_bounded_diagnostics_without_extra_method():
    _, configs = expand_screening(REPOSITORY / "configs/lyra-jersey-colors.example.yaml", device=0)
    assert list(configs) == ["baseline", "jersey-mild", "jersey-moderate"]
    for config in configs.values():
        assert config.jersey_diagnostics.enabled
        assert config.jersey_diagnostics.frame_indices == [0, 1, 2, 3]
    configs["jersey-mild"].jersey_diagnostics.frame_indices.append(4)
    assert configs["jersey-moderate"].jersey_diagnostics.frame_indices == [0, 1, 2, 3]


def test_failed_image_write_does_not_publish_manifest(tmp_path, tiny_video, monkeypatch):
    config = _config(tmp_path, tiny_video)
    runner = ExperimentRunner(config)
    prepared = runner.prepare()
    monkeypatch.setattr(cv2, "imwrite", lambda *args, **kwargs: False)
    with pytest.raises(OSError, match="diagnostic"):
        runner.prepare_diagnostics(prepared)
    assert not list((config.output_dir / "experiments").rglob("manifest.json"))


def test_failed_regeneration_keeps_pngs_atomic_and_clears_completion(tmp_path, tiny_video, monkeypatch):
    runner = ExperimentRunner(_config(tmp_path, tiny_video))
    prepared = runner.prepare()
    result = runner.prepare_diagnostics(prepared)
    manifest_path = Path(result["manifest"])
    sample = json.loads(manifest_path.read_text())["samples"][0]
    original = manifest_path.parent / sample["files"]["input"]
    before = original.read_bytes()
    (manifest_path.parent / sample["files"]["mask"]).unlink()

    def partial_write(path, *args, **kwargs):
        Path(path).write_bytes(b"truncated PNG")
        return False

    with monkeypatch.context() as patch:
        patch.setattr(cv2, "imwrite", partial_write)
        with pytest.raises(OSError, match="diagnostic"):
            runner.prepare_diagnostics(prepared)
    assert original.read_bytes() == before
    assert not manifest_path.exists()
    assert not list(manifest_path.parent.glob(".*.png"))
    assert not runner.prepare_diagnostics(prepared)["cache_hit"]
