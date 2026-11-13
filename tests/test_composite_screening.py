from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from football_pose.artifacts import iter_artifact
from football_pose.coco_screening import expand_coco_screening
from football_pose.experiments import ExperimentRunner
from football_pose.preprocessing import build_pipeline
from football_pose.preprocessing.base import ProcessorContext
from football_pose.screening import ScreeningConfig, expand_screening, load_screening, run_screening
from football_pose.video import VideoSource


REPOSITORY = Path(__file__).resolve().parents[1]
SHADED_SUITE = REPOSITORY / "configs/lyra-shaded-players.example.yaml"
ADDITIONS_SUITE = REPOSITORY / "configs/lyra-additional-methods.yaml"


def _local_suite(tmp_path: Path, tiny_video: Path) -> Path:
    base = {
        "name": "shaded-smoke", "input": str(tiny_video),
        "metadata": {"match_id": "fixture-match", "jersey_colors": {"home": ["#00FF00"]}},
        "models": [{"id": "mock", "command": ["{python}", str(REPOSITORY / "runners/mock/run.py")]}],
        "video_output": {"enabled": True, "root": str(tmp_path / "videos")},
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    payload = yaml.safe_load(SHADED_SUITE.read_text())
    payload.update(base_config="base.yaml", output_root="results", cache_root="cache")
    payload["metadata"] = {"lighting": {"weather": "sunny", "camera_heading_deg": 90}}
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_shaded_suite_matches_component_controls_and_preserves_protocol():
    suite, configs = expand_screening(SHADED_SUITE, device=4)
    assert list(configs) == ["baseline", "gamma-brighten", "clahe", "unsharp", "shaded-players"]
    assert sum(len(config.models) for config in configs.values()) == 15
    chain = configs["shaded-players"].processors
    assert [spec.type for spec in chain] == ["gamma", "clahe", "unsharp_mask"]
    original = expand_screening(REPOSITORY / "configs/lyra-preprocessing-screen.yaml", device=4)[1]
    for spec, name in zip(chain, ["gamma-brighten", "clahe", "unsharp"]):
        assert spec == configs[name].processors[0] == original[name].processors[0]
    for config in configs.values():
        assert config.models == configs["baseline"].models
        assert config.pitch_filter == configs["baseline"].pitch_filter
        assert config.metadata.model_dump()["lighting"]["weather"] is None
        assert all(model.devices == [4] for model in config.models)
    chain[1].params["tile_grid"][0] = 4
    assert configs["clahe"].processors[0].params["tile_grid"] == [8, 8]
    assert suite.methods[-1].processor_specs()[1].params["tile_grid"] == [8, 8]
    _, selected = expand_screening(SHADED_SUITE, device=4, methods={"shaded-players"})
    assert list(selected) == ["baseline", "shaded-players"]
    assert len(selected["shaded-players"].processors) == 3


def test_additional_methods_use_real_palette_and_one_shared_baseline():
    _, configs = expand_screening(ADDITIONS_SUITE, device=4)
    assert list(configs) == ["baseline", "jersey-colors", "shaded-players"]
    assert sum(len(config.models) for config in configs.values()) == 9
    assert not configs["baseline"].resolved_processors()
    jersey = configs["jersey-colors"].resolved_processors()
    assert jersey[0].params["strength"] == 0.35
    assert set(jersey[0].params["colors"]) == {"#446D64", "#769B7E", "#ECF7EC", "#A9BCBC"}
    assert configs["jersey-colors"].jersey_diagnostics.frame_indices == [0, 260, 520, 781]
    shade = configs["shaded-players"].resolved_processors()
    assert [spec.type for spec in shade] == ["gamma", "clahe", "unsharp_mask"]
    for config in configs.values():
        assert config.models == configs["baseline"].models
        assert config.pitch_filter == configs["baseline"].pitch_filter
        assert config.input == configs["baseline"].input


def test_additional_methods_complete_together_with_mock_inference(tmp_path, tiny_video, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    path = _local_suite(tmp_path, tiny_video)
    payload = yaml.safe_load(ADDITIONS_SUITE.read_text())
    payload.update(base_config="base.yaml", output_root="results", cache_root="cache")
    payload["jersey_diagnostics"]["frame_indices"] = [0, 5]
    path.write_text(yaml.safe_dump(payload))
    result = run_screening(path, device=0, progress=lambda _: None)
    report = json.loads(result.read_text())
    assert report["success"]
    assert [row["method"] for row in report["comparison"]] == ["baseline", "jersey-colors", "shaded-players"]
    assert all(row["status"] == "COMPLETE" and row["raw_records"] == 6 for row in report["comparison"])
    diagnostics = [condition["summary"]["jersey_diagnostics"] for condition in report["conditions"]]
    assert diagnostics[0]["status"] == diagnostics[2]["status"] == "not_applicable"
    assert Path(diagnostics[1]["manifest"]).is_file()
    assert diagnostics[1]["sample_count"] == 2


@pytest.mark.parametrize("methods, message", [
    ([{"name": "baseline"}, {"name": "bad", "processor": {"type": "gamma"},
                             "processors": [{"type": "clahe"}]}], "either processor or processors"),
    ([{"name": "baseline"}, {"name": "bad", "processors": []}], "at least 1"),
    ([{"name": "baseline"}, {"name": "empty"}], "baseline"),
    ([{"name": "baseline", "processors": [{"type": "gamma"}]},
      {"name": "other", "processor": {"type": "clahe"}}], "baseline"),
    ([{"name": "baseline"}, {"name": "baseline"}], "unique"),
])
def test_rejects_ambiguous_or_empty_conditions(methods, message):
    payload = load_screening(SHADED_SUITE).model_dump()
    payload["methods"] = methods
    with pytest.raises(ValueError, match=message):
        ScreeningConfig.model_validate(payload)


def test_all_chain_checkpoints_resolve_relative_to_suite(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({
        "base_config": "base.yaml", "output_root": "results", "cache_root": "cache",
        "methods": [{"name": "baseline"}, {"name": "chain", "processors": [
            {"type": "super_resolution", "params": {"checkpoint": "weights/sr.pb"}},
            {"type": "crop", "params": {"detector": {"checkpoint": "weights/detector.pt"}}},
        ]}],
    }))
    steps = load_screening(path).methods[-1].processor_specs()
    assert steps[0].params["checkpoint"] == str(tmp_path / "weights/sr.pb")
    assert steps[1].params["detector"]["checkpoint"] == str(tmp_path / "weights/detector.pt")


def test_expansion_rejects_chains_that_violate_experiment_constraints(tmp_path, tiny_video):
    path = _local_suite(tmp_path, tiny_video)
    payload = yaml.safe_load(path.read_text())
    payload["methods"] = [{"name": "baseline"}, {"name": "two-crops", "processors": [
        {"type": "crop", "params": {"detector": {"checkpoint": "unused.pt"}}},
        {"type": "crop", "params": {"detector": {"checkpoint": "unused.pt"}}},
    ]}]
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="crop"):
        expand_screening(path, device=0)


def test_chain_screening_runs_and_records_order_geometry_and_metadata(tmp_path, tiny_video, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    path = _local_suite(tmp_path, tiny_video)
    report_path = run_screening(path, device=0, progress=lambda _: None)
    report = json.loads(report_path.read_text())
    assert report["success"] and report["source_frame_count"] == 6
    assert len(report["conditions"]) == 5
    assert all(row["status"] == "COMPLETE" and row["raw_records"] == 6 for row in report["comparison"])
    summary = report["conditions"][-1]["summary"]
    assert set(summary["preprocessing_stage_timings"]) == {"00-gamma", "01-clahe", "02-unsharp_mask"}
    assert all(stage["inputs"] == stage["outputs"] == 6
               for stage in summary["preprocessing_stage_timings"].values())
    saved = yaml.safe_load((report_path.parent / "configs/shaded-players.yaml").read_text())
    assert saved["metadata"]["lighting"]["weather"] == "sunny"
    assert saved["metadata"]["jersey_colors"]["home"] == ["#00FF00"]
    assert saved["metadata"]["match_id"] == "fixture-match"
    assert "composite" in Path(saved["video_output"]["root"]).parts
    assert "Single-step" not in (report_path.parent / "comparison.md").read_text()
    assert "not accuracy" in (report_path.parent / "comparison.md").read_text()

    original = list(VideoSource(tiny_video).frames())
    cached = list(iter_artifact(Path(summary["artifact"])))
    _, configs = expand_screening(path, device=0)
    expected = original
    # Apply each separately to ensure screening actually executes the whole chain.
    for name in ("gamma-brighten", "clahe", "unsharp"):
        pipeline = build_pipeline(configs[name].processors)
        try:
            expected = list(pipeline.process(expected, ProcessorContext("test", name)))
        finally:
            pipeline.close()
    for source, output, reference in zip(original, cached, expected):
        assert output.image.shape == source.image.shape
        assert output.frame_index == source.frame_index
        assert output.timestamp_seconds == source.timestamp_seconds
        np.testing.assert_array_equal(output.to_source, source.to_source)
        np.testing.assert_array_equal(output.image, reference.image)
    assert any(not np.array_equal(source.image, output.image) for source, output in zip(original, cached))


def test_weather_context_reuses_pixels_but_chain_order_changes_cache(tmp_path, tiny_video):
    path = _local_suite(tmp_path, tiny_video)
    _, configs = expand_screening(path, device=0)
    config = configs["shaded-players"]
    original = ExperimentRunner(config).prepare()
    assert not original.cache_hit
    config.metadata = config.metadata.model_copy(update={"lighting": {"weather": "overcast"}})
    updated = ExperimentRunner(config).prepare()
    assert updated.cache_hit and updated.artifact_path == original.artifact_path
    config.processors.reverse()
    reordered = ExperimentRunner(config).prepare()
    assert not reordered.cache_hit and reordered.artifact_path != original.artifact_path


def test_coco_explicitly_rejects_composite_instead_of_silently_running_baseline(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="screen-coco requires single steps"):
        expand_coco_screening(SHADED_SUITE, device=0)
