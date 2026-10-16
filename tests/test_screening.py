from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from football_pose.cli import app
from football_pose.configuration import ProcessorSpec
from football_pose.preprocessing import REGISTRY
from football_pose.screening import (
    ScreeningConfig, comparison_rows, expand_screening, load_screening,
    preflight, run_screening,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SUITE = REPOSITORY / "configs/lyra-preprocessing-screen.yaml"


def _suite(tmp_path: Path, tiny_video: Path) -> Path:
    base = {
        "name": "cpu-screen", "input": str(tiny_video),
        "models": [{"id": "mock", "command": ["{python}", str(REPOSITORY / "runners/mock/run.py")]}],
        "video_output": {"enabled": True, "root": str(tmp_path / "videos")},
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    config = {
        "base_config": "base.yaml", "output_root": "results", "cache_root": "cache",
        "methods": [method.model_dump(mode="json") for method in load_screening(SUITE).methods
                    if method.name not in {"super-resolution", "crop"}],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_suite_covers_every_registered_method_and_fixes_shared_protocol() -> None:
    suite, expanded = expand_screening(SUITE, device=4)
    assert len(expanded) == 12
    assert {spec.type for config in expanded.values() for spec in config.processors} == (
        set(REGISTRY) - {"identity"}
    ) | {"crop"}
    baseline = expanded["baseline"]
    assert [model.id for model in baseline.models] == ["yolo26-pose", "openpose-body25", "hrnet-w32"]
    hrnet = baseline.models[-1]
    assert hrnet.auxiliary_checkpoints["detector"] == Path(
        "/mnt/storage2/vincent/football-pose/models/player-detection-best.pt"
    )
    assert "whole_image" in hrnet.command
    assert "human" not in hrnet.command
    assert baseline.processors == []
    assert all(len(config.processors) == 1 for name, config in expanded.items() if name != "baseline")
    for config in expanded.values():
        assert config.models == baseline.models
        assert config.pitch_filter == baseline.pitch_filter
        assert all(model.devices == [4] for model in config.models)
        assert config.pitch_filter.device == "4"
        assert config.cache.format == "auto"
        assert config.cache.root == suite.cache_root
        assert config.video_output.enabled
    assert expanded["crop"].processors[0].params["detector"]["device"] == "4"
    assert expanded["resize-x2"].processors[0].params["scale"] == 2
    assert expanded["super-resolution"].processors[0].params["scale"] == 2


def test_selection_always_includes_baseline_and_rejects_typos() -> None:
    _, selected = expand_screening(SUITE, device=2, methods={"clahe"}, models={"openpose-body25"})
    assert list(selected) == ["baseline", "clahe"]
    assert [model.id for model in selected["baseline"].models] == ["openpose-body25"]
    with pytest.raises(ValueError, match="unknown screening methods"):
        expand_screening(SUITE, device=2, methods={"clhae"})
    with pytest.raises(ValueError, match="unknown models"):
        expand_screening(SUITE, device=2, models={"hrnet"})


def test_suite_rejects_duplicate_names_and_missing_baseline() -> None:
    payload = load_screening(SUITE).model_dump()
    payload["methods"] = [{"name": "clahe", "processor": {"type": "clahe"}}] * 2
    with pytest.raises(ValueError, match="unique"):
        ScreeningConfig.model_validate(payload)
    payload["methods"] = [
        {"name": "clahe", "processor": {"type": "clahe"}},
        {"name": "gamma", "processor": {"type": "gamma"}},
    ]
    with pytest.raises(ValueError, match="baseline"):
        ScreeningConfig.model_validate(payload)


def test_preflight_checks_all_assets_and_reservation_before_writing(tmp_path, tiny_video, monkeypatch) -> None:
    path = _suite(tmp_path, tiny_video)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with pytest.raises(ValueError, match="outside"):
        run_screening(path, device=4)
    assert not (tmp_path / "results").exists()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    _, configs = expand_screening(path, device=4)
    configs["clahe"].processors = [ProcessorSpec(type="super_resolution", params={"checkpoint": "/missing.pb"})]
    with pytest.raises(ValueError, match="clahe.*missing.pb"):
        preflight(configs, device=4)
    assert not (tmp_path / "results").exists()


def test_hrnet_preflight_checks_the_same_player_detector_once(tmp_path, tiny_video, monkeypatch):
    import football_pose.preprocessing.cropping as cropping

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    path = _suite(tmp_path, tiny_video)
    base_path = tmp_path / "base.yaml"
    base = yaml.safe_load(base_path.read_text())
    for name in ("pose.pth", "detector.pt"):
        (tmp_path / name).write_bytes(b"fixture")
    base["models"] = [{
        "id": "hrnet-w32", "command": ["runner", "--crop-min-size", "8"],
        "checkpoint": "pose.pth", "auxiliary_checkpoints": {"detector": "detector.pt"},
    }]
    base_path.write_text(yaml.safe_dump(base))
    called = []
    class Detector:
        def __init__(self, **kwargs):
            called.append(kwargs)
        def validate(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(cropping, "UltralyticsDetectionProvider", Detector)
    _, configs = expand_screening(path, device=4, methods={"clahe"})
    preflight(configs, device=4)
    assert len(called) == 1
    assert called[0]["checkpoint"] == str(tmp_path / "detector.pt")
    assert called[0]["device"] == "cpu"
    assert called[0]["image_size"] == 1920
    assert called[0]["class_ids"] == [1, 2, 3]


def test_screen_runs_all_classical_steps_and_writes_comparison(tmp_path, tiny_video, monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    path = _suite(tmp_path, tiny_video)
    result = run_screening(path, device=0)
    report = json.loads(result.read_text())
    assert report["success"]
    assert report["source_frame_count"] == 6
    assert len(report["conditions"]) == 10
    assert all(row["status"] == "COMPLETE" for row in report["comparison"])
    tiled = next(row for row in report["comparison"] if row["method"] == "tile")
    assert tiled["raw_records"] == 24
    assert tiled["raw_per_frame"] == 4
    assert tiled["raw_records_change_percent"] == 300
    assert (result.parent / "comparison.csv").is_file()
    assert "not accuracy" in (result.parent / "comparison.md").read_text()
    for condition in report["conditions"]:
        assert Path(condition["summary_path"]).is_file()
        assert Path(condition["summary"]["jobs"][0]["outputs"]["video"]).is_file()
        assert (result.parent / "configs" / f"{condition['method']}.yaml").is_file()
    # A separate invocation makes fresh model jobs, but reuses preprocessing artifacts.
    second = run_screening(path, device=0, methods={"clahe"})
    assert second.parent != result.parent
    second_report = json.loads(second.read_text())
    assert all(row["preprocessing_cache_hit"] for row in second_report["comparison"])


def test_failed_condition_is_recorded_and_other_methods_continue(tmp_path, tiny_video, monkeypatch) -> None:
    from football_pose.experiments import ExperimentRunner

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    original = ExperimentRunner.run
    def fail_one(self):
        if self.config.name.endswith("clahe"):
            raise RuntimeError("fixture preprocessing failure")
        return original(self)
    monkeypatch.setattr(ExperimentRunner, "run", fail_one)
    result = run_screening(_suite(tmp_path, tiny_video), device=0, methods={"clahe", "gamma-brighten"})
    report = json.loads(result.read_text())
    assert not report["success"]
    assert [condition["status"] for condition in report["conditions"]] == ["COMPLETE", "FAILED", "COMPLETE"]
    failed = report["comparison"][1]
    assert failed["raw_records"] is None
    assert failed["raw_records_change_percent"] is None
    assert "fixture preprocessing failure" in failed["error"]


def test_check_only_cli_does_not_create_results(tmp_path, tiny_video, monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    path = _suite(tmp_path, tiny_video)
    result = CliRunner().invoke(app, ["screen-preprocessing", str(path), "--device", "0", "--check-only"])
    assert result.exit_code == 0, result.output
    assert "Ready: 10 conditions x 1 models" in result.output
    assert not (tmp_path / "results").exists()


def test_interrupt_keeps_partial_reports(tmp_path, tiny_video, monkeypatch) -> None:
    from football_pose.experiments import ExperimentRunner

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    original = ExperimentRunner.run
    def interrupt(self):
        if self.config.name.endswith("clahe"):
            raise KeyboardInterrupt
        return original(self)
    monkeypatch.setattr(ExperimentRunner, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_screening(_suite(tmp_path, tiny_video), device=0, methods={"clahe"})
    path = next((tmp_path / "results").glob("*/screening.json"))
    report = json.loads(path.read_text())
    assert not report["success"]
    assert [item["status"] for item in report["conditions"]] == ["COMPLETE", "INTERRUPTED"]
    assert report["comparison"][1]["raw_records"] is None


def test_comparison_never_compares_failed_or_zero_baselines() -> None:
    report = {
        "model_ids": ["m"], "source_frame_count": 6,
        "conditions": [{"method": name, "status": "COMPLETE", "summary": {
            "jobs": [{"model_id": "m", "status": "COMPLETE", "outputs": {"records": str(n)}}]
        }} for name, n in [("baseline", 0), ("clahe", 3)]],
    }
    assert comparison_rows(report)[1]["raw_records_change_percent"] is None


def test_super_resolution_real_model_when_supplied() -> None:
    # Optional downloaded asset; no network is used in the test suite.
    import numpy as np
    from football_pose.domain import FramePacket
    from football_pose.preprocessing.opencv import SuperResolutionProcessor
    from football_pose.preprocessing.base import ProcessorContext

    checkpoint = os.environ.get("BREMEN_TEST_SR_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set BREMEN_TEST_SR_CHECKPOINT to the pinned FSRCNN_x2.pb")
    spec = next(method.processor for method in load_screening(SUITE).methods
                if method.name == "super-resolution")
    processor = SuperResolutionProcessor(**{**spec.params, "checkpoint": checkpoint})
    image = np.random.default_rng(1).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    frame = FramePacket(image, 0, 0, 32, 24)
    output = processor.process(frame, ProcessorContext("e", "p"))[0]
    assert output.image.shape == (48, 64, 3)
    assert output.image.dtype == np.uint8
    np.testing.assert_allclose(output.points_to_source([[32, 24]]), [[16, 12]])


def test_crop_and_real_super_resolution_in_complete_screen(tmp_path, tiny_video, monkeypatch) -> None:
    import football_pose.preprocessing as preprocessing
    from football_pose.preprocessing.cropping import Detection

    checkpoint = os.environ.get("BREMEN_TEST_SR_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set BREMEN_TEST_SR_CHECKPOINT to the pinned FSRCNN_x2.pb")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    path = _suite(tmp_path, tiny_video)
    payload = yaml.safe_load(path.read_text())
    external = [method.model_dump(mode="json") for method in load_screening(SUITE).methods
                if method.name in {"crop", "super-resolution"}]
    detector = tmp_path / "detector.pt"
    detector.write_bytes(b"fixture")
    for method in external:
        params = method["processor"]["params"]
        if method["name"] == "super-resolution":
            params["checkpoint"] = checkpoint
        else:
            params["detector"]["checkpoint"] = str(detector)
            params["min_size"] = 8
    payload["methods"] += external
    path.write_text(yaml.safe_dump(payload))
    calls = []
    class Detector:
        def __init__(self, **kwargs):
            self.device = kwargs["device"]
        def detect(self, packet):
            calls.append(packet.frame_index)
            assert self.device == "0"
            return [Detection((8, 8, 40, 40), 0.9, str(packet.frame_index))]
        def close(self):
            pass
    monkeypatch.setattr(preprocessing, "UltralyticsDetectionProvider", Detector)
    result = run_screening(path, device=0)
    report = json.loads(result.read_text())
    assert report["success"]
    assert len(report["conditions"]) == 12
    assert calls == list(range(6))
    assert len(report["checkpoint_sha256"]) == 2
    crop = next(row for row in report["comparison"] if row["method"] == "crop")
    assert crop["raw_records"] == 6
    assert Path(crop["video"]).is_file()
