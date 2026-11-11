from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
import pytest
import yaml

from football_pose.archive import read_archive
from football_pose.artifacts import iter_artifact
from football_pose.configuration import ExperimentConfig, load_config
from football_pose.domain import FramePacket
from football_pose.experiments import ExperimentRunner
from football_pose.preprocessing import build_pipeline
from football_pose.preprocessing.base import ProcessorContext
from football_pose.preprocessing.jersey import JerseyColorEnhancementProcessor
from football_pose.screening import expand_screening, run_screening


REPOSITORY = Path(__file__).resolve().parents[1]
CONTEXT = ProcessorContext("jersey-test", "palette-test")


@pytest.fixture
def swatches() -> FramePacket:
    # Red kit, blue kit, grass, light kit, dark kit; variation imitates shading.
    bgr = np.array([[56, 50, 181], [168, 94, 49], [40, 120, 45], [230]*3, [24]*3])
    image = np.tile(np.repeat(bgr, 32, axis=0)[None, ...], (96, 1, 1))
    image = np.rint(image * np.linspace(0.9, 1.1, 96)[:, None, None]).astype(np.uint8)
    return FramePacket(
        image=image, frame_index=7, timestamp_seconds=0.7,
        source_width=320, source_height=192,
        to_source=np.diag([2.0, 2.0, 1.0]), source_id="frame-7-crop",
        crop_id="crop-7", track_id="12", source_bbox=(0, 0, 320, 192),
        metadata={"existing": "kept"},
    )


def _apply(packet: FramePacket, **params) -> FramePacket:
    return JerseyColorEnhancementProcessor(**params).process(packet, CONTEXT)[0]


def test_palette_enhances_both_kits_without_changing_grass_or_geometry(swatches) -> None:
    before = swatches.image.copy()
    output = _apply(swatches, colors=["#B53238", "#315EA8"])
    assert np.any(output.image[:, :32] != before[:, :32])
    assert np.any(output.image[:, 32:64] != before[:, 32:64])
    np.testing.assert_array_equal(output.image[:, 64:], before[:, 64:])
    np.testing.assert_array_equal(swatches.image, before)
    assert output.image.shape == before.shape
    assert output.image.dtype == np.uint8
    assert (output.frame_index, output.timestamp_seconds) == (7, 0.7)
    assert (output.source_width, output.source_height) == (320, 192)
    assert (output.source_id, output.crop_id, output.track_id) == ("frame-7-crop", "crop-7", "12")
    assert output.source_bbox == swatches.source_bbox
    assert output.metadata == swatches.metadata
    np.testing.assert_allclose(output.points_to_source([[10, 12]]), [[20, 24]])


@pytest.mark.parametrize("colors,strength", [(["#B53238"], 0), (["#FF00FF"], 0.5)])
def test_disabled_or_unmatched_filter_is_exact_identity(swatches, colors, strength) -> None:
    output = _apply(swatches, colors=colors, strength=strength)
    np.testing.assert_array_equal(output.image, swatches.image)


@pytest.mark.parametrize("color,selected", [("#E6E6E6", 3), ("#181818", 4)])
def test_neutral_kits_do_not_match_every_gray_pixel(swatches, color, selected) -> None:
    output = _apply(swatches, colors=[color], color_tolerance=15, chroma_gain=1)
    for index in range(5):
        region = slice(index * 32, (index + 1) * 32)
        if index == selected:
            assert np.any(output.image[:, region] != swatches.image[:, region])
        else:
            np.testing.assert_array_equal(output.image[:, region], swatches.image[:, region])


def test_soft_mask_fades_and_has_finite_support() -> None:
    processor = JerseyColorEnhancementProcessor(colors=["#B53238"])
    # Exact sample, nearby shade, unrelated green: decreasing selection weight.
    bgr = np.array([[[56, 50, 181], [48, 43, 154], [40, 120, 45]]], dtype=np.float32) / 255
    weights = processor._color_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))[0]
    assert weights[0] == pytest.approx(1)
    assert 0 < weights[1] < weights[0]
    assert weights[2] == 0


def test_strength_controls_blend_and_overlapping_colors_do_not_stack(swatches) -> None:
    colors = ["#B53238", "#315EA8"]
    mild = _apply(swatches, colors=colors, strength=0.25).image
    moderate = _apply(swatches, colors=colors, strength=0.5).image
    repeated = _apply(swatches, colors=colors[::-1] + colors, strength=0.5).image
    np.testing.assert_array_equal(moderate, repeated)
    mild_delta = np.abs(mild.astype(int) - swatches.image.astype(int))
    moderate_delta = np.abs(moderate.astype(int) - swatches.image.astype(int))
    assert np.all(moderate_delta >= mild_delta)
    assert moderate_delta.sum() > mild_delta.sum() > 0


@pytest.mark.parametrize("params", [
    {"colors": []}, {"colors": "#FF0000"}, {"colors": ["red"]},
    {"colors": ["#GG0000"]}, {"colors": [None]},
    {"color_tolerance": 0}, {"color_tolerance": 101},
    {"strength": -0.1}, {"strength": 1.1}, {"strength": float("nan")},
    {"chroma_gain": 0.9}, {"chroma_gain": float("inf")},
    {"clip_limit": 0}, {"tile_grid": [0, 8]}, {"tile_grid": [8.5, 8]},
])
def test_invalid_configuration_fails_before_processing(params) -> None:
    with pytest.raises(ValueError):
        JerseyColorEnhancementProcessor(**{"colors": ["#B53238"], **params})


def test_filter_requires_bgr_uint8(swatches) -> None:
    with pytest.raises(ValueError, match="uint8 BGR"):
        _apply(swatches.derived(swatches.image.astype(np.float32)), colors=["#B53238"])


def test_example_has_no_assumed_palette_and_all_conditions_read_metadata() -> None:
    _, configs = expand_screening(REPOSITORY / "configs/lyra-jersey-colors.example.yaml", device=2)
    assert list(configs) == ["baseline", "jersey-mild", "jersey-moderate"]
    assert configs["baseline"].processors == []
    for config in configs.values():
        assert config.models == configs["baseline"].models
        assert config.pitch_filter == configs["baseline"].pitch_filter
        assert not any(config.metadata.jersey_colors.values())
        if config.processors:
            with pytest.raises(ValueError, match="metadata.jersey_colors"):
                config.resolved_processors()
        config.metadata.jersey_colors["home"] = ["#B53238"]
        config.metadata.jersey_colors["away"] = ["#315EA8"]
        pipeline = build_pipeline(config.resolved_processors())
        pipeline.close()
    mild = configs["jersey-mild"].resolved_processors()[0].params
    moderate = configs["jersey-moderate"].resolved_processors()[0].params
    assert mild["colors"] == moderate["colors"]
    assert mild["strength"] == 0.25
    assert moderate["strength"] == 0.5


def test_palette_is_cached_losslessly_and_reaches_model_runner(tmp_path, tiny_video) -> None:
    config = ExperimentConfig.model_validate({
        "name": "jersey-cache", "input": tiny_video, "output_dir": tmp_path / "output",
        "metadata": {"match_id": "test-match", "jersey_colors": {"home": ["#00B400"]}},
        "processors": [{"type": "jersey_color_enhance"}],
        "cache": {"root": tmp_path / "cache"},
        "models": [{"id": "mock", "command": ["{python}", str(REPOSITORY / "runners/mock/run.py")]}],
    })
    from football_pose.video import VideoSource

    runner = ExperimentRunner(config)
    first = runner.prepare()
    assert not first.cache_hit
    assert runner.prepare().cache_hit
    cached = list(iter_artifact(first.artifact_path))
    source = list(VideoSource(tiny_video).frames())
    assert len(cached) == len(source) == 6
    for original, processed in zip(source, cached):
        expected = _apply(original, colors=["#00B400"])
        np.testing.assert_array_equal(processed.image, expected.image)
    assert any(np.any(a.image != b.image) for a, b in zip(source, cached))
    result = runner.run()
    assert result["success"] and result["cache_hit"]
    assert len(read_archive(result["jobs"][0]["outputs"]["parquet"])) == 6
    assert result["configuration"]["metadata"] == config.metadata.model_dump()
    assert first.artifact_manifest.provenance["processors"][0]["params"]["colors"] == ["#00B400"]

    changed_palette = config.model_copy(deep=True)
    changed_palette.metadata.jersey_colors["home"] = ["#0000FF"]
    prepared = ExperimentRunner(changed_palette).prepare()
    assert not prepared.cache_hit
    assert prepared.pipeline_id != first.pipeline_id

    notes_only = config.model_copy(deep=True)
    notes_only.metadata.match_id = "updated-label"
    assert ExperimentRunner(notes_only).prepare().cache_hit

    for key, value in [("colors", ["#FF0000"]), ("strength", 0.5)]:
        changed = config.model_copy(deep=True)
        changed.processors[0].params[key] = value
        prepared = ExperimentRunner(changed).prepare()
        assert not prepared.cache_hit
        assert prepared.pipeline_id != first.pipeline_id


def test_metadata_palette_resolves_from_yaml_without_freezing_the_original(tmp_path, tiny_video) -> None:
    path = tmp_path / "match.yaml"
    path.write_text(yaml.safe_dump({
        "name": "metadata", "input": str(tiny_video),
        "metadata": {"match_id": "home-v-away", "jersey_colors": {
            "home": ["#b53238", "#B53238"], "away": ["#315EA8"], "referee": [],
        }},
        "processors": [{"type": "jersey_color_enhance", "params": {"strength": 0.25}}],
    }))
    config = load_config(path)
    assert config.resolved_processors()[0].params["colors"] == ["#315EA8", "#B53238"]
    assert "colors" not in config.processors[0].params
    config.metadata.jersey_colors = {"home": ["#FFFFFF"]}
    assert config.resolved_processors()[0].params["colors"] == ["#FFFFFF"]
    config.processors[0].params["colors"] = ["#000000"]
    assert config.resolved_processors()[0].params["colors"] == ["#000000"]


@pytest.mark.parametrize("palette", [{"home": ["red"]}, {"away": "#FFFFFF"}, ["#FFFFFF"]])
def test_malformed_metadata_palette_is_rejected(palette) -> None:
    with pytest.raises(ValueError, match="jersey_colors"):
        ExperimentConfig.model_validate({
            "name": "bad-palette", "input": "unused.mp4",
            "metadata": {"jersey_colors": palette},
        })


def test_metadata_inheritance_preflight_and_saved_screening_results(tmp_path, tiny_video, monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({
        "name": "metadata-screen", "input": str(tiny_video),
        "metadata": {"match_id": "test-match", "jersey_colors": {"home": ["#00B400"]}},
        "models": [{"id": "mock", "command": ["{python}", str(REPOSITORY / "runners/mock/run.py")]}],
    }))
    suite = {
        "base_config": str(base), "output_root": str(tmp_path / "results"),
        "cache_root": str(tmp_path / "cache"),
        "methods": [{"name": "baseline"}, {"name": "jersey", "processor": {"type": "jersey_color_enhance"}}],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    _, configs = expand_screening(path, device=0)
    assert configs["jersey"].resolved_processors()[0].params["colors"] == ["#00B400"]

    # The suite can replace a base palette, while keeping other match information.
    suite["metadata"] = {"jersey_colors": {"away": ["#FFFFFF"]}, "checkpoint": "halftime"}
    path.write_text(yaml.safe_dump(suite))
    _, configs = expand_screening(path, device=0)
    assert configs["jersey"].metadata.match_id == "test-match"
    assert configs["jersey"].resolved_processors()[0].params["colors"] == ["#FFFFFF"]
    result = run_screening(path, device=0)
    report = json.loads(result.read_text())
    assert report["success"]
    for condition in report["conditions"]:
        saved = load_config(result.parent / "configs" / f"{condition['method']}.yaml")
        assert saved.metadata == configs[condition["method"]].metadata
        assert condition["summary"]["configuration"]["metadata"] == saved.metadata.model_dump()

    suite["metadata"] = {"jersey_colors": {"home": [], "away": []}}
    path.write_text(yaml.safe_dump(suite))
    with pytest.raises(ValueError, match="metadata.jersey_colors"):
        run_screening(path, device=0, check_only=True)
    assert len(list((tmp_path / "results").iterdir())) == 1
