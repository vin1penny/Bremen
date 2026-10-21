from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import av
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from football_pose.cli import app
from football_pose.pitch_filter import PitchGeometryFrame, PitchGeometryManifest, PreparedPitchGeometry
from football_pose import pitch_comparison as comparison


def _geometry(tmp_path: Path, *, dx: float = 0.0, count: int = 6) -> PreparedPitchGeometry:
    frames = [PitchGeometryFrame(
        frame_index=index, source="unavailable", landmarks=4, inliers=0,
        predicted_keypoints=[
            {"id": 0, "x": 10 + dx, "y": 10., "confidence": .9},
            {"id": 1, "x": 20 + dx, "y": 10., "confidence": .8},
            {"id": 2, "x": 30 + dx, "y": 20., "confidence": .5},
            {"id": 3, "x": 40 + dx, "y": 30., "confidence": .6},
            {"id": 4, "x": 50., "y": 30., "confidence": .1},
            {"id": 5, "x": 0., "y": 0., "confidence": .9},
        ],
    ) for index in range(count)]
    manifest = PitchGeometryManifest(
        geometry_id="test", source_sha256="source", checkpoint_sha256="checkpoint",
        settings={}, frame_count=count, observed_frames=0, bbox_frames=0,
        usable_frames=0, processing_seconds=.1, frames=frames,
    )
    path = tmp_path / f"geometry-{dx}-{count}.json"
    path.write_text(manifest.model_dump_json())
    return PreparedPitchGeometry("test", path, manifest, False)


def test_landmark_yield_is_independent_of_homography_acceptance(tmp_path):
    geometry = _geometry(tmp_path)
    stats = comparison.geometry_statistics(geometry, 64, 48)
    assert stats["visible_landmarks_mean"] == 4
    assert stats["frames_with_four_visible_landmarks"] == 6
    assert stats["accepted_geometry_frames"] == 0
    assert stats["accepted_fit_median_reprojection_cm"] is None
    # Keypoints/segments still render when no calibration is accepted.
    image = comparison.annotate_pitch(np.zeros((48, 64, 3), dtype=np.uint8), geometry.manifest.frames[0], "ours")
    assert tuple(image[64 + 10, 10]) == (255, 0, 255)


def test_disagreement_matches_frame_and_landmark_ids_not_confidence_order(tmp_path):
    pair = {"ours": _geometry(tmp_path), "roboflow": _geometry(tmp_path, dx=3)}
    pair["roboflow"].manifest.frames[0].predicted_keypoints.reverse()
    stats = comparison.landmark_disagreement(pair, 64, 48)
    assert stats["common_visible_landmark_pairs"] == 24
    assert stats["median_pixel_distance"] == 3
    assert stats["p95_pixel_distance"] == 3
    pair["roboflow"].manifest.frames[0].frame_index = 99
    with pytest.raises(ValueError, match="indices"):
        comparison.landmark_disagreement(pair, 64, 48)


def test_render_produces_three_synchronized_videos_without_reducing_resolution(tmp_path, tiny_video):
    pair = {"ours": _geometry(tmp_path), "roboflow": _geometry(tmp_path, dx=3)}
    paths = comparison.render_comparison(comparison.VideoSource(tiny_video), pair, tmp_path, lambda _: None)
    for name, path in paths.items():
        with av.open(path) as container:
            frames = list(container.decode(video=0))
        assert len(frames) == 6
        assert frames[0].width == (128 if name == "side-by-side" else 64)
        assert frames[0].height == 112
        assert float(frames[1].time - frames[0].time) == pytest.approx(.1)
    assert not list(tmp_path.glob("*.partial.mp4"))


@pytest.mark.parametrize("task,shape", [("detect", [32, 3]), ("pose", [17, 3])])
def test_reject_wrong_checkpoint_type(monkeypatch, task, shape):
    model = SimpleNamespace(task=task, model=SimpleNamespace(yaml={"kpt_shape": shape}))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: model))
    with pytest.raises(ValueError, match="32-landmark"):
        comparison.checkpoint_metadata(Path("wrong.pt"))


def _config(tmp_path, tiny_video):
    ours = tmp_path / "ours.pt"
    ours.write_bytes(b"ours")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "name": "pitch", "input": str(tiny_video),
        # A missing pose model must not be required for pitch-only comparison.
        "models": [{"id": "unused", "command": ["unused"], "checkpoint": "absent.pt"}],
        "pitch_filter": {"enabled": True, "checkpoint": str(ours), "image_size": 1280},
    }))
    return path, ours


def test_comparison_freezes_protocol_preserves_weights_and_records_identical_files(tmp_path, tiny_video, monkeypatch):
    config, ours = _config(tmp_path, tiny_video)
    reference = tmp_path / "reference.pt"
    reference.write_bytes(ours.read_bytes())
    monkeypatch.setattr(comparison, "REFERENCE_SHA256", comparison.sha256_file(reference))
    calls = []
    monkeypatch.setattr(comparison, "checkpoint_metadata", lambda _: {"task": "pose", "kpt_shape": [32, 3]})
    monkeypatch.setattr(comparison, "collect_provenance", lambda: {})
    def prepare(self, *, source_path, source_sha256, config):
        calls.append(config)
        geometry = _geometry(tmp_path)
        geometry.manifest.checkpoint_sha256 = comparison.sha256_file(config.checkpoint)
        return geometry
    monkeypatch.setattr(comparison.PitchGeometryStore, "prepare", prepare)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    messages = []
    summary = comparison.compare_pitch(config, reference_checkpoint=reference, output_root=tmp_path / "results",
                                       device=7, progress=messages.append)
    payload = json.loads(summary.read_text())
    assert payload["success"] and payload["identical_checkpoints"]
    assert any("identical" in text for text in messages)
    assert payload["disagreement"]["median_pixel_distance"] == 0
    assert set(payload["videos"]) == {"ours", "roboflow", "side-by-side"}
    assert (summary.parent / "ours-geometry.json").is_file()
    assert (summary.parent / "roboflow-geometry.json").is_file()
    assert (summary.parent / "comparison.md").is_file()
    assert ours.read_bytes() == reference.read_bytes() == b"ours"
    assert len(calls) == 2
    assert all(c.image_size == 640 and c.batch_size == 1 and c.device == "7"
               and c.max_fallback_frames == 0 and c.landmark_confidence == .5 for c in calls)
    assert calls[0].checkpoint == ours
    assert calls[1].checkpoint == reference


def test_cli_missing_reference_is_actionable_and_creates_no_output(tmp_path, tiny_video, monkeypatch):
    config, _ = _config(tmp_path, tiny_video)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    result = CliRunner().invoke(app, ["compare-pitch", str(config), "--reference-checkpoint", str(tmp_path / "absent.pt"),
                                    "--output-root", str(tmp_path / "results"), "--device", "7"])
    assert result.exit_code == 1
    assert "Required comparison input does not exist" in result.output
    assert not (tmp_path / "results").exists()


def test_reservation_mismatch_rejected_before_loading_models(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    with pytest.raises(ValueError, match="outside"):
        comparison.compare_pitch(tmp_path / "absent.yaml", reference_checkpoint=tmp_path / "absent.pt",
                                 output_root=tmp_path, device=7)


def test_wrong_reference_contents_are_rejected_before_loading_models(tmp_path, tiny_video, monkeypatch):
    config, _ = _config(tmp_path, tiny_video)
    reference = tmp_path / "lfs-pointer.pt"
    reference.write_text("version https://git-lfs.github.com/spec/v1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        comparison.compare_pitch(config, reference_checkpoint=reference, output_root=tmp_path / "results", device=7)
    assert not (tmp_path / "results").exists()
