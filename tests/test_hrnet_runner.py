from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from football_pose.domain import FramePacket
from football_pose.preprocessing.cropping import Detection
from runners.hrnet import run as hrnet_run
import football_pose.hrnet_input as hrnet_input


def _run(tmp_path, monkeypatch, packets, *, detections=None, detector=True, pose_fails=False):
    calls = {"detector_init": [], "detect": [], "closed": 0, "pose": []}
    class Detector:
        def __init__(self, **params):
            calls["detector_init"].append(params)
        def detect(self, packet):
            calls["detect"].append(packet.frame_index)
            return detections if detections is not None else [Detection((4, 8, 40, 50), .9)]
        def close(self):
            calls["closed"] += 1
    class Inferencer:
        def __init__(self, **params):
            assert params["det_model"] == "whole_image"
            assert params["device"] == "cuda:0"
            assert params["pose2d_weights"] == str(tmp_path / "pose.pth")
        def __call__(self, images, **kwargs):
            calls["pose"].extend(image.shape for image in images)
            if pose_fails:
                raise RuntimeError("fixture HRNet failure")
            for _ in images:
                yield {"predictions": [[{
                    "keypoints": [[10., 12.]] * 17, "keypoint_scores": [.8] * 17,
                }]]}
    monkeypatch.setattr(hrnet_input, "UltralyticsDetectionProvider", Detector)
    module = ModuleType("mmpose")
    apis = ModuleType("mmpose.apis")
    apis.MMPoseInferencer = Inferencer
    monkeypatch.setitem(sys.modules, "mmpose", module)
    monkeypatch.setitem(sys.modules, "mmpose.apis", apis)
    monkeypatch.setattr(hrnet_run, "iter_artifact", lambda *args, **kwargs: iter(packets))
    (tmp_path / "pose.pth").write_bytes(b"fixture")
    arguments = [
        "run.py", "--input-artifact", str(tmp_path / "artifact"),
        "--output-jsonl", str(tmp_path / "predictions.jsonl"),
        "--experiment-id", "e", "--pipeline-id", "p", "--model-id", "hrnet-w32",
        "--source-video-id", "s", "--checkpoint", str(tmp_path / "pose.pth"),
        "--crop-padding", "0", "--crop-min-size", "1", "--batch-size", "8",
    ]
    if detector:
        arguments += ["--detector-checkpoint", str(tmp_path / "detector.pt")]
    monkeypatch.setattr(sys, "argv", arguments)
    return calls


def _frame(**kwargs):
    return FramePacket(
        np.zeros((64, 80, 3), dtype=np.uint8), 3, .3, 40, 32,
        to_source=np.diag([.5, .5, 1]), **kwargs,
    )


def test_full_frame_uses_existing_yolo_and_maps_crop_keypoints_to_source(tmp_path, monkeypatch, capsys):
    calls = _run(tmp_path, monkeypatch, [_frame()])
    hrnet_run.main()
    row = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert calls["detect"] == [3]
    assert calls["closed"] == 1
    params = calls["detector_init"][0]
    assert params["checkpoint"] == str(tmp_path / "detector.pt")
    assert params["image_size"] == 1920
    assert params["class_ids"] == [1, 2, 3]
    assert params["device"] == "0"
    assert not params["track"]
    assert calls["pose"] == [(42, 36, 3)]
    assert row["keypoints"][0] == {"x": 7., "y": 10., "confidence": .8, "visibility": None}
    assert row["source_bbox"] == [2., 4., 20., 25.]
    assert row["person_score"] == .9
    assert row["frame_index"] == 3
    stats = json.loads(capsys.readouterr().out)["hrnet_input"]
    assert stats["generated_person_crops"] == 1
    assert stats["detector_calls"] == 1


def test_shared_person_crops_are_not_detected_again(tmp_path, monkeypatch):
    packet = _frame(source_bbox=(2, 4, 20, 25), crop_id="shared-person-7", track_id="7",
                    metadata={"detection_score": .95})
    calls = _run(tmp_path, monkeypatch, [packet], detector=False)
    hrnet_run.main()
    assert calls["detector_init"] == []
    assert calls["detect"] == []
    row = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert row["track_id"] == "7"
    assert row["crop_id"] == "shared-person-7"
    assert row["person_score"] == .95


def test_tiles_still_require_detection_and_get_unique_person_crop_ids(tmp_path, monkeypatch):
    packet = _frame(crop_id="tile-r00-c00")
    calls = _run(tmp_path, monkeypatch, [packet])
    hrnet_run.main()
    assert calls["detect"] == [3]
    row = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert row["crop_id"] == "tile-r00-c00-person-0000"


def test_no_player_boxes_produce_no_invented_person_pose(tmp_path, monkeypatch):
    calls = _run(tmp_path, monkeypatch, [_frame()], detections=[])
    hrnet_run.main()
    assert calls["pose"] == []
    assert (tmp_path / "predictions.jsonl").read_text() == ""
    assert calls["closed"] == 1


def test_full_frame_without_player_weights_cannot_fall_back_to_generic_detector(tmp_path, monkeypatch):
    calls = _run(tmp_path, monkeypatch, [_frame()], detector=False)
    with pytest.raises(ValueError, match="detector-checkpoint"):
        hrnet_run.main()
    assert calls["pose"] == []


def test_detector_is_released_when_hrnet_fails(tmp_path, monkeypatch):
    calls = _run(tmp_path, monkeypatch, [_frame()], pose_fails=True)
    with pytest.raises(RuntimeError, match="HRNet failure"):
        hrnet_run.main()
    assert calls["closed"] == 1
