from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from football_pose.domain import FramePacket
from runners.yolo import run as yolo_run
from runners.yolo.run import _batch_image_size, _parse_image_size


def _packet(width: int, height: int) -> FramePacket:
    return FramePacket(
        image=np.zeros((height, width, 3), dtype=np.uint8),
        frame_index=0,
        timestamp_seconds=0.0,
        source_width=width,
        source_height=height,
    )


def test_native_image_size_preserves_largest_batch_dimensions() -> None:
    packets = [_packet(1920, 1080), _packet(960, 544)]

    assert _batch_image_size(packets, "native") == (1080, 1920)


def test_fixed_image_size_is_unchanged() -> None:
    assert _batch_image_size([_packet(1920, 1080)], 640) == 640


@pytest.mark.parametrize("value", ["0", "-1", "wide"])
def test_invalid_image_size_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="positive integer or 'native'"):
        _parse_image_size(value)


def test_image_size_parser_accepts_native_or_integer() -> None:
    assert _parse_image_size("native") == "native"
    assert _parse_image_size("1920") == 1920


@pytest.mark.parametrize("nms_free", [False, True])
def test_runner_selects_nms_free_head_only_when_requested(
    monkeypatch, tmp_path, nms_free: bool
) -> None:
    options = []

    class FakeYOLO:
        def __init__(self, checkpoint):
            assert checkpoint == "weights.pt"

        def __call__(self, images, **kwargs):
            options.append(kwargs)
            assert len(images) == 1
            return [SimpleNamespace(keypoints=None)]

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    monkeypatch.setattr(yolo_run, "iter_artifact", lambda *args, **kwargs: iter([_packet(64, 48)]))
    arguments = [
        "run.py", "--input-artifact", str(tmp_path / "artifact"),
        "--output-jsonl", str(tmp_path / "predictions.jsonl"),
        "--experiment-id", "e", "--pipeline-id", "p", "--model-id", "yolo26-pose",
        "--source-video-id", "s", "--checkpoint", "weights.pt",
    ]
    if nms_free:
        arguments.append("--nms-free")
    monkeypatch.setattr(sys, "argv", arguments)
    yolo_run.main()
    assert options[0].get("nms") is (False if nms_free else None)
