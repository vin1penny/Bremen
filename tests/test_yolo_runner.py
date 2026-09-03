from __future__ import annotations

import numpy as np
import pytest

from football_pose.domain import FramePacket
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
