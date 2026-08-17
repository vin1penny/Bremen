from __future__ import annotations

import numpy as np
import pytest

from football_pose.domain import FramePacket
from football_pose.preprocessing.base import Pipeline, ProcessorContext
from football_pose.preprocessing.cropping import CropProcessor, Detection
from football_pose.preprocessing.opencv import (
    BilateralDenoiseProcessor,
    ClaheProcessor,
    GammaProcessor,
    MotionDeblurProcessor,
    NlmDenoiseProcessor,
    ResizeProcessor,
)


class StaticDetector:
    def detect(self, packet: FramePacket) -> list[Detection]:
        del packet
        return [Detection((4.0, 5.0, 20.0, 25.0), 0.9, "7")]


@pytest.fixture
def packet() -> FramePacket:
    random = np.random.default_rng(7)
    return FramePacket(
        image=random.integers(0, 256, (32, 40, 3), dtype=np.uint8),
        frame_index=3,
        timestamp_seconds=0.3,
        source_width=40,
        source_height=32,
    )


@pytest.mark.parametrize(
    "processor",
    [
        ClaheProcessor(),
        GammaProcessor(gamma=1.2),
        NlmDenoiseProcessor(template_window=3, search_window=7),
        BilateralDenoiseProcessor(diameter=3),
        MotionDeblurProcessor(length=3),
    ],
)
def test_pixel_processors_preserve_shape_type_and_range(packet: FramePacket, processor: object) -> None:
    output = processor.process(packet, ProcessorContext("e", "p"))[0]
    assert output.image.shape == packet.image.shape
    assert output.image.dtype == np.uint8
    assert output.image.min() >= 0
    assert output.image.max() <= 255


def test_resize_composes_coordinate_transform(packet: FramePacket) -> None:
    output = ResizeProcessor(width=80, height=64).process(
        packet, ProcessorContext("e", "p")
    )[0]
    assert output.image.shape == (64, 80, 3)
    np.testing.assert_allclose(output.points_to_source([[20, 16]]), [[10, 8]])


def test_crop_after_resize_returns_original_frame_coordinates(packet: FramePacket) -> None:
    pipeline = Pipeline(
        [ResizeProcessor(scale=2), CropProcessor(detector=StaticDetector(), padding_ratio=0)]
    )
    output = list(pipeline.process([packet], ProcessorContext("e", "p")))[0]
    assert output.crop_id == "000000003-0000"
    assert output.track_id == "7"
    assert output.source_bbox == pytest.approx((2.0, 2.5, 10.0, 12.5))
    np.testing.assert_allclose(output.points_to_source([[0, 0]]), [[2, 2.5]])
