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
    SuperResolutionProcessor,
    UnsharpMaskProcessor,
)
from football_pose.preprocessing.tiling import TileProcessor


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
        UnsharpMaskProcessor(kernel_size=3, sigma=0.8, amount=0.75),
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


@pytest.mark.parametrize(
    "params",
    [
        {"kernel_size": 2},
        {"kernel_size": 1},
        {"sigma": 0},
        {"amount": 0},
    ],
)
def test_unsharp_mask_rejects_invalid_configuration(params: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        UnsharpMaskProcessor(**params)


def test_crop_after_resize_returns_original_frame_coordinates(packet: FramePacket) -> None:
    pipeline = Pipeline(
        [ResizeProcessor(scale=2), CropProcessor(detector=StaticDetector(), padding_ratio=0)]
    )
    output = list(pipeline.process([packet], ProcessorContext("e", "p")))[0]
    assert output.crop_id == "000000003-0000"
    assert output.track_id == "7"
    assert output.source_bbox == pytest.approx((2.0, 2.5, 10.0, 12.5))
    np.testing.assert_allclose(output.points_to_source([[0, 0]]), [[2, 2.5]])


def test_tile_is_row_major_and_maps_back_to_source(packet: FramePacket) -> None:
    outputs = TileProcessor(rows=2, columns=2).process(
        packet, ProcessorContext("e", "p")
    )

    assert [output.crop_id for output in outputs] == [
        "000000003-tile-r00-c00",
        "000000003-tile-r00-c01",
        "000000003-tile-r01-c00",
        "000000003-tile-r01-c01",
    ]
    assert [output.image.shape for output in outputs] == [(16, 20, 3)] * 4
    assert outputs[3].source_bbox is None
    np.testing.assert_allclose(outputs[3].points_to_source([[0, 0]]), [[20, 16]])
    assert outputs[3].metadata["tile"]["input_bounds"] == [20, 16, 40, 32]
    assert outputs[3].metadata["tile"]["source_bounds"] == [20.0, 16.0, 40.0, 32.0]


def test_tile_overlap_expands_internal_edges_without_losing_coverage(
    packet: FramePacket,
) -> None:
    left, right = TileProcessor(rows=1, columns=2, overlap_ratio=0.1).process(
        packet, ProcessorContext("e", "p")
    )

    assert left.metadata["tile"]["source_bounds"] == [0.0, 0.0, 21.0, 32.0]
    assert right.metadata["tile"]["source_bounds"] == [19.0, 0.0, 40.0, 32.0]
    assert left.image.shape == right.image.shape == (32, 21, 3)


def test_crop_after_tile_has_unique_nested_identity(packet: FramePacket) -> None:
    pipeline = Pipeline(
        [
            TileProcessor(rows=1, columns=2),
            CropProcessor(detector=StaticDetector(), padding_ratio=0),
        ]
    )
    outputs = list(pipeline.process([packet], ProcessorContext("e", "p")))

    assert [output.crop_id for output in outputs] == [
        "000000003-tile-r00-c00-person-0000",
        "000000003-tile-r00-c01-person-0000",
    ]
    assert len({output.source_id for output in outputs}) == 2


@pytest.mark.parametrize(
    "params",
    [
        {"rows": 0},
        {"columns": 0},
        {"overlap_ratio": -0.1},
        {"overlap_ratio": 0.5},
    ],
)
def test_tile_rejects_invalid_configuration(params: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TileProcessor(**params)


@pytest.mark.parametrize("params", [
    {"length": 0}, {"length": 4}, {"nsr": 0},
    {"angle_degrees": float("nan")}, {"nsr": float("inf")},
])
def test_deblur_rejects_invalid_parameters(params) -> None:
    with pytest.raises(ValueError):
        MotionDeblurProcessor(**params)


@pytest.mark.parametrize("params", [
    {"template_window": 2}, {"search_window": 2},
    {"template_window": 9, "search_window": 7}, {"h": float("nan")},
])
def test_nlm_rejects_invalid_parameters(params) -> None:
    with pytest.raises(ValueError):
        NlmDenoiseProcessor(**params)


def test_sr_rejects_wrong_checksum_before_loading_graph(tmp_path) -> None:
    path = tmp_path / "model.pb"
    path.write_bytes(b"not a graph")
    with pytest.raises(ValueError, match="SHA-256"):
        SuperResolutionProcessor(checkpoint=str(path), checkpoint_sha256="0" * 64)


def test_crop_detector_is_lazy_uses_explicit_settings_and_filters_classes(tmp_path, monkeypatch, packet) -> None:
    import supervision as sv
    import ultralytics
    from football_pose.preprocessing.cropping import UltralyticsDetectionProvider

    checkpoint = tmp_path / "detector.pt"
    checkpoint.write_bytes(b"fixture")
    calls = []
    class Model:
        task = "detect"
        names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
        def __init__(self, path):
            calls.append(path)
        def __call__(self, image, **kwargs):
            calls.append(kwargs)
            return [object()]
        def to(self, device):
            calls.append(device)
    monkeypatch.setattr(ultralytics, "YOLO", Model)
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 2, 2], [3, 3, 5, 5], [8, 8, 10, 10], [15, 15, 20, 20]], dtype=float),
        confidence=np.array([.9] * 4), class_id=np.arange(4),
    )
    monkeypatch.setattr(sv.Detections, "from_ultralytics", lambda _: detections)
    provider = UltralyticsDetectionProvider(
        checkpoint=str(checkpoint), device="4", image_size=1920,
        class_ids=[1, 2, 3], track=True,
        expected_class_names={1: "goalkeeper", 2: "player", 3: "referee"},
    )
    assert calls == []
    first = provider.detect(packet)
    assert len(first) == 3
    assert all(detection.track_id is not None for detection in first)
    second = provider.detect(packet.derived(packet.image, frame_index=4, timestamp_seconds=0.4))
    assert [detection.track_id for detection in first] == [detection.track_id for detection in second]
    assert calls[1]["device"] == "4"
    assert calls[1]["imgsz"] == 1920
    provider.close()
    assert provider.model is None
    assert calls[-1] == "cpu"


def test_pipeline_closes_external_processors() -> None:
    calls = []
    class External:
        def close(self):
            calls.append("closed")
    Pipeline([External()]).close()
    assert calls == ["closed"]


@pytest.mark.parametrize("task,names,message", [
    ("pose", {1: "goalkeeper", 2: "player", 3: "referee"}, "pose weights"),
    ("detect", {1: "bicycle", 2: "car", 3: "motorcycle"}, "expected_class_names"),
])
def test_crop_preflight_rejects_wrong_task_or_class_map(tmp_path, monkeypatch, task, names, message) -> None:
    import ultralytics
    from football_pose.preprocessing.cropping import UltralyticsDetectionProvider

    checkpoint = tmp_path / "wrong.pt"
    checkpoint.write_bytes(b"fixture")
    class Model:
        def to(self, device):
            pass
    model = Model()
    model.task = task
    model.names = names
    monkeypatch.setattr(ultralytics, "YOLO", lambda _: model)
    provider = UltralyticsDetectionProvider(
        checkpoint=str(checkpoint), class_ids=[1, 2, 3],
        expected_class_names={1: "goalkeeper", 2: "player", 3: "referee"},
    )
    try:
        with pytest.raises(ValueError, match=message):
            provider.validate()
    finally:
        provider.close()
