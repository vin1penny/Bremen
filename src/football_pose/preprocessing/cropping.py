from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from football_pose.domain import FramePacket
from football_pose.preprocessing.base import ProcessorContext


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    track_id: str | None = None


class DetectionProvider(Protocol):
    def detect(self, packet: FramePacket) -> list[Detection]: ...


class UltralyticsDetectionProvider:
    """Lazy YOLO + ByteTrack provider so core-only workflows stay lightweight."""

    def __init__(
        self,
        *,
        checkpoint: str,
        confidence: float = 0.3,
        class_ids: list[int] | None = None,
        track: bool = True,
        device: str = "0",
        image_size: int = 1920,
        expected_class_names: dict[int, str] | None = None,
    ) -> None:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"player-detector checkpoint not found: {checkpoint}")
        if not 0 <= confidence <= 1 or image_size < 32:
            raise ValueError("invalid player-detector confidence or image_size")
        if class_ids is not None and (not class_ids or min(class_ids) < 0):
            raise ValueError("class_ids must contain non-negative class IDs")
        self.checkpoint = checkpoint
        self.model = None
        self.confidence = confidence
        self.class_ids = [2, 3] if class_ids is None else class_ids
        self.device = device
        self.image_size = image_size
        self.expected_class_names = {int(key): value for key, value in (expected_class_names or {}).items()}
        self.track = track
        self.tracker = None

    def validate(self) -> None:
        """Inspect task/classes on CPU before committing to a long screening run."""
        from ultralytics import YOLO

        if self.model is None:
            self.model = YOLO(self.checkpoint)
            if self.model.task != "detect":
                raise ValueError("learned cropping requires player-detection weights, not pose weights")
            if any(index not in self.model.names for index in self.class_ids):
                raise ValueError("class_ids are absent from the player detector's class map")
            if any(self.model.names.get(index) != name for index, name in self.expected_class_names.items()):
                raise ValueError("player detector class names do not match expected_class_names")

    def detect(self, packet: FramePacket) -> list[Detection]:
        import supervision as sv

        self.validate()
        if self.track and self.tracker is None:
            self.tracker = sv.ByteTrack()
            self.tracker.reset()

        result = self.model(
            packet.image, conf=self.confidence, imgsz=self.image_size,
            device=self.device, verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        if detections.class_id is not None:
            detections = detections[np.isin(detections.class_id, self.class_ids)]
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        if self.tracker is not None:
            detections = self.tracker.update_with_detections(detections)
        output: list[Detection] = []
        tracker_ids = detections.tracker_id
        confidences = detections.confidence
        for index, bbox in enumerate(detections.xyxy):
            track_id = None if tracker_ids is None else str(int(tracker_ids[index]))
            score = 1.0 if confidences is None else float(confidences[index])
            output.append(Detection(tuple(float(value) for value in bbox), score, track_id))
        return output

    def close(self) -> None:
        """Release host-side detector VRAM before the pose containers start."""
        if self.model is None:
            return
        self.model.to("cpu")
        self.model = None
        self.tracker = None
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


class CropProcessor:
    name = "crop"

    def __init__(
        self,
        *,
        detector: DetectionProvider,
        padding_ratio: float = 0.1,
        min_size: int = 8,
    ) -> None:
        if padding_ratio < 0 or min_size < 1:
            raise ValueError("invalid crop parameters")
        self.detector = detector
        self.padding_ratio = padding_ratio
        self.min_size = min_size

    def close(self) -> None:
        close = getattr(self.detector, "close", None)
        if close is not None:
            close()

    def validate(self) -> None:
        validate = getattr(self.detector, "validate", None)
        if validate is not None:
            validate()

    def process(self, packet: FramePacket, context: ProcessorContext) -> list[FramePacket]:
        del context
        crops: list[FramePacket] = []
        for detection_index, detection in enumerate(self.detector.detect(packet)):
            x1, y1, x2, y2 = detection.bbox
            pad_x = (x2 - x1) * self.padding_ratio
            pad_y = (y2 - y1) * self.padding_ratio
            left = max(0, int(np.floor(x1 - pad_x)))
            top = max(0, int(np.floor(y1 - pad_y)))
            right = min(packet.width, int(np.ceil(x2 + pad_x)))
            bottom = min(packet.height, int(np.ceil(y2 + pad_y)))
            if right - left < self.min_size or bottom - top < self.min_size:
                continue
            crop_id = (
                f"{packet.crop_id}-person-{detection_index:04d}"
                if packet.crop_id is not None
                else f"{packet.frame_index:09d}-{detection_index:04d}"
            )
            current_to_previous = np.array(
                [[1, 0, left], [0, 1, top], [0, 0, 1]], dtype=np.float64
            )
            source_bbox_points = packet.points_to_source(
                np.array([[left, top], [right, bottom]], dtype=np.float64)
            )
            source_bbox = (
                float(source_bbox_points[0, 0]),
                float(source_bbox_points[0, 1]),
                float(source_bbox_points[1, 0]),
                float(source_bbox_points[1, 1]),
            )
            crops.append(
                packet.derived(
                    packet.image[top:bottom, left:right].copy(),
                    current_to_previous,
                    source_id=crop_id,
                    crop_id=crop_id,
                    track_id=detection.track_id,
                    source_bbox=source_bbox,
                    metadata={**packet.metadata, "detection_score": detection.score},
                )
            )
        return crops
