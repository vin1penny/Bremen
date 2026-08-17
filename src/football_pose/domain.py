from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


IDENTITY_TRANSFORM = np.eye(3, dtype=np.float64)


@dataclass(slots=True)
class FramePacket:
    """An image plus the transform from its pixels to the source video frame."""

    image: np.ndarray
    frame_index: int
    timestamp_seconds: float
    source_width: int
    source_height: int
    to_source: np.ndarray = field(default_factory=lambda: IDENTITY_TRANSFORM.copy())
    source_id: str | None = None
    crop_id: str | None = None
    track_id: str | None = None
    source_bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("FramePacket.image must be an HxWx3 array")
        if self.image.size == 0:
            raise ValueError("FramePacket.image must not be empty")
        matrix = np.asarray(self.to_source, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("to_source must be a finite 3x3 matrix")
        self.to_source = matrix
        if self.source_id is None:
            suffix = self.crop_id or "frame"
            self.source_id = f"{self.frame_index:09d}-{suffix}"

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def derived(
        self,
        image: np.ndarray,
        current_to_previous: np.ndarray | None = None,
        **changes: Any,
    ) -> "FramePacket":
        transform = IDENTITY_TRANSFORM if current_to_previous is None else current_to_previous
        return replace(
            self,
            image=image,
            to_source=self.to_source @ np.asarray(transform, dtype=np.float64),
            **changes,
        )

    def points_to_source(self, points: np.ndarray) -> np.ndarray:
        points_array = np.asarray(points, dtype=np.float64)
        if points_array.size == 0:
            return points_array.reshape((-1, 2))
        flat = points_array.reshape((-1, 2))
        homogeneous = np.column_stack((flat, np.ones(len(flat))))
        mapped = (self.to_source @ homogeneous.T).T
        mapped = mapped[:, :2] / mapped[:, 2:3]
        return mapped.reshape(points_array.shape)


class Keypoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    confidence: float = Field(ge=0.0, le=1.0)
    visibility: float | None = Field(default=None, ge=0.0, le=2.0)

    @field_validator("x", "y")
    @classmethod
    def finite_coordinate(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("keypoint coordinates must be finite")
        return float(value)


class PredictionRecord(BaseModel):
    """Canonical runner output using the COCO 17-keypoint order."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    pipeline_id: str
    model_id: str
    source_video_id: str
    source_id: str
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    crop_id: str | None = None
    track_id: str | None = None
    person_id: str
    source_bbox: tuple[float, float, float, float] | None = None
    keypoints: list[Keypoint]
    coordinate_space: Literal["original_frame"] = "original_frame"
    inference_time_ms: float = Field(ge=0.0)

    @field_validator("keypoints")
    @classmethod
    def exactly_coco_17(cls, value: list[Keypoint]) -> list[Keypoint]:
        if len(value) != 17:
            raise ValueError(f"expected 17 COCO keypoints, received {len(value)}")
        return value

    @field_validator("source_bbox")
    @classmethod
    def ordered_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return value
        x1, y1, x2, y2 = value
        if not np.isfinite((x1, y1, x2, y2)).all() or x2 < x1 or y2 < y1:
            raise ValueError("source_bbox must contain finite, ordered coordinates")
        return tuple(float(item) for item in value)


class FrameManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0.0)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    to_source: list[float]
    crop_id: str | None = None
    track_id: str | None = None
    source_bbox: tuple[float, float, float, float] | None = None
    member: str | None = None
    shard: str | None = None
    sequence_index: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("to_source")
    @classmethod
    def nine_transform_values(cls, value: list[float]) -> list[float]:
        if len(value) != 9 or not np.isfinite(value).all():
            raise ValueError("to_source must contain nine finite values")
        return [float(item) for item in value]
