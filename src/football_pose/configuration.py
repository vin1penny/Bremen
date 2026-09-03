from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProcessorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    command: list[str] = Field(min_length=1)
    checkpoint: Path | None = None
    batch_size: int = Field(default=1, ge=1)
    min_batch_size: int = Field(default=1, ge=1)
    devices: list[int] = Field(default_factory=list)
    timeout_seconds: int = Field(default=3600, ge=1)
    environment: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def min_batch_not_larger(self) -> "ModelSpec":
        if self.min_batch_size > self.batch_size:
            raise ValueError("min_batch_size cannot exceed batch_size")
        return self


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path = Path("artifacts")
    format: Literal["auto", "ffv1", "png_shards"] = "auto"
    shard_size: int = Field(default=256, ge=1)
    pin: bool = True


class VideoOutputConfig(BaseModel):
    """Settings for human-reviewable, annotated experiment videos."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    root: Path = Path("rendered-videos")
    codec: str = Field(default="libx264", min_length=1)
    keypoint_confidence: float = Field(default=0.2, ge=0.0, le=1.0)
    draw_bboxes: bool = True
    draw_regions: bool = True


class PitchFilterConfig(BaseModel):
    """Shared pitch-localization and pose-filtering settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    checkpoint: Path | None = None
    cache_root: Path = Path("pitch-cache")
    device: str = "0"
    image_size: int = Field(default=640, ge=32)
    batch_size: int = Field(default=16, ge=1)
    detection_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    landmark_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    minimum_landmarks: int = Field(default=4, ge=4, le=32)
    ransac_threshold_cm: float = Field(default=150.0, gt=0.0)
    minimum_inlier_ratio: float = Field(default=0.6, gt=0.0, le=1.0)
    maximum_median_reprojection_cm: float = Field(default=200.0, gt=0.0)
    max_fallback_frames: int = Field(default=5, ge=0)
    pitch_margin_cm: float = Field(default=50.0, ge=0.0)
    pitch_bbox_margin_px: float = Field(default=10.0, ge=0.0)
    ankle_confidence: float = Field(default=0.2, ge=0.0, le=1.0)
    deduplication_iou: float = Field(default=0.5, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def checkpoint_required_when_enabled(self) -> "PitchFilterConfig":
        if self.enabled and self.checkpoint is None:
            raise ValueError("pitch_filter.checkpoint is required when enabled")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    input: Path
    output_dir: Path = Path("experiment-output")
    processors: list[ProcessorSpec] = Field(default_factory=list)
    models: list[ModelSpec] = Field(default_factory=list)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    video_output: VideoOutputConfig = Field(default_factory=VideoOutputConfig)
    pitch_filter: PitchFilterConfig = Field(default_factory=PitchFilterConfig)
    fail_fast: bool = False

    @model_validator(mode="after")
    def validate_unique_components(self) -> "ExperimentConfig":
        crop_count = sum(spec.type == "crop" for spec in self.processors)
        if crop_count > 1:
            raise ValueError("a pipeline may contain at most one crop boundary")
        model_ids = [model.id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model ids must be unique")
        return self


def _resolve_path(base: Path, value: Path | None) -> Path | None:
    if value is None or value.is_absolute():
        return value
    return (base / value).resolve()


def _resolve_checkpoint_parameters(value: Any, base: Path) -> Any:
    if isinstance(value, dict):
        resolved: dict[str, Any] = {}
        for key, item in value.items():
            if key == "checkpoint" and isinstance(item, (str, Path)):
                resolved[key] = str(_resolve_path(base, Path(item)))
            else:
                resolved[key] = _resolve_checkpoint_parameters(item, base)
        return resolved
    if isinstance(value, list):
        return [_resolve_checkpoint_parameters(item, base) for item in value]
    return value


def _parameter_checkpoints(value: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "checkpoint" and isinstance(item, (str, Path)):
                paths.append(Path(item))
            else:
                paths.extend(_parameter_checkpoints(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_parameter_checkpoints(item))
    return paths


def load_config(path: str | Path, *, check_paths: bool = True) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    config = ExperimentConfig.model_validate(raw)
    base = config_path.parent
    config.input = _resolve_path(base, config.input)  # type: ignore[assignment]
    config.output_dir = _resolve_path(base, config.output_dir)  # type: ignore[assignment]
    config.cache.root = _resolve_path(base, config.cache.root)  # type: ignore[assignment]
    config.video_output.root = _resolve_path(base, config.video_output.root)  # type: ignore[assignment]
    config.pitch_filter.checkpoint = _resolve_path(base, config.pitch_filter.checkpoint)
    config.pitch_filter.cache_root = _resolve_path(base, config.pitch_filter.cache_root)  # type: ignore[assignment]
    for processor in config.processors:
        processor.params = _resolve_checkpoint_parameters(processor.params, base)
    for model in config.models:
        model.checkpoint = _resolve_path(base, model.checkpoint)
    if check_paths:
        if not config.input.is_file():
            raise FileNotFoundError(f"input video does not exist: {config.input}")
        for model in config.models:
            if model.checkpoint is not None and not model.checkpoint.is_file():
                raise FileNotFoundError(
                    f"checkpoint for model {model.id!r} does not exist: {model.checkpoint}"
                )
        if (
            config.pitch_filter.enabled
            and config.pitch_filter.checkpoint is not None
            and not config.pitch_filter.checkpoint.is_file()
        ):
            raise FileNotFoundError(
                "pitch-filter checkpoint does not exist: "
                f"{config.pitch_filter.checkpoint}"
            )
        for processor in config.processors:
            for checkpoint in _parameter_checkpoints(processor.params):
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"checkpoint for processor {processor.type!r} does not exist: {checkpoint}"
                    )
    return config
