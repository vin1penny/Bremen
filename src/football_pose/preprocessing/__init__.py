from __future__ import annotations

from typing import Any, Callable

from football_pose.configuration import ProcessorSpec
from football_pose.preprocessing.base import Pipeline, Processor
from football_pose.preprocessing.cropping import CropProcessor, UltralyticsDetectionProvider
from football_pose.preprocessing.opencv import (
    BilateralDenoiseProcessor,
    ClaheProcessor,
    GammaProcessor,
    IdentityProcessor,
    MotionDeblurProcessor,
    NlmDenoiseProcessor,
    ResizeProcessor,
    SuperResolutionProcessor,
    UnsharpMaskProcessor,
)
from football_pose.preprocessing.tiling import TileProcessor


ProcessorFactory = Callable[[dict[str, Any]], Processor]


def _factory(processor_type: type[Processor]) -> ProcessorFactory:
    return lambda params: processor_type(**params)  # type: ignore[abstract,call-arg]


REGISTRY: dict[str, ProcessorFactory] = {
    "identity": _factory(IdentityProcessor),
    "resize": _factory(ResizeProcessor),
    "clahe": _factory(ClaheProcessor),
    "gamma": _factory(GammaProcessor),
    "nlm_denoise": _factory(NlmDenoiseProcessor),
    "bilateral_denoise": _factory(BilateralDenoiseProcessor),
    "unsharp_mask": _factory(UnsharpMaskProcessor),
    "super_resolution": _factory(SuperResolutionProcessor),
    "motion_deblur": _factory(MotionDeblurProcessor),
    "tile": _factory(TileProcessor),
}


def register_processor(name: str, factory: ProcessorFactory) -> None:
    if name in REGISTRY:
        raise ValueError(f"processor already registered: {name}")
    REGISTRY[name] = factory


def build_processor(spec: ProcessorSpec) -> Processor:
    if spec.type == "crop":
        params = dict(spec.params)
        detector_params = params.pop("detector", None)
        if not isinstance(detector_params, dict):
            raise ValueError("crop requires a detector parameter mapping")
        detector = UltralyticsDetectionProvider(**detector_params)
        return CropProcessor(detector=detector, **params)
    try:
        factory = REGISTRY[spec.type]
    except KeyError as error:
        raise ValueError(f"unknown processor: {spec.type}") from error
    return factory(dict(spec.params))


def build_pipeline(specs: list[ProcessorSpec]) -> Pipeline:
    return Pipeline([build_processor(spec) for spec in specs])


__all__ = ["Pipeline", "build_pipeline", "build_processor", "register_processor"]
