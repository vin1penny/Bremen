from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from football_pose.domain import FramePacket
from football_pose.preprocessing.base import ProcessorContext


INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def _validate_bgr_u8(packet: FramePacket) -> None:
    if packet.image.dtype != np.uint8 or packet.image.ndim != 3 or packet.image.shape[2] != 3:
        raise ValueError("OpenCV processors require uint8 BGR images")


class IdentityProcessor:
    name = "identity"

    def process(self, packet: FramePacket, context: ProcessorContext) -> list[FramePacket]:
        del context
        return [packet]


class ResizeProcessor:
    name = "resize"

    def __init__(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        scale: float | None = None,
        interpolation: str = "lanczos",
    ) -> None:
        if scale is None and width is None and height is None:
            raise ValueError("resize requires scale, width, or height")
        if scale is not None and (scale <= 0 or width is not None or height is not None):
            raise ValueError("scale must be positive and cannot be combined with width/height")
        if (width is not None and width <= 0) or (height is not None and height <= 0):
            raise ValueError("resize dimensions must be positive")
        if interpolation not in INTERPOLATIONS:
            raise ValueError(f"unknown interpolation: {interpolation}")
        self.width = width
        self.height = height
        self.scale = scale
        self.interpolation = INTERPOLATIONS[interpolation]

    def process(self, packet: FramePacket, context: ProcessorContext) -> list[FramePacket]:
        del context
        _validate_bgr_u8(packet)
        if self.scale is not None:
            width = max(1, round(packet.width * self.scale))
            height = max(1, round(packet.height * self.scale))
        elif self.width is not None and self.height is not None:
            width, height = self.width, self.height
        elif self.width is not None:
            width = self.width
            height = max(1, round(packet.height * width / packet.width))
        else:
            assert self.height is not None
            height = self.height
            width = max(1, round(packet.width * height / packet.height))
        resized = cv2.resize(packet.image, (width, height), interpolation=self.interpolation)
        current_to_previous = np.array(
            [[packet.width / width, 0, 0], [0, packet.height / height, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        return [packet.derived(resized, current_to_previous)]


class ClaheProcessor:
    name = "clahe"

    def __init__(self, *, clip_limit: float = 2.0, tile_grid: list[int] = [8, 8]) -> None:
        if clip_limit <= 0 or len(tile_grid) != 2 or min(tile_grid) <= 0:
            raise ValueError("invalid CLAHE parameters")
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))

    def process(self, packet: FramePacket, context: ProcessorContext) -> list[FramePacket]:
        del context
        _validate_bgr_u8(packet)
        lab = cv2.cvtColor(packet.image, cv2.COLOR_BGR2LAB)
        luminance, a_channel, b_channel = cv2.split(lab)
        enhanced = cv2.merge((self.clahe.apply(luminance), a_channel, b_channel))
        return [packet.derived(cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR))]


class GammaProcessor:
    name = "gamma"

    def __init__(self, *, gamma: float = 1.0) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.table = np.array(
            [((value / 255.0) ** (1.0 / gamma)) * 255 for value in range(256)],
            dtype=np.uint8,
        )

    def process(self, packet: FramePacket, context: ProcessorContext) -> list[FramePacket]:
        del context
        _validate_bgr_u8(packet)
        return [packet.derived(cv2.LUT(packet.image, self.table))]


class NlmDenoiseProcessor:
    name = "nlm_denoise"

    def __init__(
        self,
        *,
        h: float = 7.0,
        h_color: float = 7.0,
        template_window: int = 7,
        search_window: int = 21,
    ) -> None:
        if min(h, h_color) < 0 or template_window <= 0 or search_window <= 0:
            raise ValueError("invalid NLM parameters")
        self.params = (h, h_color, template_window, search_window)

    def process(self, packet: FramePacket, context: ProcessorContext) -> list[FramePacket]:
        del context
        _validate_bgr_u8(packet)
        return [packet.derived(cv2.fastNlMeansDenoisingColored(packet.image, None, *self.params))]


class BilateralDenoiseProcessor:
    name = "bilateral_denoise"

    def __init__(self, *, diameter: int = 7, sigma_color: float = 50, sigma_space: float = 50) -> None:
        if diameter <= 0 or min(sigma_color, sigma_space) <= 0:
            raise ValueError("invalid bilateral parameters")
        self.params = (diameter, sigma_color, sigma_space)

    def process(self, packet: FramePacket, context: ProcessorContext) -> list[FramePacket]:
        del context
        _validate_bgr_u8(packet)
        return [packet.derived(cv2.bilateralFilter(packet.image, *self.params))]


class SuperResolutionProcessor:
    name = "super_resolution"

    def __init__(self, *, checkpoint: str, algorithm: str = "fsrcnn", scale: int = 2) -> None:
        model_path = Path(checkpoint)
        if not model_path.is_file():
            raise FileNotFoundError(f"super-resolution checkpoint not found: {model_path}")
        if algorithm not in {"edsr", "espcn", "fsrcnn", "lapsrn"} or scale <= 1:
            raise ValueError("invalid super-resolution model or scale")
        self.scale = scale
        self.model = cv2.dnn_superres.DnnSuperResImpl_create()
        self.model.readModel(str(model_path))
        self.model.setModel(algorithm, scale)

    def process(self, packet: FramePacket, context: ProcessorContext) -> list[FramePacket]:
        del context
        _validate_bgr_u8(packet)
        upscaled = self.model.upsample(packet.image)
        current_to_previous = np.array(
            [[1 / self.scale, 0, 0], [0, 1 / self.scale, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        return [packet.derived(upscaled, current_to_previous)]


class MotionDeblurProcessor:
    """Classical Wiener deconvolution for a configured linear motion PSF."""

    name = "motion_deblur"

    def __init__(self, *, length: int = 9, angle_degrees: float = 0.0, nsr: float = 0.01) -> None:
        if length < 1 or nsr <= 0:
            raise ValueError("motion deblur requires length >= 1 and nsr > 0")
        self.length = length
        self.angle_degrees = angle_degrees
        self.nsr = nsr

    def _psf(self, shape: tuple[int, int]) -> np.ndarray:
        psf = np.zeros(shape, dtype=np.float32)
        center = np.array([shape[1] // 2, shape[0] // 2], dtype=np.float64)
        direction = np.array(
            [np.cos(np.deg2rad(self.angle_degrees)), np.sin(np.deg2rad(self.angle_degrees))]
        )
        start = np.rint(center - direction * (self.length - 1) / 2).astype(int)
        end = np.rint(center + direction * (self.length - 1) / 2).astype(int)
        cv2.line(psf, tuple(start), tuple(end), 1.0, 1)
        total = float(psf.sum())
        if total == 0:
            psf[shape[0] // 2, shape[1] // 2] = 1.0
            total = 1.0
        return np.fft.ifftshift(psf / total)

    def process(self, packet: FramePacket, context: ProcessorContext) -> list[FramePacket]:
        del context
        _validate_bgr_u8(packet)
        psf_fft = np.fft.fft2(self._psf((packet.height, packet.width)))
        inverse = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + self.nsr)
        channels = []
        for channel in cv2.split(packet.image):
            restored = np.fft.ifft2(np.fft.fft2(channel.astype(np.float32)) * inverse).real
            channels.append(np.clip(restored, 0, 255).astype(np.uint8))
        return [packet.derived(cv2.merge(channels))]
