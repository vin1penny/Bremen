"""Selective contrast and chroma enhancement from a match's sampled kit colors."""
from __future__ import annotations

import re

import cv2
import numpy as np

from football_pose.domain import FramePacket
from football_pose.preprocessing.base import ProcessorContext
from football_pose.preprocessing.opencv import _validate_bgr_u8


class JerseyColorEnhancementProcessor:
    """Blend enhanced pixels through a soft CIE Lab color-similarity mask.

    Colors are RGB hex samples from the footage, not team labels. The mask can
    also select background objects of similar color; it does not segment people.
    """

    name = "jersey_color_enhance"

    def __init__(
        self,
        *,
        colors: list[str],
        color_tolerance: float = 25.0,
        strength: float = 0.35,
        chroma_gain: float = 1.15,
        clip_limit: float = 2.0,
        tile_grid: list[int] | None = None,
    ) -> None:
        if (
            not isinstance(colors, (list, tuple)) or not colors
            or any(not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color)
                   for color in colors)
        ):
            raise ValueError("colors must be a non-empty list of RGB '#RRGGBB' samples")
        if not np.isfinite([color_tolerance, strength, chroma_gain, clip_limit]).all():
            raise ValueError("jersey enhancement parameters must be finite")
        if not 0 < color_tolerance <= 100:
            raise ValueError("color_tolerance must be in (0, 100] Lab units")
        if not 0 <= strength <= 1:
            raise ValueError("strength must be in [0, 1]")
        if not 1 <= chroma_gain <= 2:
            raise ValueError("chroma_gain must be in [1, 2]")
        grid = [8, 8] if tile_grid is None else tile_grid
        if (
            clip_limit <= 0 or not isinstance(grid, (list, tuple)) or len(grid) != 2
            or any(type(size) is not int or size <= 0 for size in grid)
        ):
            raise ValueError("clip_limit must be positive and tile_grid must contain two positive integers")

        # OpenCV's float conversion uses L=0..100 and signed a/b. Both the
        # reference swatches and image pixels must use this same representation.
        bgr = np.array(
            [[int(color[index:index + 2], 16) for index in (5, 3, 1)] for color in colors],
            dtype=np.float32,
        )[None, ...] / 255.0
        self.reference_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0]
        self.color_tolerance = color_tolerance
        self.strength = strength
        self.chroma_gain = chroma_gain
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(grid))

    def _color_mask(self, lab: np.ndarray) -> np.ndarray:
        distance_squared = np.full(lab.shape[:2], np.inf, dtype=np.float32)
        for reference in self.reference_lab:
            delta = lab - reference
            np.minimum(distance_squared, np.sum(delta * delta, axis=2), out=distance_squared)
        # Smooth falloff with finite support. Palette overlap never stacks gains.
        weight = np.clip(1.0 - distance_squared / self.color_tolerance**2, 0.0, 1.0)
        return weight * weight * (3.0 - 2.0 * weight)

    def process(self, packet: FramePacket, context: ProcessorContext) -> list[FramePacket]:
        del context
        _validate_bgr_u8(packet)
        if self.strength == 0:
            return [packet]
        original = packet.image.astype(np.float32) / 255.0
        lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        mask = self._color_mask(lab)
        if not np.any(mask):
            return [packet]

        luminance = np.rint(lab[:, :, 0] * 255.0 / 100.0).astype(np.uint8)
        lab[:, :, 0] = self.clahe.apply(luminance).astype(np.float32) * (100.0 / 255.0)
        lab[:, :, 1:] *= self.chroma_gain
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        alpha = (self.strength * mask)[:, :, None]
        blended = original + alpha * (enhanced - original)
        output = np.rint(np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)
        # Avoid even conversion/rounding changes to pixels outside the palette.
        output[mask == 0] = packet.image[mask == 0]
        return [packet.derived(output)]
