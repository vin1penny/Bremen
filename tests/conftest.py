from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=Fraction(10, 1))
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(6):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:, :, 0] = index * 20
            image[8:32, 16:48, 1] = 180
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
