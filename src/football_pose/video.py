from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av

from football_pose.domain import FramePacket


@dataclass(frozen=True, slots=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    frames: int | None
    duration_seconds: float | None
    average_rate: Fraction | None


class VideoSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"video does not exist: {self.path}")

    def probe(self) -> VideoInfo:
        with av.open(str(self.path)) as container:
            if not container.streams.video:
                raise ValueError(f"input contains no video stream: {self.path}")
            stream = container.streams.video[0]
            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            return VideoInfo(
                path=self.path,
                width=int(stream.codec_context.width),
                height=int(stream.codec_context.height),
                frames=int(stream.frames) if stream.frames else None,
                duration_seconds=duration,
                average_rate=stream.average_rate,
            )

    def frames(self) -> Iterator[FramePacket]:
        with av.open(str(self.path)) as container:
            if not container.streams.video:
                raise ValueError(f"input contains no video stream: {self.path}")
            stream = container.streams.video[0]
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            fallback_rate = float(stream.average_rate or 30)
            frame_index = 0
            for frame in container.decode(stream):
                if frame.is_corrupt:
                    raise ValueError(f"corrupt video frame at decoded index {frame_index}")
                timestamp = frame.time
                if timestamp is None:
                    timestamp = frame_index / fallback_rate
                metadata = {
                    "pts": frame.pts,
                    "time_base": str(frame.time_base) if frame.time_base is not None else None,
                }
                yield FramePacket(
                    image=frame.to_ndarray(format="bgr24"),
                    frame_index=frame_index,
                    timestamp_seconds=max(0.0, float(timestamp)),
                    source_width=width,
                    source_height=height,
                    metadata=metadata,
                )
                frame_index += 1
