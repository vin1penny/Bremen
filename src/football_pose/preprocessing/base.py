from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
import time
from typing import Any, Protocol

from football_pose.domain import FramePacket


@dataclass(slots=True)
class ProcessorContext:
    experiment_id: str
    pipeline_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Processor(Protocol):
    name: str

    def process(
        self, packet: FramePacket, context: ProcessorContext
    ) -> Iterable[FramePacket]: ...


class Pipeline:
    """Apply ordered processors; a processor may fan one frame into many crops."""

    def __init__(self, processors: list[Processor]) -> None:
        self.processors = processors

    @property
    def has_crop(self) -> bool:
        return any(processor.name == "crop" for processor in self.processors)

    @property
    def requires_png_shards(self) -> bool:
        return any(processor.name in {"crop", "tile"} for processor in self.processors)

    def process(
        self, packets: Iterable[FramePacket], context: ProcessorContext
    ) -> Iterator[FramePacket]:
        current: Iterable[FramePacket] = packets
        for stage_index, processor in enumerate(self.processors):
            previous = current
            stage_key = f"{stage_index:02d}-{processor.name}"

            def apply_stage(
                source: Iterable[FramePacket] = previous,
                stage: Processor = processor,
                timing_key: str = stage_key,
            ) -> Iterator[FramePacket]:
                for packet in source:
                    try:
                        start = time.perf_counter()
                        outputs = list(stage.process(packet, context))
                        elapsed = time.perf_counter() - start
                        timings = context.metadata.setdefault("stage_timings", {})
                        timing = timings.setdefault(
                            timing_key,
                            {"wall_seconds": 0.0, "inputs": 0, "outputs": 0},
                        )
                        timing["wall_seconds"] += elapsed
                        timing["inputs"] += 1
                        timing["outputs"] += len(outputs)
                        yield from outputs
                    except Exception as error:
                        raise RuntimeError(
                            f"processor {stage.name!r} failed at frame "
                            f"{packet.frame_index} ({packet.source_id})"
                        ) from error

            current = apply_stage()
        yield from current
