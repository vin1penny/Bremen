from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar


@dataclass(frozen=True, slots=True)
class RunnerArguments:
    input_artifact: Path
    output_jsonl: Path
    experiment_id: str
    pipeline_id: str
    model_id: str
    source_video_id: str
    checkpoint: Path | None
    batch_size: int
    shard_index: int
    shard_count: int


def contract_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--source-video-id", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser


def contract_args(namespace: argparse.Namespace) -> RunnerArguments:
    if namespace.batch_size < 1:
        raise ValueError("batch size must be positive")
    if not 0 <= namespace.shard_index < namespace.shard_count:
        raise ValueError("invalid shard index/count")
    namespace.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    return RunnerArguments(
        input_artifact=namespace.input_artifact,
        output_jsonl=namespace.output_jsonl,
        experiment_id=namespace.experiment_id,
        pipeline_id=namespace.pipeline_id,
        model_id=namespace.model_id,
        source_video_id=namespace.source_video_id,
        checkpoint=namespace.checkpoint,
        batch_size=namespace.batch_size,
        shard_index=namespace.shard_index,
        shard_count=namespace.shard_count,
    )


T = TypeVar("T")


def batches(items: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
