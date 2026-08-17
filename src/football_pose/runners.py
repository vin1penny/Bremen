from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from football_pose.configuration import ModelSpec
from football_pose.domain import PredictionRecord


OOM_MARKERS = ("out of memory", "cuda oom", "cudnn_status_alloc_failed")


class RunnerExecutionError(RuntimeError):
    def __init__(self, message: str, *, out_of_memory: bool = False) -> None:
        super().__init__(message)
        self.out_of_memory = out_of_memory


@dataclass(frozen=True, slots=True)
class RunnerResult:
    jsonl_path: Path
    batch_size: int
    attempts: int
    wall_seconds: float
    shard_count: int


class ExternalModelRunner:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def _validate_gpu_reservation(self) -> None:
        reservation = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not reservation or not self.spec.devices:
            return
        reserved = {item.strip() for item in reservation.split(",") if item.strip()}
        requested = {str(device) for device in self.spec.devices}
        outside = requested - reserved
        if outside:
            raise RunnerExecutionError(
                f"model {self.spec.id} requests GPU(s) {sorted(outside)} outside the "
                f"CUDA_VISIBLE_DEVICES reservation {sorted(reserved)}"
            )

    def _run_shard(
        self,
        *,
        artifact_path: Path,
        output_path: Path,
        experiment_id: str,
        pipeline_id: str,
        source_video_id: str,
        batch_size: int,
        shard_index: int,
        shard_count: int,
        device: int | None,
    ) -> None:
        configured_command = [
            sys.executable if part == "{python}" else part for part in self.spec.command
        ]
        command = [
            *configured_command,
            "--input-artifact",
            str(artifact_path),
            "--output-jsonl",
            str(output_path),
            "--experiment-id",
            experiment_id,
            "--pipeline-id",
            pipeline_id,
            "--model-id",
            self.spec.id,
            "--source-video-id",
            source_video_id,
            "--batch-size",
            str(batch_size),
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(shard_count),
        ]
        if self.spec.checkpoint is not None:
            command.extend(("--checkpoint", str(self.spec.checkpoint)))
        environment = os.environ.copy()
        environment.update(self.spec.environment)
        if device is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
        try:
            result = subprocess.run(
                command,
                env=environment,
                text=True,
                capture_output=True,
                timeout=self.spec.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RunnerExecutionError(
                f"model {self.spec.id} shard {shard_index} timed out after "
                f"{self.spec.timeout_seconds}s"
            ) from error
        output_path.with_suffix(".stdout.log").write_text(result.stdout, encoding="utf-8")
        output_path.with_suffix(".stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            combined = f"{result.stdout}\n{result.stderr}".lower()
            raise RunnerExecutionError(
                f"model {self.spec.id} shard {shard_index} exited with code {result.returncode}",
                out_of_memory=any(marker in combined for marker in OOM_MARKERS),
            )
        if not output_path.is_file():
            raise RunnerExecutionError(
                f"model {self.spec.id} shard {shard_index} did not create {output_path}"
            )

    @staticmethod
    def _merge(shard_paths: list[Path], output_path: Path) -> None:
        records: list[PredictionRecord] = []
        seen: set[tuple[int, str | None, str]] = set()
        for path in shard_paths:
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = PredictionRecord.model_validate_json(line)
                    except Exception as error:
                        raise RunnerExecutionError(
                            f"invalid output in {path.name} at line {line_number}"
                        ) from error
                    key = (record.frame_index, record.crop_id, record.person_id)
                    if key in seen:
                        raise RunnerExecutionError(f"duplicate prediction identity: {key}")
                    seen.add(key)
                    records.append(record)
        records.sort(key=lambda record: (record.frame_index, record.crop_id or "", record.person_id))
        temporary = output_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(record.model_dump_json() + "\n")
        temporary.replace(output_path)

    def run(
        self,
        *,
        artifact_path: Path,
        output_directory: Path,
        experiment_id: str,
        pipeline_id: str,
        source_video_id: str,
    ) -> RunnerResult:
        self._validate_gpu_reservation()
        output_directory.mkdir(parents=True, exist_ok=True)
        devices: list[int | None] = self.spec.devices or [None]
        batch_size = self.spec.batch_size
        attempts = 0
        start = time.perf_counter()
        while True:
            attempts += 1
            attempt_dir = output_directory / f"attempt-{attempts:02d}-batch-{batch_size}"
            if attempt_dir.exists():
                shutil.rmtree(attempt_dir)
            attempt_dir.mkdir()
            shard_paths = [attempt_dir / f"shard-{index:03d}.jsonl" for index in range(len(devices))]
            errors: list[RunnerExecutionError] = []
            with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                futures = [
                    executor.submit(
                        self._run_shard,
                        artifact_path=artifact_path,
                        output_path=shard_paths[index],
                        experiment_id=experiment_id,
                        pipeline_id=pipeline_id,
                        source_video_id=source_video_id,
                        batch_size=batch_size,
                        shard_index=index,
                        shard_count=len(devices),
                        device=device,
                    )
                    for index, device in enumerate(devices)
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except RunnerExecutionError as error:
                        errors.append(error)
            if not errors:
                output_path = output_directory / "predictions.jsonl"
                self._merge(shard_paths, output_path)
                return RunnerResult(
                    jsonl_path=output_path,
                    batch_size=batch_size,
                    attempts=attempts,
                    wall_seconds=time.perf_counter() - start,
                    shard_count=len(devices),
                )
            if any(not error.out_of_memory for error in errors):
                raise errors[0]
            if batch_size <= self.spec.min_batch_size:
                raise RunnerExecutionError(
                    f"model {self.spec.id} exhausted memory at minimum batch {batch_size}",
                    out_of_memory=True,
                )
            batch_size = max(self.spec.min_batch_size, batch_size // 2)
