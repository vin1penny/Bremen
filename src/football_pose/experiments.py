from __future__ import annotations

import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from football_pose import __version__
from football_pose.archive import write_archive
from football_pose.artifacts import (
    ArtifactManifest,
    ArtifactStore,
    artifact_id,
    sha256_file,
    stable_hash,
)
from football_pose.configuration import ExperimentConfig, ModelSpec
from football_pose.jobs import JobRecord, JobStatus, JobStore, atomic_write_json
from football_pose.pitch_filter import (
    PitchGeometryStore,
    PreparedPitchGeometry,
    postprocess_pitch_predictions,
)
from football_pose.preprocessing import build_pipeline
from football_pose.preprocessing.base import ProcessorContext
from football_pose.provenance import collect_provenance
from football_pose.runners import ExternalModelRunner
from football_pose.video import VideoSource
from football_pose.visualization import render_annotated_video


@dataclass(frozen=True, slots=True)
class PreparedExperiment:
    experiment_id: str
    pipeline_id: str
    source_video_id: str
    artifact_path: Path
    artifact_manifest: ArtifactManifest
    cache_hit: bool
    stage_timings: dict[str, Any]


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_store = ArtifactStore(self.config.cache.root)
        self.job_store = JobStore(self.config.output_dir / "jobs")

    def prepare(self) -> PreparedExperiment:
        source_hash = sha256_file(self.config.input)
        source_video_id = source_hash[:16]
        pipeline_payload = [spec.model_dump(mode="json") for spec in self.config.processors]
        pipeline_fingerprint = {
            "configuration": self._fingerprint_checkpoints(pipeline_payload),
            "implementation": self._implementation_fingerprint(preprocessing_only=True),
        }
        identifier, pipeline_sha = artifact_id(source_hash, pipeline_fingerprint)
        experiment_identifier = stable_hash(
            {"name": self.config.name, "source": source_hash, "pipeline": pipeline_sha}
        )[:16]
        pipeline = build_pipeline(self.config.processors)
        source = VideoSource(self.config.input)
        info = source.probe()
        context = ProcessorContext(experiment_id=experiment_identifier, pipeline_id=pipeline_sha)
        packets = pipeline.process(source.frames(), context)
        artifact_format = self.config.cache.format
        if artifact_format == "auto":
            artifact_format = "png_shards" if pipeline.requires_png_shards else "ffv1"
        provenance = {
            "football_pose_version": __version__,
            "input": str(self.config.input),
            "processors": pipeline_payload,
            "pipeline_fingerprint": pipeline_fingerprint,
            "preprocessing": context.metadata,
            **collect_provenance(),
        }
        path, manifest, cache_hit = self.artifact_store.materialize(
            packets,
            artifact_identifier=identifier,
            source_sha256=source_hash,
            pipeline_sha256=pipeline_sha,
            artifact_format=artifact_format,
            shard_size=self.config.cache.shard_size,
            rate=info.average_rate,
            pinned=self.config.cache.pin,
            provenance=provenance,
        )
        return PreparedExperiment(
            experiment_id=experiment_identifier,
            pipeline_id=pipeline_sha,
            source_video_id=source_video_id,
            artifact_path=path,
            artifact_manifest=manifest,
            cache_hit=cache_hit,
            stage_timings=dict(context.metadata.get("stage_timings", {})),
        )

    def _job_id(self, prepared: PreparedExperiment, model: ModelSpec) -> str:
        model_payload = model.model_dump(mode="json")
        if model.checkpoint is not None and model.checkpoint.is_file():
            model_payload["checkpoint_sha256"] = sha256_file(model.checkpoint)
        model_payload["implementation"] = self._implementation_fingerprint(
            preprocessing_only=False
        )
        command_files: dict[str, str] = {}
        for part in model.command[1:]:
            candidate = Path(part)
            if candidate.is_file():
                command_files[str(candidate.resolve())] = sha256_file(candidate)
        model_payload["command_file_sha256"] = command_files
        return stable_hash(
            {
                "experiment": prepared.experiment_id,
                "pipeline": prepared.pipeline_id,
                "model": model_payload,
            }
        )[:24]

    @staticmethod
    def _safe_path_component(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
        return cleaned or "experiment"

    def _video_path(
        self,
        prepared: PreparedExperiment,
        model: ModelSpec,
        job_id: str,
        pitch_geometry: PreparedPitchGeometry | None,
    ) -> Path:
        settings_id = stable_hash(
            {
                "video_output": self.config.video_output.model_dump(mode="json"),
                "pitch_filter": self.config.pitch_filter.model_dump(mode="json"),
                "pitch_geometry_id": (
                    pitch_geometry.identifier if pitch_geometry is not None else None
                ),
            }
        )[:8]
        experiment_name = self._safe_path_component(self.config.name)
        model_name = self._safe_path_component(model.id)
        return (
            self.config.video_output.root
            / experiment_name
            / prepared.experiment_id
            / f"{model_name}-{job_id}-{settings_id}.mp4"
        )

    def _ensure_video(
        self,
        record: JobRecord,
        prepared: PreparedExperiment,
        model: ModelSpec,
        pitch_geometry: PreparedPitchGeometry | None,
    ) -> JobRecord:
        if not self.config.video_output.enabled:
            return record
        parquet_path = Path(record.outputs["parquet"])
        video_path = self._video_path(prepared, model, record.job_id, pitch_geometry)
        outputs = dict(record.outputs)
        timings = dict(record.timings)
        if not video_path.is_file():
            result = render_annotated_video(
                source_path=self.config.input,
                artifact_path=prepared.artifact_path,
                prediction_parquet=parquet_path,
                output_path=video_path,
                model_id=model.id,
                settings=self.config.video_output,
                decision_parquet=record.outputs.get("pitch_decisions"),
                pitch_geometry_json=(
                    pitch_geometry.path if pitch_geometry is not None else None
                ),
            )
            timings["video_render_seconds"] = result.wall_seconds
            outputs.update(
                {
                    "video": str(result.path),
                    "video_codec": result.codec,
                    "video_frames": str(result.frames),
                }
            )
        else:
            outputs["video"] = str(video_path)
            outputs.setdefault("video_codec", self.config.video_output.codec)
        updated = record.model_copy(
            update={"outputs": outputs, "timings": timings, "updated_at_unix": time.time()}
        )
        self.job_store.save(updated)
        return updated

    def _ensure_pitch_filter(
        self,
        record: JobRecord,
        pitch_geometry: PreparedPitchGeometry | None,
    ) -> JobRecord:
        if not self.config.pitch_filter.enabled:
            return record
        if pitch_geometry is None:
            raise RuntimeError("pitch filtering is enabled but pitch geometry is unavailable")
        result = postprocess_pitch_predictions(
            prediction_parquet=record.outputs["parquet"],
            output_root=(
                self.config.output_dir
                / "jobs"
                / record.job_id
                / "archive"
                / "pitch-filter"
            ),
            geometry=pitch_geometry,
            config=self.config.pitch_filter,
        )
        outputs = dict(record.outputs)
        outputs.update(
            {
                "deduplicated_records": str(result.deduplicated_records),
                "duplicate_records": str(result.duplicate_records),
                "on_pitch_records": str(result.on_pitch_records),
                "outside_pitch_records": str(result.outside_pitch_records),
                "unclassified_records": str(result.unclassified_records),
                "on_pitch_parquet": str(result.on_pitch_parquet),
                "pitch_decisions": str(result.decisions_parquet),
                "pitch_filter_manifest": str(result.manifest_path),
            }
        )
        timings = dict(record.timings)
        timings["pitch_postprocess_seconds"] = result.wall_seconds
        updated = record.model_copy(
            update={"outputs": outputs, "timings": timings, "updated_at_unix": time.time()}
        )
        self.job_store.save(updated)
        return updated

    @staticmethod
    def _fingerprint_checkpoints(value: Any) -> Any:
        if isinstance(value, dict):
            fingerprint: dict[str, Any] = {}
            for key, item in value.items():
                if key == "checkpoint" and isinstance(item, str) and Path(item).is_file():
                    fingerprint[key] = {
                        "path": item,
                        "sha256": sha256_file(item),
                    }
                else:
                    fingerprint[key] = ExperimentRunner._fingerprint_checkpoints(item)
            return fingerprint
        if isinstance(value, list):
            return [ExperimentRunner._fingerprint_checkpoints(item) for item in value]
        return value

    @staticmethod
    def _implementation_fingerprint(*, preprocessing_only: bool) -> str:
        package = Path(__file__).resolve().parent
        if preprocessing_only:
            paths = [package / "domain.py", package / "video.py"]
            paths.extend(sorted((package / "preprocessing").glob("*.py")))
        else:
            # Rendering changes must not invalidate expensive model predictions.
            paths = [
                path
                for path in sorted(package.glob("*.py"))
                if path.name != "visualization.py"
            ]
        return stable_hash(
            {
                "football_pose_version": __version__,
                "files": {str(path.relative_to(package)): sha256_file(path) for path in paths},
            }
        )

    def _run_model(
        self,
        prepared: PreparedExperiment,
        model: ModelSpec,
        pitch_geometry: PreparedPitchGeometry | None,
    ) -> JobRecord:
        job_id = self._job_id(prepared, model)
        job_directory = self.config.output_dir / "jobs" / job_id
        record = self.job_store.load(job_id)
        if record is not None and record.status == JobStatus.COMPLETE:
            parquet = Path(record.outputs.get("parquet", ""))
            if parquet.is_file():
                try:
                    record = self._ensure_pitch_filter(record, pitch_geometry)
                    return self._ensure_video(
                        record, prepared, model, pitch_geometry
                    )
                except Exception as error:
                    failed = record.model_copy(
                        update={
                            "status": JobStatus.FAILED,
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                                "traceback": traceback.format_exc(),
                            },
                            "updated_at_unix": time.time(),
                        }
                    )
                    self.job_store.save(failed)
                    if self.config.fail_fast:
                        raise
                    return failed
            record = JobRecord(
                job_id=record.job_id,
                experiment_id=record.experiment_id,
                pipeline_id=record.pipeline_id,
                model_id=record.model_id,
                attempts=record.attempts,
                batch_size=model.batch_size,
                created_at_unix=record.created_at_unix,
                error={"type": "MissingOutput", "message": "completed archive was missing"},
            )
            self.job_store.save(record)
        if record is None:
            record = JobRecord(
                job_id=job_id,
                experiment_id=prepared.experiment_id,
                pipeline_id=prepared.pipeline_id,
                model_id=model.id,
                batch_size=model.batch_size,
            )
            self.job_store.save(record)
        elif record.status in {JobStatus.RUNNING, JobStatus.VALIDATING}:
            record = record.transition(
                JobStatus.FAILED,
                error={"type": "InterruptedJob", "message": "recovered stale in-progress job"},
            )
            self.job_store.save(record)
        record = record.transition(
            JobStatus.RUNNING,
            attempts=record.attempts + 1,
            error=None,
        )
        self.job_store.save(record)
        try:
            result = ExternalModelRunner(model).run(
                artifact_path=prepared.artifact_path,
                output_directory=job_directory / "runner",
                experiment_id=prepared.experiment_id,
                pipeline_id=prepared.pipeline_id,
                source_video_id=prepared.source_video_id,
            )
            record = record.transition(
                JobStatus.VALIDATING,
                batch_size=result.batch_size,
                timings={"model_wall_seconds": result.wall_seconds},
            )
            self.job_store.save(record)
            archive_dir = job_directory / "archive"
            parquet_path = archive_dir / "predictions.parquet"
            manifest_path = archive_dir / "manifest.json"
            provenance = {
                "experiment_id": prepared.experiment_id,
                "pipeline_id": prepared.pipeline_id,
                "model": model.model_dump(mode="json"),
                "checkpoint_sha256": (
                    sha256_file(model.checkpoint)
                    if model.checkpoint is not None and model.checkpoint.is_file()
                    else None
                ),
                "artifact": prepared.artifact_manifest.model_dump(mode="json"),
                "batch_size": result.batch_size,
                "attempts": result.attempts,
                "shard_count": result.shard_count,
                **collect_provenance(),
            }
            validation_start = time.perf_counter()
            count = write_archive(result.jsonl_path, parquet_path, manifest_path, provenance)
            timings = dict(record.timings)
            timings["validation_archive_seconds"] = time.perf_counter() - validation_start
            record = record.transition(
                JobStatus.COMPLETE,
                timings=timings,
                outputs={
                    "jsonl": str(result.jsonl_path),
                    "parquet": str(parquet_path),
                    "manifest": str(manifest_path),
                    "records": str(count),
                },
            )
            self.job_store.save(record)
            record = self._ensure_pitch_filter(record, pitch_geometry)
            record = self._ensure_video(record, prepared, model, pitch_geometry)
            return record
        except Exception as error:
            error_payload = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            if record.status == JobStatus.COMPLETE:
                failed = record.model_copy(
                    update={
                        "status": JobStatus.FAILED,
                        "error": error_payload,
                        "updated_at_unix": time.time(),
                    }
                )
            else:
                failed = record.transition(JobStatus.FAILED, error=error_payload)
            self.job_store.save(failed)
            if self.config.fail_fast:
                raise
            return failed

    def run(self) -> dict[str, Any]:
        cold_start = time.perf_counter()
        prepared = self.prepare()
        preprocessing_wall = time.perf_counter() - cold_start
        pitch_geometry: PreparedPitchGeometry | None = None
        pitch_geometry_wall = 0.0
        if self.config.pitch_filter.enabled:
            pitch_start = time.perf_counter()
            pitch_geometry = PitchGeometryStore(
                self.config.pitch_filter.cache_root
            ).prepare(
                source_path=self.config.input,
                source_sha256=prepared.artifact_manifest.source_sha256,
                config=self.config.pitch_filter,
            )
            pitch_geometry_wall = time.perf_counter() - pitch_start
        jobs = [
            self._run_model(prepared, model, pitch_geometry)
            for model in self.config.models
        ]
        summary = {
            "experiment_id": prepared.experiment_id,
            "pipeline_id": prepared.pipeline_id,
            "artifact": str(prepared.artifact_path),
            "cache_hit": prepared.cache_hit,
            "preprocessing_mode": "warm_cache" if prepared.cache_hit else "cold_materialization",
            "preprocessing_wall_seconds": preprocessing_wall,
            "artifact_processing_seconds": prepared.artifact_manifest.processing_seconds,
            "preprocessing_stage_timings": prepared.stage_timings,
            "pitch_filter": (
                {
                    "enabled": True,
                    "geometry_id": pitch_geometry.identifier,
                    "geometry": str(pitch_geometry.path),
                    "cache_hit": pitch_geometry.cache_hit,
                    "geometry_wall_seconds": pitch_geometry_wall,
                    "observed_frames": pitch_geometry.manifest.observed_frames,
                    "bbox_frames": pitch_geometry.manifest.bbox_frames,
                    "usable_frames": pitch_geometry.manifest.usable_frames,
                    "frame_count": pitch_geometry.manifest.frame_count,
                }
                if pitch_geometry is not None
                else {"enabled": False}
            ),
            "jobs": [job.model_dump(mode="json") for job in jobs],
            "success": all(job.status == JobStatus.COMPLETE for job in jobs),
            "configuration": self.config.model_dump(mode="json"),
        }
        atomic_write_json(
            self.config.output_dir / "experiments" / prepared.experiment_id / "summary.json",
            summary,
        )
        return summary
