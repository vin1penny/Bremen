"""Single-factor preprocessing runs with one shared model/pitch protocol."""
from __future__ import annotations

import csv
import argparse
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from football_pose.artifacts import load_artifact_manifest, sha256_file
from football_pose.configuration import (
    ExperimentConfig, ProcessorSpec, _resolve_checkpoint_parameters,
    load_config, validate_config_paths,
)
from football_pose.experiments import ExperimentRunner
from football_pose.jobs import atomic_write_json
from football_pose.preprocessing import build_pipeline


class ScreeningMethod(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    processor: ProcessorSpec | None = None


class ScreeningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_config: Path
    output_root: Path
    cache_root: Path
    methods: list[ScreeningMethod] = Field(min_length=2)

    @model_validator(mode="after")
    def single_factor(self) -> "ScreeningConfig":
        names = [method.name for method in self.methods]
        if len(names) != len(set(names)):
            raise ValueError("screening method names must be unique")
        baselines = [method.name for method in self.methods if method.processor is None]
        if baselines != ["baseline"]:
            raise ValueError("exactly one baseline with no processor is required")
        if next(method for method in self.methods if method.name == "baseline").processor:
            raise ValueError("baseline must not contain a processor")
        return self


def load_screening(path: Path) -> ScreeningConfig:
    path = path.resolve()
    config = ScreeningConfig.model_validate(yaml.safe_load(path.read_text()))
    for field in ("base_config", "output_root", "cache_root"):
        value = getattr(config, field)
        if not value.is_absolute():
            setattr(config, field, (path.parent / value).resolve())
    for method in config.methods:
        if method.processor is not None:
            method.processor.params = _resolve_checkpoint_parameters(
                method.processor.params, path.parent,
            )
    return config


def expand_screening(
    path: Path, *, device: int, methods: set[str] | None = None,
    models: set[str] | None = None,
) -> tuple[ScreeningConfig, dict[str, ExperimentConfig]]:
    if device < 0:
        raise ValueError("device must be a non-negative physical GPU ID")
    suite = load_screening(path)
    base = load_config(suite.base_config, check_paths=False)
    if base.processors or not base.models:
        raise ValueError("screening base must have models and no processors")
    unknown = (methods or set()) - {method.name for method in suite.methods}
    if unknown:
        raise ValueError(f"unknown screening methods: {sorted(unknown)}")
    unknown_models = (models or set()) - {model.id for model in base.models}
    if unknown_models:
        raise ValueError(f"unknown models: {sorted(unknown_models)}")
    expanded = {}
    for method in sorted(suite.methods, key=lambda item: item.name != "baseline"):
        if methods and method.name not in methods | {"baseline"}:
            continue
        config = base.model_copy(deep=True)
        config.name = f"{base.name}-{method.name}"
        config.output_dir = suite.output_root / method.name
        config.cache.root = suite.cache_root
        config.cache.format = "auto"
        config.processors = [method.processor.model_copy(deep=True)] if method.processor else []
        config.models = [model for model in config.models if not models or model.id in models]
        for model in config.models:
            model.devices = [device]
        config.pitch_filter.device = str(device)
        config.fail_fast = False
        for spec in config.processors:
            if spec.type == "crop":
                spec.params["detector"]["device"] = str(device)
        expanded[method.name] = config
    return suite, expanded


def preflight(configs: dict[str, ExperimentConfig], *, device: int) -> None:
    reservation = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if reservation and str(device) not in reservation.split(","):
        raise ValueError(f"GPU {device} is outside CUDA_VISIBLE_DEVICES={reservation}")
    errors = []
    images = set()
    validated_detectors = set()
    for name, config in configs.items():
        try:
            validate_config_paths(config)
            pipeline = build_pipeline(config.processors)
            try:
                for processor in pipeline.processors:
                    validate = getattr(processor, "validate", None)
                    if validate is not None:
                        validate()
            finally:
                pipeline.close()
            for model in config.models:
                if model.id == "hrnet-w32":
                    from football_pose.hrnet_input import add_hrnet_arguments, detector_parameters
                    from football_pose.preprocessing.cropping import CropProcessor, UltralyticsDetectionProvider

                    if model.checkpoint is None:
                        raise ValueError("HRNet requires its own pose checkpoint")
                    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
                    add_hrnet_arguments(parser)
                    namespace, _ = parser.parse_known_args(model.command)
                    namespace.detector_checkpoint = model.auxiliary_checkpoints.get("detector")
                    params = detector_parameters(namespace, device="cpu")
                    key = repr(params)
                    if key not in validated_detectors:
                        detector = UltralyticsDetectionProvider(**params)
                        cropper = CropProcessor(
                            detector=detector, padding_ratio=namespace.crop_padding,
                            min_size=namespace.crop_min_size,
                        )
                        try:
                            cropper.validate()
                        finally:
                            cropper.close()
                        validated_detectors.add(key)
                if "runners/container-runner.sh" in model.command:
                    index = model.command.index("runners/container-runner.sh")
                    images.add(model.command[index + 1])
        except Exception as error:
            errors.append(f"{name}: {error}")
    for image in sorted(images):
        try:
            subprocess.run(
                ["docker", "image", "inspect", image], check=True,
                capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            errors.append(f"Docker image unavailable: {image}; build it from the repo root")
    if errors:
        raise ValueError("Screening preflight failed:\n" + "\n".join(errors))


def _percent(value: float | int | None, reference: float | int | None) -> float | None:
    if value is None or reference is None or reference == 0:
        return None
    return 100 * (value - reference) / reference


def comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    frame_count = report.get("source_frame_count")
    for condition in report["conditions"]:
        summary = condition.get("summary", {})
        for model_id in report["model_ids"]:
            job = next((job for job in summary.get("jobs", [])
                        if job["model_id"] == model_id), {})
            complete = job.get("status") == "COMPLETE" and not condition.get("error")
            outputs = job.get("outputs", {}) if complete else {}
            timings = job.get("timings", {}) if complete else {}
            raw = int(outputs["records"]) if "records" in outputs else None
            pitch = int(outputs["on_pitch_records"]) if "on_pitch_records" in outputs else None
            model_seconds = timings.get("model_wall_seconds")
            rows.append({
                "method": condition["method"], "model": model_id,
                "status": condition["status"] if condition.get("error") else job.get("status", condition["status"]),
                "raw_records": raw, "on_pitch_records": pitch,
                "raw_per_frame": raw / frame_count if raw is not None and frame_count else None,
                "on_pitch_per_frame": pitch / frame_count if pitch is not None and frame_count else None,
                "model_seconds": model_seconds,
                "model_source_fps": frame_count / model_seconds
                if frame_count and model_seconds else None,
                "preprocessing_wall_seconds": summary.get("preprocessing_wall_seconds"),
                "artifact_processing_seconds": summary.get("artifact_processing_seconds"),
                "preprocessing_cache_hit": summary.get("cache_hit"),
                "video_render_seconds": timings.get("video_render_seconds"),
                "condition_wall_seconds": condition.get("wall_seconds"),
                "video": outputs.get("video"),
                "error": job.get("error") or condition.get("error"),
            })
    baseline = {row["model"]: row for row in rows if row["method"] == "baseline"}
    for row in rows:
        reference = baseline.get(row["model"], {})
        for key in ("raw_records", "on_pitch_records", "model_seconds"):
            row[f"{key}_change_percent"] = _percent(row[key], reference.get(key))
    return rows


def write_comparison(root: Path, report: dict[str, Any]) -> None:
    rows = comparison_rows(report)
    atomic_write_json(root / "screening.json", {**report, "comparison": rows})
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (root / "comparison.csv").write_text(csv_buffer.getvalue())
    columns = ["method", "model", "status", "raw_records", "on_pitch_records",
               "on_pitch_records_change_percent", "raw_per_frame", "model_seconds",
               "model_seconds_change_percent", "model_source_fps",
               "preprocessing_wall_seconds", "preprocessing_cache_hit"]
    headings = ["Method", "Model", "Status", "Raw", "On pitch", "Δ on pitch %",
                "Raw/frame", "Model s", "Δ time %", "Source fps", "Preproc s", "Cache hit"]
    text = ["# Single-step preprocessing comparison", "",
            "Raw/on-pitch counts are yield diagnostics, not accuracy. Percentages use the same model's baseline.",
            "Model time includes container startup and input decoding. Preprocessing is shared across models.",
            "Cold materialization and warm-cache wall times are not interchangeable. See CSV/JSON for all timings and videos.",
            "", "| " + " | ".join(headings) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = []
        for key in columns:
            value = row[key]
            cells.append("—" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value))
        text.append("| " + " | ".join(cells) + " |")
    (root / "comparison.md").write_text("\n".join(text) + "\n")


def run_screening(
    path: Path, *, device: int, methods: set[str] | None = None,
    models: set[str] | None = None, check_only: bool = False,
    progress: Callable[[str], Any] = print,
) -> Path | None:
    suite, configs = expand_screening(path, device=device, methods=methods, models=models)
    preflight(configs, device=device)
    if check_only:
        progress(f"Ready: {len(configs)} conditions x {len(configs['baseline'].models)} models")
        return None
    # Set before host inference; use only the requested reservation.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    root = suite.output_root / run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    report: dict[str, Any] = {
        "success": False, "run_id": run_id, "suite": str(path.resolve()),
        "device": device, "source_frame_count": None,
        "input_sha256": sha256_file(configs["baseline"].input),
        "model_ids": [model.id for model in configs["baseline"].models],
        "conditions": [{"method": name, "status": "PENDING"} for name in configs],
        "note": "Single-factor yield/runtime screening, not ground-truth pose accuracy.",
    }
    # Pin checkpoint contents alongside expanded YAML, before starting any condition.
    checkpoints = set()
    for name, config in configs.items():
        config.output_dir = root / name
        config.video_output.root = config.video_output.root / "single-step" / run_id
        payload = config.model_dump(mode="json")
        (root / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
        def collect(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "checkpoint" and item:
                        checkpoints.add(item)
                    elif key == "auxiliary_checkpoints":
                        checkpoints.update(item.values())
                    else:
                        collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
        collect(payload)
    report["checkpoint_sha256"] = {item: sha256_file(item) for item in sorted(checkpoints)}
    progress(f"Screening output: {root}")
    write_comparison(root, report)
    for condition in report["conditions"]:
        name = condition["method"]
        condition["status"] = "RUNNING"
        write_comparison(root, report)
        progress(f"Starting {name}")
        started = time.perf_counter()
        try:
            for checkpoint, expected_hash in report["checkpoint_sha256"].items():
                if sha256_file(checkpoint) != expected_hash:
                    raise RuntimeError(f"checkpoint changed during screening: {checkpoint}")
            summary = ExperimentRunner(configs[name]).run()
            manifest = load_artifact_manifest(Path(summary["artifact"]))
            if manifest.source_sha256 != report["input_sha256"]:
                raise RuntimeError("input video changed during screening; do not compare different sources")
            condition["summary"] = summary
            condition["summary_path"] = str(
                configs[name].output_dir / "experiments" / summary["experiment_id"] / "summary.json"
            )
            condition["status"] = "COMPLETE" if summary["success"] else "FAILED"
            if name == "baseline":
                report["source_frame_count"] = manifest.frame_count
        except KeyboardInterrupt:
            condition["status"] = "INTERRUPTED"
            raise
        except Exception as error:
            condition["status"] = "FAILED"
            condition["error"] = f"{type(error).__name__}: {error}"
        finally:
            condition["wall_seconds"] = time.perf_counter() - started
            report["success"] = all(item["status"] == "COMPLETE" for item in report["conditions"])
            write_comparison(root, report)
        progress(f"{name}: {condition['status']}")
    return root / "screening.json"
