"""Single-factor preprocessing comparison using COCO keypoint AP/AR."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from uuid import uuid4

import yaml

from football_pose.artifacts import stable_hash
from football_pose.coco_evaluation import load_coco_config, run_coco
from football_pose.hrnet_input import add_hrnet_arguments, detector_parameters
from football_pose.jobs import atomic_write_json
from football_pose.preprocessing import build_pipeline
from football_pose.preprocessing.cropping import UltralyticsDetectionProvider
from football_pose.screening import load_screening


def expand_coco_screening(path: Path, *, device: int):
    if device < 0:
        raise ValueError("device must be a non-negative physical GPU ID")
    visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if visible and str(device) not in visible:
        raise ValueError(f"GPU {device} is outside CUDA_VISIBLE_DEVICES={','.join(visible)}")
    host_device = str(visible.index(str(device))) if visible else str(device)
    suite = load_screening(path)
    base = load_coco_config(suite.base_config)
    if base.processors or not base.models:
        raise ValueError("COCO screening base must contain models and no processors")
    if len({model.id for model in base.models}) != len(base.models):
        raise ValueError("Model IDs must be unique")
    expanded = {}
    for method in sorted(suite.methods, key=lambda item: item.name != "baseline"):
        values = base.model_dump(mode="json")
        values["output_dir"] = str(suite.output_root / method.name)
        values["cache_root"] = str(suite.cache_root)
        values["processors"] = [method.processor.model_dump(mode="json")] if method.processor else []
        for model in values["models"]:
            model["devices"] = [device]
        for processor in values["processors"]:
            if processor["type"] == "crop":
                    processor["params"]["detector"]["device"] = host_device
        expanded[method.name] = type(base).model_validate(values)
    return suite, expanded


def preflight_coco(configs, *, device: int):
    import pycocotools.cocoeval  # noqa: F401

    reservation = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if reservation and str(device) not in reservation.split(","):
        raise ValueError(f"GPU {device} is outside CUDA_VISIBLE_DEVICES={reservation}")
    errors = []
    images = set()
    detectors = set()
    for name, config in configs.items():
        try:
            if not config.annotations.is_file() or not config.images.is_dir():
                raise FileNotFoundError("COCO annotation file or image directory is missing")
            pipeline = build_pipeline(config.processors)
            try:
                for processor in pipeline.processors:
                    validate = getattr(processor, "validate", None)
                    if validate:
                        validate()
            finally:
                pipeline.close()
            for model in config.models:
                for checkpoint in [model.checkpoint, *model.auxiliary_checkpoints.values()]:
                    if checkpoint is None or not checkpoint.is_file():
                        raise FileNotFoundError(f"Missing {model.id} checkpoint: {checkpoint}")
                if "runners/container-runner.sh" in model.command:
                    index = model.command.index("runners/container-runner.sh")
                    images.add(model.command[index + 1])
                if model.id == "hrnet-w32":
                    parser = argparse.ArgumentParser(add_help=False)
                    add_hrnet_arguments(parser)
                    namespace, _ = parser.parse_known_args(model.command)
                    if namespace.detector_dataset != "coco":
                        raise ValueError("COCO HRNet must select --detector-dataset coco")
                    namespace.detector_checkpoint = model.auxiliary_checkpoints.get("detector")
                    params = detector_parameters(namespace, device="cpu")
                    if stable_hash(params) not in detectors:
                        detector = UltralyticsDetectionProvider(**params)
                        try:
                            detector.validate()
                        finally:
                            detector.close()
                        detectors.add(stable_hash(params))
        except Exception as error:
            errors.append(f"{name}: {error}")
    for image in sorted(images):
        try:
            subprocess.run(["docker", "image", "inspect", image], check=True,
                           capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            errors.append(f"Docker image unavailable: {image}")
    if errors:
        raise ValueError("COCO preflight failed:\n" + "\n".join(errors))


def write_coco_comparison(root: Path, report: dict):
    rows = []
    for condition in report["conditions"]:
        summary = condition.get("summary", {})
        jobs = {job["model_id"]: job for job in summary.get("jobs", [])}
        for model_id in report["model_ids"]:
            job = jobs.get(model_id, {})
            metrics = job.get("metrics", {})
            rows.append({
                "method": condition["method"], "model": model_id,
                "status": job.get("status", "FAILED" if condition.get("error") else "PENDING"),
                **{name: metrics.get(name) for name in ("AP", "AP50", "AP75", "AR")},
                "AP_change_points": None,
                "model_seconds": job.get("model_wall_seconds"),
                "preprocessing_seconds": summary.get("artifact_processing_seconds"),
                "cache_hit": summary.get("cache_hit"),
                "image_count": summary.get("image_count"),
                "error": job.get("error") or condition.get("error"),
            })
    baselines = {row["model"]: row["AP"] for row in rows if row["method"] == "baseline"}
    for row in rows:
        baseline = baselines.get(row["model"])
        if baseline is not None and baseline >= 0 and row["AP"] is not None and row["AP"] >= 0:
            row["AP_change_points"] = 100 * (row["AP"] - baseline)
    atomic_write_json(root / "screening.json", {**report, "comparison": rows})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (root / "comparison.csv").write_text(stream.getvalue())
    lines = ["# COCO preprocessing comparison", "",
             f"Scope: {report['scope']}. AP/AR are on a 0–1 scale; -1 means unavailable.", "",
             "| Method | Model | Status | AP | Δ AP points | AR | Model seconds |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        def number(key):
            return "—" if row[key] is None else f"{row[key]:.4f}"
        lines.append(f"| {row['method']} | {row['model']} | {row['status']} | "
                     f"{number('AP')} | {number('AP_change_points')} | {number('AR')} | {number('model_seconds')} |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n")


def run_coco_screening(
    path: Path, *, device: int, limit: int | None = None,
    check_only: bool = False, resume: Path | None = None, progress=print,
):
    suite, configs = expand_coco_screening(path, device=device)
    preflight_coco(configs, device=device)
    progress(f"Preflight passed: {len(configs)} methods × {len(next(iter(configs.values())).models)} models; GPU {device}")
    if check_only:
        return None
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    signature = stable_hash({"configs": {k: v.model_dump(mode="json") for k, v in configs.items()}, "limit": limit})
    if resume is not None:
        root = resume.resolve()
        previous = json.loads((root / "screening.json").read_text())
        if previous["signature"] != signature:
            raise ValueError("Cannot resume a different suite configuration, GPU or image limit")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        root = suite.output_root / run_id
        root.mkdir(parents=True)
    report = {"success": False, "signature": signature, "device": device,
              "scope": "subset-smoke" if limit else "full-annotations",
              "model_ids": [m.id for m in next(iter(configs.values())).models],
              "conditions": [{"method": name} for name in configs]}
    # Only one process may resume/write this suite at a time.
    lock = root / ".running.lock"
    import fcntl
    with lock.open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"This suite is already running: {root}") from error
        write_coco_comparison(root, report)
        progress(f"Suite directory: {root}")
        for condition in report["conditions"]:
            name = condition["method"]
            config = configs[name]
            config_path = root / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
            progress(f"Starting {name}")
            try:
                summary_path = run_coco(config_path, limit=limit, resume_directory=root / name, progress=progress)
                condition["summary"] = json.loads(summary_path.read_text())
                condition["summary_path"] = str(summary_path)
            except Exception as error:
                condition["error"] = str(error)
                progress(f"{name}: FAILED: {error}")
            write_coco_comparison(root, report)
        report["success"] = all(c.get("summary", {}).get("success") and not c.get("error")
                                for c in report["conditions"])
        write_coco_comparison(root, report)
    return root / "screening.json"
