"""COCO keypoint evaluation in original-image coordinates, without pitch filtering."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import cv2
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from football_pose.artifacts import ArtifactStore, stable_hash
from football_pose.configuration import (
    ModelSpec, ProcessorSpec, _parameter_checkpoints, _resolve_checkpoint_parameters,
)
from football_pose.domain import FramePacket, PredictionRecord
from football_pose.jobs import atomic_write_json
from football_pose.model_mapping import COCO_KEYPOINT_NAMES
from football_pose.preprocessing import build_pipeline
from football_pose.preprocessing.base import ProcessorContext
from football_pose.runners import ExternalModelRunner


class CocoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annotations: Path
    images: Path
    output_dir: Path
    cache_root: Path
    models: list[ModelSpec]
    processors: list[ProcessorSpec] = Field(default_factory=list)
    cross_region_iou: float = Field(default=0.5, gt=0, le=1)

    @model_validator(mode="after")
    def independent_images(self):
        for processor in self.processors:
            if processor.type == "crop":
                detector = processor.params.get("detector", {})
                if detector.get("track", True):
                    raise ValueError("COCO images are independent: crop detector must set track: false")
                names = {int(key): value for key, value in detector.get("expected_class_names", {}).items()}
                if detector.get("class_ids") != [0] or names != {0: "person"}:
                    raise ValueError("COCO crops require class_ids: [0] and expected_class_names: {0: person}")
        return self


def load_coco_config(path: Path) -> CocoConfig:
    path = path.resolve()
    config = CocoConfig.model_validate(yaml.safe_load(path.read_text()))
    def resolved(value):
        return (path.parent / value).resolve() if value is not None else None
    for field in ("annotations", "images", "output_dir", "cache_root"):
        setattr(config, field, resolved(getattr(config, field)))
    for model in config.models:
        model.checkpoint = resolved(model.checkpoint)
        model.auxiliary_checkpoints = {name: resolved(value) for name, value in model.auxiliary_checkpoints.items()}
    for processor in config.processors:
        processor.params = _resolve_checkpoint_parameters(processor.params, path.parent)
    return config


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_predictions(path: Path, image_ids: list[int], *, cross_region_iou: float | None = None) -> list[dict]:
    allowed = set(image_ids)
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            record = PredictionRecord.model_validate_json(line)
            if record.frame_index not in allowed:
                raise ValueError(f"Unexpected COCO image id: {record.frame_index}")
            if record.person_score is None:
                raise ValueError("Runner must emit person_score for ranked COCO evaluation")
            records.append(record)
    if cross_region_iou is not None:
        # Suppress overlapping detections from different tiles/crops only. Full-image
        # output and distinct detections from the same model input stay untouched.
        from football_pose.pitch_filter import _iou, _pose_bbox
        grouped = defaultdict(list)
        for record in records:
            grouped[record.frame_index].append(record)
        records = []
        for group in grouped.values():
            kept = []
            for record in sorted(group, key=lambda r: -r.person_score):
                bbox = _pose_bbox(record.model_dump())
                if bbox is not None and any(
                    record.crop_id != previous.crop_id and previous_bbox is not None
                    and _iou(bbox, previous_bbox) >= cross_region_iou
                    for previous, previous_bbox in kept
                ):
                    continue
                kept.append((record, bbox))
            records.extend(record for record, _ in kept)
    predictions = []
    for record in records:
        prediction = {
            "image_id": record.frame_index, "category_id": 1,
            "score": record.person_score,
            "keypoints": [v for k in record.keypoints for v in (k.x, k.y, k.confidence)],
        }
        if record.source_bbox is not None:
            x1, y1, x2, y2 = record.source_bbox
            prediction["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        predictions.append(prediction)
    return predictions


def score_predictions(annotations: Path, predictions: list[dict], image_ids: list[int]):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        truth = COCO(str(annotations))
        if not image_ids or not set(image_ids) <= set(truth.imgs):
            raise ValueError("Evaluation requires a nonempty, valid image selection")
        if any(p["image_id"] not in image_ids for p in predictions):
            raise ValueError("Prediction image outside evaluation selection")
        if predictions:
            detected = truth.loadRes(predictions)
        else:
            # COCO.loadRes indexes the first result and cannot accept an empty list.
            detected = COCO()
            detected.dataset = {"images": list(truth.imgs.values()),
                                "categories": list(truth.cats.values()), "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(truth, detected, "keypoints")
        evaluator.params.imgIds = image_ids
        evaluator.params.catIds = [1]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    names = ("AP", "AP50", "AP75", "AP_medium", "AP_large",
             "AR", "AR50", "AR75", "AR_medium", "AR_large")
    return dict(zip(names, map(float, evaluator.stats), strict=True)), log.getvalue()


def run_coco(
    config_path: Path, limit: int | None = None, models: list[str] | None = None,
    *, resume_directory: Path | None = None, progress=None,
) -> Path:
    # Import before materializing data or starting GPU jobs.
    import pycocotools.cocoeval  # noqa: F401

    config = load_coco_config(config_path)
    progress = progress or (lambda message: None)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    ids = [model.id for model in config.models]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Models must have unique IDs and cannot be empty")
    if set(models or []) - set(ids):
        raise ValueError("Unknown model selection")
    selected_models = [m for m in config.models if not models or m.id in models]
    for model in selected_models:
        if model.checkpoint is None or not model.checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {model.id}: {model.checkpoint}")
        for checkpoint in model.auxiliary_checkpoints.values():
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing auxiliary checkpoint for {model.id}: {checkpoint}")
    data = json.loads(config.annotations.read_text())
    category = next(c for c in data["categories"] if c["id"] == 1)
    if list(category["keypoints"]) != list(COCO_KEYPOINT_NAMES):
        raise ValueError("Annotations must use standard COCO person keypoint ordering")
    images = sorted(data["images"], key=lambda im: im["id"])
    if len({im["id"] for im in images}) != len(images) or not images:
        raise ValueError("Invalid or empty image list")
    total = len(images)
    images = images[:limit] if limit else images
    image_ids = [im["id"] for im in images]
    image_root = config.images.resolve()
    sources = []
    for im in images:
        path = (image_root / im["file_name"]).resolve()
        if not path.is_relative_to(image_root):
            raise ValueError("Image filename escapes dataset directory")
        sources.append({"id": im["id"], "path": str(path), "sha256": sha256(path)})
    annotation_hash = sha256(config.annotations)
    digest = hashlib.sha256(json.dumps([annotation_hash, sources], sort_keys=True).encode()).hexdigest()
    processor_weights = {
        str(path): sha256(path) for spec in config.processors
        for path in _parameter_checkpoints(spec.params)
    }
    repository = Path(__file__).resolve().parents[2]
    code_hash = stable_hash({
        str(path.relative_to(repository)): sha256(path)
        for root in (repository / "src/football_pose", repository / "runners")
        for path in sorted(root.rglob("*")) if path.suffix in {".py", ".sh"}
    })
    pipeline_hash = stable_hash({"processors": [p.model_dump() for p in config.processors],
                                 "weights": processor_weights, "code": code_hash,
                                 "opencv": cv2.__version__})
    artifact_digest = stable_hash([digest, pipeline_hash])
    checkpoint_hashes = {
        model.id: {"pose": sha256(model.checkpoint),
                   **{name: sha256(path) for name, path in model.auxiliary_checkpoints.items()}}
        for model in selected_models
    }
    signature = stable_hash({"dataset": digest, "pipeline": pipeline_hash,
                             "config": config.model_dump(mode="json"),
                             "models": models, "weights": checkpoint_hashes,
                             "pycocotools": version("pycocotools")})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = resume_directory or config.output_dir / run_id
    output.mkdir(parents=True, exist_ok=True)
    previous = {}
    if (output / "summary.json").exists():
        previous = json.loads((output / "summary.json").read_text())
        if previous.get("signature") != signature:
            raise ValueError("Cannot resume: dataset, configuration, checkpoints or code changed")
    run_id = output.name

    def packets():
        for im, source in zip(images, sources, strict=True):
            image = cv2.imread(source["path"])
            if image is None or image.shape[:2] != (im["height"], im["width"]):
                raise ValueError(f"Image decode/size mismatch: {source['path']}")
            yield FramePacket(image=image, frame_index=im["id"], timestamp_seconds=0,
                              source_width=im["width"], source_height=im["height"],
                              source_id=f"coco-{im['id']}")

    pipeline = build_pipeline(config.processors)
    context = ProcessorContext(run_id, pipeline_hash)
    started = time.perf_counter()
    progress(f"Preparing {len(images)} COCO images")
    try:
        artifact, manifest, hit = ArtifactStore(config.cache_root).materialize(
            pipeline.process(packets(), context), artifact_identifier=artifact_digest, source_sha256=digest,
            pipeline_sha256=pipeline_hash, artifact_format="png_shards",
            provenance={"image_ids": image_ids, "processors": [p.model_dump() for p in config.processors],
                        "processor_checkpoint_sha256": processor_weights})
    finally:
        pipeline.close()
    preprocessing_seconds = time.perf_counter() - started
    report = {"success": True, "reference_match": "not assessed; reference protocol required",
              "scope": "subset-smoke" if len(images) < total else "full-annotations",
              "configuration": config.model_dump(mode="json"), "image_ids": image_ids,
              "image_count": len(images), "annotations_sha256": annotation_hash,
              "dataset_fingerprint": digest, "artifact": str(artifact), "cache_hit": hit,
              "signature": signature, "code_sha256": code_hash,
              "artifact_packets": manifest.frame_count,
              "preprocessing_wall_seconds": preprocessing_seconds,
              "artifact_processing_seconds": manifest.processing_seconds,
              "preprocessing_stage_timings": context.metadata.get("stage_timings", {}),
              "pycocotools": version("pycocotools"), "metric_scale": "0..1; -1 = unavailable",
              "protocol": "COCO keypoints OKS; maxDets=20; original-image coordinates; no pitch filtering; "
                          "independent images; cross-region bbox NMS for tile/crop inputs",
              "jobs": []}
    (output / "input-images.json").write_text(json.dumps(sources, indent=2))
    previous_jobs = {job["model_id"]: job for job in previous.get("jobs", [])}
    for index, model in enumerate(selected_models):
        directory = output / f"model-{index:02d}"
        old = previous_jobs.get(model.id, {})
        prediction_file = Path(old.get("predictions", ""))
        if (old.get("status") == "COMPLETE" and prediction_file.is_file()
                and old.get("predictions_sha256") == sha256(prediction_file)):
            progress(f"Reusing completed {model.id}")
            report["jobs"].append(old)
            atomic_write_json(output / "summary.json", report)
            continue
        progress(f"Running {model.id} on {manifest.frame_count} input packets")
        job = {"model_id": model.id, "checkpoint_sha256": checkpoint_hashes[model.id]["pose"],
               "auxiliary_checkpoint_sha256": {k: v for k, v in checkpoint_hashes[model.id].items() if k != "pose"}}
        try:
            result = ExternalModelRunner(model).run(
                artifact_path=artifact, output_directory=directory,
                experiment_id=run_id, pipeline_id=digest, source_video_id=digest)
            predictions = export_predictions(
                result.jsonl_path, image_ids,
                cross_region_iou=config.cross_region_iou
                if any(p.type in {"crop", "tile"} for p in config.processors) else None,
            )
            predictions_path = directory / "coco-predictions.json"
            predictions_path.write_text(json.dumps(predictions))
            metrics, log = score_predictions(config.annotations, predictions, image_ids)
            (directory / "coco-evaluation.txt").write_text(log)
            job.update(status="COMPLETE", metrics=metrics, records=len(predictions),
                       images_without_predictions=len(set(image_ids) - {p["image_id"] for p in predictions}),
                       predictions=str(predictions_path), model_wall_seconds=result.wall_seconds,
                       predictions_sha256=sha256(predictions_path),
                       actual_batch_size=result.batch_size, scoring="runner native person_score")
        except Exception as error:
            job.update(status="FAILED", error=str(error))
            report["success"] = False
        report["jobs"].append(job)
        atomic_write_json(output / "summary.json", report)
        progress(f"{model.id}: {job['status']}")
    return output / "summary.json"
