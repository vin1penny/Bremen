"""Full-image COCO keypoint evaluation, separate from football/pitch filtering."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import cv2
import yaml
from pydantic import BaseModel, ConfigDict

from football_pose.artifacts import ArtifactStore
from football_pose.configuration import ModelSpec
from football_pose.domain import FramePacket, PredictionRecord
from football_pose.model_mapping import COCO_KEYPOINT_NAMES
from football_pose.runners import ExternalModelRunner


class CocoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annotations: Path
    images: Path
    output_dir: Path
    cache_root: Path
    models: list[ModelSpec]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_predictions(path: Path, image_ids: list[int]) -> list[dict]:
    allowed = set(image_ids)
    predictions = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = PredictionRecord.model_validate_json(line)
        if record.frame_index not in allowed:
            raise ValueError(f"Unexpected COCO image id: {record.frame_index}")
        if record.person_score is None:
            raise ValueError("Runner must emit person_score for ranked COCO evaluation")
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


def run_coco(config_path: Path, limit: int | None = None, models: list[str] | None = None) -> Path:
    # Import before materializing data or starting GPU jobs.
    import pycocotools.cocoeval  # noqa: F401

    config = CocoConfig.model_validate(yaml.safe_load(config_path.read_text()))
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

    def packets():
        for im, source in zip(images, sources, strict=True):
            image = cv2.imread(source["path"])
            if image is None or image.shape[:2] != (im["height"], im["width"]):
                raise ValueError(f"Image decode/size mismatch: {source['path']}")
            yield FramePacket(image=image, frame_index=im["id"], timestamp_seconds=0,
                              source_width=im["width"], source_height=im["height"],
                              source_id=f"coco-{im['id']}")

    artifact, _, hit = ArtifactStore(config.cache_root).materialize(
        packets(), artifact_identifier=digest, source_sha256=digest,
        pipeline_sha256=hashlib.sha256(b"coco-original-bgr-v1").hexdigest(),
        artifact_format="png_shards", provenance={"image_ids": image_ids})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = config.output_dir / run_id
    output.mkdir(parents=True)
    report = {"success": True, "reference_match": "not assessed; reference protocol required",
              "scope": "subset-smoke" if len(images) < total else "full-annotations",
              "configuration": config.model_dump(mode="json"), "image_ids": image_ids,
              "image_count": len(images), "annotations_sha256": annotation_hash,
              "dataset_fingerprint": digest, "artifact": str(artifact), "cache_hit": hit,
              "pycocotools": version("pycocotools"), "metric_scale": "0..1; -1 = unavailable",
              "protocol": "COCO keypoints OKS; maxDets=20; original images; no pitch filtering",
              "jobs": []}
    (output / "input-images.json").write_text(json.dumps(sources, indent=2))
    for index, model in enumerate(selected_models):
        directory = output / f"model-{index:02d}"
        job = {"model_id": model.id, "checkpoint_sha256": sha256(model.checkpoint)}
        try:
            result = ExternalModelRunner(model).run(
                artifact_path=artifact, output_directory=directory,
                experiment_id=run_id, pipeline_id=digest, source_video_id=digest)
            predictions = export_predictions(result.jsonl_path, image_ids)
            predictions_path = directory / "coco-predictions.json"
            predictions_path.write_text(json.dumps(predictions))
            metrics, log = score_predictions(config.annotations, predictions, image_ids)
            (directory / "coco-evaluation.txt").write_text(log)
            job.update(status="COMPLETE", metrics=metrics, records=len(predictions),
                       images_without_predictions=len(set(image_ids) - {p["image_id"] for p in predictions}),
                       predictions=str(predictions_path), model_wall_seconds=result.wall_seconds,
                       actual_batch_size=result.batch_size, scoring="runner native person_score")
        except Exception as error:
            job.update(status="FAILED", error=str(error))
            report["success"] = False
        report["jobs"].append(job)
        (output / "summary.json").write_text(json.dumps(report, indent=2))
    return output / "summary.json"
