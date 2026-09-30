import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

pytest.importorskip("pycocotools")

from football_pose.artifacts import iter_artifact
from football_pose.coco_evaluation import export_predictions, run_coco, score_predictions
from football_pose.model_mapping import COCO_KEYPOINT_NAMES


@pytest.fixture
def dataset(tmp_path):
    keypoints = [v for i in range(17) for v in (20 + i, 20 + i, 2)]
    data = {
        "info": {},
        "images": [{"id": i, "file_name": f"{i}.jpg", "height": 100, "width": 100 + i}
                   for i in (3, 9)],
        "categories": [{"id": 1, "name": "person", "keypoints": list(COCO_KEYPOINT_NAMES)}],
        "annotations": [{"id": i, "image_id": i, "category_id": 1, "iscrowd": 0,
                         "bbox": [10, 10, 60, 70], "area": 4200, "num_keypoints": 17,
                         "keypoints": keypoints} for i in (3, 9)],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data))
    for image in data["images"]:
        cv2.imwrite(str(tmp_path / image["file_name"]), np.zeros((100, image["width"], 3), np.uint8))
    return path, data


def test_official_evaluator_perfect_missing_and_empty(dataset):
    path, data = dataset
    predictions = [{"image_id": a["image_id"], "category_id": 1, "score": .9,
                    "keypoints": a["keypoints"]} for a in data["annotations"]]
    metrics, _ = score_predictions(path, predictions, [3, 9])
    assert metrics["AP"] == pytest.approx(1)
    missing, _ = score_predictions(path, predictions[:1], [3, 9])
    assert missing["AR"] == pytest.approx(.5)
    empty, _ = score_predictions(path, [], [3, 9])
    assert empty["AP"] == 0
    predictions[0]["keypoints"] = [v + 100 if i % 3 != 2 else v
                                    for i, v in enumerate(predictions[0]["keypoints"])]
    displaced, _ = score_predictions(path, predictions, [3, 9])
    assert displaced["AP"] < metrics["AP"]


def test_coco_artifact_and_report(dataset, tmp_path, monkeypatch):
    path, _ = dataset
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"test")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "annotations": str(path), "images": str(tmp_path),
        "output_dir": str(tmp_path / "results"), "cache_root": str(tmp_path / "cache"),
        "models": [{"id": "test", "command": ["unused"], "checkpoint": str(checkpoint)}]}))

    def fake_run(self, **kwargs):
        packets = list(iter_artifact(kwargs["artifact_path"]))
        assert [p.frame_index for p in packets] == [3, 9]
        assert [p.width for p in packets] == [103, 109]
        directory = kwargs["output_directory"]
        directory.mkdir()
        jsonl = directory / "predictions.jsonl"
        jsonl.write_text("")
        return SimpleNamespace(jsonl_path=jsonl, wall_seconds=0, batch_size=1)

    monkeypatch.setattr("football_pose.coco_evaluation.ExternalModelRunner.run", fake_run)
    report_path = run_coco(config)
    report = json.loads(report_path.read_text())
    assert report["success"]
    assert report["scope"] == "full-annotations"
    assert report["jobs"][0]["images_without_predictions"] == 2
    assert report["jobs"][0]["metrics"]["AP"] == 0
    assert run_coco(config) != report_path  # Never overwrite earlier evaluations.


def test_export_requires_native_score_and_valid_id(tmp_path):
    path = tmp_path / "predictions.jsonl"
    record = dict(experiment_id="a", pipeline_id="b", model_id="c", source_video_id="d",
                  source_id="coco-3", frame_index=3, timestamp_seconds=0, person_id="p",
                  keypoints=[dict(x=12, y=15, confidence=.8)] * 17, inference_time_ms=0)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="person_score"):
        export_predictions(path, [3])
    record["person_score"] = .7
    path.write_text(json.dumps(record))
    assert export_predictions(path, [3])[0]["score"] == .7
    with pytest.raises(ValueError, match="image id"):
        export_predictions(path, [9])


def test_real_external_runner_subset_and_failure(dataset, tmp_path, monkeypatch):
    path, _ = dataset
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    checkpoint = tmp_path / "mock-weights"
    checkpoint.write_bytes(b"mock")
    config = tmp_path / "config.yaml"
    repository = Path(__file__).resolve().parents[1]
    values = {
        "annotations": str(path), "images": str(tmp_path),
        "output_dir": str(tmp_path / "results"), "cache_root": str(tmp_path / "cache"),
        "models": [{"id": "mock", "checkpoint": str(checkpoint),
                    "command": ["{python}", str(repository / "runners/mock/run.py")]}],
    }
    config.write_text(yaml.safe_dump(values))
    report = json.loads(run_coco(config, limit=1).read_text())
    assert report["success"]
    assert report["scope"] == "subset-smoke"
    assert report["image_ids"] == [3]
    predictions = json.loads(Path(report["jobs"][0]["predictions"]).read_text())
    assert predictions[0]["image_id"] == 3
    assert len(predictions[0]["keypoints"]) == 51
    values["models"][0]["command"] = ["{python}", "-c", "raise SystemExit(1)"]
    config.write_text(yaml.safe_dump(values))
    report = json.loads(run_coco(config, limit=1).read_text())
    assert not report["success"]
    assert report["jobs"][0]["status"] == "FAILED"
