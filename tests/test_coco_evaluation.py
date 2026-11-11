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


def _mock_config(dataset, tmp_path, processors=None):
    path, _ = dataset
    checkpoint = tmp_path / "mock-weights"
    checkpoint.write_bytes(b"mock")
    config = tmp_path / "config.yaml"
    repository = Path(__file__).resolve().parents[1]
    values = {
        "annotations": str(path), "images": str(tmp_path),
        "output_dir": str(tmp_path / "results"), "cache_root": str(tmp_path / "cache"),
        "processors": processors or [],
        "models": [{"id": "mock", "checkpoint": str(checkpoint),
                    "command": ["{python}", str(repository / "runners/mock/run.py")]}],
    }
    config.write_text(yaml.safe_dump(values))
    return config, values


@pytest.mark.parametrize("processor,packets", [
    ({"type": "resize", "params": {"scale": 2}}, 2),
    ({"type": "tile", "params": {"rows": 2, "columns": 2, "overlap_ratio": .1}}, 8),
])
def test_coco_preprocessing_retains_ids_and_coordinate_transforms(dataset, tmp_path, processor, packets):
    config, values = _mock_config(dataset, tmp_path, [processor])
    report = json.loads(run_coco(config).read_text())
    assert report["success"]
    assert report["image_ids"] == [3, 9]
    materialized = list(iter_artifact(report["artifact"]))
    assert len(materialized) == packets
    assert {p.frame_index for p in materialized} == {3, 9}
    if processor["type"] == "resize":
        assert materialized[0].width == 206
        assert np.allclose(materialized[0].points_to_source(np.array([[40., 40.]])), [[20., 20.]])
    else:
        assert materialized[-1].to_source[0, 2] > 0
        assert materialized[-1].to_source[1, 2] > 0
    values["processors"] = []
    config.write_text(yaml.safe_dump(values))
    baseline = json.loads(run_coco(config).read_text())
    assert baseline["artifact"] != report["artifact"]
    assert baseline["dataset_fingerprint"] == report["dataset_fingerprint"]


def test_coco_resume_reuses_verified_results_and_rejects_changed_weights(dataset, tmp_path, monkeypatch):
    config, values = _mock_config(dataset, tmp_path)
    directory = tmp_path / "resumed"
    first = json.loads(run_coco(config, resume_directory=directory).read_text())
    from football_pose.runners import ExternalModelRunner
    original = ExternalModelRunner.run
    calls = []
    def counted(self, **kwargs):
        calls.append(self.spec.id)
        return original(self, **kwargs)
    monkeypatch.setattr(ExternalModelRunner, "run", counted)
    resumed = json.loads(run_coco(config, resume_directory=directory).read_text())
    assert not calls
    assert resumed["jobs"] == first["jobs"]
    Path(first["jobs"][0]["predictions"]).write_text("[]")
    assert json.loads(run_coco(config, resume_directory=directory).read_text())["success"]
    assert calls == ["mock"]
    Path(values["models"][0]["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Cannot resume"):
        run_coco(config, resume_directory=directory)


def test_coco_empty_crop_artifact_still_scores_all_images(dataset, tmp_path, monkeypatch):
    from football_pose.preprocessing.base import Pipeline
    class EmptyCrop:
        name = "crop"
        def process(self, packet, context):
            return []
    monkeypatch.setattr("football_pose.coco_evaluation.build_pipeline", lambda specs: Pipeline([EmptyCrop()]))
    config, _ = _mock_config(dataset, tmp_path)
    report = json.loads(run_coco(config).read_text())
    assert report["success"]
    assert report["image_count"] == 2
    assert report["artifact_packets"] == 0
    assert report["jobs"][0]["metrics"]["AP"] == 0
    assert report["jobs"][0]["images_without_predictions"] == 2


def test_cross_region_nms_keeps_native_score_and_same_region_people(tmp_path):
    record = dict(experiment_id="a", pipeline_id="b", model_id="c", source_video_id="d",
                  source_id="tile-a", frame_index=3, timestamp_seconds=0, person_id="p",
                  crop_id="tile-a", source_bbox=[0, 0, 20, 20], person_score=.9,
                  keypoints=[dict(x=12, y=15, confidence=.8)] * 17, inference_time_ms=0)
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        record, {**record, "person_id": "p2", "person_score": .8},
        {**record, "crop_id": "tile-b", "source_id": "tile-b", "person_score": .7},
    ]))
    assert len(export_predictions(path, [3])) == 3
    assert [p["score"] for p in export_predictions(path, [3], cross_region_iou=.5)] == [.9, .8]


def test_coco_crop_rejects_tracking_and_football_detector(dataset, tmp_path):
    from football_pose.coco_evaluation import CocoConfig
    _, values = _mock_config(dataset, tmp_path)
    values["processors"] = [{"type": "crop", "params": {"detector": {
        "track": True, "class_ids": [0], "expected_class_names": {0: "person"}}}}]
    with pytest.raises(ValueError, match="independent"):
        CocoConfig.model_validate(values)
    values["processors"][0]["params"]["detector"] = {"track": False, "class_ids": [1, 2, 3]}
    with pytest.raises(ValueError, match="COCO crops"):
        CocoConfig.model_validate(values)


def test_full_screening_matrix_and_separate_coco_detector(monkeypatch):
    from football_pose.coco_screening import expand_coco_screening
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    repository = Path(__file__).resolve().parents[1]
    _, expanded = expand_coco_screening(repository / "configs/lyra-coco-preprocessing-screen.yaml", device=4)
    assert len(expanded) == 12
    assert len(expanded["baseline"].models) == 3
    assert not expanded["baseline"].processors
    assert all(model.devices == [4] for config in expanded.values() for model in config.models)
    detector = expanded["crop"].processors[0].params["detector"]
    assert detector["device"] == "4"
    assert detector["track"] is False
    assert detector["class_ids"] == [0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    _, reserved = expand_coco_screening(repository / "configs/lyra-coco-preprocessing-screen.yaml", device=4)
    assert reserved["crop"].processors[0].params["detector"]["device"] == "0"
    assert reserved["baseline"].models[0].devices == [4]


def test_screening_runs_and_resumes_with_mock_models(dataset, tmp_path, monkeypatch):
    from football_pose.coco_screening import run_coco_screening
    config, _ = _mock_config(dataset, tmp_path)
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump({
        "base_config": str(config), "output_root": str(tmp_path / "screen"),
        "cache_root": str(tmp_path / "cache"),
        "methods": [{"name": "baseline"}, {"name": "resize", "processor": {
            "type": "resize", "params": {"scale": 2}}}],
    }))
    path = run_coco_screening(suite, device=0, progress=lambda _: None)
    report = json.loads(path.read_text())
    assert report["success"]
    assert len(report["comparison"]) == 2
    assert (path.parent / "comparison.csv").is_file()
    def unexpected(*args, **kwargs):
        raise AssertionError("completed models should not run again")
    monkeypatch.setattr("football_pose.coco_evaluation.ExternalModelRunner.run", unexpected)
    resumed = run_coco_screening(suite, device=0, resume=path.parent, progress=lambda _: None)
    assert json.loads(resumed.read_text())["success"]
    with pytest.raises(ValueError, match="Cannot resume"):
        run_coco_screening(suite, device=0, limit=1, resume=path.parent, progress=lambda _: None)
