"""Pitch-only equivalent of train_remote.ipynb for a Lyra tmux session."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    import yaml
    from ultralytics import YOLO

    data = args.data.resolve()
    config = yaml.safe_load(data.read_text())
    if config.get("kpt_shape") != [32, 3]:
        raise ValueError("Expected the notebook's 32-landmark pitch dataset ([32, 3]).")
    name = "pitch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    # Roboflow exports often contain ../train paths or paths from another machine.
    # Resolve the known exported layout without modifying the source dataset.
    config["path"] = str(data.parent)
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        images = data.parent / folder / "images"
        if not images.is_dir():
            raise ValueError(f"Dataset split directory missing: {images}")
        config[split] = str(images)
    args.project.mkdir(parents=True, exist_ok=True)
    normalized_data = args.project.resolve() / f"{name}-data.yaml"
    normalized_data.write_text(yaml.safe_dump(config))
    model = YOLO("yolov8n-pose.pt")
    model.train(data=str(normalized_data), project=str(args.project.resolve()), name=name,
                epochs=args.epochs, batch=16, imgsz=640, mosaic=0.0,
                device=0, workers=4, seed=0, deterministic=True, exist_ok=False)
    checkpoint = Path(model.trainer.best)
    metrics = YOLO(str(checkpoint)).val(data=str(normalized_data), device=0, imgsz=640,
                                       batch=16, workers=4)
    with checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    payload = {
        "checkpoint": str(checkpoint),
        "sha256": checkpoint_hash,
        "data": str(data), "metric": "pose_mAP50-95",
        "pose_mAP50": float(metrics.pose.map50),
        "pose_mAP50-95": float(metrics.pose.map),
        "metrics": metrics.results_dict,
        "note": "Validate landmarks on the experiment video before adopting this checkpoint.",
    }
    (checkpoint.parent.parent / "pitch-validation.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
