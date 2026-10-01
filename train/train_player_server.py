"""Train a football-player crop detector with a YOLO26 detection backbone."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--name", default="player-yolo26-production")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    import yaml
    from ultralytics import YOLO

    data = args.data.resolve()
    config = yaml.safe_load(data.read_text())
    if not isinstance(config, dict) or "train" not in config or "val" not in config:
        raise ValueError("Expected a YOLO detection data.yaml with train and val splits")
    if "kpt_shape" in config:
        raise ValueError("Player crop training needs detection labels, not keypoint labels")

    args.project.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    if model.task != "detect":
        raise ValueError(f"Expected a detection backbone, got {model.task!r}")
    model.train(
        data=str(data), project=str(args.project.resolve()), name=args.name,
        epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=0,
        workers=4, seed=0, deterministic=True, exist_ok=True,
    )
    checkpoint = Path(model.trainer.best)
    validated = YOLO(str(checkpoint))
    metrics = validated.val(data=str(data), device=0, imgsz=args.imgsz,
                            batch=args.batch, workers=4)
    with checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    payload = {
        "checkpoint": str(checkpoint),
        "sha256": checkpoint_hash,
        "data": str(data),
        "initial_model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "class_names": validated.names,
        "box_mAP50": float(metrics.box.map50),
        "box_mAP50-95": float(metrics.box.map),
        "metrics": metrics.results_dict,
        "note": "Verify class IDs and football-video crops before adopting this checkpoint.",
    }
    (checkpoint.parent.parent / "player-validation.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
