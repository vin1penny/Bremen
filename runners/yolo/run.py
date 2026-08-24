from __future__ import annotations

import os
import time

import numpy as np

from football_pose.artifacts import iter_artifact
from football_pose.domain import PredictionRecord
from football_pose.model_mapping import bbox_to_source, canonical_keypoints
from football_pose.runner_cli import batches, contract_args, contract_parser


def main() -> None:
    parser = contract_parser("Ultralytics YOLO Pose artifact runner")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Ultralytics inference image size; recorded as part of the model command.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum person detection confidence.",
    )
    namespace = parser.parse_args()
    args = contract_args(namespace)
    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    if namespace.imgsz < 1:
        parser.error("--imgsz must be positive")
    if not 0.0 <= namespace.confidence <= 1.0:
        parser.error("--confidence must be in [0.0, 1.0]")
    from ultralytics import YOLO

    model = YOLO(str(args.checkpoint))
    packets = iter_artifact(
        args.input_artifact, shard_index=args.shard_index, shard_count=args.shard_count
    )
    device = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else None
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for packet_batch in batches(packets, args.batch_size):
            start = time.perf_counter()
            results = model(
                [packet.image for packet in packet_batch],
                verbose=False,
                device=device,
                imgsz=namespace.imgsz,
                conf=namespace.confidence,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000 / len(packet_batch)
            for packet, result in zip(packet_batch, results, strict=True):
                if result.keypoints is None:
                    continue
                coordinates = result.keypoints.xy.detach().cpu().numpy()
                confidence_tensor = result.keypoints.conf
                confidences = (
                    np.ones(coordinates.shape[:2], dtype=np.float32)
                    if confidence_tensor is None
                    else confidence_tensor.detach().cpu().numpy()
                )
                boxes = (
                    np.zeros((len(coordinates), 4), dtype=np.float32)
                    if result.boxes is None
                    else result.boxes.xyxy.detach().cpu().numpy()
                )
                for person_index, (xy, scores, bbox) in enumerate(
                    zip(coordinates, confidences, boxes, strict=True)
                ):
                    record = PredictionRecord(
                        experiment_id=args.experiment_id,
                        pipeline_id=args.pipeline_id,
                        model_id=args.model_id,
                        source_video_id=args.source_video_id,
                        source_id=str(packet.source_id),
                        frame_index=packet.frame_index,
                        timestamp_seconds=packet.timestamp_seconds,
                        crop_id=packet.crop_id,
                        track_id=packet.track_id,
                        person_id=packet.track_id or f"yolo-{person_index}",
                        source_bbox=packet.source_bbox or bbox_to_source(packet, bbox),
                        keypoints=canonical_keypoints(packet, xy, scores),
                        inference_time_ms=elapsed_ms,
                    )
                    output.write(record.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
