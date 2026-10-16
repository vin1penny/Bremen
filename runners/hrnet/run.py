from __future__ import annotations

import time
import json

import numpy as np

from football_pose.artifacts import iter_artifact
from football_pose.domain import PredictionRecord
from football_pose.hrnet_input import add_hrnet_arguments, person_packets
from football_pose.preprocessing.base import ProcessorContext
from football_pose.model_mapping import bbox_to_source, canonical_keypoints
from football_pose.runner_cli import batches, contract_args, contract_parser


def _instances(result: dict) -> list[dict]:
    predictions = result.get("predictions", [])
    if predictions and isinstance(predictions[0], list):
        return predictions[0]
    return predictions


def main() -> None:
    parser = contract_parser("MMPose HRNet-W32 artifact runner")
    add_hrnet_arguments(parser)
    namespace = parser.parse_args()
    args = contract_args(namespace)
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise ValueError("HRNet requires an existing HRNet pose checkpoint via --checkpoint")
    from mmpose.apis import MMPoseInferencer

    inferencer = MMPoseInferencer(
        pose2d=namespace.pose_config,
        pose2d_weights=str(args.checkpoint) if args.checkpoint else None,
        det_model="whole_image",
        device="cuda:0",
    )
    packets = iter_artifact(
        args.input_artifact, shard_index=args.shard_index, shard_count=args.shard_count
    )
    stats = {"input_packets": 0, "detector_calls": 0, "generated_person_crops": 0,
             "reused_person_crops": 0, "detector_crop_seconds": 0.0, "pose_seconds": 0.0}
    crops = person_packets(
        packets, namespace=namespace,
        context=ProcessorContext(args.experiment_id, args.pipeline_id), stats=stats,
    )
    try:
        _write_predictions(args, inferencer, crops, stats)
    finally:
        crops.close()
    print(json.dumps({"hrnet_input": stats}, sort_keys=True))


def _write_predictions(args, inferencer, crops, stats) -> None:
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for packet_batch in batches(crops, args.batch_size):
            start = time.perf_counter()
            results = list(
                inferencer(
                    [packet.image for packet in packet_batch],
                    batch_size=len(packet_batch),
                    show=False,
                    return_vis=False,
                )
            )
            elapsed = time.perf_counter() - start
            stats["pose_seconds"] += elapsed
            elapsed_ms = elapsed * 1000 / len(packet_batch)
            for packet, result in zip(packet_batch, results, strict=True):
                for person_index, instance in enumerate(_instances(result)):
                    xy = np.asarray(instance["keypoints"], dtype=np.float64)
                    scores = np.asarray(
                        instance.get("keypoint_scores", np.ones(17)), dtype=np.float64
                    )
                    raw_bbox = np.asarray(
                        instance.get("bbox", [[0, 0, packet.width, packet.height]])
                    ).reshape(-1, 4)[0]
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
                        person_id=packet.track_id or f"hrnet-{person_index}",
                        person_score=packet.metadata.get("detection_score"),
                        source_bbox=packet.source_bbox or bbox_to_source(packet, raw_bbox),
                        keypoints=canonical_keypoints(packet, xy, scores),
                        inference_time_ms=elapsed_ms,
                    )
                    output.write(record.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
