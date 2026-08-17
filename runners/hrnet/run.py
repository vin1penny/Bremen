from __future__ import annotations

import time

import numpy as np

from football_pose.artifacts import iter_artifact
from football_pose.domain import PredictionRecord
from football_pose.model_mapping import bbox_to_source, canonical_keypoints
from football_pose.runner_cli import batches, contract_args, contract_parser


def _instances(result: dict) -> list[dict]:
    predictions = result.get("predictions", [])
    if predictions and isinstance(predictions[0], list):
        return predictions[0]
    return predictions


def main() -> None:
    parser = contract_parser("MMPose HRNet-W32 artifact runner")
    parser.add_argument(
        "--pose-config",
        default="td-hm_hrnet-w32_8xb64-210e_coco-256x192",
    )
    parser.add_argument(
        "--det-model",
        default="human",
        help="Use 'whole_image' for player-crop artifacts and 'human' for full frames.",
    )
    namespace = parser.parse_args()
    args = contract_args(namespace)
    from mmpose.apis import MMPoseInferencer

    inferencer = MMPoseInferencer(
        pose2d=namespace.pose_config,
        pose2d_weights=str(args.checkpoint) if args.checkpoint else None,
        det_model=namespace.det_model,
        device="cuda:0",
    )
    packets = iter_artifact(
        args.input_artifact, shard_index=args.shard_index, shard_count=args.shard_count
    )
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for packet_batch in batches(packets, args.batch_size):
            start = time.perf_counter()
            results = list(
                inferencer(
                    [packet.image for packet in packet_batch],
                    batch_size=len(packet_batch),
                    show=False,
                    return_vis=False,
                )
            )
            elapsed_ms = (time.perf_counter() - start) * 1000 / len(packet_batch)
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
                        source_bbox=packet.source_bbox or bbox_to_source(packet, raw_bbox),
                        keypoints=canonical_keypoints(packet, xy, scores),
                        inference_time_ms=elapsed_ms,
                    )
                    output.write(record.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
