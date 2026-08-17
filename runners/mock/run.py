from __future__ import annotations

from football_pose.artifacts import iter_artifact
from football_pose.domain import Keypoint, PredictionRecord
from football_pose.runner_cli import contract_args, contract_parser


def main() -> None:
    args = contract_args(contract_parser("Deterministic contract-test runner").parse_args())
    packets = iter_artifact(
        args.input_artifact, shard_index=args.shard_index, shard_count=args.shard_count
    )
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for packet in packets:
            center = packet.points_to_source([[packet.width / 2, packet.height / 2]])[0]
            keypoints = [
                Keypoint(x=float(center[0]), y=float(center[1]), confidence=0.5)
                for _ in range(17)
            ]
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
                person_id=packet.track_id or "mock-0",
                source_bbox=packet.source_bbox,
                keypoints=keypoints,
                inference_time_ms=0.0,
            )
            output.write(record.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
