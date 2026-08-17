from __future__ import annotations

import sys

from football_pose.artifacts import iter_artifact
from football_pose.domain import Keypoint, PredictionRecord
from football_pose.runner_cli import contract_args, contract_parser


def main() -> None:
    args = contract_args(contract_parser("OOM retry fixture").parse_args())
    if args.batch_size > 2:
        print("CUDA out of memory", file=sys.stderr)
        raise SystemExit(1)
    keypoints = [Keypoint(x=0, y=0, confidence=0) for _ in range(17)]
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for packet in iter_artifact(
            args.input_artifact,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        ):
            record = PredictionRecord(
                experiment_id=args.experiment_id,
                pipeline_id=args.pipeline_id,
                model_id=args.model_id,
                source_video_id=args.source_video_id,
                source_id=str(packet.source_id),
                frame_index=packet.frame_index,
                timestamp_seconds=packet.timestamp_seconds,
                person_id="fixture",
                keypoints=keypoints,
                inference_time_ms=0,
            )
            output.write(record.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
