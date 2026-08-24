from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

from football_pose.artifacts import iter_artifact
from football_pose.domain import PredictionRecord
from football_pose.model_mapping import BODY25_TO_COCO17, canonical_keypoints
from football_pose.runner_cli import contract_args, contract_parser


def _stage_body25_checkpoint(
    bundled_model_folder: Path, checkpoint: Path, destination: Path
) -> Path:
    """Combine bundled model definitions with an externally mounted BODY_25 weight."""
    shutil.copytree(bundled_model_folder, destination)
    weight = destination / "pose/body_25/pose_iter_584000.caffemodel"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.unlink(missing_ok=True)
    weight.symlink_to(checkpoint)
    return destination


def main() -> None:
    parser = contract_parser("CMU OpenPose BODY_25 artifact runner")
    parser.add_argument("--openpose-python", default="/opt/openpose/build/python")
    parser.add_argument("--model-folder", default="/opt/openpose/models")
    parser.add_argument(
        "--net-resolution",
        default="-1x368",
        help="OpenPose network resolution; recorded as part of the model command.",
    )
    namespace = parser.parse_args()
    args = contract_args(namespace)
    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    runtime_models = tempfile.TemporaryDirectory(prefix="openpose-models-")
    model_folder = _stage_body25_checkpoint(
        Path(namespace.model_folder),
        args.checkpoint,
        Path(runtime_models.name) / "models",
    )
    sys.path.append(namespace.openpose_python)
    from openpose import pyopenpose as op

    wrapper = op.WrapperPython()
    wrapper.configure(
        {
            "model_folder": str(model_folder),
            "model_pose": "BODY_25",
            "net_resolution": namespace.net_resolution,
            "keypoint_scale": 0,
            "render_pose": 0,
        }
    )
    wrapper.start()
    packets = iter_artifact(
        args.input_artifact, shard_index=args.shard_index, shard_count=args.shard_count
    )
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for packet in packets:
            datum = op.Datum()
            datum.cvInputData = packet.image
            start = time.perf_counter()
            wrapper.emplaceAndPop(op.VectorDatum([datum]))
            elapsed_ms = (time.perf_counter() - start) * 1000
            if datum.poseKeypoints is None:
                continue
            for person_index, pose in enumerate(np.asarray(datum.poseKeypoints)):
                xy = pose[:, :2]
                scores = pose[:, 2]
                visible = xy[scores > 0]
                if len(visible):
                    bbox_points = packet.points_to_source(
                        np.array(
                            [
                                [visible[:, 0].min(), visible[:, 1].min()],
                                [visible[:, 0].max(), visible[:, 1].max()],
                            ]
                        )
                    )
                    bbox = tuple(float(value) for value in bbox_points.reshape(-1))
                else:
                    bbox = packet.source_bbox
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
                    person_id=packet.track_id or f"openpose-{person_index}",
                    source_bbox=packet.source_bbox or bbox,
                    keypoints=canonical_keypoints(
                        packet, xy, scores, indices=BODY25_TO_COCO17
                    ),
                    inference_time_ms=elapsed_ms,
                )
                output.write(record.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
