"""The fixed YOLO-player-detector input adapter used by top-down HRNet."""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path
import time
from typing import Any

from football_pose.domain import FramePacket
from football_pose.preprocessing.base import ProcessorContext
from football_pose.preprocessing.cropping import CropProcessor, UltralyticsDetectionProvider


def add_hrnet_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pose-config", default="td-hm_hrnet-w32_8xb64-210e_coco-256x192")
    parser.add_argument("--det-model", choices=["whole_image"], default="whole_image",
                        help="HRNet receives person crops; no fallback MMDetection detector is used.")
    parser.add_argument("--detector-checkpoint", type=Path)
    parser.add_argument("--detector-image-size", type=int, default=1920)
    parser.add_argument("--detector-confidence", type=float, default=0.3)
    parser.add_argument("--detector-class-ids", default="1,2,3")
    parser.add_argument("--crop-padding", type=float, default=0.1)
    parser.add_argument("--crop-min-size", type=int, default=32)


def detector_parameters(namespace: argparse.Namespace, *, device: str) -> dict[str, Any]:
    if namespace.detector_checkpoint is None:
        raise ValueError("HRNet requires --detector-checkpoint for full-frame or tiled inputs")
    classes = [int(item) for item in namespace.detector_class_ids.split(",")]
    names = {1: "goalkeeper", 2: "player", 3: "referee"}
    if not classes or any(index not in names for index in classes):
        raise ValueError("HRNet football detector class IDs must be a subset of 1,2,3")
    return {
        "checkpoint": str(namespace.detector_checkpoint),
        "confidence": namespace.detector_confidence,
        "class_ids": classes, "expected_class_names": {index: names[index] for index in classes},
        "image_size": namespace.detector_image_size, "device": device,
        # Tiled packets are not successive video frames. This adapter does not
        # advance a temporal tracker once per tile; the shared crop step may track.
        "track": False,
    }


def person_packets(
    packets: Iterable[FramePacket], *, namespace: argparse.Namespace,
    context: ProcessorContext, stats: dict[str, Any],
) -> Iterator[FramePacket]:
    cropper = None
    try:
        for packet in packets:
            stats["input_packets"] += 1
            if packet.source_bbox is not None:
                # The shared learned-crop artifact already contains one person.
                stats["reused_person_crops"] += 1
                yield packet
                continue
            if cropper is None:
                cropper = CropProcessor(
                    detector=UltralyticsDetectionProvider(**detector_parameters(namespace, device="0")),
                    padding_ratio=namespace.crop_padding, min_size=namespace.crop_min_size,
                )
            started = time.perf_counter()
            crops = cropper.process(packet, context)
            stats["detector_crop_seconds"] += time.perf_counter() - started
            stats["detector_calls"] += 1
            stats["generated_person_crops"] += len(crops)
            yield from crops
    finally:
        if cropper is not None:
            cropper.close()
