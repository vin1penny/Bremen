from __future__ import annotations

import numpy as np

from football_pose.domain import FramePacket, Keypoint


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# BODY_25 indices corresponding to the COCO order above.
BODY25_TO_COCO17 = (0, 15, 16, 17, 18, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11)


def canonical_keypoints(
    packet: FramePacket,
    coordinates: np.ndarray,
    confidences: np.ndarray | None = None,
    *,
    indices: tuple[int, ...] | None = None,
) -> list[Keypoint]:
    xy = np.asarray(coordinates, dtype=np.float64)
    if indices is not None:
        xy = xy[list(indices)]
        if confidences is not None:
            confidences = np.asarray(confidences)[list(indices)]
    if xy.shape != (17, 2):
        raise ValueError(f"expected keypoint coordinates with shape (17, 2), received {xy.shape}")
    mapped = packet.points_to_source(xy)
    scores = np.ones(17, dtype=np.float64) if confidences is None else np.asarray(confidences)
    if scores.shape != (17,):
        raise ValueError(f"expected 17 confidence values, received {scores.shape}")
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    scores = np.clip(scores, 0.0, 1.0)
    return [
        Keypoint(x=float(point[0]), y=float(point[1]), confidence=float(score))
        for point, score in zip(mapped, scores, strict=True)
    ]


def bbox_to_source(
    packet: FramePacket, bbox: np.ndarray | list[float] | tuple[float, ...]
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    points = packet.points_to_source(np.array([[x1, y1], [x2, y2]], dtype=np.float64))
    return float(points[0, 0]), float(points[0, 1]), float(points[1, 0]), float(points[1, 1])
