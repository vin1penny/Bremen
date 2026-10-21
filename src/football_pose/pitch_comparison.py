"""Pitch-only checkpoint comparison; no player models or production-weight changes."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from fractions import Fraction
import gc
import os
from pathlib import Path
import shutil
from typing import Any, Callable
from uuid import uuid4

import av
import cv2
import numpy as np

from football_pose.artifacts import sha256_file, stable_hash
from football_pose.configuration import load_config
from football_pose.jobs import atomic_write_json
from football_pose.pitch_filter import PitchGeometryFrame, PitchGeometryStore, PreparedPitchGeometry
from football_pose.pitch_models import ROBOFLOW_SPORTS_REVISION, ROBOFLOW_PITCH_URL, ROBOFLOW_PITCH_SHA256
from football_pose.provenance import collect_provenance
from football_pose.video import VideoSource
from football_pose.visualization import _draw_pitch_region


UPSTREAM_REVISION = ROBOFLOW_SPORTS_REVISION
REFERENCE_URL = ROBOFLOW_PITCH_URL
REFERENCE_SHA256 = ROBOFLOW_PITCH_SHA256
# Edges from sports/configs/soccer.py at UPSTREAM_REVISION, converted to zero-based IDs.
PITCH_EDGES = tuple((a - 1, b - 1) for a, b in (
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
    (10, 11), (11, 12), (12, 13), (14, 15), (15, 16), (16, 17),
    (18, 19), (19, 20), (20, 21), (23, 24),
    (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
    (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
    (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
))


def visible_points(frame: PitchGeometryFrame, width: int, height: int) -> dict[int, dict]:
    return {
        int(point["id"]): point for point in frame.predicted_keypoints
        if point["confidence"] >= 0.5
        and 1 < point["x"] < width and 1 < point["y"] < height
    }


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(path))
    try:
        shape = model.model.yaml.get("kpt_shape")
        if model.task != "pose" or shape != [32, 3]:
            raise ValueError(f"Expected 32-landmark pose weights, got task={model.task}, kpt_shape={shape}: {path}")
        return {
            "task": model.task, "kpt_shape": shape,
            "scale": model.model.yaml.get("scale"),
            "parameters": sum(parameter.numel() for parameter in model.model.parameters()),
        }
    finally:
        del model
        gc.collect()


def geometry_statistics(geometry: PreparedPitchGeometry, width: int, height: int) -> dict[str, Any]:
    frames = geometry.manifest.frames
    counts = [len(visible_points(frame, width, height)) for frame in frames]
    errors = [frame.median_reprojection_cm for frame in frames
              if frame.homography is not None and frame.median_reprojection_cm is not None]
    return {
        "frames": len(frames), "visible_landmarks_mean": float(np.mean(counts)) if counts else 0.0,
        "frames_with_four_visible_landmarks": sum(count >= 4 for count in counts),
        "accepted_geometry_frames": sum(frame.homography is not None for frame in frames),
        "accepted_fit_median_reprojection_cm": float(np.median(errors)) if errors else None,
        "inference_and_geometry_seconds": geometry.manifest.processing_seconds,
        "cache_hit": geometry.cache_hit,
    }


def landmark_disagreement(
    geometries: dict[str, PreparedPitchGeometry], width: int, height: int,
) -> dict[str, Any]:
    ours, reference = (geometries[name].manifest.frames for name in ("ours", "roboflow"))
    distances = []
    for left, right in zip(ours, reference, strict=True):
        if left.frame_index != right.frame_index:
            raise ValueError("Pitch comparison frame indices do not match")
        a, b = visible_points(left, width, height), visible_points(right, width, height)
        for index in a.keys() & b.keys():
            distances.append(float(np.hypot(a[index]["x"] - b[index]["x"], a[index]["y"] - b[index]["y"])))
    return {
        "common_visible_landmark_pairs": len(distances),
        "median_pixel_distance": float(np.median(distances)) if distances else None,
        "p95_pixel_distance": float(np.percentile(distances, 95)) if distances else None,
        "note": "Inter-model disagreement in original-frame pixels, not ground-truth error.",
    }


def annotate_pitch(image: np.ndarray, frame: PitchGeometryFrame, name: str) -> np.ndarray:
    height, width = image.shape[:2]
    output = image.copy()
    thickness = max(1, round(width / 960))
    points = visible_points(frame, width, height)
    for start, end in PITCH_EDGES:
        if start in points and end in points:
            a = (round(points[start]["x"]), round(points[start]["y"]))
            b = (round(points[end]["x"]), round(points[end]["y"]))
            cv2.line(output, a, b, (0, 200, 0), thickness, cv2.LINE_AA)
    _draw_pitch_region(output, frame.pitch_polygon, thickness)
    for index, point in points.items():
        xy = (round(point["x"]), round(point["y"]))
        cv2.circle(output, xy, 3 * thickness, (255, 0, 255), -1)
        cv2.putText(output, f"P{index} {point['confidence']:.2f}", (xy[0] + 5, xy[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thickness, (255, 0, 255), thickness)
    # Header is outside the original picture; no input resizing or hidden landmarks.
    output = cv2.copyMakeBorder(output, 64, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
    scale = max(0.35, min(1.0, width / 1400))
    cv2.putText(output, f"{name} | frame {frame.frame_index} | points {len(points)} | fit {frame.source}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, "Magenta: landmarks | Green: detected segments | Cyan: accepted boundary",
                (8, 49), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def render_comparison(
    source: VideoSource, geometries: dict[str, PreparedPitchGeometry], root: Path,
    progress: Callable[[str], Any],
) -> dict[str, str]:
    info = source.probe()
    rate = info.average_rate or Fraction(30, 1)
    width, height = info.width + info.width % 2, info.height + 64 + info.height % 2
    paths = {name: root / f"{name}.mp4" for name in ("ours", "roboflow", "side-by-side")}
    expected = len(geometries["ours"].manifest.frames)
    if expected == 0 or len(geometries["roboflow"].manifest.frames) != expected:
        raise ValueError("Pitch geometries must cover the same nonempty video")
    frames = {name: {frame.frame_index: frame for frame in value.manifest.frames}
              for name, value in geometries.items()}
    streams = {}
    count = 0
    with ExitStack() as stack:
        for name, path in paths.items():
            container = stack.enter_context(av.open(str(path.with_suffix(".partial.mp4")), mode="w"))
            stream = container.add_stream("libx264", rate=rate)
            stream.width = width * (2 if name == "side-by-side" else 1)
            stream.height, stream.pix_fmt = height, "yuv420p"
            stream.options = {"crf": "20", "preset": "fast"}
            streams[name] = (container, stream)
        for packet in source.frames():
            panels = {name: annotate_pitch(packet.image, rows[packet.frame_index], name)
                      for name, rows in frames.items()}
            panels["side-by-side"] = np.concatenate([panels["ours"], panels["roboflow"]], axis=1)
            for name, image in panels.items():
                container, stream = streams[name]
                frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                frame.pts, frame.time_base = count, Fraction(rate.denominator, rate.numerator)
                for encoded in stream.encode(frame):
                    container.mux(encoded)
            count += 1
            if count % 100 == 0:
                progress(f"Rendered {count}/{expected} frames")
        if count != expected:
            raise ValueError("Input frame count changed between inference and rendering")
        for container, stream in streams.values():
            for encoded in stream.encode():
                container.mux(encoded)
    for path in paths.values():
        os.replace(path.with_suffix(".partial.mp4"), path)
    return {name: str(path) for name, path in paths.items()}


def compare_pitch(
    config_path: Path, *, reference_checkpoint: Path, output_root: Path, device: int,
    progress: Callable[[str], Any] = print,
) -> Path:
    if device < 0:
        raise ValueError("device must be a non-negative physical GPU ID")
    reservation = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if reservation and str(device) not in reservation.split(","):
        raise ValueError(f"GPU {device} is outside CUDA_VISIBLE_DEVICES={reservation}")
    config = load_config(config_path, check_paths=False)
    if config.pitch_filter.checkpoint is None:
        raise ValueError("Config must name our pitch-filter checkpoint")
    checkpoints = {"ours": config.pitch_filter.checkpoint, "roboflow": reference_checkpoint.resolve()}
    for path in (config.input, *checkpoints.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Required comparison input does not exist: {path}")
    if sha256_file(checkpoints["roboflow"]) != REFERENCE_SHA256:
        raise ValueError("Roboflow reference SHA-256 mismatch; download the pinned checkpoint from the documented upstream source")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    metadata = {}
    for name, checkpoint in checkpoints.items():
        progress(f"Inspecting {name}: {checkpoint}")
        metadata[name] = {"checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint),
                          **checkpoint_metadata(checkpoint)}
    identical = metadata["ours"]["sha256"] == metadata["roboflow"]["sha256"]
    if identical:
        progress("WARNING: checkpoint files are identical; this is not a two-weight comparison")
    source = VideoSource(config.input)
    info = source.probe()
    source_hash = sha256_file(config.input)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    root = output_root.resolve() / run_id
    root.mkdir(parents=True, exist_ok=False)
    settings = config.pitch_filter.model_copy(update={
        "enabled": True, "device": str(device), "image_size": 640, "batch_size": 1,
        "detection_confidence": 0.25, "landmark_confidence": 0.5, "max_fallback_frames": 0,
    })
    report = {
        "success": False, "input": str(config.input), "input_sha256": source_hash,
        "config": str(config_path.resolve()), "device": device, "run_id": run_id,
        "upstream_revision": UPSTREAM_REVISION, "reference_download_url": REFERENCE_URL,
        "reference_expected_sha256": REFERENCE_SHA256,
        "identical_checkpoints": identical, "models": metadata,
        "settings": {key: value for key, value in settings.model_dump(mode="json").items()
                     if key not in {"checkpoint", "checkpoint_sha256", "cache_root"}},
        "pitch_template": "Weserstadion only: 105 x 68 m; shared 32-landmark ordering",
        "landmark_order_assumption": "Both checkpoints must use the Roboflow football-field 32-slot semantic order; shape validation alone cannot prove semantic correspondence.",
        "implementation_sha256": sha256_file(Path(__file__)),
        "note": "Weight comparison under shared inference/calibration, not an exact replay of the upstream radar demo or a ground-truth accuracy evaluation.",
        **collect_provenance(),
    }
    summary = root / "summary.json"
    atomic_write_json(summary, report)
    progress(f"Pitch comparison output: {root}")
    geometries = {}
    # Do not reuse detections produced with a different SDK/runtime or comparison implementation.
    cache_namespace = stable_hash({"packages": report.get("packages", {}),
                                   "implementation": report["implementation_sha256"]})
    store = PitchGeometryStore(output_root.resolve() / "cache" / cache_namespace)
    for name, checkpoint in checkpoints.items():
        progress(f"Inferring {name}: every frame, imgsz=640, batch=1 (or reuse matching geometry cache)")
        geometry = store.prepare(source_path=config.input, source_sha256=source_hash,
                                 config=settings.model_copy(update={
                                     "checkpoint": checkpoint,
                                     "checkpoint_sha256": metadata[name]["sha256"],
                                 }))
        if geometry.manifest.checkpoint_sha256 != metadata[name]["sha256"]:
            raise RuntimeError(f"Checkpoint changed during comparison: {checkpoint}")
        geometries[name] = geometry
        geometry_path = root / f"{name}-geometry.json"
        shutil.copyfile(geometry.path, geometry_path)
        metadata[name].update({"geometry": str(geometry_path),
                               **geometry_statistics(geometry, info.width, info.height)})
        atomic_write_json(summary, report)
        progress(f"{name}: {metadata[name]['accepted_geometry_frames']}/{metadata[name]['frames']} accepted pitch fits")
    if sha256_file(config.input) != source_hash:
        raise RuntimeError("Source video changed during comparison")
    report["disagreement"] = landmark_disagreement(geometries, info.width, info.height)
    progress("Rendering ours.mp4, roboflow.mp4 and side-by-side.mp4")
    report["videos"] = render_comparison(source, geometries, root, progress)
    report["success"] = True
    atomic_write_json(summary, report)
    lines = ["# Pitch checkpoint comparison", "", report["note"], "",
             f"Identical checkpoint files: {identical}", "",
             "| Model | Visible landmarks/frame | Frames with >=4 landmarks | Accepted pitch fits | Total frames |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, item in metadata.items():
        lines.append(f"| {name} | {item['visible_landmarks_mean']:.2f} | {item['frames_with_four_visible_landmarks']} | {item['accepted_geometry_frames']} | {item['frames']} |")
    lines += ["", "Visible = confidence >=0.5, coordinates inside the original frame; IDs are zero-based P0-P31.",
              "Counts and fitting residuals measure yield/internal consistency, NOT landmark accuracy.",
              f"Common-landmark median disagreement (source pixels): {report['disagreement']['median_pixel_distance']}",
              "", "Videos: ours.mp4, roboflow.mp4, side-by-side.mp4 (ours left, Roboflow right).",
              "Magenta: detected landmarks; green: segments with two confident endpoints; cyan: accepted fitted boundary.",
              "The separate JSON geometry files retain all SDK-returned landmark confidences, including those not displayed."]
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return summary
