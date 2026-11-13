"""Bounded, matched inspection exports, independent of inference artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any

import cv2
import numpy as np

from football_pose.artifacts import sha256_file, stable_hash
from football_pose.configuration import ExperimentConfig
from football_pose.jobs import atomic_write_json
from football_pose.preprocessing.jersey import JerseyColorEnhancementProcessor
from football_pose.video import VideoSource


def _write_png(path: Path, image: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-",
                                     suffix=".png", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f"could not write jersey diagnostic: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _comparison(input_image: np.ndarray, mask: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Labelled preview; separate PNGs retain native pixel dimensions."""
    scale = min(1.0, 960 / input_image.shape[1])
    size = (max(1, round(input_image.shape[1] * scale)),
            max(1, round(input_image.shape[0] * scale)))
    panels = []
    for title, pixels in (
        ("Input", input_image),
        ("Colour selection: black=0, white=1", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
        ("Enhanced input to model", output),
    ):
        resized = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
        panel = cv2.copyMakeBorder(resized, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def render_jersey_diagnostics(
    config: ExperimentConfig, *, experiment_id: str, artifact_id: str,
) -> dict[str, Any]:
    """Recompute only selected source frames, also when inference is cached.

    Standalone jersey conditions have original source frames as their exact stage
    inputs. Combined chains are rejected instead of showing a misleading mask
    computed from a different input or replaying a stateful detector/tracker.
    """
    if not config.jersey_diagnostics.enabled:
        return {"enabled": False}
    specs = config.resolved_processors()
    if not any(spec.type == "jersey_color_enhance" for spec in specs):
        return {"enabled": True, "status": "not_applicable"}
    start = time.perf_counter()
    settings = config.jersey_diagnostics.model_dump(mode="json")
    identity = stable_hash({
        "artifact_id": artifact_id, "settings": settings,
        "renderer": sha256_file(Path(__file__)),
    })[:16]
    root = config.output_dir / "experiments" / experiment_id / "jersey-diagnostics" / identity
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if all((root / name).is_file()
               for sample in manifest["samples"] for name in sample["files"].values()):
            return {
                "enabled": True, "status": "complete", "manifest": str(manifest_path),
                "sample_count": len(manifest["samples"]), "cache_hit": True,
                "missing_frame_indices": manifest["missing_frame_indices"],
                "wall_seconds": time.perf_counter() - start,
            }

    processor = JerseyColorEnhancementProcessor(**specs[0].params)
    requested = set(config.jersey_diagnostics.frame_indices)
    root.mkdir(parents=True, exist_ok=True)
    # A failed regeneration must not leave an old completion marker behind.
    manifest_path.unlink(missing_ok=True)
    samples = []
    frames = VideoSource(config.input).frames()
    try:
        for packet in frames:
            if packet.frame_index in requested:
                enhanced, mask = processor.inspect(packet)
                mask_image = np.rint(np.clip(mask, 0, 1) * 255).astype(np.uint8)
                prefix = f"frame-{packet.frame_index:09d}"
                files = {name: f"{prefix}-{name}.png"
                         for name in ("input", "mask", "enhanced", "comparison")}
                for name, pixels in (
                    ("input", packet.image), ("mask", mask_image),
                    ("enhanced", enhanced.image),
                    ("comparison", _comparison(packet.image, mask_image, enhanced.image)),
                ):
                    _write_png(root / files[name], pixels)
                samples.append({
                    "frame_index": packet.frame_index,
                    "timestamp_seconds": packet.timestamp_seconds,
                    "width": packet.width, "height": packet.height,
                    "selected_pixel_fraction": float(np.mean(mask > 0)),
                    "mean_selection_weight": float(mask.mean()),
                    "mean_blend_weight": float(mask.mean()) * processor.strength,
                    "files": files,
                })
            if packet.frame_index >= max(requested):
                break
    finally:
        frames.close()
    missing = sorted(requested - {sample["frame_index"] for sample in samples})
    manifest = {
        "artifact_id": artifact_id,
        "input": str(config.input), "processor": specs[0].model_dump(mode="json"),
        "effective_parameters": {
            "colors": specs[0].params["colors"],
            "color_tolerance": processor.color_tolerance,
            "strength": processor.strength, "chroma_gain": processor.chroma_gain,
            "clip_limit": processor.clahe.getClipLimit(),
            "tile_grid": list(processor.clahe.getTilesGridSize()),
        },
        "settings": settings, "samples": samples, "missing_frame_indices": missing,
        "mask_encoding": "uint8 PNG: round(255 * colour-similarity weight); not blend weight or player confidence",
        "statistics": "Computed from the floating-point mask before PNG quantisation; not accuracy metrics.",
        "wall_seconds": time.perf_counter() - start,
    }
    # A manifest marks completion only after every requested available sample was written.
    atomic_write_json(manifest_path, manifest)
    return {
        "enabled": True, "status": "complete", "manifest": str(manifest_path),
        "sample_count": len(samples), "cache_hit": False,
        "missing_frame_indices": missing, "wall_seconds": time.perf_counter() - start,
    }
