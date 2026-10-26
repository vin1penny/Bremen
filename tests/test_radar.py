from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from football_pose.archive import PREDICTION_SCHEMA
from football_pose.cli import app
from football_pose.configuration import ExperimentConfig, PitchFilterConfig, VideoOutputConfig
from football_pose.experiments import ExperimentRunner
from football_pose.pitch_filter import (
    PitchGeometryFrame, PitchGeometryManifest, PreparedPitchGeometry, postprocess_pitch_predictions,
)
from football_pose.visualization import (
    _draw_radar, _radar_canvas, render_annotated_video,
)


def _location(x, y, **changes):
    return {"pitch_x_m": x, "pitch_y_m": y, "kept": True,
            "projection_status": "projected", **changes}


def _green_pixels(image):
    # H.264 4:2:0 softens the color of tiny dots; distinguish green chroma from
    # white pitch lines/dark turf rather than demanding uncompressed RGB values.
    pixels = image.astype(np.int16)
    return np.count_nonzero((pixels[:, :, 1] > 125) &
                            (pixels[:, :, 1] - np.maximum(pixels[:, :, 0], pixels[:, :, 2]) > 45))


def test_radar_dots_follow_export_positions_and_exclude_unkept():
    template = _radar_canvas()
    empty = np.zeros((720, 1280, 3), dtype=np.uint8)
    _draw_radar(empty, [], geometry_source="observed", template=template)
    image = np.zeros_like(empty)
    _draw_radar(image, [_location(52.5, 34.), _location(20., 20., kept=False),
                        _location(80., 40., projection_status="no_geometry"),
                        _location(None, None), _location(float("nan"), 10.)],
                geometry_source="observed", template=template)
    assert _green_pixels(image) > 0
    panel, scale, origin = template
    factor = min(1280 * .28 / panel.shape[1], 720 * .32 / panel.shape[0])
    pw, ph = round(panel.shape[1] * factor), round(panel.shape[0] * factor)
    px = (1280 - pw) // 2 + round((origin[0] + 52.5 * scale) * pw / panel.shape[1])
    py = 720 - ph - 8 + round((origin[1] + 34 * scale) * ph / panel.shape[0])
    assert _green_pixels(image[py-4:py+5, px-4:px+5]) > 0
    # The inset is bottom-centred and never paints over the top of the source.
    assert not image[:400].any()
    stale = np.zeros_like(empty)
    _draw_radar(stale, [_location(52.5, 34.)], geometry_source="unavailable", template=template)
    assert _green_pixels(stale) == 0
    # An available fit with zero detections is also an empty map, not the previous frame.
    assert _green_pixels(empty) == 0
    assert not np.array_equal(stale, empty)  # Distinct no-fit and zero-player labels.
    assert _green_pixels(template[0]) == 0  # The cached template was not modified.


def test_tiny_frames_are_not_overwritten_by_an_illegible_radar():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    _draw_radar(image, [_location(20., 20.)], geometry_source="observed", template=_radar_canvas())
    assert not image.any()


def test_saved_run_can_render_radar_without_model_inference(tmp_path):
    # Use a real encoded video large enough to show the inset, rather than only
    # testing the drawing helper or the tiny-frame compatibility path.
    source = tmp_path / "source.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        for _ in range(6):
            frame = av.VideoFrame.from_ndarray(np.zeros((360, 640, 3), dtype=np.uint8), format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    config = ExperimentConfig.model_validate({
        "name": "radar", "input": source, "output_dir": tmp_path / "output",
        "cache": {"root": tmp_path / "cache", "format": "png_shards"},
    })
    prepared = ExperimentRunner(config).prepare()
    raw = tmp_path / "raw.parquet"
    row = {
        "experiment_id": "experiment", "pipeline_id": "pipeline", "model_id": "model",
        "source_video_id": "video", "source_id": "frame", "frame_index": 0,
        "timestamp_seconds": 0., "person_id": "person", "source_bbox": [10., 10., 30., 30.],
        "keypoints": [{"x": 20., "y": 30., "confidence": .9} for _ in range(17)],
        "coordinate_space": "original_frame", "inference_time_ms": 0.,
    }
    pq.write_table(pa.Table.from_pylist([row], schema=PREDICTION_SCHEMA), raw)
    manifest = PitchGeometryManifest(
        geometry_id="geometry", source_sha256="source", checkpoint_sha256="checkpoint",
        settings={}, frame_count=6, observed_frames=1, bbox_frames=0, usable_frames=1,
        processing_seconds=0., frames=[PitchGeometryFrame(
            frame_index=0, homography=[100., 0., 0., 0., 100., 0., 0., 0., 1.],
            source="observed", landmarks=8, inliers=8,
        )],
    )
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(manifest.model_dump_json())
    geometry = PreparedPitchGeometry("geometry", geometry_path, manifest, True)
    pitch = PitchFilterConfig(enabled=True, checkpoint=tmp_path / "absent.pt")
    filtered = postprocess_pitch_predictions(
        prediction_parquet=raw, output_root=tmp_path / "filter", geometry=geometry, config=pitch,
    )
    config.pitch_filter = pitch
    report = {
        "configuration": config.model_dump(mode="json"), "artifact": str(prepared.artifact_path),
        "pitch_filter": {"geometry": str(geometry_path)},
        "jobs": [{"status": "COMPLETE", "model_id": "model", "job_id": "job",
                  "outputs": {"parquet": str(raw), "pitch_decisions": str(filtered.decisions_parquet)}}],
    }
    summary_path = tmp_path / "summary.json"
    original = json.dumps(report)
    summary_path.write_text(original)
    runner = CliRunner()
    result = runner.invoke(app, ["render-videos", str(summary_path), "--output-root", str(tmp_path / "videos")])
    assert result.exit_code == 0, result.output
    index = json.loads(next((tmp_path / "videos").glob("videos-*.json")).read_text())
    path = Path(index["videos"][0]["video"])
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="bgr24") for frame in container.decode(video=0)]
        assert len(frames) == 6
        assert container.streams.video[0].width == 640
    assert _green_pixels(frames[0][240:354, 230:410]) > 0
    assert _green_pixels(frames[1][240:354, 230:410]) == 0
    assert summary_path.read_text() == original
    previous = path.stat().st_mtime_ns
    result = runner.invoke(app, ["render-videos", str(summary_path), "--output-root", str(tmp_path / "videos")])
    assert result.exit_code == 0, result.output
    assert path.stat().st_mtime_ns == previous

    screening_path = tmp_path / "screening.json"
    screening_path.write_text(json.dumps({"conditions": [
        {"method": "baseline", "summary": report}, {"method": "crop", "status": "FAILED"},
    ]}))
    result = runner.invoke(app, ["render-videos", str(screening_path), "--output-root", str(tmp_path / "videos")])
    assert result.exit_code == 0, result.output
    indices = [json.loads(p.read_text()) for p in (tmp_path / "videos").glob("videos-*.json")]
    screen_index = next(index for index in indices if index["source_report"] == str(screening_path))
    assert screen_index["videos"][0]["condition"] == "baseline"
    assert screen_index["skipped"][0]["condition"] == "crop"

    # Disabling the inset permits the older rendering inputs without a location file.
    settings = VideoOutputConfig(draw_radar=False)
    rendered = render_annotated_video(
        source_path=source, artifact_path=prepared.artifact_path, prediction_parquet=raw,
        output_path=tmp_path / "disabled.mp4", model_id="model", settings=settings,
        pitch_geometry_json=geometry_path,
    )
    assert rendered.frames == 6
