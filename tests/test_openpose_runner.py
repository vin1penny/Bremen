from __future__ import annotations

from pathlib import Path

from runners.openpose.run import _stage_body25_checkpoint


def test_external_body25_checkpoint_is_staged_with_bundled_definition(
    tmp_path: Path,
) -> None:
    bundled = tmp_path / "bundled"
    definition = bundled / "pose/body_25/pose_deploy.prototxt"
    definition.parent.mkdir(parents=True)
    definition.write_text("model definition", encoding="utf-8")
    checkpoint = tmp_path / "pose_iter_584000.caffemodel"
    checkpoint.write_bytes(b"model weights")

    staged = _stage_body25_checkpoint(bundled, checkpoint, tmp_path / "runtime")

    assert (staged / "pose/body_25/pose_deploy.prototxt").read_text(
        encoding="utf-8"
    ) == "model definition"
    assert (staged / "pose/body_25/pose_iter_584000.caffemodel").resolve() == checkpoint
