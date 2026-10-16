from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from football_pose.configuration import ExperimentConfig, ModelSpec
from football_pose.experiments import ExperimentRunner
from football_pose.runners import ExternalModelRunner


REPOSITORY = Path(__file__).resolve().parents[1]


def test_external_runner_passes_auxiliary_detector_checkpoint(tmp_path, monkeypatch):
    detector = tmp_path / "detector.pt"
    detector.write_bytes(b"weights")
    output = tmp_path / "predictions.jsonl"
    output.write_text("")
    captured = {}
    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = ModelSpec(id="hrnet-w32", command=["runner"], auxiliary_checkpoints={"detector": detector})
    ExternalModelRunner(spec)._run_shard(
        artifact_path=tmp_path / "artifact", output_path=output,
        experiment_id="e", pipeline_id="p", source_video_id="s",
        batch_size=1, shard_index=0, shard_count=1, device=4,
    )
    index = captured["command"].index("--detector-checkpoint")
    assert captured["command"][index + 1] == str(detector)
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "4"


def test_detector_weight_content_changes_job_identity(tmp_path, tiny_video):
    detector = tmp_path / "detector.pt"
    detector.write_bytes(b"first weights")
    config = ExperimentConfig.model_validate({
        "name": "auxiliary", "input": tiny_video, "output_dir": tmp_path / "out",
        "cache": {"root": tmp_path / "cache"},
        "models": [{"id": "m", "command": ["runner"],
                    "auxiliary_checkpoints": {"detector": detector}}],
    })
    runner = ExperimentRunner(config)
    prepared = runner.prepare()
    first = runner._job_id(prepared, config.models[0])
    detector.write_bytes(b"different weights")
    assert first != runner._job_id(prepared, config.models[0])


def test_wrapper_mounts_detector_and_pose_weights_read_only(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    docker.chmod(0o755)
    detector = tmp_path / "detector.pt"
    pose = tmp_path / "pose.pth"
    result = subprocess.run([
        "bash", str(REPOSITORY / "runners/container-runner.sh"), "hrnet-image",
        "--input-artifact", str(tmp_path / "artifact"),
        "--checkpoint", str(pose), "--detector-checkpoint", str(detector),
        "--output-jsonl", str(tmp_path / "output/predictions.jsonl"),
    ], check=True, capture_output=True, text=True, env={
        **os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "CUDA_VISIBLE_DEVICES": "4",
    })
    args = json.loads(result.stdout)
    assert f"{detector}:{detector}:ro" in args
    assert f"{pose}:{pose}:ro" in args
    assert "device=4" in args
    assert args[args.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert "TORCH_HOME=/tmp/football-pose-torch" in args
