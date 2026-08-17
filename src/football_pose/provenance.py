from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from typing import Any


PACKAGES = (
    "av",
    "numpy",
    "opencv-contrib-python",
    "pandas",
    "pyarrow",
    "pydantic",
    "PyYAML",
    "torch",
    "ultralytics",
    "supervision",
)


def collect_provenance() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        hardware["cuda_available"] = torch.cuda.is_available()
        hardware["cuda_version"] = torch.version.cuda
        hardware["gpu_count"] = torch.cuda.device_count()
        hardware["gpus"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except ImportError:
        hardware["cuda_available"] = False
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    return {"packages": versions, "hardware": hardware, "git_commit": git_commit}
