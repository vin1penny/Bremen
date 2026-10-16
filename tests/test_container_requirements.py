from __future__ import annotations

from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def test_yolo_numpy_pins_cover_container_and_host_python_versions() -> None:
    requirements = (REPOSITORY / "requirements/yolo.txt").read_text(encoding="utf-8")

    assert 'numpy==2.2.6; python_version < "3.11"' in requirements
    assert 'numpy==2.3.5; python_version >= "3.11"' in requirements


def test_hrnet_container_has_existing_detector_runtime_and_matching_cuda_wheel():
    requirements = (REPOSITORY / "requirements/hrnet.txt").read_text()
    dockerfile = (REPOSITORY / "containers/Dockerfile.hrnet").read_text()
    assert "ultralytics==8.4.19" in requirements
    assert "supervision==0.27.0.post1" in requirements
    assert "mmpose==1.3.2" in requirements
    assert "https://download.pytorch.org/whl/cu118" in dockerfile
    assert "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html" in dockerfile
    assert "--only-binary=mmcv" in dockerfile
