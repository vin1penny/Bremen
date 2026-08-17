from __future__ import annotations

from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def test_yolo_numpy_pins_cover_container_and_host_python_versions() -> None:
    requirements = (REPOSITORY / "requirements/yolo.txt").read_text(encoding="utf-8")

    assert 'numpy==2.2.6; python_version < "3.11"' in requirements
    assert 'numpy==2.3.5; python_version >= "3.11"' in requirements
