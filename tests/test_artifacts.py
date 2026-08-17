from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from football_pose.artifacts import ArtifactStore, iter_artifact
from football_pose.domain import FramePacket


def packets(count: int = 5) -> list[FramePacket]:
    return [
        FramePacket(
            image=np.full((12, 16, 3), index * 31, dtype=np.uint8),
            frame_index=index,
            timestamp_seconds=index / 10,
            source_width=16,
            source_height=12,
        )
        for index in range(count)
    ]


@pytest.mark.parametrize("artifact_format", ["png_shards", "ffv1"])
def test_lossless_artifact_round_trip_and_cache_hit(
    tmp_path: Path, artifact_format: str
) -> None:
    source = packets()
    store = ArtifactStore(tmp_path)
    path, manifest, cache_hit = store.materialize(
        iter(source),
        artifact_identifier=f"artifact-{artifact_format}",
        source_sha256="source",
        pipeline_sha256="pipeline",
        artifact_format=artifact_format,
        shard_size=2,
    )
    assert not cache_hit
    assert manifest.frame_count == len(source)
    decoded = list(iter_artifact(path))
    assert [item.frame_index for item in decoded] == list(range(len(source)))
    for expected, actual in zip(source, decoded, strict=True):
        np.testing.assert_array_equal(actual.image, expected.image)

    _, cached_manifest, cache_hit = store.materialize(
        iter(()),
        artifact_identifier=f"artifact-{artifact_format}",
        source_sha256="source",
        pipeline_sha256="pipeline",
        artifact_format=artifact_format,
    )
    assert cache_hit
    assert cached_manifest.frame_count == len(source)


def test_artifact_sharding_is_disjoint_and_complete(tmp_path: Path) -> None:
    path, _, _ = ArtifactStore(tmp_path).materialize(
        iter(packets(7)),
        artifact_identifier="sharded",
        source_sha256="source",
        pipeline_sha256="pipeline",
        artifact_format="png_shards",
        shard_size=2,
    )
    shards = [list(iter_artifact(path, shard_index=index, shard_count=3)) for index in range(3)]
    indexes = [[packet.frame_index for packet in shard] for shard in shards]
    assert indexes == [[0, 3, 6], [1, 4], [2, 5]]
