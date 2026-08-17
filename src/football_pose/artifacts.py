from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import time
import uuid
from collections.abc import Iterable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import av
import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from football_pose.domain import FrameManifestRecord, FramePacket


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    source_sha256: str
    pipeline_sha256: str
    format: Literal["ffv1", "png_shards"]
    frame_count: int = Field(ge=0)
    created_at_unix: float
    processing_seconds: float = Field(ge=0.0)
    pinned: bool
    complete: bool
    provenance: dict[str, Any] = Field(default_factory=dict)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_id(source_sha256: str, pipeline_payload: Any) -> tuple[str, str]:
    pipeline_sha256 = stable_hash(pipeline_payload)
    return stable_hash({"source": source_sha256, "pipeline": pipeline_sha256}), pipeline_sha256


def _record_for(packet: FramePacket, sequence_index: int, **storage: Any) -> FrameManifestRecord:
    return FrameManifestRecord(
        source_id=str(packet.source_id),
        frame_index=packet.frame_index,
        timestamp_seconds=packet.timestamp_seconds,
        source_width=packet.source_width,
        source_height=packet.source_height,
        width=packet.width,
        height=packet.height,
        to_source=packet.to_source.reshape(-1).tolist(),
        crop_id=packet.crop_id,
        track_id=packet.track_id,
        source_bbox=packet.source_bbox,
        sequence_index=sequence_index,
        metadata=packet.metadata,
        **storage,
    )


class PngShardWriter:
    def __init__(self, directory: Path, shard_size: int) -> None:
        self.directory = directory
        self.shard_size = shard_size
        self.manifest_stream = (directory / "frames.jsonl").open("w", encoding="utf-8")
        self.archive: tarfile.TarFile | None = None
        self.shard_index = -1
        self.count = 0

    def _open_shard(self) -> str:
        if self.archive is not None:
            self.archive.close()
        self.shard_index += 1
        name = f"frames-{self.shard_index:05d}.tar"
        self.archive = tarfile.open(self.directory / name, mode="w")
        return name

    def write(self, packet: FramePacket) -> None:
        if self.count % self.shard_size == 0:
            shard = self._open_shard()
        else:
            shard = f"frames-{self.shard_index:05d}.tar"
        success, encoded = cv2.imencode(".png", packet.image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not success:
            raise RuntimeError(f"PNG encoding failed for {packet.source_id}")
        member = f"frames/{self.count:09d}-{packet.source_id}.png"
        payload = encoded.tobytes()
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        assert self.archive is not None
        self.archive.addfile(info, io.BytesIO(payload))
        record = _record_for(packet, self.count, member=member, shard=shard)
        self.manifest_stream.write(record.model_dump_json() + "\n")
        self.count += 1

    def close(self) -> int:
        if self.archive is not None:
            self.archive.close()
        self.manifest_stream.close()
        return self.count


class Ffv1Writer:
    def __init__(self, directory: Path, rate: Fraction | None = None) -> None:
        self.directory = directory
        self.rate = rate or Fraction(30, 1)
        self.manifest_stream = (directory / "frames.jsonl").open("w", encoding="utf-8")
        self.container = av.open(str(directory / "frames.mkv"), mode="w")
        self.stream: av.VideoStream | None = None
        self.size: tuple[int, int] | None = None
        self.count = 0

    def write(self, packet: FramePacket) -> None:
        size = (packet.width, packet.height)
        if self.stream is None:
            self.stream = self.container.add_stream("ffv1", rate=self.rate)
            self.stream.width, self.stream.height = size
            self.stream.pix_fmt = "bgr0"
            self.size = size
        elif self.size != size:
            raise ValueError("FFV1 artifacts require a constant frame size; use png_shards")
        video_frame = av.VideoFrame.from_ndarray(packet.image, format="bgr24")
        video_frame.pts = self.count
        video_frame.time_base = Fraction(self.rate.denominator, self.rate.numerator)
        assert self.stream is not None
        for encoded_packet in self.stream.encode(video_frame):
            self.container.mux(encoded_packet)
        record = _record_for(packet, self.count)
        self.manifest_stream.write(record.model_dump_json() + "\n")
        self.count += 1

    def close(self) -> int:
        if self.stream is not None:
            for encoded_packet in self.stream.encode():
                self.container.mux(encoded_packet)
        self.container.close()
        self.manifest_stream.close()
        return self.count


def _read_records(directory: Path) -> list[FrameManifestRecord]:
    records: list[FrameManifestRecord] = []
    with (directory / "frames.jsonl").open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                records.append(FrameManifestRecord.model_validate_json(line))
            except Exception as error:
                raise ValueError(f"invalid artifact frame record at line {line_number}") from error
    return records


def _packet_from_record(record: FrameManifestRecord, image: np.ndarray) -> FramePacket:
    return FramePacket(
        image=image,
        frame_index=record.frame_index,
        timestamp_seconds=record.timestamp_seconds,
        source_width=record.source_width,
        source_height=record.source_height,
        to_source=np.asarray(record.to_source, dtype=np.float64).reshape(3, 3),
        source_id=record.source_id,
        crop_id=record.crop_id,
        track_id=record.track_id,
        source_bbox=record.source_bbox,
        metadata=record.metadata,
    )


def iter_artifact(
    directory: str | Path, *, shard_index: int = 0, shard_count: int = 1
) -> Iterator[FramePacket]:
    root = Path(directory)
    artifact = load_artifact_manifest(root)
    records = _read_records(root)
    selected = [record for record in records if record.sequence_index % shard_count == shard_index]
    if artifact.format == "ffv1":
        selected_by_index = {record.sequence_index: record for record in selected}
        with av.open(str(root / "frames.mkv")) as container:
            decoded_index = 0
            for frame in container.decode(video=0):
                record = selected_by_index.get(decoded_index)
                if record is not None:
                    yield _packet_from_record(record, frame.to_ndarray(format="bgr24"))
                decoded_index += 1
        return
    by_shard: dict[str, list[FrameManifestRecord]] = {}
    for record in selected:
        assert record.shard is not None
        by_shard.setdefault(record.shard, []).append(record)
    for shard_name in sorted(by_shard):
        with tarfile.open(root / shard_name, mode="r") as archive:
            for record in sorted(by_shard[shard_name], key=lambda item: item.sequence_index):
                assert record.member is not None
                extracted = archive.extractfile(record.member)
                if extracted is None:
                    raise ValueError(f"missing member {record.member} in {shard_name}")
                image = cv2.imdecode(np.frombuffer(extracted.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"invalid PNG member {record.member}")
                yield _packet_from_record(record, image)


def load_artifact_manifest(directory: str | Path) -> ArtifactManifest:
    path = Path(directory) / "artifact.json"
    manifest = ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if not manifest.complete:
        raise ValueError(f"artifact is incomplete: {directory}")
    return manifest


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, artifact_identifier: str) -> Path:
        return self.root / artifact_identifier

    def materialize(
        self,
        packets: Iterable[FramePacket],
        *,
        artifact_identifier: str,
        source_sha256: str,
        pipeline_sha256: str,
        artifact_format: Literal["ffv1", "png_shards"],
        shard_size: int = 256,
        rate: Fraction | None = None,
        pinned: bool = True,
        provenance: dict[str, Any] | None = None,
    ) -> tuple[Path, ArtifactManifest, bool]:
        target = self.path_for(artifact_identifier)
        if (target / "artifact.json").is_file():
            return target, load_artifact_manifest(target), True
        lock_path = self.root / f".{artifact_identifier}.lock"
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"artifact is currently being written: {artifact_identifier}") from error
        os.close(lock_fd)
        temporary = self.root / f".{artifact_identifier}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir(parents=False)
        start = time.perf_counter()
        try:
            writer: PngShardWriter | Ffv1Writer
            if artifact_format == "ffv1":
                writer = Ffv1Writer(temporary, rate=rate)
            else:
                writer = PngShardWriter(temporary, shard_size=shard_size)
            try:
                for packet in packets:
                    writer.write(packet)
            finally:
                count = writer.close()
            manifest = ArtifactManifest(
                artifact_id=artifact_identifier,
                source_sha256=source_sha256,
                pipeline_sha256=pipeline_sha256,
                format=artifact_format,
                frame_count=count,
                created_at_unix=time.time(),
                processing_seconds=time.perf_counter() - start,
                pinned=pinned,
                complete=True,
                provenance=provenance or {},
            )
            (temporary / "artifact.json").write_text(
                manifest.model_dump_json(indent=2), encoding="utf-8"
            )
            if target.exists():
                shutil.rmtree(temporary)
                return target, load_artifact_manifest(target), True
            os.replace(temporary, target)
            return target, manifest, False
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        finally:
            lock_path.unlink(missing_ok=True)
