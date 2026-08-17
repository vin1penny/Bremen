from __future__ import annotations

import json
import os
import time
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    VALIDATING = "VALIDATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


ALLOWED_TRANSITIONS = {
    JobStatus.PENDING: {JobStatus.RUNNING},
    JobStatus.RUNNING: {JobStatus.VALIDATING, JobStatus.FAILED},
    JobStatus.VALIDATING: {JobStatus.COMPLETE, JobStatus.FAILED},
    JobStatus.FAILED: {JobStatus.RUNNING},
    JobStatus.COMPLETE: set(),
}


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    experiment_id: str
    pipeline_id: str
    model_id: str
    status: JobStatus = JobStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    batch_size: int = Field(default=1, ge=1)
    created_at_unix: float = Field(default_factory=time.time)
    updated_at_unix: float = Field(default_factory=time.time)
    error: dict[str, Any] | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)

    def transition(self, status: JobStatus, **updates: Any) -> "JobRecord":
        if status not in ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(f"invalid job transition: {self.status} -> {status}")
        payload = self.model_dump()
        payload.update(updates)
        payload["status"] = status
        payload["updated_at_unix"] = time.time()
        return JobRecord.model_validate(payload)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class JobStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, job_id: str) -> Path:
        return self.root / job_id / "job.json"

    def load(self, job_id: str) -> JobRecord | None:
        path = self.path_for(job_id)
        if not path.is_file():
            return None
        return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, record: JobRecord) -> None:
        atomic_write_json(self.path_for(record.job_id), record.model_dump(mode="json"))
