#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Versioned job state, retention and event contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def _json_safe(value: Any) -> Any:
    """Convert contract payloads without serializing full numerical matrices."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item) and getattr(value, "ndim", 1) == 0:
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            return str(value)
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "nbytes"):
        return JobResultSummary(
            result_type=f"{type(value).__module__}.{type(value).__qualname__}",
            estimated_bytes=max(0, int(value.nbytes)),
            shape=tuple(int(item) for item in value.shape),
            dtype=str(value.dtype),
            message="Numerical result omitted from serialized snapshot",
        ).to_dict()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    return str(value)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED}


class JobEventType(str, Enum):
    QUEUED = "job_queued"
    STARTED = "job_started"
    PROGRESS = "progress_changed"
    WARNING = "warning_raised"
    ARTIFACT = "artifact_created"
    COMPLETED = "job_completed"
    FAILED = "job_failed"
    CANCELLED = "job_cancelled"


@dataclass(frozen=True, slots=True)
class JobRetentionPolicy:
    """Bound memory retained by the in-process job registry."""

    max_events_per_job: int = 256
    max_terminal_jobs: int = 128
    terminal_ttl_seconds: float = 3600.0
    max_result_bytes: int = 16 * 1024 * 1024
    max_total_result_bytes: int = 128 * 1024 * 1024
    max_event_payload_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_events_per_job", max(8, int(self.max_events_per_job)))
        object.__setattr__(self, "max_terminal_jobs", max(1, int(self.max_terminal_jobs)))
        object.__setattr__(self, "terminal_ttl_seconds", max(0.0, float(self.terminal_ttl_seconds)))
        object.__setattr__(self, "max_result_bytes", max(0, int(self.max_result_bytes)))
        object.__setattr__(self, "max_total_result_bytes", max(0, int(self.max_total_result_bytes)))
        object.__setattr__(self, "max_event_payload_bytes", max(0, int(self.max_event_payload_bytes)))


@dataclass(frozen=True, slots=True)
class JobResultSummary:
    """Lightweight replacement for a result that was too large to retain."""

    result_type: str
    estimated_bytes: int
    shape: tuple[int, ...] = ()
    dtype: str = ""
    message: str = "Result released by retention policy"
    schema_version: str = "mygpr.job_result_summary.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "result_type": self.result_type,
            "estimated_bytes": self.estimated_bytes,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class JobEvent:
    job_id: str
    event_type: JobEventType
    sequence: int
    timestamp_utc: str
    message: str = ""
    completed: int = 0
    total: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "mygpr.job_event.v1"

    @property
    def progress(self) -> float | None:
        if self.total <= 0:
            return None
        return min(1.0, max(0.0, float(self.completed) / float(self.total)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "event_type": self.event_type.value,
            "sequence": self.sequence,
            "timestamp_utc": self.timestamp_utc,
            "message": self.message,
            "completed": self.completed,
            "total": self.total,
            "payload": _json_safe(self.payload),
        }

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        event_type: JobEventType,
        sequence: int,
        message: str = "",
        completed: int = 0,
        total: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> "JobEvent":
        return cls(
            job_id=str(job_id),
            event_type=event_type,
            sequence=int(sequence),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            message=str(message),
            completed=max(0, int(completed)),
            total=max(0, int(total)),
            payload=dict(payload or {}),
        )


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    job_id: str
    title: str
    status: JobStatus
    completed: int = 0
    total: int = 0
    message: str = ""
    result: Any = None
    result_released: bool = False
    error_type: str = ""
    error_code: str = ""
    error_message: str = ""
    error_details: dict[str, Any] = field(default_factory=dict)
    resource_keys: tuple[str, ...] = ()
    created_at_utc: str = ""
    updated_at_utc: str = ""
    schema_version: str = "mygpr.job_snapshot.v1"

    @property
    def progress(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, max(0.0, float(self.completed) / float(self.total)))

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    def to_dict(self) -> dict[str, Any]:
        result = _json_safe(self.result)
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "title": self.title,
            "status": self.status.value,
            "completed": self.completed,
            "total": self.total,
            "message": self.message,
            "result": result,
            "result_released": self.result_released,
            "error_type": self.error_type,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_details": _json_safe(self.error_details),
            "resource_keys": list(self.resource_keys),
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }


__all__ = [
    "JobEvent",
    "JobEventType",
    "JobResultSummary",
    "JobRetentionPolicy",
    "JobSnapshot",
    "JobStatus",
]
