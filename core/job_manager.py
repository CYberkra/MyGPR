#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable, resource-aware background job contracts.

Execution remains owned by the Qt/application adapter.  This module is pure
Python and provides lifecycle records, cooperative cancellation, persistent
journals, restart recovery, checkpoints, dependencies and project resource
conflict control.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, List, Iterable

from core.schema_registry import DEFAULT_SCHEMA_REGISTRY
from core.storage_primitives import atomic_write_json, utc_now

JOB_JOURNAL_SCHEMA = "mygpr.job_journal.v2"


def _now() -> str:
    return utc_now()


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    BLOCKED = "blocked"


TERMINAL_JOB_STATES = {
    JobState.CANCELLED,
    JobState.SUCCEEDED,
    JobState.FAILED,
    JobState.INTERRUPTED,
}


class JobCancelled(RuntimeError):
    """Raised by a worker when cooperative cancellation is observed."""


class JobResourceConflict(RuntimeError):
    """Raised when a job tries to acquire a conflicting project resource."""


class JobDependencyError(RuntimeError):
    """Raised when a job starts before its dependencies have succeeded."""


@dataclass(frozen=True)
class JobResource:
    key: str
    mode: str = "write"

    def __post_init__(self) -> None:
        if not str(self.key).strip():
            raise ValueError("Job resource key cannot be empty")
        if self.mode not in {"read", "write"}:
            raise ValueError("Job resource mode must be 'read' or 'write'")

    @classmethod
    def parse(cls, value: "JobResource | str | dict[str, Any]") -> "JobResource":
        if isinstance(value, JobResource):
            return value
        if isinstance(value, str):
            if value.startswith("read:"):
                return cls(value[5:], "read")
            if value.startswith("write:"):
                return cls(value[6:], "write")
            return cls(value, "write")
        return cls(str(value.get("key") or ""), str(value.get("mode") or "write"))


@dataclass
class JobProgress:
    completed: int = 0
    total: int = 0
    message: str = "等待执行"
    detail: str = ""

    @property
    def percent(self) -> int:
        if self.total <= 0:
            return 0
        return max(0, min(100, int(round(self.completed * 100.0 / self.total))))

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "JobProgress":
        payload = dict(payload or {})
        return cls(
            completed=int(payload.get("completed") or 0),
            total=int(payload.get("total") or 0),
            message=str(payload.get("message") or "等待执行"),
            detail=str(payload.get("detail") or ""),
        )


@dataclass
class JobRecord:
    job_id: str
    title: str
    category: str = "通用任务"
    state: JobState = JobState.QUEUED
    cancellable: bool = True
    created_at: str = field(default_factory=_now)
    started_at: str = ""
    finished_at: str = ""
    progress: JobProgress = field(default_factory=JobProgress)
    error: str = ""
    result_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostic_id: str = field(default_factory=lambda: f"JOB-{uuid.uuid4().hex[:12].upper()}")
    resources: list[JobResource] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    priority: int = 0
    worker_kind: str = "io_thread"
    checkpoint: dict[str, Any] = field(default_factory=dict)
    attempt: int = 1
    retry_limit: int = 0
    interrupted_reason: str = ""

    @property
    def active(self) -> bool:
        return self.state not in TERMINAL_JOB_STATES

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["progress"]["percent"] = self.progress.percent
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JobRecord":
        state_raw = str(payload.get("state") or JobState.QUEUED.value)
        try:
            state = JobState(state_raw)
        except ValueError:
            state = JobState.FAILED
        return cls(
            job_id=str(payload.get("job_id") or uuid.uuid4().hex),
            title=str(payload.get("title") or "未命名任务"),
            category=str(payload.get("category") or "通用任务"),
            state=state,
            cancellable=bool(payload.get("cancellable", True)),
            created_at=str(payload.get("created_at") or _now()),
            started_at=str(payload.get("started_at") or ""),
            finished_at=str(payload.get("finished_at") or ""),
            progress=JobProgress.from_dict(payload.get("progress")),
            error=str(payload.get("error") or ""),
            result_summary=str(payload.get("result_summary") or ""),
            metadata=dict(payload.get("metadata") or {}),
            diagnostic_id=str(payload.get("diagnostic_id") or f"JOB-{uuid.uuid4().hex[:12].upper()}"),
            resources=[JobResource.parse(item) for item in payload.get("resources", [])],
            dependencies=[str(item) for item in payload.get("dependencies", [])],
            priority=int(payload.get("priority") or 0),
            worker_kind=str(payload.get("worker_kind") or "io_thread"),
            checkpoint=dict(payload.get("checkpoint") or {}),
            attempt=int(payload.get("attempt") or 1),
            retry_limit=int(payload.get("retry_limit") or 0),
            interrupted_reason=str(payload.get("interrupted_reason") or ""),
        )


class JobJournal:
    """Atomic state snapshot plus append-only event stream."""

    def __init__(self, path: str | Path) -> None:
        candidate = Path(path)
        if candidate.suffix.lower() != ".json":
            candidate = candidate / "jobs.json"
        self.path = candidate.resolve()
        self.events_path = self.path.with_name("job_events.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def load(self) -> list[JobRecord]:
        if not self.path.exists():
            return []
        loaded = DEFAULT_SCHEMA_REGISTRY.load_path(
            self.path,
            family="mygpr.job_journal",
            write_migrated=True,
            quarantine_root=self.path.parent / "quarantine",
        )
        return [JobRecord.from_dict(row) for row in loaded.payload.get("jobs", []) if isinstance(row, dict)]

    def save(self, records: Iterable[JobRecord]) -> None:
        with self._lock:
            atomic_write_json(self.path, {
                "schema": JOB_JOURNAL_SCHEMA,
                "updated_at": _now(),
                "process_id": os.getpid(),
                "jobs": [record.to_dict() for record in records],
            })

    def event(self, event_type: str, record: JobRecord, **detail: Any) -> None:
        entry = {
            "schema": "mygpr.job_event.v1",
            "timestamp": _now(),
            "event": str(event_type),
            "job_id": record.job_id,
            "diagnostic_id": record.diagnostic_id,
            "state": record.state.value,
            "detail": detail,
        }
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())


class JobContext:
    """Worker-facing progress, cancellation and checkpoint facade."""

    def __init__(
        self,
        job_id: str,
        cancel_event: threading.Event,
        progress_callback: Callable[[str, JobProgress], None],
        checkpoint_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.job_id = job_id
        self._cancel_event = cancel_event
        self._progress_callback = progress_callback
        self._checkpoint_callback = checkpoint_callback

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.cancel_requested:
            raise JobCancelled("任务已取消")

    def raise_if_cancelled(self) -> None:
        self.check_cancelled()

    def report(self, completed: int, total: int, message: str, detail: str = "") -> None:
        self.check_cancelled()
        self._progress_callback(
            self.job_id,
            JobProgress(
                completed=max(int(completed), 0),
                total=max(int(total), 0),
                message=str(message or "处理中"),
                detail=str(detail or ""),
            ),
        )

    def stage(self, message: str, *, detail: str = "") -> None:
        self.report(0, 0, message, detail)

    def checkpoint(self, **payload: Any) -> None:
        self.check_cancelled()
        if self._checkpoint_callback is not None:
            self._checkpoint_callback(self.job_id, dict(payload))


class JobRegistry:
    """Thread-safe lifecycle registry with optional project-local persistence."""

    def __init__(
        self,
        *,
        history_limit: int = 200,
        journal_path: str | Path | None = None,
    ) -> None:
        self._records: dict[str, JobRecord] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._resource_holders: dict[str, dict[str, str]] = {}
        self._lock = threading.RLock()
        self.history_limit = max(int(history_limit), 20)
        self.journal = JobJournal(journal_path) if journal_path is not None else None
        if self.journal is not None:
            self._restore()

    def bind_journal(self, path: str | Path) -> None:
        with self._lock:
            if any(record.active for record in self._records.values()):
                raise RuntimeError("存在活动任务时不能切换任务日志")
            self.journal = JobJournal(path)
            self._records.clear()
            self._cancel_events.clear()
            self._resource_holders.clear()
            self._restore()

    def _restore(self) -> None:
        assert self.journal is not None
        changed = False
        for record in self.journal.load():
            if record.state in {JobState.RUNNING, JobState.CANCELLING, JobState.QUEUED, JobState.BLOCKED}:
                record.state = JobState.INTERRUPTED
                record.finished_at = _now()
                record.interrupted_reason = "应用上次退出时任务尚未完成"
                record.progress.message = "上次运行中断"
                changed = True
            self._records[record.job_id] = record
            self._cancel_events[record.job_id] = threading.Event()
        if changed:
            self._persist_locked()

    def _persist_locked(self, event_type: str | None = None, record: JobRecord | None = None, **detail: Any) -> None:
        if self.journal is None:
            return
        self.journal.save(self.list())
        if event_type and record is not None:
            self.journal.event(event_type, record, **detail)

    def create(
        self,
        title: str,
        *,
        category: str = "通用任务",
        cancellable: bool = True,
        metadata: dict[str, Any] | None = None,
        resources: Iterable[JobResource | str | dict[str, Any]] | None = None,
        dependencies: Iterable[str] | None = None,
        priority: int = 0,
        worker_kind: str = "io_thread",
        retry_limit: int = 0,
    ) -> JobRecord:
        with self._lock:
            job_id = uuid.uuid4().hex
            record = JobRecord(
                job_id=job_id,
                title=str(title),
                category=str(category),
                cancellable=bool(cancellable),
                metadata=dict(metadata or {}),
                resources=[JobResource.parse(item) for item in resources or ()],
                dependencies=[str(item) for item in dependencies or ()],
                priority=int(priority),
                worker_kind=str(worker_kind),
                retry_limit=max(int(retry_limit), 0),
            )
            self._records[job_id] = record
            self._cancel_events[job_id] = threading.Event()
            self._trim_locked()
            self._persist_locked("created", record)
            return record

    def _trim_locked(self) -> None:
        terminal = [record for record in self._records.values() if not record.active]
        if len(terminal) <= self.history_limit:
            return
        terminal.sort(key=lambda record: (record.finished_at, record.created_at))
        for record in terminal[: len(terminal) - self.history_limit]:
            self._records.pop(record.job_id, None)
            self._cancel_events.pop(record.job_id, None)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            return self._records[job_id]

    def event(self, job_id: str) -> threading.Event:
        with self._lock:
            return self._cancel_events[job_id]

    def list(self, *, active_only: bool = False) -> list[JobRecord]:
        with self._lock:
            records = list(self._records.values())
            if active_only:
                records = [record for record in records if record.active]
            return sorted(records, key=lambda record: (record.priority, record.created_at, record.job_id), reverse=True)

    def snapshot(self) -> List[dict[str, Any]]:
        return [record.to_dict() for record in self.list()]

    def _dependencies_ready(self, record: JobRecord) -> None:
        for dependency_id in record.dependencies:
            dependency = self._records.get(dependency_id)
            if dependency is None or dependency.state != JobState.SUCCEEDED:
                raise JobDependencyError(f"任务依赖尚未成功：{dependency_id}")

    def _claim_resources(self, record: JobRecord) -> None:
        for resource in record.resources:
            holders = self._resource_holders.get(resource.key, {})
            for holder_id, holder_mode in holders.items():
                if holder_id == record.job_id:
                    continue
                if resource.mode == "write" or holder_mode == "write":
                    raise JobResourceConflict(
                        f"资源正在被任务 {holder_id} 占用：{resource.key} ({holder_mode})"
                    )
        for resource in record.resources:
            self._resource_holders.setdefault(resource.key, {})[record.job_id] = resource.mode

    def _release_resources(self, record: JobRecord) -> None:
        for resource in record.resources:
            holders = self._resource_holders.get(resource.key)
            if holders is None:
                continue
            holders.pop(record.job_id, None)
            if not holders:
                self._resource_holders.pop(resource.key, None)

    def mark_running(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._records[job_id]
            if record.state == JobState.RUNNING:
                return record
            self._dependencies_ready(record)
            self._claim_resources(record)
            record.state = JobState.RUNNING
            record.started_at = _now()
            record.progress.message = "正在执行"
            self._persist_locked("started", record)
            return record

    def update_progress(self, job_id: str, progress: JobProgress) -> JobRecord:
        with self._lock:
            record = self._records[job_id]
            if record.active:
                record.progress = progress
                self._persist_locked("progress", record, percent=progress.percent)
            return record

    def save_checkpoint(self, job_id: str, checkpoint: dict[str, Any]) -> JobRecord:
        with self._lock:
            record = self._records[job_id]
            record.checkpoint = dict(checkpoint)
            self._persist_locked("checkpoint", record)
            return record

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            record = self._records.get(job_id)
            if record is None or not record.active or not record.cancellable:
                return False
            self._cancel_events[job_id].set()
            record.state = JobState.CANCELLING
            record.progress.message = "正在取消"
            self._persist_locked("cancel_requested", record)
            return True

    def _mark_terminal(self, record: JobRecord, state: JobState, *, summary: str = "", error: str = "") -> JobRecord:
        record.state = state
        record.finished_at = _now()
        record.result_summary = str(summary or "")
        record.error = str(error or "")
        record.progress.message = {
            JobState.SUCCEEDED: "已完成",
            JobState.CANCELLED: "已取消",
            JobState.FAILED: "执行失败",
            JobState.INTERRUPTED: "执行中断",
        }.get(state, state.value)
        if state == JobState.SUCCEEDED and record.progress.total > 0:
            record.progress.completed = record.progress.total
        self._release_resources(record)
        self._persist_locked(state.value, record)
        return record

    def mark_succeeded(self, job_id: str, *, summary: str = "") -> JobRecord:
        with self._lock:
            return self._mark_terminal(self._records[job_id], JobState.SUCCEEDED, summary=summary)

    def mark_cancelled(self, job_id: str, *, summary: str = "") -> JobRecord:
        with self._lock:
            return self._mark_terminal(self._records[job_id], JobState.CANCELLED, summary=summary)

    def mark_failed(self, job_id: str, error: str) -> JobRecord:
        with self._lock:
            return self._mark_terminal(self._records[job_id], JobState.FAILED, error=error)


__all__ = [
    "JOB_JOURNAL_SCHEMA",
    "JobCancelled",
    "JobContext",
    "JobDependencyError",
    "JobJournal",
    "JobProgress",
    "JobRecord",
    "JobRegistry",
    "JobResource",
    "JobResourceConflict",
    "JobState",
    "TERMINAL_JOB_STATES",
]
