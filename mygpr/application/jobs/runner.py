#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thread-backed job runner with bounded retention and graceful shutdown."""
from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, wait
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from threading import RLock
import logging
import sys
import time
from typing import Any, Callable, Iterable
from uuid import uuid4

import numpy as np

from mygpr.application.jobs.cancellation import CancellationTokenSource, JobCancelledError
from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.jobs.models import (
    JobEvent,
    JobEventType,
    JobResultSummary,
    JobRetentionPolicy,
    JobSnapshot,
    JobStatus,
)
from mygpr.domain.common.errors import MyGPRError, error_info_from_exception

JobOperation = Callable[[ExecutionContext], Any]
JobSubscriber = Callable[[JobEvent], None]
JobFinalizer = Callable[[], None]

_LOGGER = logging.getLogger(__name__)


class JobRunnerClosedError(MyGPRError):
    """Raised when work is submitted after graceful shutdown has started."""

    error_code = "MYGPR_JOB_RUNNER_CLOSED"
    category = "job"
    default_hint = "创建新的后端实例，或等待当前关闭流程完成。"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_object_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Conservative bounded-size estimate used only for retention decisions."""
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, np.memmap):
        return int(value.nbytes)
    seen = seen or set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = int(sys.getsizeof(value, 0))
    if isinstance(value, dict):
        return size + sum(_estimate_object_bytes(k, seen) + _estimate_object_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list, set, frozenset)):
        return size + sum(_estimate_object_bytes(item, seen) for item in value)
    if is_dataclass(value):
        return size + sum(_estimate_object_bytes(getattr(value, item.name), seen) for item in fields(value))
    return size


def _result_summary(value: Any, estimated_bytes: int) -> JobResultSummary:
    shape: tuple[int, ...] = ()
    dtype = ""
    if isinstance(value, np.ndarray):
        shape = tuple(int(item) for item in value.shape)
        dtype = str(value.dtype)
    elif hasattr(value, "data") and isinstance(getattr(value, "data", None), np.ndarray):
        array = getattr(value, "data")
        shape = tuple(int(item) for item in array.shape)
        dtype = str(array.dtype)
    return JobResultSummary(
        result_type=f"{type(value).__module__}.{type(value).__qualname__}",
        estimated_bytes=max(0, int(estimated_bytes)),
        shape=shape,
        dtype=dtype,
    )


@dataclass(slots=True)
class _MutableJob:
    job_id: str
    title: str
    token_source: CancellationTokenSource
    status: JobStatus = JobStatus.QUEUED
    completed: int = 0
    total: int = 0
    message: str = ""
    result: Any = None
    result_released: bool = False
    retained_result_bytes: int = 0
    error_type: str = ""
    error_code: str = ""
    error_message: str = ""
    error_details: dict[str, Any] | None = None
    resource_keys: tuple[str, ...] = ()
    created_at_utc: str = ""
    updated_at_utc: str = ""
    terminal_at_monotonic: float | None = None
    sequence: int = 0
    future: Future[Any] | None = None
    finalizer: JobFinalizer | None = None
    finalizer_called: bool = False


class InMemoryJobRunner:
    """Execute backend jobs without Qt and publish ordered, bounded events."""

    def __init__(
        self,
        max_workers: int = 2,
        *,
        retention_policy: JobRetentionPolicy | None = None,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix="mygpr-job")
        self._event_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mygpr-job-events")
        self._lock = RLock()
        self._jobs: dict[str, _MutableJob] = {}
        self._subscribers: dict[str, list[JobSubscriber]] = {}
        self._events: dict[str, list[JobEvent]] = {}
        self._retention = retention_policy or JobRetentionPolicy()
        self._accepting = True
        self._shutdown = False

    @property
    def accepting(self) -> bool:
        with self._lock:
            return self._accepting and not self._shutdown

    def stop_accepting(self) -> None:
        with self._lock:
            self._accepting = False

    def submit(
        self,
        title: str,
        operation: JobOperation,
        *,
        resource_keys: Iterable[str] = (),
        finalizer: JobFinalizer | None = None,
    ) -> str:
        if not callable(operation):
            raise TypeError("operation must be callable")
        with self._lock:
            if not self._accepting or self._shutdown:
                raise JobRunnerClosedError("job runner is not accepting new work")
        self.prune()
        job_id = uuid4().hex
        now = _now()
        record = _MutableJob(
            job_id=job_id,
            title=str(title or "MyGPR job"),
            token_source=CancellationTokenSource(),
            resource_keys=tuple(sorted({str(item) for item in resource_keys if str(item)})),
            created_at_utc=now,
            updated_at_utc=now,
            finalizer=finalizer,
        )
        with self._lock:
            self._jobs[job_id] = record
            self._subscribers[job_id] = []
            self._events[job_id] = []
        self._emit(job_id, JobEventType.QUEUED, message=record.title)
        try:
            future = self._executor.submit(self._run, job_id, operation)
        except (RuntimeError, TypeError):
            with self._lock:
                self._jobs.pop(job_id, None)
                self._subscribers.pop(job_id, None)
                self._events.pop(job_id, None)
            if finalizer is not None:
                finalizer()
            raise
        with self._lock:
            record.future = future
        future.add_done_callback(lambda completed: self._on_future_done(job_id, completed))
        return job_id

    def subscribe(self, job_id: str, callback: JobSubscriber) -> Callable[[], None]:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._require(job_id)
            self._subscribers[job_id].append(callback)

        def unsubscribe() -> None:
            with self._lock:
                callbacks = self._subscribers.get(str(job_id), [])
                if callback in callbacks:
                    callbacks.remove(callback)

        return unsubscribe

    def active_job_ids(self, *, resource_key: str | None = None) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                job_id
                for job_id, record in self._jobs.items()
                if not record.status.is_terminal
                and (resource_key is None or str(resource_key) in record.resource_keys)
            )

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            record = self._require(job_id)
            if record.status.is_terminal:
                return False
            record.token_source.cancel()
            record.message = "正在取消"
            record.updated_at_utc = _now()
            future = record.future
        if future is not None and future.cancel():
            self._mark_cancelled(job_id, "任务在启动前取消")
        return True

    def cancel_all(self, *, resource_key: str | None = None) -> tuple[str, ...]:
        job_ids = self.active_job_ids(resource_key=resource_key)
        for job_id in job_ids:
            self.cancel(job_id)
        return job_ids

    def snapshot(self, job_id: str) -> JobSnapshot:
        with self._lock:
            return self._snapshot(self._require(job_id))

    def get(self, job_id: str) -> JobSnapshot:
        return self.snapshot(job_id)

    def list_snapshots(self, *, include_terminal: bool = True) -> tuple[JobSnapshot, ...]:
        with self._lock:
            records = tuple(self._jobs.values())
            return tuple(
                self._snapshot(record)
                for record in records
                if include_terminal or not record.status.is_terminal
            )

    def events(self, job_id: str, *, after_sequence: int = 0) -> tuple[JobEvent, ...]:
        threshold = max(0, int(after_sequence))
        with self._lock:
            self._require(job_id)
            return tuple(event for event in self._events.get(str(job_id), ()) if event.sequence > threshold)

    def wait(self, job_id: str, timeout: float | None = None) -> JobSnapshot:
        with self._lock:
            future = self._require(job_id).future
        if future is not None:
            try:
                future.exception(timeout=timeout)
            except (CancelledError, FuturesTimeoutError):
                return self.snapshot(job_id)
            self._on_future_done(job_id, future)
        return self.snapshot(job_id)

    def wait_all(self, timeout: float | None = None) -> tuple[str, ...]:
        with self._lock:
            futures = [record.future for record in self._jobs.values() if record.future is not None and not record.status.is_terminal]
        if not futures:
            return ()
        _done, pending = wait(futures, timeout=timeout)
        return tuple(
            record.job_id
            for record in self._jobs.values()
            if record.future in pending and not record.status.is_terminal
        )

    def release_result(self, job_id: str) -> bool:
        with self._lock:
            record = self._require(job_id)
            if not record.status.is_terminal or record.result is None:
                return False
            if isinstance(record.result, JobResultSummary):
                return False
            estimated = _estimate_object_bytes(record.result)
            record.result = _result_summary(record.result, estimated)
            record.result_released = True
            record.retained_result_bytes = 0
            record.updated_at_utc = _now()
            return True

    def forget(self, job_id: str) -> bool:
        with self._lock:
            record = self._require(job_id)
            if not record.status.is_terminal:
                return False
            self._jobs.pop(str(job_id), None)
            self._subscribers.pop(str(job_id), None)
            self._events.pop(str(job_id), None)
            return True

    def prune(self) -> tuple[str, ...]:
        now = time.monotonic()
        with self._lock:
            terminal = [record for record in self._jobs.values() if record.status.is_terminal]
            terminal.sort(key=lambda item: item.terminal_at_monotonic or now)
            remove: set[str] = set()
            ttl = self._retention.terminal_ttl_seconds
            if ttl > 0:
                remove.update(
                    record.job_id
                    for record in terminal
                    if record.terminal_at_monotonic is not None and now - record.terminal_at_monotonic >= ttl
                )
            survivors = [record for record in terminal if record.job_id not in remove]
            overflow = max(0, len(survivors) - self._retention.max_terminal_jobs)
            remove.update(record.job_id for record in survivors[:overflow])
            for job_id in remove:
                self._jobs.pop(job_id, None)
                self._subscribers.pop(job_id, None)
                self._events.pop(job_id, None)
            self._enforce_result_budget_locked()
            return tuple(sorted(remove))

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_futures: bool = False,
        cancel_running: bool = False,
        timeout: float | None = None,
    ) -> tuple[str, ...]:
        self.stop_accepting()
        if cancel_running:
            self.cancel_all()
        pending = self.wait_all(timeout=timeout) if wait else self.active_job_ids()
        if pending and wait:
            return pending
        with self._lock:
            if self._shutdown:
                return ()
            self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        self._event_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        self.prune()
        return pending

    def _run(self, job_id: str, operation: JobOperation) -> None:
        with self._lock:
            record = self._require(job_id)
            if record.token_source.token.is_cancelled:
                self._mark_cancelled(job_id, "任务在启动前取消")
                return
            record.status = JobStatus.RUNNING
            record.updated_at_utc = _now()
        self._emit(job_id, JobEventType.STARTED, message="任务开始")

        def on_progress(completed: int, total: int, message: str) -> None:
            with self._lock:
                current = self._require(job_id)
                current.completed = max(0, int(completed))
                current.total = max(0, int(total))
                current.message = str(message)
                current.updated_at_utc = _now()
            self._emit(job_id, JobEventType.PROGRESS, message=message, completed=completed, total=total)

        def on_warning(payload: dict[str, Any]) -> None:
            self._emit(job_id, JobEventType.WARNING, message=str(payload.get("message", "")), payload=payload)

        def on_artifact(payload: dict[str, Any]) -> None:
            self._emit(job_id, JobEventType.ARTIFACT, message=str(payload.get("label", payload.get("path", ""))), payload=payload)

        context = ExecutionContext(
            cancellation_token=record.token_source.token,
            progress_callback=on_progress,
            warning_callback=on_warning,
            artifact_callback=on_artifact,
            metadata={"job_id": job_id, "resource_keys": record.resource_keys},
        )
        try:
            context.raise_if_cancelled()
            result = operation(context)
            context.raise_if_cancelled()
        except JobCancelledError:
            self._mark_cancelled(job_id, "任务已取消")
            return
        except RuntimeError as exc:
            # core 侧 24 处协作取消抛 core.job_manager.JobCancelled（同为
            # RuntimeError；application 层受架构政策限制不能 import core），
            # 按 duck-type 识别为取消，避免 _on_future_done 把 CANCELLED
            # 标成 FAILED 误导用户。1.1.0 取消异常统一后移除。
            if type(exc).__name__ == "JobCancelled" and type(exc).__module__ == "core.job_manager":
                self._mark_cancelled(job_id, str(exc) or "任务已取消")
                return
            raise

        estimated = _estimate_object_bytes(result)
        released = estimated > self._retention.max_result_bytes
        retained = _result_summary(result, estimated) if released else result
        with self._lock:
            current = self._require(job_id)
            current.status = JobStatus.COMPLETED
            current.result = retained
            current.result_released = released
            current.retained_result_bytes = 0 if released else estimated
            current.completed = max(current.completed, current.total or 10_000)
            current.total = max(current.total, 10_000)
            current.message = "任务完成"
            current.updated_at_utc = _now()
            current.terminal_at_monotonic = time.monotonic()
            self._enforce_result_budget_locked()
        self._emit(job_id, JobEventType.COMPLETED, message="任务完成", completed=10_000, total=10_000)
        with self._lock:
            self._subscribers[job_id] = []

    def _mark_cancelled(self, job_id: str, message: str) -> None:
        with self._lock:
            current = self._require(job_id)
            if current.status.is_terminal:
                return
            current.status = JobStatus.CANCELLED
            current.message = str(message)
            current.updated_at_utc = _now()
            current.terminal_at_monotonic = time.monotonic()
        self._emit(job_id, JobEventType.CANCELLED, message=message)
        with self._lock:
            self._subscribers[job_id] = []

    def _emit(
        self,
        job_id: str,
        event_type: JobEventType,
        *,
        message: str = "",
        completed: int = 0,
        total: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            record = self._require(job_id)
            record.sequence += 1
            event = JobEvent.create(
                job_id=job_id,
                event_type=event_type,
                sequence=record.sequence,
                message=message,
                completed=completed,
                total=total,
                payload=self._bounded_event_payload(payload),
            )
            retained = self._events.setdefault(job_id, [])
            retained.append(event)
            overflow = len(retained) - self._retention.max_events_per_job
            if overflow > 0:
                del retained[:overflow]
            callbacks = tuple(self._subscribers.get(job_id, ()))
        for callback in callbacks:
            future = self._event_executor.submit(callback, event)
            future.add_done_callback(lambda completed, current_job_id=job_id: self._log_subscriber_result(current_job_id, completed))

    def _bounded_event_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        value = dict(payload or {})
        estimated = _estimate_object_bytes(value)
        limit = self._retention.max_event_payload_bytes
        if limit <= 0 or estimated <= limit:
            return value
        return {
            "payload_released": True,
            "estimated_bytes": estimated,
            "keys": sorted(str(key) for key in value),
        }

    def _enforce_result_budget_locked(self) -> None:
        budget = self._retention.max_total_result_bytes
        retained = [
            record
            for record in self._jobs.values()
            if record.status.is_terminal
            and record.retained_result_bytes > 0
            and record.result is not None
            and not isinstance(record.result, JobResultSummary)
        ]
        total = sum(record.retained_result_bytes for record in retained)
        if total <= budget:
            return
        retained.sort(key=lambda item: item.terminal_at_monotonic or time.monotonic())
        for record in retained:
            if total <= budget:
                break
            estimated = record.retained_result_bytes
            record.result = _result_summary(record.result, estimated)
            record.result_released = True
            record.retained_result_bytes = 0
            record.updated_at_utc = _now()
            total -= estimated

    def _on_future_done(self, job_id: str, future: Future[Any]) -> None:
        try:
            if future.cancelled():
                self._mark_cancelled(job_id, "任务在启动前取消")
            else:
                error = future.exception()
                if error is not None:
                    info = error_info_from_exception(error, category="job", context={"job_id": job_id})
                    marked_failed = False
                    with self._lock:
                        current = self._require(job_id)
                        if not current.status.is_terminal:
                            current.status = JobStatus.FAILED
                            current.error_type = info.error_type
                            current.error_code = info.error_code
                            current.error_message = info.user_message
                            current.error_details = info.to_dict()
                            current.message = info.user_message
                            current.updated_at_utc = _now()
                            current.terminal_at_monotonic = time.monotonic()
                            marked_failed = True
                    if marked_failed:
                        self._emit(job_id, JobEventType.FAILED, message=info.user_message, payload=info.to_dict())
                        with self._lock:
                            self._subscribers[job_id] = []
        finally:
            self._call_finalizer(job_id)

    def _call_finalizer(self, job_id: str) -> None:
        finalizer: JobFinalizer | None = None
        with self._lock:
            record = self._jobs.get(str(job_id))
            if record is not None and not record.finalizer_called:
                record.finalizer_called = True
                finalizer = record.finalizer
                record.finalizer = None
        if finalizer is not None:
            try:
                finalizer()
            except (RuntimeError, OSError, ValueError):
                _LOGGER.exception("job finalizer failed", extra={"job_id": job_id})

    @staticmethod
    def _log_subscriber_result(job_id: str, future: Future[Any]) -> None:
        error = future.exception()
        if error is not None:
            _LOGGER.error("job subscriber failed", exc_info=(type(error), error, error.__traceback__), extra={"job_id": job_id})

    def _require(self, job_id: str) -> _MutableJob:
        try:
            return self._jobs[str(job_id)]
        except KeyError as exc:
            raise KeyError(f"unknown job: {job_id}") from exc

    @staticmethod
    def _snapshot(record: _MutableJob) -> JobSnapshot:
        return JobSnapshot(
            job_id=record.job_id,
            title=record.title,
            status=record.status,
            completed=record.completed,
            total=record.total,
            message=record.message,
            result=record.result,
            result_released=record.result_released,
            error_type=record.error_type,
            error_code=record.error_code,
            error_message=record.error_message,
            error_details=dict(record.error_details or {}),
            resource_keys=record.resource_keys,
            created_at_utc=record.created_at_utc,
            updated_at_utc=record.updated_at_utc,
        )


__all__ = ["InMemoryJobRunner", "JobRunnerClosedError"]
