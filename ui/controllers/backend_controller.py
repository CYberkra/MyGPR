#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend lifecycle controller and job-event bridge.

``BackendController`` owns the one-time creation of ``MyGPRBackend`` on a
``QThread``; ``JobBridge`` converts the backend's worker-thread job events
into Qt signals that are safe to consume from the GUI thread.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Protocol

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from ui.desktop_backend_facade import JobEventType, JobResultSummary, UiJobSnapshot, job_snapshot_from_raw
from mygpr.interfaces.backend import MyGPRBackend  # noqa: F401 — type-check only

_LOGGER = logging.getLogger(__name__)

PROJECT_BUSY_MESSAGE = "项目正被其他任务占用，请稍后重试"


class WorkerCommand(Protocol):
    def execute(self) -> None: ...


def run_command(command: WorkerCommand, *, name: str = "mygpr-ui-worker") -> threading.Thread:
    """Start ``command.execute()`` on a daemon worker thread and return the thread."""
    thread = threading.Thread(target=command.execute, name=name, daemon=True)
    thread.start()
    return thread


_STATUS_BY_EVENT = {
    JobEventType.QUEUED: "queued",
    JobEventType.STARTED: "running",
    JobEventType.PROGRESS: "running",
    JobEventType.COMPLETED: "completed",
    JobEventType.FAILED: "failed",
    JobEventType.CANCELLED: "cancelled",
}
_TERMINAL_EVENTS = {JobEventType.COMPLETED, JobEventType.FAILED, JobEventType.CANCELLED}
_POLL_INTERVAL_S = 0.2


def friendly_error_message(exc: BaseException | None) -> str:
    """Map backend errors to user-facing Chinese messages (error_code first).

    MyGPRError 子类（含 error_code 属性）显示 ``[错误码] 消息 — 建议``，
    让用户可操作地排障而非纯 str(exc)。无 error_code 的异常保持原行为。
    """
    if exc is None:
        return "未知错误"
    # P1：MyGPRError 子类含 error_code，按结构输出建议（不 import domain 层）
    exc_error_code = str(getattr(exc, "error_code", "") or "")
    if exc_error_code:
        if exc_error_code == "MYGPR_PROJECT_BUSY":
            return PROJECT_BUSY_MESSAGE
        hint = str(getattr(exc, "hint", "") or getattr(exc, "default_hint", "") or "")
        text = f"[{exc_error_code}] {str(exc).strip()}"
        if hint:
            text += f" — {hint}"
        return text
    # 无 error_code 的通用异常（绝大多数 Value/Type/KeyError），保持原行为
    message = str(exc).strip()
    return message or type(exc).__name__


def snapshot_error_message(snapshot: UiJobSnapshot) -> str:
    """Build the user-facing message for a failed/cancelled job snapshot."""
    if snapshot.error_code == "MYGPR_PROJECT_BUSY":
        return PROJECT_BUSY_MESSAGE
    return snapshot.error_message or snapshot.message or "任务失败"


def extract_small_result(result: Any) -> Any:
    """Pass small results through to the GUI; drop large array payloads."""
    if result is None:
        return None
    if isinstance(result, (np.ndarray, JobResultSummary)):
        return None
    data = getattr(result, "data", None)
    if isinstance(data, np.ndarray):
        return None
    return result


def run_worker(target: Callable[[], None], *, name: str = "mygpr-ui-worker") -> threading.Thread:
    """Start ``target`` on a daemon worker thread and return the thread."""
    thread = threading.Thread(target=target, name=name, daemon=True)
    thread.start()
    return thread


class JobBridge(QObject):
    """Poll ``backend.jobs`` on watcher threads and re-emit as Qt signals."""

    progress_changed = pyqtSignal(str, int, int, str)   # job_id, completed, total, message
    status_changed = pyqtSignal(str, str)               # job_id, status.value
    job_completed = pyqtSignal(str, bool, str, object)  # job_id, success, message, result_or_none

    def __init__(self, backend: MyGPRBackend, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self._lock = threading.Lock()
        self._titles: dict[str, str] = {}
        self._stops: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._shutdown = threading.Event()

    def watch(self, job_id: str, title: str = "") -> None:
        """Watch ``job_id`` until a terminal state, emitting Qt signals."""
        job_id = str(job_id)
        with self._lock:
            if self._shutdown.is_set():
                return
            self._titles[job_id] = str(title or "")
            stop = threading.Event()
            self._stops[job_id] = stop
        try:
            snapshot = self._backend.jobs.snapshot(job_id)
            self.status_changed.emit(job_id, snapshot.status.value)
        except KeyError:
            self.status_changed.emit(job_id, "queued")
        thread = threading.Thread(
            target=self._watch_loop,
            args=(job_id, stop),
            name=f"mygpr-job-watch-{job_id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._threads[job_id] = thread
        thread.start()

    def cancel(self, job_id: str) -> None:
        """Cooperatively cancel a watched job (watcher exits at terminal)."""
        try:
            self._backend.jobs.cancel(str(job_id))
        except (KeyError, RuntimeError) as exc:
            _LOGGER.warning("job cancel failed for %s: %s", job_id, exc)

    def titles(self) -> dict[str, str]:
        with self._lock:
            return dict(self._titles)

    def shutdown(self) -> None:
        """Stop every watcher thread; safe to call multiple times."""
        self._shutdown.set()
        with self._lock:
            stops = list(self._stops.values())
            threads = list(self._threads.values())
        for stop in stops:
            stop.set()
        for thread in threads:
            thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    def _watch_loop(self, job_id: str, stop: threading.Event) -> None:
        sequence = 0
        last_status = ""
        terminal_seen = False
        try:
            while not stop.is_set() and not self._shutdown.is_set():
                try:
                    events = self._backend.jobs.events(job_id, after_sequence=sequence)
                except KeyError:
                    return
                for event in events:
                    sequence = max(sequence, int(event.sequence))
                    if event.event_type is JobEventType.PROGRESS:
                        self.progress_changed.emit(
                            job_id, int(event.completed), int(event.total), str(event.message)
                        )
                    status = _STATUS_BY_EVENT.get(event.event_type, "")
                    if status and status != last_status:
                        last_status = status
                        self.status_changed.emit(job_id, status)
                    if event.event_type in _TERMINAL_EVENTS:
                        terminal_seen = True
                if terminal_seen:
                    break
                stop.wait(_POLL_INTERVAL_S)
        except Exception:  # pragma: no cover - defensive: never kill a watcher silently
            _LOGGER.exception("job watcher crashed for %s", job_id)
        finally:
            self._finalize(job_id)

    def _finalize(self, job_id: str) -> None:
        success = False
        message = ""
        result: Any = None
        try:
            raw_snapshot = self._backend.jobs.snapshot(job_id)
        except KeyError:
            message = "任务状态不可用"
        else:
            snapshot = job_snapshot_from_raw(raw_snapshot)
            status = snapshot.status
            success = status == "completed"
            if success:
                message = snapshot.message or "任务完成"
                result = extract_small_result(raw_snapshot.result)
            elif status == "cancelled":
                message = snapshot.message or "任务已取消"
            else:
                message = snapshot_error_message(snapshot)
        try:
            self._backend.jobs.forget(job_id)
        except (KeyError, RuntimeError) as exc:
            _LOGGER.debug("job forget failed for %s: %s", job_id, exc)
        with self._lock:
            self._titles.pop(job_id, None)
            self._stops.pop(job_id, None)
            self._threads.pop(job_id, None)
        self.job_completed.emit(job_id, success, message, result)


class _BackendInitWorker(QObject):
    """Runs ``MyGPRBackend.create_default`` inside a dedicated QThread."""

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, max_workers: int = 2) -> None:
        super().__init__()
        self._max_workers = int(max(1, max_workers))

    def run(self) -> None:
        try:
            backend = MyGPRBackend.create_default(max_workers=self._max_workers)
        except Exception as exc:  # noqa: BLE001 - surface any init failure to the GUI
            _LOGGER.exception("backend initialisation failed")
            self.failed.emit(friendly_error_message(exc))
        else:
            self.succeeded.emit(backend)


class BackendController(QObject):
    """Owns backend creation, readiness and graceful shutdown."""

    log_message = pyqtSignal(str)
    backend_ready = pyqtSignal()
    backend_failed = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.backend: MyGPRBackend | None = None
        self.job_bridge: JobBridge | None = None
        self._thread: QThread | None = None
        self._worker: _BackendInitWorker | None = None

    def start(self, *, max_workers: int = 2) -> None:
        """Create the backend on a QThread; emits backend_ready/failed."""
        if self.backend is not None or self._thread is not None:
            return
        self.log_message.emit("后端初始化中…")
        thread = QThread(self)
        worker = _BackendInitWorker(max_workers=int(max_workers))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._on_backend_created)
        worker.failed.connect(self._on_backend_failed)
        worker.succeeded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    @property
    def is_ready(self) -> bool:
        return self.backend is not None and self.job_bridge is not None

    def shutdown(self) -> None:
        """Shut the backend down; exceptions are swallowed and logged."""
        if self.job_bridge is not None:
            try:
                self.job_bridge.shutdown()
            except Exception:  # noqa: BLE001
                _LOGGER.exception("job bridge shutdown failed")
        backend = self.backend
        if backend is None:
            return
        try:
            backend.shutdown(wait=True)
            self.log_message.emit("后端已关闭")
        except Exception as exc:  # noqa: BLE001 - shutdown must never crash the app
            _LOGGER.exception("backend shutdown failed")
            self.log_message.emit(f"后端关闭异常：{friendly_error_message(exc)}")

    # ------------------------------------------------------------------
    def _on_backend_created(self, backend: MyGPRBackend) -> None:
        self.backend = backend
        self.job_bridge = JobBridge(backend, parent=self)
        self.log_message.emit("后端就绪")
        self.backend_ready.emit()

    def _on_backend_failed(self, message: str) -> None:
        self.log_message.emit(f"后端初始化失败：{message}")
        self.backend_failed.emit(message)

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None


__all__ = [
    "BackendController",
    "JobBridge",
    "PROJECT_BUSY_MESSAGE",
    "WorkerCommand",
    "extract_small_result",
    "friendly_error_message",
    "run_command",
    "run_worker",
    "snapshot_error_message",
]
