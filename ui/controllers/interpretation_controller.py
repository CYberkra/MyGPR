#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interface-annotation edit session controller (QObject, no QWidget)."""
from __future__ import annotations

import logging
from typing import Any, Callable

from PyQt6.QtCore import QObject, pyqtSignal

from ui.controllers.backend_controller import friendly_error_message, run_worker

_LOGGER = logging.getLogger(__name__)


class InterpretationController(QObject):
    """Wraps ``InterpretationEditService`` sessions with Qt signals."""

    log_message = pyqtSignal(str)
    session_opened = pyqtSignal(object)    # InterfaceEditSnapshot
    session_updated = pyqtSignal(object)
    session_failed = pyqtSignal(str)
    saved = pyqtSignal(str)                # message
    busy_changed = pyqtSignal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._backend_controller = None
        self._session_id: str | None = None
        self._busy = False

    # ------------------------------------------------------------------
    def set_backend(self, backend_controller) -> None:
        self._backend_controller = backend_controller

    def _backend(self):
        controller = self._backend_controller
        backend = getattr(controller, "backend", None) if controller is not None else None
        if backend is None:
            self.log_message.emit("后端尚未就绪，请稍后再试")
        return backend

    def _set_busy(self, value: bool) -> None:
        value = bool(value)
        if value != self._busy:
            self._busy = value
            self.busy_changed.emit(value)

    # ------------------------------------------------------------------
    def open_session(self, project_id: str, line_id: str) -> None:
        backend = self._backend()
        if backend is None:
            self.session_failed.emit("后端尚未就绪")
            return
        if self._busy:
            self.log_message.emit("操作进行中，请稍后…")
            return
        self._set_busy(True)

        def runner() -> None:
            try:
                snapshot = backend.interpretation_edit.open_session(
                    str(project_id), str(line_id)
                )
            except Exception as exc:  # noqa: BLE001
                message = friendly_error_message(exc)
                self.log_message.emit(f"打开标注会话失败：{message}")
                self.session_failed.emit(message)
            else:
                self._session_id = snapshot.session_id
                self.log_message.emit(f"标注会话已打开：{line_id}")
                self.session_opened.emit(snapshot)
            finally:
                self._set_busy(False)

        run_worker(runner, name="mygpr-interpretation-open")

    def _run_edit(
        self,
        operation_name: str,
        operation: Callable[[Any, str], Any],
        *,
        name: str = "mygpr-interpretation-edit",
    ) -> None:
        backend = self._backend()
        session_id = self._session_id
        if backend is None:
            return
        if session_id is None:
            self.session_failed.emit("请先打开标注会话")
            return
        if self._busy:
            self.log_message.emit("操作进行中，请稍后…")
            return
        self._set_busy(True)

        def runner() -> None:
            try:
                snapshot = operation(backend.interpretation_edit, session_id)
            except Exception as exc:  # noqa: BLE001
                message = friendly_error_message(exc)
                self.log_message.emit(f"{operation_name}失败：{message}")
                self.session_failed.emit(message)
            else:
                self.session_updated.emit(snapshot)
            finally:
                self._set_busy(False)

        run_worker(runner, name=name)

    # ------------------------------------------------------------------
    def replace_points(self, points: list[tuple[int, int]]) -> None:
        normalized = [(float(trace), float(sample)) for trace, sample in (points or [])]
        self._run_edit(
            "更新标注点",
            lambda service, sid: service.replace_points(sid, normalized),
        )

    def auto_trace(self) -> None:
        self._run_edit("自动追踪", lambda service, sid: service.auto_trace(sid))

    def snap(self, radius_samples: int = 8) -> None:
        self._run_edit(
            "吸附",
            lambda service, sid: service.snap_to_signal(sid, radius_samples=int(radius_samples)),
        )

    def smooth(self, radius: int = 2) -> None:
        self._run_edit(
            "平滑",
            lambda service, sid: service.smooth(sid, radius=int(radius)),
        )

    def undo(self) -> None:
        self._run_edit("撤销", lambda service, sid: service.undo(sid))

    def redo(self) -> None:
        self._run_edit("重做", lambda service, sid: service.redo(sid))

    # ------------------------------------------------------------------
    def save(self, status: str = "draft") -> None:
        backend = self._backend()
        session_id = self._session_id
        if backend is None:
            return
        if session_id is None:
            self.session_failed.emit("请先打开标注会话")
            return
        if self._busy:
            self.log_message.emit("操作进行中，请稍后…")
            return
        self._set_busy(True)

        def runner() -> None:
            try:
                backend.interpretation_edit.save_session(session_id, status=str(status or "draft"))
            except Exception as exc:  # noqa: BLE001
                message = friendly_error_message(exc)
                self.log_message.emit(f"保存标注失败：{message}")
                self.session_failed.emit(message)
            else:
                self.log_message.emit("标注已保存")
                self.saved.emit("标注已保存")
            finally:
                self._set_busy(False)

        run_worker(runner, name="mygpr-interpretation-save")

    def close_session(self) -> None:
        backend = self._backend()
        session_id = self._session_id
        self._session_id = None
        if backend is None or session_id is None:
            return
        try:
            backend.interpretation_edit.close_session(session_id)
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"关闭标注会话失败：{friendly_error_message(exc)}")
        else:
            self.log_message.emit("标注会话已关闭")


__all__ = ["InterpretationController"]
