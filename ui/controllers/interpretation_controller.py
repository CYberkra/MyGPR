#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interface-annotation edit session controller (QObject, no QWidget)."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from ui.controllers.backend_controller import friendly_error_message, run_command
from ui.desktop_backend_facade import BoreholeRecord, InterpretationZone

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _SessionContext:
    generation: int
    session_id: str | None
    service: Any
    busy_lock: Any
    edit_token: object | None = None


@dataclass(frozen=True, slots=True)
class _Completion:
    context: _SessionContext
    kind: str
    snapshot: Any = None
    message: str = ""


class InterpretationController(QObject):
    """Wraps ``InterpretationEditService`` sessions with Qt signals."""

    log_message = pyqtSignal(str)
    session_opened = pyqtSignal(object)    # InterfaceEditSnapshot
    session_updated = pyqtSignal(object)
    session_failed = pyqtSignal(str)
    saved = pyqtSignal(str)                # message
    saved_with_status = pyqtSignal(str)
    boreholes_loaded = pyqtSignal(list)
    borehole_comparisons_loaded = pyqtSignal(list)
    borehole_failed = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)
    _completed = pyqtSignal(object)
    _worker_busy = pyqtSignal(int, bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._backend_controller = None
        self._session_id: str | None = None
        self._project_id = ""
        self._line_id = ""
        self._generation = 0
        self._state_lock = threading.RLock()
        self._dirty = False
        self._pending_edits: set[object] = set()
        self._busy = False
        # execute 跑在 run_command 每次新建的线程上，"检查 _busy 再置位"
        # 分离会产生 TOCTOU；用非阻塞锁把占用判定与置位合成原子操作。
        self._busy_lock = threading.Lock()
        self._completed.connect(self._apply_completion)
        self._worker_busy.connect(self._apply_busy)

    # ------------------------------------------------------------------
    def set_backend(self, backend_controller) -> None:
        self.close_session()
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

    def _capture_context(self, *, pending_edit: bool = False) -> _SessionContext:
        backend = self._backend()
        with self._state_lock:
            edit_token = object() if pending_edit and self._session_id else None
            if edit_token is not None:
                self._pending_edits.add(edit_token)
            return _SessionContext(
                self._generation, self._session_id,
                None if backend is None else backend.interpretation_edit,
                self._busy_lock,
                edit_token,
            )

    def _is_current_context(self, context: _SessionContext) -> bool:
        with self._state_lock:
            return (context.generation == self._generation
                    and context.session_id == self._session_id)

    def is_current_session(self, snapshot: Any) -> bool:
        """Recheck snapshot ownership when a queued UI consumer receives it."""
        with self._state_lock:
            return bool(self._session_id) and (
                getattr(snapshot, "session_id", None) == self._session_id
                and getattr(snapshot, "project_id", None) == self._project_id
                and getattr(snapshot, "line_id", None) == self._line_id
            )

    def has_unsaved_changes(self) -> bool:
        """Include submitted edits whose result has not reached the Qt thread."""
        with self._state_lock:
            return bool(self._session_id and (self._dirty or self._pending_edits))

    def has_active_session(self) -> bool:
        """A failed edit/save can be retried without discarding a live session."""
        with self._state_lock:
            return bool(self._session_id)

    @pyqtSlot(int, bool)
    def _apply_busy(self, generation: int, value: bool) -> None:
        if generation == self._generation:
            self._set_busy(value)

    @pyqtSlot(object)
    def _apply_completion(self, completion: _Completion) -> None:
        context = completion.context
        if not self._is_current_context(context):
            if completion.kind == "opened":
                run_command(_CloseSessionCommand(self, context.service,
                            completion.snapshot.session_id),
                            name="mygpr-interpretation-close-orphan")
            return
        with self._state_lock:
            self._pending_edits.discard(context.edit_token)
        if completion.kind == "opened":
            snapshot = completion.snapshot
            with self._state_lock:
                self._session_id = snapshot.session_id
                self._dirty = bool(getattr(snapshot, "dirty", False))
            self.log_message.emit(f"标注会话已打开：{self._line_id}")
            if self.is_current_session(snapshot):
                self.session_opened.emit(snapshot)
        elif completion.kind == "updated":
            if self.is_current_session(completion.snapshot):
                with self._state_lock:
                    self._dirty = bool(getattr(completion.snapshot, "dirty", False))
                self.session_updated.emit(completion.snapshot)
        elif completion.kind == "saved":
            with self._state_lock:
                self._dirty = False
            self.log_message.emit("标注已保存")
            if self._is_current_context(context):
                self.saved.emit("标注已保存")
                self.saved_with_status.emit(str(getattr(snapshot, 'status', 'draft') or 'draft'))
        elif completion.kind == "failed":
            self.log_message.emit(completion.message)
            if self._is_current_context(context):
                self.session_failed.emit(completion.message)
        elif completion.kind == "busy":
            self.log_message.emit("操作进行中，请稍后…")

    # ------------------------------------------------------------------
    def open_session(self, project_id: str, line_id: str, input_artifact_id: str = "") -> None:
        self.close_session()
        with self._state_lock:
            self._project_id, self._line_id = str(project_id), str(line_id)
        run_command(
            _OpenSessionCommand(self, project_id, line_id, input_artifact_id),
            name="mygpr-interpretation-open",
        )

    def _run_edit(
        self,
        operation_name: str,
        operation: Callable[[Any, str], Any],
        *,
        name: str = "mygpr-interpretation-edit",
    ) -> None:
        run_command(
            _InterpretationEditCommand(self, operation_name, operation),
            name=name,
        )

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

    def update_properties(self, values: dict) -> None:
        values = dict(values or {})

        def apply(service, session_id):
            if 'uncertainty_samples' in values:
                service.set_uncertainty(
                    session_id, width_samples=float(values['uncertainty_samples']))
            return service.update_properties(
                session_id,
                name=str(values.get('name') or '基覆界面'),
                note=str(values.get('note') or ''),
                interface_type=str(values.get('interface_type') or 'basal_interface'),
                confidence=float(values.get('confidence', 0.8)),
            )

        self._run_edit("更新解释属性", apply)

    def add_zone(self, values: dict) -> None:
        values = dict(values or {})
        self._run_edit(
            "添加语义区段",
            lambda service, sid: service.add_zone(sid, InterpretationZone(**values)),
        )

    def load_boreholes(self) -> None:
        run_command(_BoreholeCommand(self, 'load'), name='mygpr-boreholes-load')

    def save_borehole(self, values: dict) -> None:
        run_command(_BoreholeCommand(self, 'save', values), name='mygpr-borehole-save')

    def compare_boreholes(self, threshold_m: float = 1.0) -> None:
        run_command(
            _BoreholeCommand(self, 'compare', {'threshold_m': float(threshold_m)}),
            name='mygpr-boreholes-compare',
        )

    # ------------------------------------------------------------------
    def save(self, status: str = "draft") -> None:
        run_command(
            _SaveSessionCommand(self, status),
            name="mygpr-interpretation-save",
        )

    def close_session(self) -> None:
        """异步关闭标注会话（fire-and-forget）。

        主窗口 closeEvent / 关闭项目链都不等待结果：先本地失效 session_id，
        实际 close 放到 worker 线程，成败只记日志。
        """
        backend = getattr(self._backend_controller, "backend", None)
        with self._state_lock:
            session_id = self._session_id
            self._generation += 1
            self._session_id = None
            self._dirty = False
            self._pending_edits.clear()
            self._project_id = self._line_id = ""
            # Old work releases its own lock and cannot block a new context.
            self._busy_lock = threading.Lock()
        self._set_busy(False)
        if backend is None or session_id is None:
            return
        run_command(
            _CloseSessionCommand(self, backend.interpretation_edit, session_id),
            name="mygpr-interpretation-close",
        )


# ------------------------------------------------------------------
# Worker commands (replaces run_worker closures)
# ------------------------------------------------------------------

class _SessionCommand:
    """Freeze ownership at submission; marshal completion to the Qt thread."""

    def __init__(self, controller: InterpretationController, kind: str) -> None:
        self._controller = controller
        self._context = controller._capture_context(pending_edit=kind == "updated")
        self._kind = kind

    def _perform(self, service: Any) -> Any:
        raise NotImplementedError

    def execute(self) -> None:
        c, context = self._controller, self._context
        if not c._is_current_context(context):
            return
        if context.service is None:
            c._completed.emit(_Completion(context, "failed", message="后端尚未就绪"))
            return
        if self._kind != "opened" and context.session_id is None:
            c._completed.emit(_Completion(context, "failed", message="请先打开标注会话"))
            return
        if not context.busy_lock.acquire(blocking=False):
            c._completed.emit(_Completion(context, "busy"))
            return
        try:
            if not c._is_current_context(context):
                return
            c._worker_busy.emit(context.generation, True)
            try:
                snapshot = self._perform(context.service)
            except Exception as exc:  # noqa: BLE001 - failures cross the Qt boundary
                completion = _Completion(context, "failed", message=friendly_error_message(exc))
            else:
                completion = _Completion(context, self._kind, snapshot=snapshot)
            c._completed.emit(completion)
        finally:
            context.busy_lock.release()
            c._worker_busy.emit(context.generation, False)


class _OpenSessionCommand(_SessionCommand):

    def __init__(
        self,
        controller: InterpretationController,
        project_id: str,
        line_id: str,
        input_artifact_id: str = "",
    ) -> None:
        super().__init__(controller, "opened")
        self._project_id = project_id
        self._line_id = line_id
        self._input_artifact_id = input_artifact_id

    def _perform(self, service: Any) -> Any:
        return service.open_session(
            str(self._project_id), str(self._line_id),
            input_artifact_id=str(self._input_artifact_id or ""),
        )


class _InterpretationEditCommand(_SessionCommand):

    def __init__(
        self,
        controller: InterpretationController,
        operation_name: str,
        operation: Callable[[Any, str], Any],
    ) -> None:
        super().__init__(controller, "updated")
        self._operation_name = operation_name
        self._operation = operation

    def _perform(self, service: Any) -> Any:
        return self._operation(service, self._context.session_id)


class _SaveSessionCommand(_SessionCommand):

    def __init__(self, controller: InterpretationController, status: str = "draft") -> None:
        super().__init__(controller, "saved")
        self._status = status

    def _perform(self, service: Any) -> Any:
        return service.save_session(self._context.session_id, status=str(self._status or "draft"))


class _BoreholeCommand:
    __slots__ = ("_controller", "_action", "_values", "_project_id", "_line_id")

    def __init__(self, controller, action, values=None):
        self._controller = controller
        self._action = str(action)
        self._values = dict(values or {})
        self._project_id = controller._project_id
        self._line_id = controller._line_id

    def execute(self):
        c = self._controller
        backend = c._backend()
        if backend is None:
            c.borehole_failed.emit("后端尚未就绪")
            return
        service = backend.interpretation
        try:
            if self._action == 'save':
                values = {**self._values, 'line_id': self._line_id}
                service.save_borehole(self._project_id, BoreholeRecord(**values))
            if self._action == 'compare':
                comparisons = service.compare_boreholes(
                    self._project_id,
                    self._line_id,
                    threshold_m=float(self._values.get('threshold_m', 1.0)),
                )
                c.borehole_comparisons_loaded.emit(list(comparisons))
            else:
                c.boreholes_loaded.emit(list(service.list_boreholes(self._project_id)))
        except Exception as exc:  # noqa: BLE001
            c.borehole_failed.emit(friendly_error_message(exc))


class _CloseSessionCommand:
    """worker 线程关闭标注会话；成败经 log_message 回 GUI 线程。"""

    __slots__ = ("_controller", "_service", "_session_id")

    def __init__(self, controller: InterpretationController, service: Any,
                 session_id: str) -> None:
        self._controller = controller
        self._service = service
        self._session_id = session_id

    def execute(self) -> None:
        c = self._controller
        try:
            self._service.close_session(self._session_id)
        except Exception as exc:  # noqa: BLE001
            c.log_message.emit(
                f"关闭标注会话失败：{friendly_error_message(exc)}")
        else:
            c.log_message.emit("标注会话已关闭")


__all__ = ["InterpretationController"]
