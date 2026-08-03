#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Processing method catalog / pipeline / AutoTune controller."""
from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from ui.desktop_backend_facade import (
    method_catalog,
    pipeline_from_dicts,
    pipeline_to_raw,
)
from ui.controllers.backend_controller import friendly_error_message, run_command

_LOGGER = logging.getLogger(__name__)


def _schema_to_list(parameter_schema: Mapping[str, Any]) -> list[dict]:
    params: list[dict] = []
    for key, item in dict(parameter_schema or {}).items():
        entry = dict(item) if isinstance(item, Mapping) else {"name": str(key)}
        entry.setdefault("name", str(key))
        params.append(entry)
    return params


def build_method_dicts(entries: tuple[UiMethodEntry, ...]) -> list[dict]:
    """Convert facade method catalog entries to the list-of-dicts shape the UI expects."""
    items: list[dict] = []
    for entry in entries:
        items.append(
            {
                "method_id": entry.method_id,
                "name": entry.name,
                "display_name": entry.display_name,
                "category": entry.category,
                "category_label": entry.category_label,
                "tags": list(entry.tags),
                "auto_tune_enabled": entry.auto_tune_enabled,
                "parameter_schema": list(entry.parameter_schema),
                "description": entry.description,
            }
        )
    return items


class ProcessingController(QObject):
    """Method catalog loading, pipeline runs and AutoTune submission."""

    log_message = pyqtSignal(str)
    methods_loaded = pyqtSignal(list)       # list[dict], see build_method_dicts
    run_submitted = pyqtSignal(str)         # job_id
    run_finished = pyqtSignal(bool, str)    # success, message
    autotune_finished = pyqtSignal(str, dict)   # method_id, {best_params, ...}
    autotune_failed = pyqtSignal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._backend_controller = None
        self._loading = False

    # ------------------------------------------------------------------
    def set_backend(self, backend_controller) -> None:
        self._backend_controller = backend_controller

    def _backend(self):
        controller = self._backend_controller
        backend = getattr(controller, "backend", None) if controller is not None else None
        if backend is None:
            self.log_message.emit("后端尚未就绪，请稍后再试")
        return backend

    def _job_bridge(self):
        return getattr(self._backend_controller, "job_bridge", None)

    def _watch_with_callback(
        self,
        job_id: str,
        title: str,
        callback: Callable[[bool, str, Any], None],
    ) -> None:
        bridge = self._job_bridge()
        if bridge is None:
            return
        bridge.watch(job_id, title=title)

        def _on_done(done_id: str, success: bool, message: str, result: Any) -> None:
            if done_id != job_id:
                return
            try:
                bridge.job_completed.disconnect(_on_done)
            except TypeError:
                pass
            callback(success, message, result)

        bridge.job_completed.connect(_on_done)

    # ------------------------------------------------------------------
    def load_methods(self) -> None:
        if self._loading:
            return
        self._loading = True
        try:
            entries = method_catalog()
            methods = build_method_dicts(entries)
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"加载方法库失败：{friendly_error_message(exc)}")
        else:
            self.methods_loaded.emit(methods)
            self.log_message.emit(f"方法库已加载：{len(methods)} 个方法")
        finally:
            self._loading = False

    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        project_id,
        line_id,
        pipeline_def: dict,
        result_name: str,
        input_artifact_id: str = "",
    ) -> str | None:
        backend = self._backend()
        bridge = self._job_bridge()
        if backend is None or bridge is None:
            return None
        try:
            ui_pipeline = pipeline_from_dicts(
                list(dict(pipeline_def or {}).get("steps", [])),
                name=str(result_name or dict(pipeline_def or {}).get("name", "")),
            )
            pipeline = pipeline_to_raw(ui_pipeline)
        except Exception as exc:  # noqa: BLE001
            message = friendly_error_message(exc)
            self.log_message.emit(f"处理链定义无效：{message}")
            self.run_finished.emit(False, message)
            return None
        try:
            job_id = backend.submit_project_pipeline(
                str(project_id),
                str(line_id),
                pipeline,
                result_name=str(result_name or ""),
                input_artifact_id=str(input_artifact_id or ""),
            )
        except Exception as exc:  # noqa: BLE001
            message = friendly_error_message(exc)
            self.log_message.emit(f"处理链提交失败：{message}")
            self.run_finished.emit(False, message)
            return None
        self.log_message.emit(f"处理链已提交：{line_id}（{len(ui_pipeline.steps)} 步）")
        self.run_submitted.emit(job_id)
        self._watch_with_callback(
            job_id,
            f"处理流水线 {line_id}",
            lambda success, message, _result: self.run_finished.emit(success, message),
        )
        return job_id

    # ------------------------------------------------------------------
    def run_autotune(self, project_id, line_id, method_id: str, params_hint: dict) -> None:
        run_command(
            _AutotuneSubmitCommand(self, project_id, line_id, method_id, params_hint),
            name="mygpr-autotune-submit",
        )


# ------------------------------------------------------------------
# Worker commands (replaces run_worker closures)
# ------------------------------------------------------------------

class _AutotuneSubmitCommand:
    __slots__ = (
        "_controller",
        "_project_id",
        "_line_id",
        "_method_id",
        "_params_hint",
    )

    def __init__(
        self,
        controller: ProcessingController,
        project_id: Any,
        line_id: Any,
        method_id: str,
        params_hint: dict,
    ) -> None:
        self._controller = controller
        self._project_id = project_id
        self._line_id = line_id
        self._method_id = str(method_id)
        self._params_hint = params_hint

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        bridge = c._job_bridge()
        if backend is None or bridge is None:
            return
        try:
            dataset = backend.projects.read_dataset(str(self._project_id), str(self._line_id))
            data = np.asarray(dataset.data, dtype=np.float32)
            kwargs: dict[str, Any] = {}
            if self._params_hint:
                kwargs["candidate_params"] = [dict(self._params_hint)]
            job_id = backend.submit_autotune(data, self._method_id, **kwargs)
        except Exception as exc:  # noqa: BLE001
            message = friendly_error_message(exc)
            c.log_message.emit(f"自动调参提交失败：{message}")
            c.autotune_failed.emit(self._method_id, message)
            return
        c.log_message.emit(f"自动调参已提交：{self._method_id}")

        def _done(success: bool, message: str, result: Any) -> None:
            if success:
                c.autotune_finished.emit(
                    self._method_id, dict(result) if isinstance(result, Mapping) else {}
                )
                c.log_message.emit(f"自动调参完成：{self._method_id}")
            else:
                c.autotune_failed.emit(self._method_id, message)
                c.log_message.emit(f"自动调参失败：{message}")

        c._watch_with_callback(job_id, f"自动调参 {self._method_id}", _done)


__all__ = ["ProcessingController", "build_method_dicts"]
