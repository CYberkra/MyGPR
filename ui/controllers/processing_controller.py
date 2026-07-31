#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Processing method catalog / pipeline / AutoTune controller."""
from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from core.method_registry_metadata import (
    METHOD_CATEGORY_LABELS,
    METHOD_METADATA,
    METHOD_TAGS,
    PREFERRED_METHOD_ORDER,
)
from core.methods_registry import PROCESSING_METHODS
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from ui.controllers.backend_controller import friendly_error_message, run_worker

_LOGGER = logging.getLogger(__name__)


def _schema_to_list(parameter_schema: Mapping[str, Any]) -> list[dict]:
    params: list[dict] = []
    for key, item in dict(parameter_schema or {}).items():
        entry = dict(item) if isinstance(item, Mapping) else {"name": str(key)}
        entry.setdefault("name", str(key))
        params.append(entry)
    return params


def build_method_dicts(descriptors) -> list[dict]:
    """Merge service descriptors with registry display metadata."""
    order = {method_id: index for index, method_id in enumerate(PREFERRED_METHOD_ORDER)}
    items: list[dict] = []
    for descriptor in descriptors:
        method_id = str(descriptor.method_id)
        meta = METHOD_METADATA.get(method_id, {})
        raw = PROCESSING_METHODS.get(method_id, {})
        category = str(meta.get("category") or descriptor.category or "experimental")
        tag = METHOD_TAGS.get(method_id)
        items.append(
            {
                "method_id": method_id,
                "name": str(descriptor.name),
                "display_name": str(meta.get("display_name") or descriptor.name),
                "category": category,
                "category_label": str(METHOD_CATEGORY_LABELS.get(category, category)),
                "tags": [str(tag)] if tag else [],
                "auto_tune_enabled": bool(descriptor.auto_tune_enabled),
                "parameter_schema": _schema_to_list(descriptor.parameter_schema),
                "description": str(raw.get("description") or ""),
            }
        )
    items.sort(key=lambda item: (order.get(item["method_id"], len(order)), item["method_id"]))
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
        backend = self._backend()
        if backend is None or self._loading:
            return
        self._loading = True

        def runner() -> None:
            try:
                descriptors = backend.processing.list_methods(public_only=True)
                methods = build_method_dicts(descriptors)
            except Exception as exc:  # noqa: BLE001
                self.log_message.emit(f"加载方法库失败：{friendly_error_message(exc)}")
            else:
                self.methods_loaded.emit(methods)
                self.log_message.emit(f"方法库已加载：{len(methods)} 个方法")
            finally:
                self._loading = False

        run_worker(runner, name="mygpr-methods-load")

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
            steps = [
                PipelineStep(
                    method_id=str(step.get("method_id", "")),
                    params=dict(step.get("params") or {}),
                    enabled=bool(step.get("enabled", True)),
                    label=str(step.get("label", "")),
                )
                for step in dict(pipeline_def or {}).get("steps", [])
            ]
            pipeline = PipelineDefinition(
                steps=steps,
                name=str(result_name or dict(pipeline_def or {}).get("name", "")),
            )
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
        self.log_message.emit(f"处理链已提交：{line_id}（{len(steps)} 步）")
        self.run_submitted.emit(job_id)
        self._watch_with_callback(
            job_id,
            f"处理流水线 {line_id}",
            lambda success, message, _result: self.run_finished.emit(success, message),
        )
        return job_id

    # ------------------------------------------------------------------
    def run_autotune(self, project_id, line_id, method_id: str, params_hint: dict) -> None:
        backend = self._backend()
        bridge = self._job_bridge()
        if backend is None or bridge is None:
            return
        method_id = str(method_id)

        def runner() -> None:
            try:
                dataset = backend.projects.read_dataset(str(project_id), str(line_id))
                data = np.asarray(dataset.data, dtype=np.float32)
                kwargs: dict[str, Any] = {}
                if params_hint:
                    kwargs["candidate_params"] = [dict(params_hint)]
                job_id = backend.submit_autotune(data, method_id, **kwargs)
            except Exception as exc:  # noqa: BLE001
                message = friendly_error_message(exc)
                self.log_message.emit(f"自动调参提交失败：{message}")
                self.autotune_failed.emit(method_id, message)
                return
            self.log_message.emit(f"自动调参已提交：{method_id}")

            def _done(success: bool, message: str, result: Any) -> None:
                if success:
                    self.autotune_finished.emit(
                        method_id, dict(result) if isinstance(result, Mapping) else {}
                    )
                    self.log_message.emit(f"自动调参完成：{method_id}")
                else:
                    self.autotune_failed.emit(method_id, message)
                    self.log_message.emit(f"自动调参失败：{message}")

            self._watch_with_callback(job_id, f"自动调参 {method_id}", _done)

        run_worker(runner, name="mygpr-autotune-submit")


__all__ = ["ProcessingController", "build_method_dicts"]
