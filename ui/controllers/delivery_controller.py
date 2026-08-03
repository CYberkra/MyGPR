#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial results / report / backup delivery controller."""
from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from ui.controllers.backend_controller import friendly_error_message, run_command

_LOGGER = logging.getLogger(__name__)


class DeliveryController(QObject):
    """Delivery-side jobs: spatial synthesis, reports, backup/restore."""

    log_message = pyqtSignal(str)
    spatial_results_updated = pyqtSignal(list)
    report_generated = pyqtSignal(object)     # ReportPackage

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._backend_controller = None

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

    def _submit_and_watch(self, title: str, submit, on_done=None) -> str | None:
        backend = self._backend()
        bridge = self._job_bridge()
        if backend is None or bridge is None:
            return None
        try:
            job_id = submit(backend)
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"{title}提交失败：{friendly_error_message(exc)}")
            return None
        self.log_message.emit(f"{title}已提交")
        bridge.watch(job_id, title=title)
        if on_done is not None:

            def _on_completed(done_id: str, success: bool, message: str, result: Any) -> None:
                if done_id != job_id:
                    return
                try:
                    bridge.job_completed.disconnect(_on_completed)
                except TypeError:
                    pass
                on_done(success, message, result)

            bridge.job_completed.connect(_on_completed)
        return job_id

    # ------------------------------------------------------------------
    def refresh_spatial(self, project_id: str) -> None:
        run_command(
            _SpatialRefreshCommand(self, project_id),
            name="mygpr-spatial-refresh",
        )

    def create_spatial(self, project_id: str, name: str, line_ids: list[str]) -> str | None:
        selected = [str(item) for item in (line_ids or [])]
        return self._submit_and_watch(
            f"生成空间成果 {name}",
            lambda backend: backend.submit_spatial_result(
                str(project_id),
                name=str(name),
                line_ids=selected or None,
            ),
        )

    def generate_report(self, project_id: str, package_name: str = "") -> str | None:
        def _on_done(success: bool, message: str, result: Any) -> None:
            if success and result is not None:
                self.report_generated.emit(result)
                self.log_message.emit("报告包已生成")
            elif not success:
                self.log_message.emit(f"报告生成失败：{message}")

        return self._submit_and_watch(
            "生成项目报告",
            lambda backend: backend.submit_project_report(
                str(project_id),
                package_name=str(package_name) or None,
            ),
            on_done=_on_done,
        )

    def backup_project(self, project_id: str, dest_dir: str) -> str | None:
        return self._submit_and_watch(
            "项目备份",
            lambda backend: backend.submit_project_backup(
                str(project_id),
                destination_dir=str(dest_dir),
            ),
        )

    def restore_project(self, archive_path: str, dest_root: str) -> str | None:
        return self._submit_and_watch(
            "恢复项目备份",
            lambda backend: backend.submit_project_restore(
                str(archive_path),
                str(dest_root),
            ),
        )


# ------------------------------------------------------------------
# Worker commands (replaces run_worker closures)
# ------------------------------------------------------------------

class _SpatialRefreshCommand:
    __slots__ = ("_controller", "_project_id")

    def __init__(self, controller: DeliveryController, project_id: str) -> None:
        self._controller = controller
        self._project_id = project_id

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            return
        try:
            results = list(backend.spatial.list_results(str(self._project_id)))
        except Exception as exc:  # noqa: BLE001
            c.log_message.emit(f"刷新空间成果失败：{friendly_error_message(exc)}")
        else:
            c.spatial_results_updated.emit(results)


__all__ = ["DeliveryController"]
