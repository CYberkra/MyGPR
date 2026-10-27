#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial results / report / backup delivery controller."""
from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from ui.controllers.backend_controller import EXPECTED_BACKEND_ERRORS, friendly_error_message, run_command

_LOGGER = logging.getLogger(__name__)


class DeliveryController(QObject):
    """Delivery-side jobs: spatial synthesis, reports, backup/restore."""

    log_message = pyqtSignal(str)
    spatial_results_updated = pyqtSignal(list)
    report_generated = pyqtSignal(object)     # ReportPackage
    processing_package_generated = pyqtSignal(object)
    report_list_updated = pyqtSignal(list)    # [ReportPackage, ...] 全量列表
    _project_read_ready = pyqtSignal(object, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._backend_controller = None
        self._project_id = ""
        self._project_context_generation = 0
        self._project_read_ready.connect(self._deliver_project_read)

    # ------------------------------------------------------------------
    def set_backend(self, backend_controller) -> None:
        self.set_project_context("")
        self._backend_controller = backend_controller

    def set_project_context(self, project_id: str) -> None:
        """Invalidate reads on every open/close, including reopening the same ID."""
        self._project_context_generation += 1
        self._project_id = str(project_id or "")

    @pyqtSlot(object, object)
    def _deliver_project_read(self, command, payload) -> None:
        # A worker can finish before a switch but remain queued until afterwards.
        if command._context_is_current():
            signal_name, args = payload
            getattr(self, signal_name).emit(*args)

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
        except EXPECTED_BACKEND_ERRORS as exc:
            self.log_message.emit(f"{title}提交失败：{friendly_error_message(exc)}")
            return None
        self.log_message.emit(f"{title}已提交")
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
        bridge.watch(job_id, title=title)
        return job_id

    # ------------------------------------------------------------------
    def refresh_spatial(self, project_id: str) -> None:
        run_command(
            _SpatialRefreshCommand(self, project_id),
            name="mygpr-spatial-refresh",
        )

    def refresh_reports(self, project_id: str) -> None:
        """项目报告包全量列表（文件树「项目报告」组 + 成果页共用数据源）。"""
        run_command(
            _ReportsRefreshCommand(self, project_id),
            name="mygpr-reports-refresh",
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

    def export_spatial(self, project_id: str, options: dict) -> str | None:
        options = dict(options or {})
        result_id = str(options.get('result_id') or '')
        destination = str(options.get('destination') or '')
        if not result_id or not destination:
            return None
        return self._submit_and_watch(
            f"导出空间成果 {result_id}",
            lambda backend: backend.submit_spatial_export(
                str(project_id), result_id, destination,
                format_name=str(options.get('format_name') or 'zip'),
            ),
        )

    def delete_spatial(self, project_id: str, result_id: str) -> str | None:
        def _on_done(success, message, _result):
            if success:
                self.refresh_spatial(project_id)
            else:
                self.log_message.emit(f"移除空间成果失败：{message}")
        return self._submit_and_watch(
            f"移除空间成果 {result_id}",
            lambda backend: backend.submit_spatial_delete(str(project_id), str(result_id)),
            on_done=_on_done,
        )

    def set_spatial_current(self, project_id: str, result_id: str) -> None:
        run_command(
            _SpatialCurrentCommand(self, str(project_id), str(result_id)),
            name='mygpr-spatial-current',
        )

    def generate_report(self, project_id: str, package_name: str = "", report_profile: dict | None = None) -> str | None:
        context = _DeliveryReadCommand(self, str(project_id))

        def _on_done(success: bool, message: str, result: Any) -> None:
            if not context._context_is_current():
                return
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
                report_profile=dict(report_profile or {}),
            ),
            on_done=_on_done,
        )

    def export_processing_package(self, project_id: str, line_id: str, destination: str, *,
                                  artifact_id: str = '') -> str | None:
        context = _DeliveryReadCommand(self, str(project_id))

        def _on_done(success: bool, message: str, result: Any) -> None:
            if not context._context_is_current():
                return
            if success and result is not None:
                self.processing_package_generated.emit(result)
                self.log_message.emit(f'处理结果包已保存：{result.archive_path}')
            elif not success:
                self.log_message.emit(f'处理结果打包未完成：{message}')

        return self._submit_and_watch(
            f'打包处理结果 {line_id}',
            lambda backend: backend.submit_processing_result_package(
                str(project_id), str(line_id), str(destination), artifact_id=str(artifact_id)),
            on_done=_on_done)

    def backup_project(
        self,
        project_id: str,
        dest_dir: str,
        *,
        incremental: bool = False,
        retention_keep: int | None = None,
    ) -> str | None:
        return self._submit_and_watch(
            "项目备份",
            lambda backend: backend.submit_project_backup(
                str(project_id),
                destination_dir=str(dest_dir),
                incremental=bool(incremental),
                retention_keep=retention_keep,
            ),
        )

    def restore_project(self, archive_path: str, dest_root: str, *,
                        project_dir_name: str = "", open_after_restore: bool = True) -> str | None:
        return self._submit_and_watch(
            "恢复项目备份为新项目",
            lambda backend: backend.submit_project_restore(
                str(archive_path),
                str(dest_root),
                project_dir_name=str(project_dir_name or "") or None,
                open_after_restore=bool(open_after_restore),
            ),
        )


# ------------------------------------------------------------------
# Worker commands (replaces run_worker closures)
# ------------------------------------------------------------------

class _DeliveryReadCommand:
    __slots__ = ("_controller", "_project_id", "_context_generation")

    def __init__(self, controller: DeliveryController, project_id: str) -> None:
        self._controller = controller
        self._project_id = str(project_id)
        self._context_generation = controller._project_context_generation

    def _context_is_current(self) -> bool:
        c = self._controller
        return (c._project_id == self._project_id
                and c._project_context_generation == self._context_generation)

    def _publish(self, signal_name: str, *args) -> None:
        if self._context_is_current():
            self._controller._project_read_ready.emit(self, (signal_name, args))


class _SpatialRefreshCommand(_DeliveryReadCommand):
    __slots__ = ()

    def execute(self) -> None:
        c = self._controller
        if not self._context_is_current():
            return
        backend = c._backend()
        if backend is None:
            return
        try:
            results = list(backend.spatial.list_results(str(self._project_id)))
        except EXPECTED_BACKEND_ERRORS as exc:
            self._publish("log_message", f"刷新空间成果失败：{friendly_error_message(exc)}")
        else:
            self._publish("spatial_results_updated", results)


class _SpatialCurrentCommand(_SpatialRefreshCommand):
    __slots__ = ("_result_id",)

    def __init__(self, controller, project_id, result_id):
        super().__init__(controller, project_id)
        self._result_id = str(result_id)

    def execute(self):
        if not self._context_is_current():
            return
        backend = self._controller._backend()
        if backend is None:
            return
        try:
            backend.spatial.set_current(self._project_id, self._result_id)
        except EXPECTED_BACKEND_ERRORS as exc:
            self._publish("log_message", f"设置当前空间成果失败：{friendly_error_message(exc)}")
        else:
            super().execute()


class _ReportsRefreshCommand(_DeliveryReadCommand):
    __slots__ = ()

    def execute(self) -> None:
        c = self._controller
        if not self._context_is_current():
            return
        backend = c._backend()
        if backend is None:
            return
        try:
            packages = list(
                backend.projects.list_report_packages(str(self._project_id)))
        except EXPECTED_BACKEND_ERRORS as exc:
            self._publish("log_message", f"刷新项目报告列表失败：{friendly_error_message(exc)}")
        else:
            self._publish("report_list_updated", packages)


__all__ = ["DeliveryController"]
