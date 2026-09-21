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
    spatial_current_changed = pyqtSignal(str)   # 设为当前成果成功（result_id）
    report_generated = pyqtSignal(object)     # ReportPackage
    report_list_updated = pyqtSignal(list)    # [ReportPackage, ...] 全量列表
    backend_not_ready = pyqtSignal()          # _backend() 兜底：后端未就绪（接线层弹 InfoBar）
    busy_changed = pyqtSignal(bool)           # 交付任务在飞态（接线层禁用成果页按钮）

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._backend_controller = None
        self._busy = False

    # ------------------------------------------------------------------
    def set_backend(self, backend_controller) -> None:
        self._backend_controller = backend_controller

    def _backend(self):
        controller = self._backend_controller
        backend = getattr(controller, "backend", None) if controller is not None else None
        if backend is None:
            self.log_message.emit("后端尚未就绪，请稍后再试")
            self.backend_not_ready.emit()
        return backend

    def _set_busy(self, value: bool) -> None:
        value = bool(value)
        if value != self._busy:
            self._busy = value
            self.busy_changed.emit(value)

    def _job_bridge(self):
        return getattr(self._backend_controller, "job_bridge", None)

    def _submit_and_watch(self, title: str, submit, on_done=None) -> str | None:
        """提交交付类任务并监视：在飞互斥 + busy 信号（接线层禁用成果页按钮）。

        返回 None = 未提交（后端未就绪 / 已有交付任务在飞 / 提交异常），
        调用方不得把 None 当成功。busy 在任务到达终态（成功/失败/取消）
        后统一释放，防重复提交。
        """
        backend = self._backend()
        bridge = self._job_bridge()
        if backend is None or bridge is None:
            return None
        if self._busy:
            self.log_message.emit(f"{title}：已有交付任务在执行，请稍候…")
            return None
        try:
            job_id = submit(backend)
        except Exception as exc:  # noqa: BLE001
            self.log_message.emit(f"{title}提交失败：{friendly_error_message(exc)}")
            return None
        self.log_message.emit(f"{title}已提交")
        self._set_busy(True)

        def _on_completed(done_id: str, success: bool, message: str, result: Any) -> None:
            if done_id != job_id:
                return
            try:
                bridge.job_completed.disconnect(_on_completed)
            except TypeError:
                pass
            self._set_busy(False)
            if on_done is not None:
                on_done(success, message, result)

        # 先接完成回调再 watch：秒终态任务（提交即失败）的 job_completed
        # 可能在 watch 启动的竞态窗口内到达，先连接保证 busy 必然释放。
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

    def set_current_spatial(self, project_id: str, result_id: str) -> None:
        """设为当前空间成果（成果页表格双击/右键）。"""
        run_command(
            _SetCurrentSpatialCommand(self, project_id, result_id),
            name="mygpr-spatial-set-current",
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


class _SetCurrentSpatialCommand:
    __slots__ = ("_controller", "_project_id", "_result_id")

    def __init__(self, controller: DeliveryController, project_id: str,
                 result_id: str) -> None:
        self._controller = controller
        self._project_id = project_id
        self._result_id = result_id

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            return
        try:
            backend.spatial.set_current(
                str(self._project_id), str(self._result_id))
        except Exception as exc:  # noqa: BLE001
            c.log_message.emit(f"设置当前空间成果失败：{friendly_error_message(exc)}")
        else:
            c.log_message.emit(f"SUCCESS 已设为当前空间成果：{self._result_id}")
            c.spatial_current_changed.emit(str(self._result_id))


class _ReportsRefreshCommand:
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
            packages = list(
                backend.projects.list_report_packages(str(self._project_id)))
        except Exception as exc:  # noqa: BLE001
            c.log_message.emit(f"刷新项目报告列表失败：{friendly_error_message(exc)}")
        else:
            c.report_list_updated.emit(packages)


__all__ = ["DeliveryController"]
