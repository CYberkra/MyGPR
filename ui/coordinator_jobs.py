# -*- coding: utf-8 -*-
"""Distribute task events to the bottom panel and project startup view."""
from __future__ import annotations


class JobHub:
    """任务事件分发枢纽；纯 Python 接线类，构造时拿 PageCoordinator 门面。"""

    def __init__(self, coordinator) -> None:
        self._co = coordinator
        self.known_job_ids: set[str] = set()   # 已 upsert 到任务控件的任务
        self.import_job_ids: set[str] = set()  # 测线导入/传感器同步任务（完成后刷新测线）
        self.import_forms: dict[str, str] = {}
        self.spatial_job_ids: set[str] = set()  # 空间成果任务（完成后刷新空间成果表）
        self._terminal_statuses: dict[str, str] = {}
        self._progress_counts: dict[str, tuple[int, int]] = {}

    # ============================================================ 信号注册
    def connect_all(self) -> None:
        co = self._co
        output_panel = co.output_panel
        if output_panel is not None:
            output_panel.cancel_job_requested.connect(self.on_cancel)

        home_jobs = co.page('homeInterface').mini_jobs()
        if home_jobs is not None:
            home_jobs.cancel_requested.connect(self.on_cancel)

    # ============================================================ 视图扇出
    def _views(self) -> tuple:
        co = self._co
        views = []
        output_panel = co.output_panel
        if output_panel is not None:
            views.append(output_panel.mini_jobs())
        views.append(co.page('homeInterface').mini_jobs())
        return tuple(v for v in views if v is not None)

    def _upsert(self, job_id: str) -> None:
        if job_id in self.known_job_ids:
            return
        self.known_job_ids.add(job_id)
        title = job_id
        bridge = self._co.job_bridge()
        if bridge is not None:
            title = bridge.titles().get(job_id) or job_id
        for view in self._views():
            view.upsert_job(job_id, title)

    # ============================================================ 任务事件
    def on_status(self, job_id: str, status: str) -> None:
        job_id = str(job_id)
        self._upsert(job_id)
        for view in self._views():
            view.set_status(job_id, str(status))
        if str(status) in ('completed', 'failed', 'cancelled'):
            self._terminal_statuses[job_id] = str(status)
    def on_progress(self, job_id: str, completed: int, total: int,
                    message: str) -> None:
        job_id = str(job_id)
        self._upsert(job_id)
        self._progress_counts[job_id] = (int(completed), int(total))
        for view in self._views():
            view.update_progress(job_id, int(completed), int(total), str(message))
        if job_id == self._co.processing.processing_job_id:
            self._co.page('processingInterface').set_progress(
                int(completed), int(total), str(message))

    def on_notice(self, job_id: str, payload: dict) -> None:
        job_id = str(job_id)
        self._upsert(job_id)
        message = str(payload.get('message') or payload.get('code') or '处理提示')
        level = 'INFO' if payload.get('level') == 'info' else 'WARNING'
        self._co.log_message(f'{level} 任务 {str(job_id)[:8]}… {message}')
        completed, total = self._progress_counts.get(job_id, (0, 0))
        for view in self._views():
            view.update_progress(job_id, completed, total, message)

    def on_completed(self, job_id: str, success: bool, message: str,
                     result) -> None:
        co = self._co
        job_id = str(job_id)
        self._upsert(job_id)
        cancelled = self._terminal_statuses.pop(job_id, '') == 'cancelled'
        if success:
            co.log_message(
                f'SUCCESS 任务 {job_id[:8]}… 完成：{message}')
        elif cancelled:
            co.log_message(f'INFO 任务 {job_id[:8]}… 已取消：{message}')
        else:
            co.log_message(
                f'WARNING 任务 {job_id[:8]}… 结束：{message}')
            # P0-1：失败必须显式反馈；处理任务由 run_finished 单独弹窗，避免重复
            if job_id != co.processing.processing_job_id:
                co.infobar('error', '任务失败',
                           message or '任务执行失败，详见底栏任务', duration=8000)
        status = 'cancelled' if cancelled else 'completed' if success else 'failed'
        completed, total = self._progress_counts.pop(job_id, (0, 0))
        for view in self._views():
            view.update_progress(job_id, completed, total, str(message))
            view.set_status(job_id, status)
        self.known_job_ids.discard(job_id)
        home_jobs = co.page('homeInterface').mini_jobs()
        if home_jobs is not None:
            home_jobs.remove_inactive()
        panel = co.output_panel.mini_jobs() if co.output_panel is not None else None
        if panel is not None:
            action = co.project.archive_task_action(result) if success and result is not None else None
            panel.finish_job(job_id, str(message), action)
        # 导入/同步完成 → 刷新测线列表
        if job_id in self.import_job_ids:
            self.import_job_ids.discard(job_id)
            submitted_source = self.import_forms.pop(job_id, None)
            if success and submitted_source is not None:
                failed_sources = [str(item.source) for item in getattr(result, 'results', ())
                                  if not item.success]
                co.page('projectInterface').finish_import(submitted_source, failed_sources)
            if success and co.project_controller is not None:
                co.project.invalidate_spatial_tracks()
                co.project_controller.refresh_lines()
        # 空间成果完成 → 刷新空间成果表
        if job_id in self.spatial_job_ids:
            self.spatial_job_ids.discard(job_id)
            project_id = co.current_project_id()
            if success and project_id and co.delivery_controller is not None:
                co.delivery_controller.refresh_spatial(project_id)

    # ============================================================ 取消 / 清理
    def on_cancel(self, job_id: str) -> None:
        bridge = self._co.job_bridge()
        if bridge is not None:
            job_id = str(job_id)
            if job_id and job_id == self._co.processing.processing_job_id:
                # 从底栏或启动区取消处理任务同样按"已取消"提示，而非失败
                self._co.processing.processing_cancel_requested = True
            bridge.cancel(job_id)
            self._co.log_message(f'INFO 已请求取消任务 {job_id[:8]}…')
