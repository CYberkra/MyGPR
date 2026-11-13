"""Route job events and actions to the bottom output panel."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from uuid import uuid4


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
        self._contexts = {}
        self._actions = {}

    def connect_all(self) -> None:
        panel = self._co.output_panel
        if panel is not None:
            panel.cancel_job_requested.connect(self.on_cancel)
            panel.job_controls.prune_requested.connect(self.on_prune_jobs)
            panel.job_table().action_requested.connect(self.on_job_action)

    def _views(self) -> tuple:
        panel = self._co.output_panel
        return (panel.job_table(),) if panel is not None else ()

    def register_processing(self, job_id, context):
        self._contexts[str(job_id)] = deepcopy(context)

    def record_submission_failure(self, context, message):
        job_id = uuid4().hex
        self.register_processing(job_id, context)
        self.known_job_ids.add(job_id)
        for view in self._views():
            view.upsert_job(job_id, self._job_title(job_id, f'处理流水线 {context["line_id"]}'))
            view.update_progress(job_id, 0, 0, str(message))
            view.set_status(job_id, 'failed')
        self._attach_action(job_id, 'failed', None)
        return job_id

    def register_batch(self, job_ids):
        if job_ids:
            for job_id in job_ids:
                self._upsert(str(job_id))
            for view in self._views():
                view.add_batch(uuid4().hex, job_ids)

    def _upsert(self, job_id: str) -> None:
        if job_id in self.known_job_ids:
            return
        self.known_job_ids.add(job_id)
        self._contexts.setdefault(job_id, {'project_id': str(self._co.current_project_id() or '')})
        title = job_id
        bridge = self._co.job_bridge()
        if bridge is not None:
            title = bridge.titles().get(job_id) or job_id
        for view in self._views():
            view.upsert_job(job_id, self._job_title(job_id, title))

    def _job_title(self, job_id, title):
        controller = getattr(self._co, 'project_controller', None)
        project = getattr(controller, 'current_project', None)
        owner = self._contexts[job_id].get('project_id')
        if owner and owner == getattr(project, 'project_id', None):
            name = str(getattr(project, 'name', '') or '')
            if name:
                return f'{title} · {name}'
        return title

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
                           message or '任务执行失败，详见底部任务面板', duration=8000)
        status = 'cancelled' if cancelled else 'completed' if success else 'failed'
        completed, total = self._progress_counts.pop(job_id, (0, 0))
        for view in self._views():
            view.update_progress(job_id, completed, total, str(message))
            view.set_status(job_id, status)
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
        self._attach_action(job_id, status, result)
        # 空间成果完成 → 刷新空间成果表
        if job_id in self.spatial_job_ids:
            self.spatial_job_ids.discard(job_id)
            project_id = co.current_project_id()
            if success and project_id and co.delivery_controller is not None:
                result_id = self._field(result, 'result_id')
                if result_id and self._contexts.get(job_id, {}).get('project_id') == str(project_id):
                    co.page('spatialInterface').delivery_tools.select_when_available(result_id)
                co.delivery_controller.refresh_spatial(project_id)

    @staticmethod
    def _field(result, name):
        value = result.get(name) if isinstance(result, Mapping) else getattr(result, name, '')
        return str(value or '')

    def _attach_action(self, job_id, status, result):
        context = self._contexts.get(job_id, {})
        label, action = '', None
        if status == 'completed':
            artifact_id = self._field(result, 'artifact_id')
            line_id = self._field(result, 'line_id') or context.get('line_id', '')
            result_id = self._field(result, 'result_id')
            directory = self._field(result, 'package_dir') or self._field(result, 'project_path')
            archive = self._field(result, 'archive_path')
            if isinstance(result, Path):
                archive = str(result)
            metadata = result.get('metadata') if isinstance(result, Mapping) else getattr(result, 'metadata', None)
            if directory and isinstance(metadata, Mapping) and metadata.get('project_root'):
                directory = str(Path(metadata['project_root']) / directory)
            if artifact_id and line_id:
                label, action = '查看结果', ('artifact', line_id, artifact_id)
            elif result_id and job_id in self.spatial_job_ids:
                label, action = '查看结果', ('spatial', result_id)
            elif directory or archive:
                path = Path(directory) if directory else Path(archive).parent
                if path.is_absolute():
                    label, action = '打开目录', ('directory', str(path))
        elif status == 'failed' and context.get('steps'):
            label, action = '返回配置', ('configuration',)
        if action:
            self._actions[job_id] = action
            for view in self._views():
                view.set_job_action(job_id, label)
        else:
            self._contexts.pop(job_id, None)

    def on_job_action(self, job_id):
        action = self._actions.get(job_id)
        if not action:
            return
        co = self._co
        kind, *target = action
        context = self._contexts.get(job_id, {})
        if kind == 'directory':
            from PyQt6.QtCore import QUrl
            from PyQt6.QtGui import QDesktopServices

            if Path(target[0]).is_dir() and QDesktopServices.openUrl(QUrl.fromLocalFile(target[0])):
                return
            co.infobar('warning', '打开目录', '输出目录不可用')
        elif str(co.current_project_id() or '') != context.get('project_id'):
            co.infobar('warning', '任务', '请先打开该任务所属项目')
        elif kind == 'artifact':
            co.project.on_artifact_focus_requested(*target)
        elif kind == 'spatial':
            co.project.on_delivery_focus_requested('spatial', target[0])
        else:
            co.project.on_job_configuration_requested(context)

    # ============================================================ 取消 / 清理
    def on_cancel(self, job_id: str) -> None:
        bridge = self._co.job_bridge()
        if bridge is not None:
            job_id = str(job_id)
            if job_id and job_id == self._co.processing.processing_job_id:
                self._co.processing.processing_cancel_requested = True
            bridge.cancel(job_id)
            self._co.log_message(f'INFO 已请求取消任务 {job_id[:8]}…')

    def on_prune_jobs(self) -> None:
        """清理已完成任务：BackendController.prune_jobs（工作线程，不阻塞 UI）。"""
        for view in self._views():
            removed = {job_id for job_id in self._actions if not view.job_snapshot(job_id)}
            for job_id in removed:
                self._actions.pop(job_id, None)
                self._contexts.pop(job_id, None)
        backend_controller = self._co.backend_controller
        if backend_controller is None or not backend_controller.prune_jobs():
            self._co.infobar('warning', '任务中心', '后端尚未就绪')
