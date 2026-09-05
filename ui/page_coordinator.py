# -*- coding: utf-8 -*-
"""PageCoordinator — 跨页业务信号链的集中接线器（任务 F 候选 1）。

职责边界：
- ``main_window.py``：窗口组装（页面创建 / 导航 / 主题 / 面板 / 快捷键 /
  后端就绪门控）与项目对话框；
- 本模块：全部跨页业务信号链（page 信号 → controller 调用 → page 刷新），
  以及处理 / 导入 / 任务域的运行态（运行中任务、提交时测线快照、待选测线等）。

单条信号链 = 本类一个方法。除 Qt 信号连接外不依赖窗口内部实现，
仅通过鸭子类型的 window 服务接口（``_page`` / ``_infobar`` / ``log_message`` /
``_require_project`` 等）访问窗口，因此每条链可用假窗口独立做单元测试；
``connect_all()`` 是唯一的信号注册入口，由主窗口组装完成后调用一次。

本模块不得创建 QWidget（与 controllers 同一纪律）。
"""
from __future__ import annotations

from typing import Any
import uuid

from ui import constants
from ui.logger_config import setup_logger


# run_command（backend_controller，依赖 PyQt6）在 _on_prune_jobs 内按需导入，
# 使本模块在无 Qt 的打包/测试环境也可导入（tests/test_page_coordinator.py 依赖此性质）。

_LOGGER = setup_logger('mygpr_window', 'logs/mygpr_window.log')


class PageCoordinator:
    """跨页业务信号链与运行态的唯一持有者。"""

    def __init__(self, window) -> None:
        self._win = window
        # ---- 业务接线状态（自 main_window 迁入，SPEC §7）----
        self._current_line_id = ''          # 当前测线（项目/处理/解释页共用）
        self._known_job_ids = set()         # 已 upsert 到任务控件的任务
        self._import_job_ids = set()        # 测线导入/传感器同步任务（完成后刷新测线）
        self._spatial_job_ids = set()       # 空间成果任务（完成后刷新空间成果表）
        self._processing_job_id = ''        # 处理页当前运行任务
        self._velocity_token = None         # 速度分析当前提交代（None = 无在飞提交）
        self._processing_line_id = ''       # 运行提交时的测线（防运行中切测线竞态）
        self._processing_cancel_requested = False  # 用户主动取消（区别于失败）
        self._preview_newest_artifact = False  # 处理完成后自动预览最新成果
        self._show_run_completion_notice = False  # 处理完成后提示一次
        self._pending_select_line_id = ''        # 导入完成后要选中的测线

    # ------------------------------------------------------------ 窗口服务接口（鸭子类型，便于假窗口测试）
    def _page(self, object_name: str):
        return self._win._page(object_name)

    def _infobar(self, level: str, title: str, content: str,
                 duration: int = None) -> None:
        self._win._infobar(level, title, content, duration)

    def log_message(self, msg: str) -> None:
        self._win.log_message(msg)

    def _goto_page(self, object_name: str) -> None:
        self._win._goto_page(object_name)

    @property
    def _backend_ready(self) -> bool:
        return self._win._backend_ready

    def _current_project_id(self):
        return self._win._current_project_id()

    def _require_project(self) -> bool:
        return self._win._require_project()

    def _require_line(self) -> str:
        return self._win._require_line()

    def _job_bridge(self):
        return self._win._job_bridge()

    def connect_job_bridge(self, bridge) -> None:
        """把 JobBridge 三个任务信号接到本接线器的槽（任务 F 候选 1 迁移后唯一接线点）。

        显式属性访问：槽位被改名/移走时立刻 AttributeError 暴露，
        而不是 hasattr 探测静默跳过（那次事故：load_methods 不执行 → 方法库为空）。
        """
        bridge.progress_changed.connect(self._on_job_progress)
        bridge.status_changed.connect(self._on_job_status)
        bridge.job_completed.connect(self._on_job_completed)

    def current_line_id(self) -> str:
        """当前测线（窗口 _require_line 的委托目标，避免读私有属性）。"""
        return self._current_line_id

    # ---- 窗口属性透传（控制器 / 设置 / 日志面板）----
    @property
    def backend_controller(self):
        return self._win.backend_controller

    @property
    def project_controller(self):
        return self._win.project_controller

    @property
    def processing_controller(self):
        return self._win.processing_controller

    @property
    def interpretation_controller(self):
        return self._win.interpretation_controller

    @property
    def delivery_controller(self):
        return self._win.delivery_controller

    @property
    def settings(self):
        return self._win.settings

    @property
    def log_panel(self):
        return self._win.log_panel

    @property
    def _log_signal(self):
        return self._win._log_signal

    # ============================================================ 信号注册（唯一入口）
    def connect_all(self) -> None:
        """全量业务接线：Page 信号 → 本模块槽；Controller 信号 → 本模块 / Page。"""
        home = self._page('homeInterface')
        project = self._page('projectInterface')
        processing = self._page('processingInterface')
        interpretation = self._page('interpretationInterface')
        spatial = self._page('spatialInterface')
        delivery = self._page('deliveryInterface')
        jobs = self._page('jobsInterface')

        # ---------------- 主页（SPEC §6.2）
        if hasattr(home, 'new_project_requested'):
            home.new_project_requested.connect(self._win._show_new_project_dialog)
            home.open_project_requested.connect(self._win._open_project_dialog)
            home.import_line_requested.connect(
                lambda: self._goto_page('projectInterface'))
            home.goto_page.connect(self._goto_page)
            home_jobs = home.mini_jobs()
            if home_jobs is not None and hasattr(home_jobs, 'cancel_requested'):
                home_jobs.cancel_requested.connect(self._cancel_job)

        # ---------------- 项目页（SPEC §6.3）
        if hasattr(project, 'import_requested'):
            project.import_requested.connect(self._on_import_requested)
            project.sync_requested.connect(self._on_sync_requested)
            project.line_selected.connect(self._on_line_selected)
            project.line_process_requested.connect(self._on_line_process_requested)
            project.line_delete_requested.connect(self._on_line_delete_requested)
            project.artifact_preview_requested.connect(
                self._on_artifact_preview_requested)
            project.close_project_requested.connect(self._on_close_project_requested)
            # 测线表右键"复制路径/打开所在文件夹"的路径查询回调
            project.set_source_path_resolver(
                self.project_controller.line_source_path)

        # ---------------- 处理页（SPEC §6.5）
        if hasattr(processing, 'run_requested'):
            processing.run_requested.connect(self._on_run_requested)
            processing.cancel_requested.connect(self._on_processing_cancel)
            processing.autotune_requested.connect(self._on_autotune_requested)
            processing.line_changed.connect(self._on_processing_line_changed)
            processing.artifact_selected.connect(
                self._on_processing_artifact_selected)

        # ---------------- 解释页（SPEC §6.6）
        if hasattr(interpretation, 'open_session_requested'):
            interpretation.open_session_requested.connect(
                self._on_open_session_requested)
            interpretation.points_changed.connect(self._on_points_changed)
            interpretation.velocity_requested.connect(
                self._on_velocity_requested)
            if self.interpretation_controller is not None:
                interpretation.auto_trace_requested.connect(
                    self.interpretation_controller.auto_trace)
                interpretation.snap_requested.connect(
                    self.interpretation_controller.snap)
                interpretation.smooth_requested.connect(
                    self.interpretation_controller.smooth)
                interpretation.undo_requested.connect(
                    self.interpretation_controller.undo)
                interpretation.redo_requested.connect(
                    self.interpretation_controller.redo)
                interpretation.save_requested.connect(
                    self.interpretation_controller.save)

        # ---------------- 空间信息页
        if hasattr(spatial, 'current_line_requested'):
            spatial.current_line_requested.connect(self._on_spatial_current_line)

        # ---------------- 成果页（SPEC §6.7）
        if hasattr(delivery, 'spatial_requested'):
            delivery.spatial_requested.connect(self._on_spatial_requested)
            delivery.report_requested.connect(self._on_report_requested)
            delivery.backup_requested.connect(self._on_backup_requested)
            delivery.restore_requested.connect(self._on_restore_requested)

        # ---------------- 任务页（SPEC §6.8）
        if hasattr(jobs, 'cancel_requested'):
            jobs.cancel_requested.connect(self._cancel_job)
            jobs.prune_requested.connect(self._on_prune_jobs)

        # ---------------- 右侧日志面板任务取消
        if hasattr(self.log_panel, 'cancel_job_requested'):
            self.log_panel.cancel_job_requested.connect(self._cancel_job)

        # ---------------- 控制器 → 本模块 / 页面
        pc = self.project_controller
        if pc is not None:
            pc.project_opened.connect(self._on_project_opened)
            pc.project_closed.connect(self._on_project_closed)
            pc.open_failed.connect(self._on_open_failed)
            pc.lines_updated.connect(self._on_lines_updated)
            pc.artifacts_updated.connect(self._on_artifacts_updated)
            pc.dataset_preview_ready.connect(self._on_dataset_preview)
            pc.artifact_preview_ready.connect(self._on_artifact_preview)
            pc.preflight_ready.connect(self._on_preflight_ready)
            pc.preflight_failed.connect(self._on_preflight_failed)
            if hasattr(spatial, 'set_tracks') and hasattr(pc, 'spatial_tracks_ready'):
                pc.spatial_tracks_ready.connect(spatial.set_tracks)
            if hasattr(project, 'set_busy'):
                pc.busy_changed.connect(project.set_busy)

        prc = self.processing_controller
        if prc is not None:
            if hasattr(processing, 'set_methods'):
                prc.methods_loaded.connect(self._on_methods_loaded)
            prc.run_finished.connect(self._on_run_finished)
            prc.autotune_finished.connect(self._on_autotune_finished)
            prc.autotune_failed.connect(self._on_autotune_failed)
            prc.velocity_finished.connect(self._on_velocity_finished)
            prc.velocity_failed.connect(self._on_velocity_failed)

        ic = self.interpretation_controller
        if ic is not None and hasattr(interpretation, 'set_points'):
            ic.session_opened.connect(self._on_session_opened)
            ic.session_updated.connect(self._on_session_updated)
            ic.session_failed.connect(self._on_session_failed)
            ic.saved.connect(self._on_annotation_saved)
            ic.busy_changed.connect(interpretation.set_busy)

        dc = self.delivery_controller
        if dc is not None and hasattr(delivery, 'set_spatial_results'):
            dc.spatial_results_updated.connect(delivery.set_spatial_results)
            dc.report_generated.connect(self._on_report_generated)

    # ============================================================ 项目生命周期
    def _on_close_project_requested(self) -> None:
        if self.project_controller is not None and self._require_project():
            # The edit service keeps an in-memory session bound to this project.
            # Release it before the backend closes the project, otherwise a later
            # project open can retain stale annotation state.
            if self.interpretation_controller is not None:
                self.interpretation_controller.close_session()
            self.project_controller.close_current()

    def _on_project_opened(self, summary) -> None:
        """project_opened → 主页卡片 + 项目页信息 + 最近项目 + 成果页刷新。"""
        home = self._page('homeInterface')
        project = self._page('projectInterface')
        if hasattr(home, 'set_current_project'):
            home.set_current_project(summary)
        if hasattr(project, 'set_project_info'):
            project.set_project_info(summary)
        root = str(getattr(summary, 'root_path', '') or '')
        if root:
            self.settings.add_recent_project(root)
            self.settings.save()
        self._current_line_id = ''
        self._update_line_labels()
        # 测线列表由 controller 在 project_opened 后自行 refresh_lines
        project_id = self._current_project_id()
        if project_id and self.delivery_controller is not None:
            self.delivery_controller.refresh_spatial(project_id)
        # 空间信息页：项目打开后加载空间轨迹
        if self.project_controller is not None and hasattr(
                self.project_controller, 'load_spatial_tracks'):
            self.project_controller.load_spatial_tracks()
        name = getattr(summary, 'name', '')
        self._infobar('success', '项目', f'项目已打开：{name}')
        self.log_message(f'SUCCESS 当前项目：{name}（{root}）')

    def _on_project_closed(self) -> None:
        """project_closed → 清空各页项目态并恢复无项目门控。"""
        self._current_line_id = ''
        home = self._page('homeInterface')
        project = self._page('projectInterface')
        processing = self._page('processingInterface')
        interpretation = self._page('interpretationInterface')
        delivery = self._page('deliveryInterface')
        if hasattr(home, 'set_current_project'):
            home.set_current_project(None)
            home.set_preview_bundle(None)
        if hasattr(project, 'set_project_info'):
            project.set_project_info(None)
            project.set_lines([])
            project.set_artifacts([])
            project.set_preview_bundle(None)
        if hasattr(processing, 'set_line_label'):
            processing.set_line_label('')
            processing.set_original_bundle(None)
            processing.set_result_bundle(None)
        self._velocity_token = None  # 关闭项目即失效当前提交代，迟到回调全部丢弃
        if hasattr(interpretation, 'set_session_active'):
            interpretation.set_session_active(False)
        if hasattr(interpretation, 'set_line_label'):
            interpretation.set_line_label('')
            interpretation.set_session_info('未打开会话')
            interpretation.set_points([])
            interpretation.set_velocity_failed('')  # 项目关闭 = 状态重置，非失败
        if hasattr(delivery, 'set_lines'):
            delivery.set_lines([])
            delivery.set_spatial_results([])
        spatial = self._page('spatialInterface')
        if hasattr(spatial, 'set_tracks'):
            spatial.set_tracks([])
            spatial.set_lines([])
        self.log_message('INFO 项目已关闭，相关页面恢复未打开项目状态')

    def _on_open_failed(self, message: str) -> None:
        self._infobar('error', '项目', message)

    # ============================================================ 测线 / 成果 / 预览
    def _update_line_labels(self) -> None:
        """当前测线标签：处理页 / 解释页共用。"""
        processing = self._page('processingInterface')
        interpretation = self._page('interpretationInterface')
        if hasattr(processing, 'set_line_label'):
            processing.set_line_label(self._current_line_id)
        if hasattr(interpretation, 'set_line_label'):
            interpretation.set_line_label(self._current_line_id)

    def _on_lines_updated(self, lines: list) -> None:
        """lines_updated → 项目页测线表（自动选中首行）+ 成果页/空间页测线多选。"""
        lines = list(lines or [])
        project = self._page('projectInterface')
        delivery = self._page('deliveryInterface')
        spatial = self._page('spatialInterface')
        processing = self._page('processingInterface')
        if hasattr(project, 'set_lines'):
            project.set_lines(lines)
        # 导入完成后：若记录了目标测线，选中并预览它
        pending = self._pending_select_line_id
        if pending and hasattr(project, 'select_line'):
            if project.select_line(pending):
                self._on_line_selected(pending)
            self._pending_select_line_id = ''
        elif hasattr(project, 'select_line') and lines:
            # 保持用户当前测线；尚未选中时默认首条。
            target = self._current_line_id or str(
                getattr(lines[0], 'line_id', '') or '')
            if target and project.select_line(target):
                if target != self._current_line_id:
                    self._on_line_selected(target)
        if hasattr(processing, 'set_lines'):
            processing.set_lines(lines)
        if hasattr(delivery, 'set_lines'):
            delivery.set_lines(lines)
        if hasattr(spatial, 'set_lines'):
            spatial.set_lines(lines)
        # 测线集合变化后空间轨迹同步重载（导入/同步完成均触发 lines_updated）
        if self.project_controller is not None and hasattr(
                self.project_controller, 'load_spatial_tracks'):
            self.project_controller.load_spatial_tracks()
        valid_ids = [str(getattr(line, 'line_id', '') or '') for line in lines]
        if self._current_line_id not in valid_ids:
            self._current_line_id = valid_ids[0] if valid_ids else ''
            self._update_line_labels()
        if not lines:
            if hasattr(project, 'set_artifacts'):
                project.set_artifacts([])

    def _on_line_selected(self, line_id: str) -> None:
        """line_selected → 记录当前测线 + 预览 + 刷新成果列表。"""
        line_id = str(line_id or '')
        if not line_id:
            return
        if line_id != self._current_line_id:
            # 切换测线：清掉处理页"处理结果"分段里上一条测线的残留预览
            processing = self._page('processingInterface')
            if hasattr(processing, 'set_result_bundle'):
                processing.set_result_bundle(None)
            # 在飞速度分析回调带旧测线，会被 line 守卫丢弃；
            # 同步失效 token 并复位解释页，避免新测线永久停留在"拟合中"
            self._velocity_token = None
            interpretation = self._page('interpretationInterface')
            if hasattr(interpretation, 'set_velocity_failed'):
                interpretation.set_velocity_failed('')  # 空串 = 状态重置，非失败
        self._current_line_id = line_id
        self._update_line_labels()
        if self.project_controller is not None:
            self.project_controller.preview_line(line_id)
            self.project_controller.refresh_artifacts(line_id)

    def _on_line_process_requested(self, line_id: str) -> None:
        """项目页双击测线 → 设为当前测线并加载数据，跳转处理页。"""
        line_id = str(line_id or '')
        if not line_id:
            return
        self._on_line_selected(line_id)
        self._goto_page('processingInterface')
        self._infobar('info', '数据预览', f'正在加载测线 {line_id} …')

    def _on_spatial_current_line(self, line_id: str) -> None:
        """空间信息页"设为当前测线" → 复用测线选择逻辑 + InfoBar 提示。"""
        line_id = str(line_id or '')
        if not line_id or line_id == self._current_line_id:
            return
        self._on_line_selected(line_id)
        self._infobar('success', '空间信息', f'已设为当前测线：{line_id}')

    def _on_artifacts_updated(self, line_id: str, artifacts: list) -> None:
        """artifacts_updated → 项目页/处理页成果表；处理完成后自动预览最新成果。"""
        if str(line_id) != self._current_line_id:
            return
        artifacts = list(artifacts or [])
        project = self._page('projectInterface')
        processing = self._page('processingInterface')
        interpretation = self._page('interpretationInterface')
        if hasattr(project, 'set_artifacts'):
            project.set_artifacts(artifacts)
        if hasattr(processing, 'set_artifacts'):
            processing.set_artifacts(artifacts)
        if hasattr(interpretation, 'set_artifacts'):
            interpretation.set_artifacts(artifacts)
        if self._preview_newest_artifact:
            self._preview_newest_artifact = False
            if artifacts and self.project_controller is not None:
                artifact_id = str(getattr(artifacts[0], 'artifact_id', '') or '')
                if artifact_id:
                    # 同步处理页成果下拉的选中项（静默），再预览
                    if hasattr(processing, 'select_artifact'):
                        processing.select_artifact(artifact_id)
                    self.project_controller.preview_artifact(
                        self._current_line_id, artifact_id)

    def _on_artifact_preview_requested(self, line_id: str, artifact_id: str) -> None:
        if self.project_controller is not None and self._require_project():
            self.project_controller.preview_artifact(str(line_id), str(artifact_id))

    def _on_dataset_preview(self, bundle) -> None:
        """原始数据预览 → 主页 / 项目页 / 处理页（原始）/ 解释页剖面。"""
        home = self._page('homeInterface')
        project = self._page('projectInterface')
        processing = self._page('processingInterface')
        interpretation = self._page('interpretationInterface')
        if hasattr(home, 'set_preview_bundle'):
            home.set_preview_bundle(bundle)
        if hasattr(project, 'set_preview_bundle'):
            project.set_preview_bundle(bundle)
        if hasattr(processing, 'set_original_bundle'):
            processing.set_original_bundle(bundle)
        if hasattr(interpretation, 'set_bundle'):
            interpretation.set_bundle(bundle)

    def _on_artifact_preview(self, artifact_id: str, bundle) -> None:
        """成果预览 → 项目页预览 + 处理页（处理结果）。"""
        project = self._page('projectInterface')
        processing = self._page('processingInterface')
        if hasattr(project, 'set_preview_bundle'):
            project.set_preview_bundle(bundle)
        if hasattr(processing, 'set_result_bundle'):
            processing.set_result_bundle(bundle)
        if self._show_run_completion_notice:
            self._show_run_completion_notice = False
            self._infobar('success', '处理完成', '已更新处理结果预览')
            # P1-4：运行完成自动切到"处理结果"分段，避免提示与画面矛盾
            if hasattr(processing, 'show_result_segment'):
                processing.show_result_segment()

    def _on_line_delete_requested(self, line_ids: list[str]) -> None:
        """项目页删除所选测线（页面已弹确认框）→ 交给 ProjectController。"""
        line_ids = [str(lid) for lid in (line_ids or []) if lid]
        if not line_ids:
            return
        if self.project_controller is None or not self._require_project():
            return
        self.project_controller.delete_lines(line_ids)

    # ============================================================ 导入 / 预检 / 传感器同步
    def _on_import_requested(self, payload: dict) -> None:
        """import_requested：preflight=True→预检；False→提交导入任务。"""
        if self.project_controller is None or not self._require_project():
            return
        payload = dict(payload or {})
        line_id = str(payload.get('line_id', '') or 'L01')
        if payload.get('preflight'):
            self.project_controller.preflight_import(
                str(payload.get('source', '')),
                line_id,
                float(payload.get('dielectric', constants.DEFAULT_DIELECTRIC)))
            return
        job_id = self.project_controller.import_line(
            str(payload.get('source', '')),
            line_id,
            str(payload.get('name', '') or ''),
            float(payload.get('dielectric', constants.DEFAULT_DIELECTRIC)))
        if job_id:
            self._import_job_ids.add(str(job_id))
            self._pending_select_line_id = line_id
            self._infobar('info', '导入测线', '导入任务已提交，完成后会自动选中该测线')

    def _on_preflight_ready(self, result) -> None:
        """preflight_ready → 项目页预检结果区（鸭子类型取字段）。"""
        project = self._page('projectInterface')
        if not hasattr(project, 'set_preflight_result'):
            return
        can_import = bool(getattr(result, 'can_import', False))
        message = str(getattr(result, 'message', '') or '')
        fmt = str(getattr(result, 'format_name', '') or '')
        samples = int(getattr(result, 'sample_count', 0) or 0)
        traces = int(getattr(result, 'trace_count', 0) or 0)
        parts = [message] if message else []
        if fmt:
            parts.append(f'格式: {fmt}')
        if samples and traces:
            parts.append(f'数据: {samples} 采样 × {traces} 道')
        suggestions = [str(s) for s in (getattr(result, 'suggestions', ()) or ())]
        parts.extend(suggestions)
        project.set_preflight_result('\n'.join(parts) or '预检完成', can_import)

    def _on_preflight_failed(self, message: str) -> None:
        project = self._page('projectInterface')
        if hasattr(project, 'set_preflight_result'):
            project.set_preflight_result(f'预检失败: {message}', False)

    def _on_sync_requested(self, payload: dict) -> None:
        if self.project_controller is None or not self._require_project():
            return
        payload = dict(payload or {})
        job_id = self.project_controller.sync_sensors(
            str(payload.get('line_id', '') or self._current_line_id),
            dict(payload.get('paths') or {}),
            dict(payload.get('settings') or {}))
        if job_id:
            self._import_job_ids.add(str(job_id))
            self._infobar('info', '传感器同步', '同步任务已提交')

    # ============================================================ 处理域
    def _on_methods_loaded(self, methods: list) -> None:
        """方法库 → 处理页 MethodBrowser。"""
        processing = self._page('processingInterface')
        if hasattr(processing, 'set_methods'):
            processing.set_methods(methods)

    def _on_processing_line_changed(self, line_id: str) -> None:
        """处理页测线下拉变化 → 同步为当前测线并刷新原始数据/成果列表。"""
        line_id = str(line_id or '')
        if not line_id or line_id == self._current_line_id:
            return
        self._on_line_selected(line_id)
        self._infobar('info', '切换测线', f'已切换到 {line_id}')

    def _on_processing_artifact_selected(self, artifact_id: str) -> None:
        """处理页成果下拉变化 → 预览所选处理结果。"""
        line_id = self._require_line()
        if line_id and artifact_id and self.project_controller is not None:
            self.project_controller.preview_artifact(line_id, str(artifact_id))

    def _on_run_requested(self, payload: dict) -> None:
        """run_requested(dict) → run_pipeline（含结果名回退与链式输入）。"""
        if self.processing_controller is None:
            return
        line_id = self._require_line()
        if not line_id:
            return
        payload = dict(payload or {})
        steps = list(payload.get('steps') or [])
        if not steps:
            self._infobar('warning', '处理链', '处理链为空，请先添加处理步骤')
            return
        result_name = str(payload.get('result_name') or ''
                          ).strip() or f'处理结果_{line_id}'
        input_artifact_id = str(payload.get('input_artifact_id') or '')
        job_id = self.processing_controller.run_pipeline(
            self._current_project_id(), line_id, {'steps': steps}, result_name,
            input_artifact_id=input_artifact_id)
        if job_id:
            self._processing_job_id = str(job_id)
            # 快照提交时的测线：运行期间用户可能切换测线，
            # 完成回调必须回到本条测线刷新成果，而不是"完成那一刻的当前测线"
            self._processing_line_id = line_id
            self._processing_cancel_requested = False
            self._show_run_completion_notice = True
            processing = self._page('processingInterface')
            if hasattr(processing, 'set_running'):
                processing.set_running(True, str(job_id))

    def _on_processing_cancel(self) -> None:
        bridge = self._job_bridge()
        if bridge is not None and self._processing_job_id:
            self._processing_cancel_requested = True
            bridge.cancel(self._processing_job_id)
            self.log_message(f'INFO 已请求取消处理任务 {self._processing_job_id}')

    def _on_run_finished(self, success: bool, message: str) -> None:
        """run_finished → 恢复运行态 + InfoBar + 刷新成果并自动预览。

        用户主动取消按 info 提示而非错误；成果刷新回到提交时的测线，
        只有该测线仍是当前测线时才自动预览最新成果。
        """
        self._processing_job_id = ''
        cancelled = self._processing_cancel_requested
        self._processing_cancel_requested = False
        run_line_id = str(self._processing_line_id or self._current_line_id)
        self._processing_line_id = ''
        processing = self._page('processingInterface')
        if hasattr(processing, 'set_running'):
            processing.set_running(False)
        if success:
            self._infobar('success', '处理链',
                          message or f'处理链运行完成：{run_line_id}')
            if self.project_controller is not None and run_line_id:
                # 只有成果属于当前查看的测线时才自动预览，避免抢走用户视图
                self._preview_newest_artifact = (
                    run_line_id == self._current_line_id)
                self.project_controller.refresh_artifacts(run_line_id)
                self.project_controller.refresh_lines()
        elif cancelled:
            self._infobar('info', '处理链', f'处理链已取消：{run_line_id}')
        else:
            self._infobar('error', '处理链', message or '处理链运行失败')

    def _on_autotune_requested(self, method_id: str, params_hint: dict,
                               input_artifact_id: str = "") -> None:
        if self.processing_controller is None:
            return
        line_id = self._require_line()
        if not line_id:
            return
        self.processing_controller.run_autotune(
            self._current_project_id(), line_id, str(method_id),
            dict(params_hint or {}),
            input_artifact_id=str(input_artifact_id or ""))
        # P2-7：运行期间禁用"开始调参"，防重复提交
        processing = self._page('processingInterface')
        if hasattr(processing, 'set_autotune_running'):
            processing.set_autotune_running(True)
        self._infobar('info', 'AutoTune 自动调参', f'已提交调参任务：{method_id}')

    def _on_autotune_finished(self, method_id: str, result: dict) -> None:
        processing = self._page('processingInterface')
        if hasattr(processing, 'set_autotune_running'):
            processing.set_autotune_running(False)
        if hasattr(processing, 'set_autotune_result'):
            processing.set_autotune_result(method_id, result)
        self._infobar('success', 'AutoTune 自动调参', f'调参完成：{method_id}')

    def _on_velocity_requested(self, points: list) -> None:
        """解释页「拟合速度模型」→ 处理控制器提交速度分析 job。"""
        if self.processing_controller is None:
            return
        line_id = self._require_line()
        if not line_id:
            return
        if len(points or []) < 3:
            self._infobar('warning', '速度分析',
                          '双曲线拟合至少需要 3 个拾取点（当前 %d 个）'
                          % len(points or []))
            return
        interpretation = self._page('interpretationInterface')
        if hasattr(interpretation, 'set_velocity_running'):
            interpretation.set_velocity_running(True)
        # token 先生成并登记，再启动 worker：保证任何时点的失败都能匹配
        token = uuid.uuid4().hex
        self._velocity_token = token
        self.processing_controller.run_velocity_analysis(
            token, self._current_project_id(), line_id, list(points or []))
        self._infobar('info', '速度分析',
                      '已提交速度分析任务：%s（%d 个拾取点）'
                      % (line_id, len(points or [])))

    def _velocity_callback_current(self, token: str, project_id: str,
                                   line_id: str) -> bool:
        """回调有效性：token 匹配当前提交代 + 项目/测线仍一致。

        token 是每次提交的代标识：重开同项目/测线的新提交会换 token，
        关闭项目会失效 token，迟到回调因此不会误伤新会话或在飞任务。
        """
        if token != getattr(self, '_velocity_token', None):
            self.log_message('INFO 速度分析回调代次过期，已忽略')
            return False
        if str(project_id) != str(self._current_project_id() or ''):
            self.log_message(
                f'INFO 速度分析回调来自已关闭项目 {project_id}，已忽略')
            return False
        if str(line_id) != str(self._current_line_id or ''):
            self.log_message(
                f'INFO 速度分析回调测线 {line_id} 非当前测线，已忽略')
            return False
        return True

    def _on_velocity_finished(self, token: str, project_id: str,
                              line_id: str, result: dict) -> None:
        """速度分析完成 → 解释页卡片回填 + InfoBar（仅限当前代次/项目/测线）。"""
        if not self._velocity_callback_current(token, project_id, line_id):
            return
        interpretation = self._page('interpretationInterface')
        if hasattr(interpretation, 'set_velocity_result'):
            interpretation.set_velocity_result(line_id, result)
        self._infobar('success', '速度分析', f'速度模型已写回：{line_id}')

    def _on_velocity_failed(self, token: str, project_id: str,
                            line_id: str, message: str) -> None:
        if not self._velocity_callback_current(token, project_id, line_id):
            return
        interpretation = self._page('interpretationInterface')
        if hasattr(interpretation, 'set_velocity_failed'):
            interpretation.set_velocity_failed(message)
        self._infobar('error', '速度分析', f'{line_id}: {message}')

    def _on_autotune_failed(self, method_id: str, message: str) -> None:
        processing = self._page('processingInterface')
        if hasattr(processing, 'set_autotune_running'):
            processing.set_autotune_running(False)
        self._infobar('error', 'AutoTune 自动调参',
                      f'{method_id}: {message}')

    # ============================================================ 解释域
    @staticmethod
    def _snapshot_points(snapshot) -> list:
        """InterfaceEditSnapshot → [(trace, sample), ...]（鸭子类型）。"""
        annotation = getattr(snapshot, 'annotation', None)
        points = getattr(annotation, 'points', ()) or ()
        result = []
        for point in points:
            if isinstance(point, (tuple, list)) and len(point) >= 2:
                result.append((int(point[0]), int(point[1])))
            else:
                result.append((int(getattr(point, 'trace_index', 0)),
                               int(getattr(point, 'sample_index', 0))))
        return result

    def _on_open_session_requested(self, artifact_id: str = "") -> None:
        if self.interpretation_controller is None:
            return
        line_id = self._require_line()
        if line_id:
            self.interpretation_controller.open_session(
                self._current_project_id(), line_id,
                input_artifact_id=str(artifact_id or ""),
            )

    def _on_session_opened(self, snapshot) -> None:
        interpretation = self._page('interpretationInterface')
        if not hasattr(interpretation, 'set_points'):
            return
        interpretation.set_points(self._snapshot_points(snapshot))
        line_id = str(getattr(snapshot, 'line_id', '') or self._current_line_id)
        interpretation.set_session_info(f'会话已打开（{line_id}）')
        if hasattr(interpretation, 'set_session_active'):
            interpretation.set_session_active(True)
        self._infobar('success', '界面解释标注', f'标注会话已打开：{line_id}')
        if self.project_controller is not None and self._current_line_id:
            artifact_id = str(getattr(snapshot, 'input_artifact_id', '') or '')
            if artifact_id:
                # 在成果上标注：预览该成果而非原始数据
                self.project_controller.preview_artifact(
                    self._current_line_id, artifact_id)
            else:
                self.project_controller.preview_line(self._current_line_id)

    def _on_session_updated(self, snapshot) -> None:
        interpretation = self._page('interpretationInterface')
        if hasattr(interpretation, 'set_points'):
            interpretation.set_points(self._snapshot_points(snapshot))

    def _on_session_failed(self, message: str) -> None:
        interpretation = self._page('interpretationInterface')
        if hasattr(interpretation, 'set_session_active'):
            interpretation.set_session_active(False)
        self._infobar('error', '界面解释标注', message)

    def _on_points_changed(self, points: list) -> None:
        if self.interpretation_controller is not None:
            self.interpretation_controller.replace_points(list(points or []))

    def _on_annotation_saved(self, message: str) -> None:
        interpretation = self._page('interpretationInterface')
        if hasattr(interpretation, 'set_session_info'):
            interpretation.set_session_info('已保存')
        self._infobar('success', '界面解释标注', message or '标注已保存')

    # ============================================================ 成果域
    def _on_spatial_requested(self, payload: dict) -> None:
        if self.delivery_controller is None or not self._require_project():
            return
        payload = dict(payload or {})
        job_id = self.delivery_controller.create_spatial(
            self._current_project_id(),
            str(payload.get('name', '') or '空间成果'),
            list(payload.get('line_ids') or []))
        if job_id:
            self._spatial_job_ids.add(str(job_id))
            self._infobar('info', '空间成果', '空间成果任务已提交')

    def _on_report_requested(self, payload: dict) -> None:
        if self.delivery_controller is None or not self._require_project():
            return
        job_id = self.delivery_controller.generate_report(
            self._current_project_id(),
            str(dict(payload or {}).get('package_name', '') or ''))
        if job_id:
            self._infobar('info', '项目报告', '报告生成任务已提交')

    def _on_report_generated(self, result) -> None:
        delivery = self._page('deliveryInterface')
        if hasattr(delivery, 'set_report_result'):
            delivery.set_report_result(result)

    def _on_backup_requested(self, options: dict) -> None:
        if self.delivery_controller is None or not self._require_project():
            return
        dest = str(options.get('destination_dir', ''))
        if not dest:
            return
        job_id = self.delivery_controller.backup_project(
            self._current_project_id(), dest,
            incremental=bool(options.get('incremental', False)),
            retention_keep=options.get('retention_keep'),
        )
        if job_id:
            mode = '增量' if options.get('incremental') else '全量'
            self._infobar('info', '项目备份', f'{mode}备份任务已提交 → {dest}')

    def _on_restore_requested(self, archive_path: str) -> None:
        if not self._backend_ready:
            self._infobar('warning', '恢复备份', '后端尚未就绪，请稍后再试')
            return
        if self.delivery_controller is None:
            return
        dest_root = str(self.settings.get(
            'project_root', constants.DEFAULT_PROJECT_ROOT))
        job_id = self.delivery_controller.restore_project(
            str(archive_path), dest_root)
        if job_id:
            self._infobar('info', '恢复备份', f'恢复任务已提交 → {dest_root}')

    # ============================================================ 任务中心
    def _job_views(self) -> tuple:
        """(JobTable, LogPanel.mini_jobs, HomePage.mini_jobs)（None 已过滤）。"""
        views = []
        jobs = self._page('jobsInterface')
        if hasattr(jobs, 'job_table'):
            views.append(jobs.job_table())
        if hasattr(self.log_panel, 'mini_jobs'):
            views.append(self.log_panel.mini_jobs())
        home = self._page('homeInterface')
        if hasattr(home, 'mini_jobs'):
            views.append(home.mini_jobs())
        return tuple(v for v in views if v is not None)

    def _upsert_job(self, job_id: str) -> None:
        if job_id in self._known_job_ids:
            return
        self._known_job_ids.add(job_id)
        title = job_id
        bridge = self._job_bridge()
        if bridge is not None:
            title = bridge.titles().get(job_id) or job_id
        for view in self._job_views():
            view.upsert_job(job_id, title)

    def _on_job_status(self, job_id: str, status: str) -> None:
        job_id = str(job_id)
        self._upsert_job(job_id)
        for view in self._job_views():
            view.set_status(job_id, str(status))
        if str(status) in ('completed', 'failed', 'cancelled'):
            for view in self._job_views():
                if hasattr(view, 'remove_inactive'):
                    view.remove_inactive()

    def _on_job_progress(self, job_id: str, completed: int, total: int,
                         message: str) -> None:
        job_id = str(job_id)
        self._upsert_job(job_id)
        for view in self._job_views():
            view.update_progress(job_id, int(completed), int(total), str(message))
        if job_id == self._processing_job_id:
            processing = self._page('processingInterface')
            if hasattr(processing, 'set_progress'):
                processing.set_progress(int(completed), int(total), str(message))

    def _on_job_completed(self, job_id: str, success: bool, message: str,
                          result) -> None:
        job_id = str(job_id)
        if success:
            self.log_message(
                f'SUCCESS 任务 {job_id[:8]}… 完成：{message}')
        else:
            self.log_message(
                f'WARNING 任务 {job_id[:8]}… 结束：{message}')
            # P0-1：失败必须显式反馈；处理任务由 run_finished 单独弹窗，避免重复
            if job_id != self._processing_job_id:
                self._infobar('error', '任务失败',
                              message or '任务执行失败，详见任务页', duration=8000)
        status = 'completed' if success else 'failed'
        for view in self._job_views():
            if hasattr(view, 'set_status'):
                view.set_status(job_id, status)
        # 导入/同步完成 → 刷新测线列表
        if job_id in self._import_job_ids:
            self._import_job_ids.discard(job_id)
            if success and self.project_controller is not None:
                self.project_controller.refresh_lines()
        # 空间成果完成 → 刷新空间成果表
        if job_id in self._spatial_job_ids:
            self._spatial_job_ids.discard(job_id)
            project_id = self._current_project_id()
            if success and project_id and self.delivery_controller is not None:
                self.delivery_controller.refresh_spatial(project_id)

    def _cancel_job(self, job_id: str) -> None:
        bridge = self._job_bridge()
        if bridge is not None:
            job_id = str(job_id)
            if job_id and job_id == self._processing_job_id:
                # 从任务页/日志面板取消处理任务同样按"已取消"提示，而非失败
                self._processing_cancel_requested = True
            bridge.cancel(job_id)
            self.log_message(f'INFO 已请求取消任务 {job_id[:8]}…')

    def _on_prune_jobs(self) -> None:
        """清理已完成任务：backend.jobs.prune（工作线程，不阻塞 UI）。"""
        backend = getattr(self.backend_controller, 'backend', None) \
            if self.backend_controller is not None else None
        if backend is None:
            self._infobar('warning', '任务中心', '后端尚未就绪')
            return

        from ui.controllers.backend_controller import run_command
        run_command(
            _JobsPruneCommand(self, backend),
            name='mygpr-jobs-prune',
        )


class _JobsPruneCommand:
    __slots__ = ("_coordinator", "_backend")

    def __init__(self, coordinator: PageCoordinator, backend: Any) -> None:
        self._coordinator = coordinator
        self._backend = backend

    def execute(self) -> None:
        try:
            removed = self._backend.jobs.prune()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning('jobs prune 失败: %s', exc)
            self._coordinator._log_signal.emit(f'WARNING 清理已完成任务失败: {exc}')
        else:
            self._coordinator._log_signal.emit(f'INFO 已清理 {len(removed)} 个终态任务记录')
