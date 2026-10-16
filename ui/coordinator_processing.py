# -*- coding: utf-8 -*-
"""ProcessingChain — 处理/解释/深度切片 的子接线器。

域归属（PageCoordinator 拆分的三段之一）：
- 处理执行（run_requested/cancel/run_finished）与运行态快照语义；
- AutoTune 调参；方法库装载；处理页 B-scan 成果预览选择；
- 速度分析（velocity token 门卫：代次 + 项目 + 测线三重校验）；
- 界面解释标注会话（打开/点位/保存/自动追踪）；
- 深度切片（空间页预览请求 → 代数交付门卫 → 存为图层）。

运行态（处理任务 id、提交代、取消标记、完成提示旗标、预览旗标、
velocity token）的本类是唯一所有者；当前测线经
``coordinator.project.current_line_id`` 读取，不得跨类读私有属性。

本模块不得创建 QWidget（与 controllers 同一纪律）。
"""
from __future__ import annotations

import uuid
from copy import deepcopy


class ProcessingChain:
    """处理/解释域跨页信号链；纯 Python 接线类，构造时拿 PageCoordinator 门面。"""

    def __init__(self, coordinator) -> None:
        self._co = coordinator
        self.processing_job_id = ''          # 处理页当前运行任务
        self.processing_line_id = ''         # 运行提交时的测线（防运行中切测线竞态）
        self.processing_cancel_requested = False  # 用户主动取消（区别于失败）
        self.show_run_completion_notice = False   # 处理完成后提示一次
        self.preview_newest_artifact = False      # 处理完成后自动预览最新成果
        self.velocity_token = None           # 速度分析当前提交代（None = 无在飞提交）
        self._pending_reconfigure = None
        self._submission_context = None
        self._submission_started_job = ''

    # ============================================================ 信号注册
    def connect_all(self) -> None:
        """处理域接线：处理页/解释页/空间页(深度切片) + 处理/解释/项目控制器。"""
        co = self._co
        processing = co.page('processingInterface')
        interpretation = co.page('interpretationInterface')
        spatial = co.page('spatialInterface')

        # ---------------- 处理页（SPEC §6.5）
        processing.run_requested.connect(self.on_run_requested)
        processing.cancel_requested.connect(self.on_processing_cancel)
        processing.autotune_requested.connect(self.on_autotune_requested)
        processing.line_changed.connect(self.on_processing_line_changed)
        processing.artifact_selected.connect(
            self.on_processing_artifact_selected)
        # tab 模型懒加载：可见面板缺 bundle → 按需预览（异步回填）
        processing.artifact_preview_requested.connect(
            self.on_step_preview_requested)
        if hasattr(processing, 'artifact_delete_requested'):
            processing.artifact_delete_requested.connect(co.project.on_artifact_delete_requested)
        if hasattr(processing, 'artifact_focus_requested'):
            processing.artifact_focus_requested.connect(co.project.on_artifact_focus_requested)
        project_page = co.page('projectInterface')
        if hasattr(processing, 'input_activated') and hasattr(project_page, 'set_line_resume_available'):
            processing.input_activated.connect(lambda line_id, _aid:
                project_page.set_line_resume_available(line_id, processing.has_line_session(line_id)))
        tree = co.file_tree()
        if tree is not None:
            if hasattr(processing, 'project_resources_toggled'):
                processing.project_resources_toggled.connect(
                    lambda visible: tree.set_collapsed(not visible, animate=False))
            if hasattr(processing, 'input_activated') and hasattr(tree, 'set_active_source'):
                processing.input_activated.connect(tree.set_active_source)
            if hasattr(tree, 'raw_focus_requested'):
                tree.raw_focus_requested.connect(self.on_original_focus_requested)
            if hasattr(tree, 'artifact_reconfigure_requested'):
                tree.artifact_reconfigure_requested.connect(self.on_reconfigure_requested)

        # ---------------- 解释页（SPEC §6.6）
        interpretation.open_session_requested.connect(
            self.on_open_session_requested)
        interpretation.points_changed.connect(self.on_points_changed)
        interpretation.velocity_requested.connect(
            self.on_velocity_requested)
        ic0 = co.interpretation_controller
        if ic0 is not None:
            interpretation.auto_trace_requested.connect(ic0.auto_trace)
            interpretation.snap_requested.connect(ic0.snap)
            interpretation.smooth_requested.connect(ic0.smooth)
            interpretation.undo_requested.connect(ic0.undo)
            interpretation.redo_requested.connect(ic0.redo)
            interpretation.save_requested.connect(ic0.save)

        # ---------------- 空间信息页：深度切片域
        spatial.depth_preview_requested.connect(self.on_depth_preview_requested)
        spatial.save_depth_layer_requested.connect(
            self.on_save_depth_layer_requested)

        # ---------------- 处理控制器 → 本链
        prc = co.processing_controller
        if prc is not None:
            prc.methods_loaded.connect(self.on_methods_loaded)
            if hasattr(prc, 'run_submitted'):
                prc.run_submitted.connect(self.on_run_submitted)
            if hasattr(prc, 'run_context_finished'):
                prc.run_context_finished.connect(self.on_run_context_finished)
            else:
                prc.run_finished.connect(self.on_run_finished)
            if hasattr(prc, 'run_result_ready'):
                prc.run_result_ready.connect(processing.set_run_result)
            prc.autotune_finished.connect(self.on_autotune_finished)
            prc.autotune_failed.connect(self.on_autotune_failed)
            prc.velocity_finished.connect(self.on_velocity_finished)
            prc.velocity_failed.connect(self.on_velocity_failed)

        # ---------------- 解释控制器 → 本链 / 解释页
        ic = co.interpretation_controller
        if ic is not None:
            ic.session_opened.connect(self.on_session_opened)
            ic.session_updated.connect(self.on_session_updated)
            ic.session_failed.connect(self.on_session_failed)
            ic.saved.connect(self.on_annotation_saved)
            ic.busy_changed.connect(interpretation.set_busy)

        # ---------------- 项目控制器：深度切片回包（代数门卫在本链）
        pc = co.project_controller
        if pc is not None:
            if hasattr(pc, 'project_opened') and hasattr(processing, 'set_project_read_only'):
                pc.project_opened.connect(lambda summary:
                    processing.set_project_read_only(bool(getattr(summary, 'read_only', False))))
            if hasattr(pc, 'preview_context_failed'):
                pc.preview_context_failed.connect(processing.set_preview_failure)
            if hasattr(pc, 'dataset_revision_ready'):
                pc.dataset_revision_ready.connect(processing.set_original_revision)
            if hasattr(processing, 'original_preview_requested'):
                processing.original_preview_requested.connect(self.on_original_preview_requested)
            pc.artifacts_updated.connect(self._on_reconfigure_artifacts)
            pc.depth_preview_ready.connect(self.on_depth_preview_ready)
            pc.depth_layer_saved.connect(self.on_depth_layer_saved)
            pc.depth_save_failed.connect(self.on_depth_save_failed)

    # ============================================================ 处理域
    def on_methods_loaded(self, methods: list) -> None:
        """方法库 → 处理页 MethodBrowser。"""
        self._co.page('processingInterface').set_methods(methods)

    def on_processing_line_changed(self, line_id: str) -> None:
        """处理页测线下拉变化 → 同步为当前测线并刷新原始数据/成果列表。"""
        line_id = str(line_id or '')
        if not line_id or line_id == self._co.project.current_line_id:
            return
        self._co.project.on_line_selected(line_id)
        self._co.infobar('info', '切换测线', f'已切换到 {line_id}')

    def on_processing_artifact_selected(self, artifact_id: str) -> None:
        """处理页成果下拉变化 → 预览所选处理结果。"""
        line_id = self._co.require_line()
        if line_id and artifact_id and self._co.project_controller is not None:
            self._co.project_controller.preview_artifact(line_id, str(artifact_id))

    def on_original_focus_requested(self, line_id: str) -> None:
        self._co.project.on_line_selected(str(line_id))
        self._co.goto_page('processingInterface')
        self._co.page('processingInterface').select_original()

    def on_reconfigure_requested(self, line_id: str, artifact_id: str) -> None:
        self._pending_reconfigure = (str(line_id), str(artifact_id))
        self._co.project.on_line_selected(str(line_id))
        self._co.goto_page('processingInterface')
        self._try_reconfigure(str(line_id))

    def _on_reconfigure_artifacts(self, line_id: str, _artifacts: list) -> None:
        self._try_reconfigure(str(line_id))

    def _try_reconfigure(self, line_id: str) -> None:
        pending = self._pending_reconfigure
        if pending is None or pending[0] != line_id:
            return
        page = self._co.page('processingInterface')
        if not page.has_artifact(pending[1]):
            return
        self._pending_reconfigure = None
        if not page.reconfigure_artifact(pending[1]):
            self._co.infobar('warning', '重新配置', '该成果缺少可读取的直接步骤输入，或用户保留了原草稿')

    def on_step_preview_requested(self, artifact_id: str) -> None:
        """tab 模型懒加载：可见面板缺 bundle → 按需预览该成果。

        异步回填走 on_artifact_preview → set_artifact_bundle，与手选
        成果同一条链路；步骤批量预览挂 strict_generation=False——按
        artifact_id 寻址回填各自槽位，不共用"最新代数"守卫（并发互相
        踩：先完成的被后提交的静默丢弃，真机骨架挂死根因之一）。
        """
        line_id = self._co.require_line()
        if line_id and artifact_id and self._co.project_controller is not None:
            self._co.project_controller.preview_artifact(
                line_id, str(artifact_id), strict_generation=False)

    def on_original_preview_requested(self) -> None:
        line_id = self._co.require_line()
        if line_id and self._co.project_controller is not None:
            self._co.project_controller.preview_line(line_id)

    def on_run_requested(self, payload: dict) -> None:
        """run_requested(dict) → run_pipeline（含结果名回退与链式输入）。"""
        if self._co.processing_controller is None:
            return
        if self.processing_job_id:
            self._co.infobar('warning', '处理任务', '当前任务运行中，请等待完成或取消')
            return
        line_id = self._co.require_line()
        if not line_id:
            return
        payload = dict(payload or {})
        steps = deepcopy(list(payload.get('steps') or []))
        if not steps:
            self._co.infobar('warning', '处理链', '处理链为空，请先添加处理步骤')
            return
        result_name = str(payload.get('result_name') or ''
                          ).strip() or f'处理结果_{line_id}'
        input_artifact_id = str(payload.get('input_artifact_id') or '')
        self._submission_context = (line_id, input_artifact_id)
        self._submission_started_job = ''
        try:
            job_id = self._co.processing_controller.run_pipeline(
                self._co.current_project_id(), line_id, {'steps': steps}, result_name,
                input_artifact_id=input_artifact_id)
            # Legacy controllers may not publish run_submitted. Do not restart a
            # job whose immediate completion has already been handled.
            if job_id and str(job_id) != self._submission_started_job:
                self.on_run_submitted(str(job_id))
        finally:
            self._submission_context = None

    def on_run_submitted(self, job_id: str) -> None:
        if self._submission_context is None:
            return
        line_id, input_artifact_id = self._submission_context
        self._submission_started_job = self.processing_job_id = str(job_id)
        self.processing_line_id = line_id
        self.processing_cancel_requested = False
        self.show_run_completion_notice = True
        self._co.page('processingInterface').set_running(True, str(job_id))
        tree = self._co.file_tree()
        if tree is not None and hasattr(tree, 'set_processing_input'):
            tree.set_processing_input(line_id, input_artifact_id, True)

    def on_processing_cancel(self) -> None:
        bridge = self._co.job_bridge()
        if bridge is not None and self.processing_job_id:
            self.processing_cancel_requested = True
            bridge.cancel(self.processing_job_id)
            self._co.log_message(
                f'INFO 已请求取消处理任务 {self.processing_job_id}')

    def on_run_context_finished(self, job_id: str, success: bool, message: str) -> None:
        if str(job_id) != self.processing_job_id:
            return
        self.on_run_finished(success, message)

    def on_run_finished(self, success: bool, message: str) -> None:
        """run_finished → 恢复运行态 + InfoBar + 刷新成果并自动预览。

        用户主动取消按 info 提示而非错误；成果刷新回到提交时的测线，
        只有该测线仍是当前测线时才自动预览最新成果。
        """
        self.processing_job_id = ''
        cancelled = self.processing_cancel_requested
        self.processing_cancel_requested = False
        run_line_id = str(self.processing_line_id
                          or self._co.project.current_line_id)
        self.processing_line_id = ''
        processing = self._co.page('processingInterface')
        processing.set_running(False, success=bool(success))
        self.show_run_completion_notice = False
        self.preview_newest_artifact = False  # result tabs manage activation from frozen UI context
        tree = self._co.file_tree()
        if tree is not None and hasattr(tree, 'set_processing_input'):
            tree.set_processing_input(run_line_id, '', False)
        if success:
            self._co.infobar('success', '处理链',
                             message or f'处理链运行完成：{run_line_id}')
            if self._co.project_controller is not None and run_line_id:
                # 只有成果属于当前查看的测线时才自动预览，避免抢走用户视图
                self._co.project_controller.refresh_artifacts(run_line_id)
                self._co.project_controller.refresh_lines()
        elif cancelled:
            self._co.infobar('info', '处理链', f'处理链已取消：{run_line_id}')
        else:
            self._co.infobar('error', '处理链', message or '处理链运行失败')

    def on_autotune_requested(self, method_id: str, params_hint: dict,
                              input_artifact_id: str = "") -> None:
        if self._co.processing_controller is None:
            return
        line_id = self._co.require_line()
        if not line_id:
            return
        self._co.processing_controller.run_autotune(
            self._co.current_project_id(), line_id, str(method_id),
            dict(params_hint or {}),
            input_artifact_id=str(input_artifact_id or ""))
        # P2-7：运行期间禁用"开始调参"，防重复提交
        self._co.page('processingInterface').set_autotune_running(True)
        self._co.infobar('info', 'AutoTune 自动调参', f'已提交调参任务：{method_id}')

    def on_autotune_finished(self, method_id: str, result: dict) -> None:
        processing = self._co.page('processingInterface')
        processing.set_autotune_running(False)
        processing.set_autotune_result(method_id, result)
        self._co.infobar('success', 'AutoTune 自动调参', f'调参完成：{method_id}')

    def on_autotune_failed(self, method_id: str, message: str) -> None:
        self._co.page('processingInterface').set_autotune_running(False)
        self._co.infobar('error', 'AutoTune 自动调参',
                         f'{method_id}: {message}')

    # ============================================================ 速度分析
    def on_velocity_requested(self, points: list) -> None:
        """解释页「拟合速度模型」→ 处理控制器提交速度分析 job。"""
        if self._co.processing_controller is None:
            return
        line_id = self._co.require_line()
        if not line_id:
            return
        if len(points or []) < 3:
            self._co.infobar('warning', '速度分析',
                             '双曲线拟合至少需要 3 个拾取点（当前 %d 个）'
                             % len(points or []))
            return
        interpretation = self._co.page('interpretationInterface')
        interpretation.set_velocity_running(True)
        # token 先生成并登记，再启动 worker：保证任何时点的失败都能匹配
        token = uuid.uuid4().hex
        self.velocity_token = token
        self._co.processing_controller.run_velocity_analysis(
            token, self._co.current_project_id(), line_id, list(points or []))
        self._co.infobar('info', '速度分析',
                         '已提交速度分析任务：%s（%d 个拾取点）'
                         % (line_id, len(points or [])))

    def _velocity_callback_current(self, token: str, project_id: str,
                                   line_id: str) -> bool:
        """回调有效性：token 匹配当前提交代 + 项目/测线仍一致。

        token 是每次提交的代标识：重开同项目/测线的新提交会换 token，
        关闭项目会失效 token，迟到回调因此不会误伤新会话或在飞任务。
        """
        if token != self.velocity_token:
            self._co.log_message('INFO 速度分析回调代次过期，已忽略')
            return False
        if str(project_id) != str(self._co.current_project_id() or ''):
            self._co.log_message(
                f'INFO 速度分析回调来自已关闭项目 {project_id}，已忽略')
            return False
        if str(line_id) != str(self._co.project.current_line_id or ''):
            self._co.log_message(
                f'INFO 速度分析回调测线 {line_id} 非当前测线，已忽略')
            return False
        return True

    def on_velocity_finished(self, token: str, project_id: str,
                             line_id: str, result: dict) -> None:
        """速度分析完成 → 解释页卡片回填 + InfoBar（仅限当前代次/项目/测线）。"""
        if not self._velocity_callback_current(token, project_id, line_id):
            return
        self._co.page('interpretationInterface').set_velocity_result(
            line_id, result)
        self._co.infobar('success', '速度分析', f'速度模型已写回：{line_id}')

    def on_velocity_failed(self, token: str, project_id: str,
                           line_id: str, message: str) -> None:
        if not self._velocity_callback_current(token, project_id, line_id):
            return
        self._co.page('interpretationInterface').set_velocity_failed(message)
        self._co.infobar('error', '速度分析', f'{line_id}: {message}')

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

    def on_open_session_requested(self, artifact_id: str = "") -> None:
        if self._co.interpretation_controller is None:
            return
        line_id = self._co.require_line()
        if line_id:
            self._co.interpretation_controller.open_session(
                self._co.current_project_id(), line_id,
                input_artifact_id=str(artifact_id or ""),
            )

    def on_session_opened(self, snapshot) -> None:
        interpretation = self._co.page('interpretationInterface')
        interpretation.set_points(self._snapshot_points(snapshot))
        line_id = str(getattr(snapshot, 'line_id', '') or ''
                      ) or self._co.project.current_line_id
        interpretation.set_session_info(f'会话已打开（{line_id}）')
        interpretation.set_session_active(True)
        self._co.infobar('success', '界面解释标注', f'标注会话已打开：{line_id}')
        if self._co.project_controller is not None and self._co.project.current_line_id:
            artifact_id = str(getattr(snapshot, 'input_artifact_id', '') or '')
            if artifact_id:
                # 在成果上标注：预览该成果而非原始数据
                self._co.project_controller.preview_artifact(
                    self._co.project.current_line_id, artifact_id)
            else:
                self._co.project_controller.preview_line(
                    self._co.project.current_line_id)

    def on_session_updated(self, snapshot) -> None:
        self._co.page('interpretationInterface').set_points(
            self._snapshot_points(snapshot))

    def on_session_failed(self, message: str) -> None:
        self._co.page('interpretationInterface').set_session_active(False)
        self._co.infobar('error', '界面解释标注', message)

    def on_points_changed(self, points: list) -> None:
        if self._co.interpretation_controller is not None:
            self._co.interpretation_controller.replace_points(list(points or []))

    def on_annotation_saved(self, message: str) -> None:
        self._co.page('interpretationInterface').set_session_info('已保存')
        self._co.infobar('success', '界面解释标注', message or '标注已保存')

    # ============================================================ 深度切片域
    def on_depth_preview_requested(self, line_ids: list) -> None:
        """空间页切到深度切片段 / 勾选变化 → 请求界面深度预览。"""
        pc = self._co.project_controller
        if pc is None or not self._co.require_project():
            return
        pc.request_depth_preview(list(line_ids or []))

    def on_depth_preview_ready(self, payload: dict, line_ids: list,
                               cell_size_m: float, generation: int) -> None:
        """深度预览回包交付门卫：代数过期即丢弃，再交给空间页渲染。

        双层防护 ②：worker 发射时的代数经信号快照传递；本 slot 在主线程
        执行，若期间项目关闭/新请求已推进代数，则此回包已失效。
        """
        pc = self._co.project_controller
        if pc is None:
            return
        if generation != pc.depth_preview_generation:
            return
        self._co.page('spatialInterface').set_depth_grid(
            payload, line_ids, cell_size_m)

    def on_save_depth_layer_requested(self, line_ids: list, cell_size_m: float) -> None:
        """空间页「存为图层」→ 提交网格化界面深度图层 job。"""
        pc = self._co.project_controller
        if pc is None or not self._co.require_project():
            return
        pc.submit_depth_layer(list(line_ids or []), float(cell_size_m or 1.0))

    def on_depth_layer_saved(self, job_id: str, line_ids: list, cell_size_m: float) -> None:
        """深度图层 job 完成 → 日志 + InfoBar（无图层列表页，无需刷新）。"""
        self._co.infobar('success', '深度图层',
                         f'已保存 {len(line_ids)} 条测线的界面深度图层'
                         f'（格网 {cell_size_m:.2f} m）')
        self._co.log_message(
            f'SUCCESS 深度图层已保存：{len(line_ids)} 条测线，任务 {job_id[:8]}…')

    def on_depth_save_failed(self, message: str) -> None:
        self._co.infobar('error', '深度图层', message or '深度图层任务失败', duration=8000)
