# -*- coding: utf-8 -*-
"""ProjectChain — 项目/测线/成果/预览/导入/删除链 的子接线器。

域归属（PageCoordinator 拆分的三段之一）：
- 项目生命周期（打开/关闭/失败）与全页项目态重置；
- 测线/成果/预览扇出（lines_updated、line_selected、artifacts_updated、
  dataset/artifact 预览、预览失效）；
- 导入 / 预检 / 传感器同步；
- 成果删除链（后代闭包查询 → 级联确认框 → 提交删除）；
- 成果（交付）页与备份/恢复/报告链、空间页"设为当前测线"。

运行态（当前测线、导入后待选测线）的本类是唯一所有者；
处理域状态（velocity token 等）经 ``coordinator.processing`` 公共属性访问，
不得跨类读私有属性。

本模块不得创建 QWidget（与 controllers 同一纪律）；级联确认框
经 ``ui.dialogs.ask_artifact_delete`` 函数式 API（按需导入，保持本模块
无 Qt 依赖，可独立单测）。
"""
from __future__ import annotations

from ui import constants


class ProjectChain:
    """项目域跨页信号链；纯 Python 接线类，构造时拿 PageCoordinator 门面。"""

    def __init__(self, coordinator) -> None:
        self._co = coordinator
        self.current_line_id = ''         # 当前测线（项目/处理/解释页共用）
        self.pending_select_line_id = ''  # 导入完成后要选中的测线
        self.pending_focus_artifact = ''  # 文件树点成果：待选中预览的 artifact_id
        # 空间轨迹加载去重（见 on_project_opened / on_lines_updated）：
        # 一次「打开项目」里两个槽都要"确保轨迹已加载"，此前各发一次
        # ``load_spatial_tracks()``，导致同一份轨迹被后端完整解析两遍
        # （实测 570–695 ms/次，是 open 链路最大的单项命令成本）。
        self._spatial_reload_pending = False
        # 上次真正发起重载时的测线集合；集合变化才需要重新解析轨迹
        self._spatial_reload_line_ids: tuple[str, ...] | None = None

    # ============================================================ 信号注册
    def connect_all(self) -> None:
        """项目域接线：主页/项目页/空间页(当前测线)/成果页 + 项目/交付控制器。"""
        co = self._co
        home = co.page('homeInterface')
        project = co.page('projectInterface')
        spatial = co.page('spatialInterface')
        delivery = co.page('deliveryInterface')

        # ---------------- 主页（SPEC §6.2）
        home.new_project_requested.connect(co.show_new_project_dialog)
        home.open_project_requested.connect(co.open_project_dialog)
        home.import_line_requested.connect(
            lambda: co.goto_page('projectInterface'))
        home.goto_page.connect(co.goto_page)

        # ---------------- 项目页（SPEC §6.3）
        project.import_requested.connect(self.on_import_requested)
        project.sync_requested.connect(self.on_sync_requested)
        project.line_selected.connect(self.on_line_selected)
        project.line_process_requested.connect(self.on_line_process_requested)
        project.line_delete_requested.connect(self.on_line_delete_requested)
        project.artifact_preview_requested.connect(
            self.on_artifact_preview_requested)
        project.artifact_delete_requested.connect(
            self.on_artifact_delete_requested)
        project.close_project_requested.connect(self.on_close_project_requested)
        # 测线表右键"复制路径/打开所在文件夹"：异步路径查询回包缓存进页面
        pc0 = co.project_controller
        if pc0 is not None:
            pc0.line_source_path_ready.connect(project.set_line_source_path)

        # ---------------- 左侧常驻文件树（与项目页测线表同一信号语义）
        tree = co.file_tree()
        if tree is not None:
            tree.line_selected.connect(self.on_line_selected)
            tree.line_process_requested.connect(self.on_line_process_requested)
            tree.line_delete_requested.connect(self.on_line_delete_requested)
            # 空间成果/项目报告叶子点击 → 跳成果页
            tree.delivery_focus_requested.connect(
                lambda _kind: co.goto_page('deliveryInterface'))
            # 成果叶子点击 → 换线（如需）+ 跳处理页选中预览
            tree.artifact_focus_requested.connect(
                self.on_artifact_focus_requested)
            # 成果叶子右键「删除成果」→ 与项目页同一删除链路
            # （异步查后代闭包 → 级联确认框 → 回收站）
            tree.artifact_delete_requested.connect(
                lambda line_id, artifact_id:
                self.on_artifact_delete_requested(line_id, [artifact_id]))

        # ---------------- 空间信息页：设为当前测线（测线归属项目域）
        spatial.current_line_requested.connect(self.on_spatial_current_line)

        # ---------------- 成果页（SPEC §6.7）
        delivery.spatial_requested.connect(self.on_spatial_requested)
        delivery.report_requested.connect(self.on_report_requested)
        delivery.backup_requested.connect(self.on_backup_requested)
        delivery.restore_requested.connect(self.on_restore_requested)

        # ---------------- 项目控制器 → 本链 / 页面
        pc = co.project_controller
        if pc is not None:
            pc.project_opened.connect(self.on_project_opened)
            pc.project_closed.connect(self.on_project_closed)
            pc.open_failed.connect(self.on_open_failed)
            pc.lines_updated.connect(self.on_lines_updated)
            pc.artifacts_updated.connect(self.on_artifacts_updated)
            pc.preview_failed.connect(
                lambda msg: self._co.infobar('error', '成果预览失败', msg))
            pc.dataset_preview_ready.connect(self.on_dataset_preview)
            pc.artifact_preview_ready.connect(self.on_artifact_preview)
            pc.preview_invalidated.connect(self.on_preview_invalidated)
            pc.preflight_ready.connect(self.on_preflight_ready)
            pc.preflight_failed.connect(self.on_preflight_failed)
            pc.spatial_tracks_ready.connect(spatial.set_tracks)
            pc.artifact_descendants_ready.connect(
                self.on_artifact_descendants_ready)
            pc.busy_changed.connect(project.set_busy)
            if tree is not None:
                pc.busy_changed.connect(tree.set_busy)
                # 文件树「成果」视图：全项目成果列表（与按线的 artifacts_updated 互补）
                pc.all_artifacts_updated.connect(tree.set_artifacts)

        dc = co.delivery_controller
        if dc is not None:
            dc.spatial_results_updated.connect(delivery.set_spatial_results)
            dc.report_generated.connect(self.on_report_generated)
            if tree is not None:
                # 文件树「空间成果 / 项目报告」组与成果页同一数据源
                dc.spatial_results_updated.connect(tree.set_spatial_results)
                dc.report_list_updated.connect(tree.set_reports)

    # ============================================================ 项目生命周期
    def on_close_project_requested(self) -> None:
        if self._co.project_controller is not None and self._co.require_project():
            # The edit service keeps an in-memory session bound to this project.
            # Release it before the backend closes the project, otherwise a later
            # project open can retain stale annotation state.
            if self._co.interpretation_controller is not None:
                self._co.interpretation_controller.close_session()
            self._co.project_controller.close_current()

    def on_project_opened(self, summary) -> None:
        """project_opened → 主页卡片 + 项目页信息 + 最近项目 + 成果页刷新。"""
        co = self._co
        home = co.page('homeInterface')
        project = co.page('projectInterface')
        home.set_current_project(summary)
        project.set_project_info(summary)
        if co.file_tree() is not None:
            co.file_tree().set_project_info(summary)
        root = str(getattr(summary, 'root_path', '') or '')
        if root:
            co.settings.add_recent_project(root)
            co.settings.save()
        self.current_line_id = ''
        self.update_line_labels()
        # 测线列表由 controller 在 project_opened 后自行 refresh_lines
        project_id = co.current_project_id()
        if project_id and co.delivery_controller is not None:
            co.delivery_controller.refresh_spatial(project_id)
            co.delivery_controller.refresh_reports(project_id)
        # 空间信息页：项目打开后加载空间轨迹
        #
        # 去重（性能，见 __init__ 注释）：这里只**记意图**，不直接发起。
        # ``refresh_lines()`` 刚在上面被 controller 发起，其回包
        # ``on_lines_updated`` 紧随其后到达，由它统一发起一次加载。
        self._spatial_reload_pending = True
        self._spatial_reload_line_ids = None
        if co.project_controller is not None:
            co.project_controller.refresh_all_artifacts()  # 文件树「成果」视图
        # A→B 直切：清空空间页残留的 A 项目深度切片与勾选态，
        # 否则非空 payload 守卫会压制 B 项目首次进入深度段的预览请求。
        spatial = co.page('spatialInterface')
        spatial.clear_depth_grid()
        name = getattr(summary, 'name', '')
        co.infobar('success', '项目', f'项目已打开：{name}')
        co.log_message(f'SUCCESS 当前项目：{name}（{root}）')

    def on_project_closed(self) -> None:
        """project_closed → 清空各页项目态并恢复无项目门控。"""
        co = self._co
        self.current_line_id = ''
        # 关闭项目即重置轨迹去重态：否则重新打开**同一个**项目时，测线集合
        # 与上次相同 → 会被误判为"无需重载"，空间页停在空轨迹。
        self._spatial_reload_pending = False
        self._spatial_reload_line_ids = None
        home = co.page('homeInterface')
        project = co.page('projectInterface')
        processing = co.page('processingInterface')
        interpretation = co.page('interpretationInterface')
        delivery = co.page('deliveryInterface')
        home.set_current_project(None)
        home.set_preview_bundle(None)
        project.set_project_info(None)
        if co.file_tree() is not None:
            co.file_tree().set_project_info(None)
        project.set_lines([])
        project.set_artifacts([])
        processing.set_line_label('')
        processing.set_original_bundle(None)
        processing.close_all_artifact_tabs()
        co.processing.velocity_token = None  # 关闭项目即失效当前提交代，迟到回调全部丢弃
        interpretation.set_session_active(False)
        interpretation.set_line_label('')
        interpretation.set_session_info('未打开会话')
        interpretation.set_points([])
        interpretation.set_velocity_failed('')  # 项目关闭 = 状态重置，非失败
        delivery.set_lines([])
        delivery.set_spatial_results([])
        spatial = co.page('spatialInterface')
        spatial.set_tracks([])
        spatial.set_lines([])
        pc = co.project_controller
        if pc is not None:
            pc.invalidate_depth_previews()
        spatial.clear_depth_grid()
        co.log_message('INFO 项目已关闭，相关页面恢复未打开项目状态')

    def on_open_failed(self, message: str) -> None:
        self._co.infobar('error', '项目', message)

    # ============================================================ 测线 / 成果 / 预览
    def update_line_labels(self) -> None:
        """当前测线标签：处理页 / 解释页 / 文件树共用。"""
        co = self._co
        co.page('processingInterface').set_line_label(self.current_line_id)
        co.page('interpretationInterface').set_line_label(self.current_line_id)
        if co.file_tree() is not None:
            co.file_tree().set_current_line(self.current_line_id)

    def on_lines_updated(self, lines: list) -> None:
        """lines_updated → 项目页测线表（自动选中首行）+ 成果页/空间页测线多选。"""
        co = self._co
        lines = list(lines or [])
        project = co.page('projectInterface')
        delivery = co.page('deliveryInterface')
        spatial = co.page('spatialInterface')
        processing = co.page('processingInterface')
        project.set_lines(lines)
        # 预热右键菜单"复制路径"缓存：异步查询各测线源文件路径，
        # 回包经 line_source_path_ready → project.set_line_source_path 入缓存
        if co.project_controller is not None:
            for line in lines:
                lid = str(getattr(line, 'line_id', '') or '')
                if lid:
                    co.project_controller.line_source_path(lid)
        # 导入完成后：若记录了目标测线，选中并预览它
        pending = self.pending_select_line_id
        if pending:
            if project.select_line(pending):
                self.on_line_selected(pending)
            self.pending_select_line_id = ''
        elif lines:
            # 保持用户当前测线；尚未选中时默认首条。
            target = self.current_line_id or str(
                getattr(lines[0], 'line_id', '') or '')
            if target and project.select_line(target):
                if target != self.current_line_id:
                    self.on_line_selected(target)
        processing.set_lines(lines)
        delivery.set_lines(lines)
        spatial.set_lines(lines)
        if co.file_tree() is not None:
            co.file_tree().set_lines(lines)
        # 测线集合变化后空间轨迹同步重载（导入/同步完成均触发 lines_updated）。
        #
        # 去重（性能）：本槽在一次「打开项目」里紧随 on_project_opened 到达，
        # 此时测线集合并未变化，重跑一遍只是把同一份轨迹再解析一次
        # （570–695 ms）。故只在"确有待加载意图"或"测线集合与上次不同"
        # 时才真正发起——后者覆盖导入 / 同步完成等真实变更场景。
        if co.project_controller is not None:
            line_ids = tuple(
                str(getattr(line, 'line_id', '') or '') for line in (lines or ()))
            if self._spatial_reload_pending \
                    or line_ids != self._spatial_reload_line_ids:
                self._spatial_reload_line_ids = line_ids
                co.project_controller.load_spatial_tracks()
            self._spatial_reload_pending = False
        valid_ids = [str(getattr(line, 'line_id', '') or '') for line in lines]
        if self.current_line_id not in valid_ids:
            self.current_line_id = valid_ids[0] if valid_ids else ''
            self.update_line_labels()
        if not lines:
            project.set_artifacts([])

    def on_line_selected(self, line_id: str) -> None:
        """line_selected → 记录当前测线 + 预览 + 刷新成果列表。"""
        line_id = str(line_id or '')
        if not line_id:
            return
        co = self._co
        if line_id != self.current_line_id:
            # 切换测线：关闭上一条测线的成果/步骤 tab（原始锚点保留换内容）
            processing = co.page('processingInterface')
            processing.close_all_artifact_tabs()
            # 在飞速度分析回调带旧测线，会被 line 守卫丢弃；
            # 同步失效 token 并复位解释页，避免新测线永久停留在"拟合中"
            co.processing.velocity_token = None
            # 在飞成果预览同理作废，防旧测线成果渲染进新测线"处理结果"；
            # 成果预览代数与测线预览独立，同测线重复选中不受影响
            if co.project_controller is not None:
                co.project_controller.invalidate_artifact_previews()
            interpretation = co.page('interpretationInterface')
            interpretation.set_velocity_failed('')  # 空串 = 状态重置，非失败
        self.current_line_id = line_id
        self.update_line_labels()
        # 反向同步项目页测线表选中（select_line 内部抑止回发，无双倍预览）
        project = co.page('projectInterface')
        if project is not None and hasattr(project, 'select_line'):
            project.select_line(line_id)
        if co.project_controller is not None:
            co.project_controller.preview_line(line_id)
            co.project_controller.refresh_artifacts(line_id)

    def on_line_process_requested(self, line_id: str) -> None:
        """项目页双击测线 → 设为当前测线并加载数据，跳转处理页。"""
        line_id = str(line_id or '')
        if not line_id:
            return
        self.on_line_selected(line_id)
        self._co.goto_page('processingInterface')
        self._co.infobar('info', '数据预览', f'正在加载测线 {line_id} …')

    def on_spatial_current_line(self, line_id: str) -> None:
        """空间信息页"设为当前测线" → 复用测线选择逻辑 + InfoBar 提示。"""
        line_id = str(line_id or '')
        if not line_id or line_id == self.current_line_id:
            return
        self.on_line_selected(line_id)
        self._co.infobar('success', '空间信息', f'已设为当前测线：{line_id}')

    def on_artifacts_updated(self, line_id: str, artifacts: list) -> None:
        """artifacts_updated → 项目页/处理页成果表；处理完成后自动预览最新成果。"""
        if str(line_id) != self.current_line_id:
            return
        artifacts = list(artifacts or [])
        co = self._co
        project = co.page('projectInterface')
        processing = co.page('processingInterface')
        interpretation = co.page('interpretationInterface')
        project.set_artifacts(artifacts)
        processing.set_artifacts(artifacts)
        interpretation.set_artifacts(artifacts)
        # 文件树「成果」视图跟随刷新（面板按 artifact_id 签名去重，不换内容不重建）
        if co.project_controller is not None:
            co.project_controller.refresh_all_artifacts()
        # 文件树点成果的跨线场景：该线成果列表到达后完成选中 + 预览
        pending = self.pending_focus_artifact
        if pending:
            if processing.select_artifact(pending):
                self.pending_focus_artifact = ''
                if co.project_controller is not None:
                    co.project_controller.preview_artifact(
                        self.current_line_id, pending)
                return
        processing_chain = co.processing
        if processing_chain.preview_newest_artifact:
            processing_chain.preview_newest_artifact = False
            if artifacts and co.project_controller is not None:
                artifact_id = str(getattr(artifacts[0], 'artifact_id', '') or '')
                if artifact_id:
                    # 同步处理页成果下拉的选中项（静默），再预览
                    processing.select_artifact(artifact_id)
                    co.project_controller.preview_artifact(
                        self.current_line_id, artifact_id)

    def on_artifact_preview_requested(self, line_id: str, artifact_id: str) -> None:
        if self._co.project_controller is not None and self._co.require_project():
            self._co.project_controller.preview_artifact(str(line_id), str(artifact_id))

    def on_artifact_focus_requested(self, line_id: str, artifact_id: str) -> None:
        """文件树「成果」叶子点击 → 换线（如需）+ 跳处理页选中并预览该成果。

        跨线时该线成果列表未到，select_artifact 会落空：记下
        ``pending_focus_artifact``，由 on_artifacts_updated 在列表到达后
        完成选中与预览（与导入后 pending_select_line_id 同一异步模式）。
        """
        line_id = str(line_id or '')
        artifact_id = str(artifact_id or '')
        co = self._co
        if not artifact_id or co.project_controller is None \
                or not co.require_project():
            return
        self.pending_focus_artifact = artifact_id
        if line_id and line_id != self.current_line_id:
            self.on_line_selected(line_id)
        co.goto_page('processingInterface')
        processing = co.page('processingInterface')
        if processing.select_artifact(artifact_id):
            # 同线且列表已在：直接预览，无需等刷新
            self.pending_focus_artifact = ''
            co.project_controller.preview_artifact(
                line_id or self.current_line_id, artifact_id)

    def on_dataset_preview(self, bundle) -> None:
        """原始数据预览 → 主页 / 处理页（原始）/ 解释页剖面。

        项目页不再接收 B-Scan 预览（2026-09-22 移除，见 ProjectPage 模块
        文档）：该页右列三卡纵向分割后预览区只有 ~198px 高，B-Scan 必被
        压扁；查看请走处理页/解释页。
        """
        co = self._co
        co.page('homeInterface').set_preview_bundle(bundle)
        co.page('processingInterface').set_original_bundle(bundle)
        co.page('interpretationInterface').set_bundle(bundle)

    def on_artifact_preview(self, artifact_id: str, bundle) -> None:
        """成果预览 → 处理页对应 tab（tab 模型：artifact_id 定位数据源）。"""
        co = self._co
        processing = co.page('processingInterface')
        processing.set_artifact_bundle(str(artifact_id), bundle)
        processing_chain = co.processing
        if processing_chain.show_run_completion_notice:
            processing_chain.show_run_completion_notice = False
            co.infobar('success', '处理完成', '已更新处理结果预览')
            # 跑完自动选中末位 tab（= 最终结果），避免提示与画面矛盾
            processing.show_latest_result()

    def on_preview_invalidated(self, artifact_id: str = '') -> None:
        """当前预览的成果已被删除 → 关闭其 tab（窗口数随 tab 收敛）。"""
        processing = self._co.page('processingInterface')
        processing.close_artifact_tab(str(artifact_id or ''))

    def on_line_delete_requested(self, line_ids: list[str]) -> None:
        """项目页删除所选测线（页面已弹确认框）→ 交给 ProjectController。"""
        line_ids = [str(lid) for lid in (line_ids or []) if lid]
        if not line_ids:
            return
        if self._co.project_controller is None or not self._co.require_project():
            return
        self._co.project_controller.delete_lines(line_ids)

    def on_artifact_delete_requested(self, line_id: str,
                                     artifact_ids: list) -> None:
        """项目页删除成果 → 异步查后代闭包 → 级联确认框 → 提交删除。"""
        if self._co.project_controller is None or not self._co.require_project():
            return
        line_id = str(line_id or '')
        ids = [str(a) for a in (artifact_ids or []) if a]
        if not line_id or not ids:
            return
        self._co.project_controller.get_artifact_descendants(line_id, ids[:1])

    def on_artifact_descendants_ready(self, line_id: str,
                                      descendants: list,
                                      names: dict) -> None:
        """后代闭包回包（GUI 线程）→ 级联确认框 → 确认后提交删除。"""
        descendants = [str(a) for a in (descendants or []) if a]
        if not descendants:
            return
        co = self._co
        pc = co.project_controller
        if pc is None or not co.require_project():
            return
        shown_names = [str(dict(names or {}).get(aid, aid[:8]))
                       for aid in descendants]
        # 对话框创建在 ui.dialogs（本模块纪律：不建 QWidget，按需导入）
        from ui.dialogs import ask_artifact_delete
        if ask_artifact_delete(co.page('projectInterface'), shown_names):
            pc.delete_artifacts(line_id, descendants)

    # ============================================================ 导入 / 预检 / 传感器同步
    def on_import_requested(self, payload: dict) -> None:
        """import_requested：preflight=True→预检；False→提交导入任务。"""
        if self._co.project_controller is None or not self._co.require_project():
            return
        payload = dict(payload or {})
        line_id = str(payload.get('line_id', '') or 'L01')
        if payload.get('preflight'):
            self._co.project_controller.preflight_import(
                str(payload.get('source', '')),
                line_id,
                float(payload.get('dielectric', constants.DEFAULT_DIELECTRIC)))
            return
        job_id = self._co.project_controller.import_line(
            str(payload.get('source', '')),
            line_id,
            str(payload.get('name', '') or ''),
            float(payload.get('dielectric', constants.DEFAULT_DIELECTRIC)))
        if job_id:
            self._co.jobs.import_job_ids.add(str(job_id))
            self.pending_select_line_id = line_id
            self._co.infobar('info', '导入测线', '导入任务已提交，完成后会自动选中该测线')

    def on_preflight_ready(self, result) -> None:
        """preflight_ready → 项目页预检结果区（鸭子类型取字段）。"""
        project = self._co.page('projectInterface')
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

    def on_preflight_failed(self, message: str) -> None:
        self._co.page('projectInterface').set_preflight_result(
            f'预检失败: {message}', False)

    def on_sync_requested(self, payload: dict) -> None:
        if self._co.project_controller is None or not self._co.require_project():
            return
        payload = dict(payload or {})
        job_id = self._co.project_controller.sync_sensors(
            str(payload.get('line_id', '') or self.current_line_id),
            dict(payload.get('paths') or {}),
            dict(payload.get('settings') or {}))
        if job_id:
            self._co.jobs.import_job_ids.add(str(job_id))
            self._co.infobar('info', '传感器同步', '同步任务已提交')

    # ============================================================ 成果（交付）域
    def on_spatial_requested(self, payload: dict) -> None:
        if self._co.delivery_controller is None or not self._co.require_project():
            return
        payload = dict(payload or {})
        job_id = self._co.delivery_controller.create_spatial(
            self._co.current_project_id(),
            str(payload.get('name', '') or '空间成果'),
            list(payload.get('line_ids') or []))
        if job_id:
            self._co.jobs.spatial_job_ids.add(str(job_id))
            self._co.infobar('info', '空间成果', '空间成果任务已提交')

    def on_report_requested(self, payload: dict) -> None:
        if self._co.delivery_controller is None or not self._co.require_project():
            return
        job_id = self._co.delivery_controller.generate_report(
            self._co.current_project_id(),
            str(dict(payload or {}).get('package_name', '') or ''))
        if job_id:
            self._co.infobar('info', '项目报告', '报告生成任务已提交')

    def on_report_generated(self, result) -> None:
        self._co.page('deliveryInterface').set_report_result(result)
        # 报告列表刷新（文件树「项目报告」组）；失败仅记日志，不影响主流程
        project_id = self._co.current_project_id()
        if project_id and self._co.delivery_controller is not None:
            self._co.delivery_controller.refresh_reports(project_id)

    def on_backup_requested(self, options: dict) -> None:
        if self._co.delivery_controller is None or not self._co.require_project():
            return
        dest = str(options.get('destination_dir', ''))
        if not dest:
            return
        job_id = self._co.delivery_controller.backup_project(
            self._co.current_project_id(), dest,
            incremental=bool(options.get('incremental', False)),
            retention_keep=options.get('retention_keep'),
        )
        if job_id:
            mode = '增量' if options.get('incremental') else '全量'
            self._co.infobar('info', '项目备份', f'{mode}备份任务已提交 → {dest}')

    def on_restore_requested(self, archive_path: str) -> None:
        co = self._co
        if not co.backend_ready:
            co.infobar('warning', '恢复备份', '后端尚未就绪，请稍后再试')
            return
        if co.delivery_controller is None:
            return
        dest_root = str(co.settings.get(
            'project_root', constants.DEFAULT_PROJECT_ROOT))
        job_id = co.delivery_controller.restore_project(
            str(archive_path), dest_root)
        if job_id:
            co.infobar('info', '恢复备份', f'恢复任务已提交 → {dest_root}')
