# -*- coding: utf-8 -*-
"""Mixin classes for MyGPRMainWindow signal handlers (split from 1360-line monolith).

Each mixin groups handlers for one logical domain.  MyGPRMainWindow inherits
from all mixins via multiple inheritance; signal/slot wiring stays in
main_window.py.
"""
from __future__ import annotations

import datetime
import logging
from ui.controllers.backend_controller import run_command

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QPushButton, QTextBrowser, QToolButton, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    InfoBar, InfoBarPosition, LineEdit, PrimaryPushButton, PushButton,
)

from ui import constants
from ui.logger_config import setup_logger

_LOGGER = setup_logger('mygpr_window', 'logs/mygpr_window.log', level=logging.DEBUG)


# ================================================================ 项目生命周期
class _ProjectLifecycleMixin:
    """新建/打开/关闭项目对话框 + project_opened/project_closed 分发。"""

    def _show_new_project_dialog(self) -> None:
        """新建项目对话框：选目录 + 项目名 + 可选元数据 → create_project。"""
        if not self._backend_ready:
            self._infobar('warning', '新建项目', '后端尚未就绪，请稍后再试')
            return
        if self.project_controller is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('新建项目')
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        form.setSpacing(10)

        root_row = QHBoxLayout()
        root_edit = LineEdit(dialog)
        root_edit.setText(str(self.settings.get(
            'project_root', constants.DEFAULT_PROJECT_ROOT)))
        browse_btn = PushButton('浏览', dialog)
        browse_btn.setFixedWidth(70)

        def _browse() -> None:
            path = QFileDialog.getExistingDirectory(
                dialog, '选择项目根目录', root_edit.text().strip()
                or constants.DEFAULT_PROJECT_ROOT)
            if path:
                root_edit.setText(path)

        browse_btn.clicked.connect(_browse)
        root_row.addWidget(root_edit, 1)
        root_row.addWidget(browse_btn)
        form.addRow('项目根目录:', root_row)

        name_edit = LineEdit(dialog)
        name_edit.setPlaceholderText('例如: 新区道路探测')
        form.addRow('项目名称:', name_edit)

        # P1-9：6 个可选元数据折叠进"项目详情"，首屏只留根目录+名称
        detail_btn = QToolButton(dialog)
        detail_btn.setText('项目详情（可选）')
        detail_btn.setCheckable(True)
        detail_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        detail_btn.setArrowType(Qt.ArrowType.RightArrow)
        form.addRow(detail_btn)

        detail_widget = QWidget(dialog)
        detail_form = QFormLayout(detail_widget)
        detail_form.setContentsMargins(0, 0, 0, 0)
        detail_form.setSpacing(8)
        meta_edits = {}
        for key, label in (('location', '位置(可选):'), ('operator', '操作员(可选):'),
                           ('project_no', '项目编号(可选):'),
                           ('device_model', '设备型号(可选):'),
                           ('coordinate_system', '坐标系(可选):'),
                           ('vertical_datum', '高程基准(可选):')):
            edit = LineEdit(dialog)
            detail_form.addRow(label, edit)
            meta_edits[key] = edit
        detail_widget.setVisible(False)
        form.addRow(detail_widget)

        def _toggle_detail(checked: bool) -> None:
            detail_widget.setVisible(checked)
            detail_btn.setArrowType(
                Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

        detail_btn.toggled.connect(_toggle_detail)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        ok_btn = PrimaryPushButton('创建', dialog)
        cancel_btn = PushButton('取消', dialog)
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        def _accept() -> None:
            name = name_edit.text().strip()
            root = root_edit.text().strip()
            if not root:
                self._infobar('warning', '新建项目', '项目根目录不能为空')
                return
            if not name:
                self._infobar('warning', '新建项目', '项目名称不能为空')
                return
            meta = {key: edit.text().strip()
                    for key, edit in meta_edits.items() if edit.text().strip()}
            dialog.accept()
            self.project_controller.create_project(root, name, meta)

        ok_btn.clicked.connect(_accept)
        dialog.exec()

    def _open_project_dialog(self) -> None:
        """打开项目：目录对话框 → open_project。"""
        if not self._backend_ready:
            self._infobar('warning', '打开项目', '后端尚未就绪，请稍后再试')
            return
        if self.project_controller is None:
            return
        root = QFileDialog.getExistingDirectory(
            self, '选择项目目录', str(self.settings.get(
                'project_root', constants.DEFAULT_PROJECT_ROOT)))
        if root:
            self.project_controller.open_project(root)

    def _on_close_project_requested(self) -> None:
        if self.project_controller is not None and self._require_project():
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
        if hasattr(interpretation, 'set_line_label'):
            interpretation.set_line_label('')
            interpretation.set_session_info('未打开会话')
            interpretation.set_points([])
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


# ================================================================ 测线 / 成果 / 预览
class _LineArtifactMixin:
    """测线选择、数据预览、成果预览、批量删除。"""

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
        pending = getattr(self, '_pending_select_line_id', '')
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
        if getattr(self, '_show_run_completion_notice', False):
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


# ================================================================ 导入 / 预检 / 传感器同步
class _ImportPreflightMixin:
    """import_requested / preflight / sync 信号路由。"""

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


# ================================================================ 处理页
class _ProcessingMixin:
    """处理页信号路由：run / cancel / autotune / 测线切换 / 成果选择。"""

    def _on_methods_loaded(self, methods: list) -> None:
        """方法库 → 处理页 MethodBrowser。"""
        processing = self._page('processingInterface')
        if hasattr(processing, 'set_methods'):
            processing.set_methods(methods)

    def _on_line_load_requested(self) -> None:
        """处理页"加载测线数据" → 预览当前测线原始数据。"""
        line_id = self._require_line()
        if line_id and self.project_controller is not None:
            self.project_controller.preview_line(line_id)
            self._infobar('info', '数据预览', f'正在加载测线 {line_id} …')

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
        cancelled = bool(getattr(self, '_processing_cancel_requested', False))
        self._processing_cancel_requested = False
        run_line_id = str(getattr(self, '_processing_line_id', '')
                          or self._current_line_id)
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

    def _on_autotune_failed(self, method_id: str, message: str) -> None:
        processing = self._page('processingInterface')
        if hasattr(processing, 'set_autotune_running'):
            processing.set_autotune_running(False)
        self._infobar('error', 'AutoTune 自动调参',
                      f'{method_id}: {message}')


# ================================================================ 解释页
class _InterpretationMixin:
    """解释页信号路由：会话打开 / 更新 / 点替换 / 保存。"""

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


# ================================================================ 成果页
class _DeliveryMixin:
    """成果页信号路由：空间成果 / 报告 / 备份 / 恢复。"""

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

    def _on_backup_requested(self, dest_dir: str) -> None:
        if self.delivery_controller is None or not self._require_project():
            return
        job_id = self.delivery_controller.backup_project(
            self._current_project_id(), str(dest_dir))
        if job_id:
            self._infobar('info', '项目备份', f'备份任务已提交 → {dest_dir}')

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


# ================================================================ 任务中心
class _JobCenterMixin:
    """JobBridge → 任务控件（JobTable / LogPanel.mini_jobs / HomePage.mini_jobs）。"""

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

        run_command(
            _JobsPruneCommand(self, backend),
            name='mygpr-jobs-prune',
        )


# ------------------------------------------------------------------
# Worker commands (replaces run_worker closures)
# ------------------------------------------------------------------

class _JobsPruneCommand:
    __slots__ = ("_mixins", "_backend")

    def __init__(self, mixins: Any, backend: Any) -> None:
        self._mixins = mixins
        self._backend = backend

    def execute(self) -> None:
        try:
            removed = self._backend.jobs.prune()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning('jobs prune 失败: %s', exc)
            self._mixins._log_signal.emit(f'WARNING 清理已完成任务失败: {exc}')
        else:
            self._mixins._log_signal.emit(f'INFO 已清理 {len(removed)} 个终态任务记录')
