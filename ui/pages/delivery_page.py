# -*- coding: utf-8 -*-
"""DeliveryPage — 成果与交付（SPEC §6.7）。

- 卡片1"空间成果"：名称 LineEdit + 测线多选 + PrimaryPushButton('生成空间成果')
  + 结果表格
- 卡片2"项目报告"：包名 LineEdit(可空) + PrimaryPushButton('生成报告包')
  + 结果区（PDF/HTML/Excel/ZIP 路径 + PushButton('打开目录')）
- 卡片3"备份与恢复"：PushButton('备份当前项目') + PushButton('从备份恢复')
  （目录/文件选择对话框）

页面纯展示 + 发信号，不直接调 controller/backend。
"""

import os
import json
from collections.abc import Mapping

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QHeaderView, QListWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    CaptionLabel, CheckBox, ComboBox, InfoBar, InfoBarPosition, LineEdit,
    MessageBox, PrimaryPushButton, PushButton, ScrollArea, SpinBox,
    StrongBodyLabel,
)
from qfluentwidgets import FluentIcon as FIF

from ui import constants, file_dialogs
from ui.page_scaffold import (make_card, make_form_row, make_hint,
                              rebuild_check_list, style_transparent_scroll,
                              wrap_centered)
from ui.widgets import EmptyStateOverlay
from ui.widgets import (clear_invalid, make_separator, mark_invalid,
                        validate_non_empty)
from ui.widgets.object_context_menu import ObjectMenuBinding, object_action, object_menu, show_details

# 报告结果字段（鸭子类型：dict 键或对象属性）
_REPORT_FIELDS = (
    ('pdf_path', 'PDF:'),
    ('html_path', 'HTML:'),
    ('xlsx_path', 'Excel:'),
    ('delivery_zip_path', 'ZIP:'),
)
_REPORT_DIR_KEYS = ('package_dir', 'output_dir', 'root_dir', 'dir')


def _get(obj, key, default=''):
    """鸭子类型取值：dict 键优先，其次对象属性。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _line_display(line) -> str:
    """测线显示文本：'名称 (线号)'，名称缺失或与线号相同时只显示线号。"""
    line_id = str(_get(line, 'line_id', '') or _get(line, 'id', ''))
    name = str(_get(line, 'name', '') or line_id)
    return '%s (%s)' % (name, line_id) if name != line_id else line_id


def _report_output_path(result, key: str) -> str:
    """Resolve project-relative backend paths while keeping legacy absolutes."""
    path = str(_get(result, key, '') or '')
    metadata = _get(result, 'metadata', {})
    root = str(metadata.get('project_root') or '') if isinstance(metadata, Mapping) else ''
    if path and root and not os.path.isabs(path):
        return os.path.normpath(os.path.join(root, path))
    return path


class DeliveryPage(QWidget):
    """成果与交付页面。"""

    spatial_requested = pyqtSignal(dict)   # {'name': str, 'line_ids': list[str]}
    spatial_export_requested = pyqtSignal(dict)
    spatial_delete_requested = pyqtSignal(str)
    spatial_current_requested = pyqtSignal(str)
    report_requested = pyqtSignal(dict)    # {'package_name': str}
    backup_requested = pyqtSignal(dict)    # {'destination_dir': str, 'incremental': bool, 'retention_keep': int|None}
    restore_requested = pyqtSignal(dict)   # archive_path, destination_root, project_dir_name, open_after_restore

    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = False
        self._report_dir = ''
        self._spatial_results = []
        self._spatial_revision = 0
        self._project_root = ''
        self._artifacts = []
        self._report_packages = []
        self._line_records = []
        self._build_ui()
        self._connect_internal()

    # ============================================================ UI 构建
    def _build_ui(self) -> None:
        scroll = ScrollArea(self)
        style_transparent_scroll(scroll)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget(scroll)
        content.setObjectName('deliveryScrollContent')
        content.setStyleSheet(
            'QWidget#deliveryScrollContent { background-color: transparent; }')
        root = QVBoxLayout(content)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)
        scroll.setWidget(wrap_centered(content, constants.FORM_COLUMN_MAX_WIDTH))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # 三张卡各成一个方法：每张是一组独立交付能力（空间成果 / 报告包 /
        # 备份恢复），各自有主操作、未就绪门控与空态引导，混在一个方法里
        # 会让「这张卡有哪些控件」只能靠通读 116 行才看得清。
        self._build_spatial_card(root)
        self._build_report_card(root)
        self._build_backup_card(root)
        root.addStretch(1)

    def _build_spatial_card(self, root: QVBoxLayout) -> None:
        """卡片1：空间成果（主操作进卡头行，P2-3）。"""
        self._spatial_btn = PrimaryPushButton('生成空间成果', self)
        # 未就绪门控：无测线时禁用，set_lines 导入测线后点亮
        self._spatial_btn.setEnabled(False)
        self._spatial_btn.setToolTip('导入测线后可用')
        spatial_card, spatial_layout = make_card(
            '空间成果', parent=self, header_action=self._spatial_btn)
        self._spatial_name_edit = LineEdit(spatial_card)
        self._spatial_name_edit.setPlaceholderText('例如：全场剖面拼接成果')
        spatial_layout.addLayout(make_form_row(
            '成果名称:', self._spatial_name_edit, parent=spatial_card,
            trailing_stretch=False))

        lines_label = StrongBodyLabel('选择测线（可多选）：', spatial_card)
        spatial_layout.addWidget(lines_label)
        self._lines_list = QListWidget(spatial_card)
        self._lines_list.setMinimumHeight(120)
        spatial_layout.addWidget(self._lines_list)
        # 空态引导：无测线时列表藏起、提示占位（同时说明点亮前置条件）
        self._lines_empty_hint = make_hint(
            '暂无测线——在项目管理页导入测线后，这里可勾选并生成空间成果',
            parent=spatial_card)
        self._lines_empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spatial_layout.addWidget(self._lines_empty_hint)
        self._lines_list.setVisible(False)

        spatial_layout.addWidget(make_separator())

        self._spatial_table = QTableWidget(0, 3, spatial_card)
        self._spatial_table.setHorizontalHeaderLabels(
            ['名称', '测线数', '创建时间'])
        self._spatial_table.verticalHeader().setVisible(False)
        self._spatial_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._spatial_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._spatial_table.setMinimumHeight(140)
        spatial_layout.addWidget(self._spatial_table)
        # 空态引导浮层（评审 P0-1）：无成果时盖在表格上
        self._spatial_empty = EmptyStateOverlay(
            self._spatial_table, icon=FIF.SEND, title='暂无空间成果',
            hint='勾选测线并生成空间成果后，结果会列在这里')
        root.addWidget(spatial_card)

    def _build_report_card(self, root: QVBoxLayout) -> None:
        """卡片2：项目报告（主操作进卡头行，P2-3）。"""
        self._report_btn = PrimaryPushButton('生成报告包', self)
        report_card, report_layout = make_card(
            '项目报告', parent=self, header_action=self._report_btn)
        self._report_name_edit = LineEdit(report_card)
        self._report_name_edit.setPlaceholderText('可空，留空使用默认包名')
        report_layout.addLayout(make_form_row(
            '报告包名:', self._report_name_edit, parent=report_card,
            trailing_stretch=False))
        self._report_history_combo = ComboBox(report_card)
        self._report_history_combo.setMinimumWidth(240)
        self._report_history_combo.currentIndexChanged.connect(self._on_report_history_selected)
        report_layout.addLayout(make_form_row('历史报告:', self._report_history_combo, parent=report_card))
        report_layout.addWidget(make_separator())

        self._report_path_labels = {}
        for key, caption in _REPORT_FIELDS:
            value = make_hint('--', parent=report_card)
            report_layout.addLayout(make_form_row(
                caption, value, parent=report_card, trailing_stretch=False))
            self._report_path_labels[key] = value
        # 「打开目录」未就绪（无报告包）时禁用 + 前置条件 hint（P2-3）
        open_row = QHBoxLayout()
        open_row.setSpacing(constants.CARD_SPACING)
        self._open_dir_hint = make_hint('生成报告包后，可在此打开输出目录',
                                        parent=report_card)
        open_row.addWidget(self._open_dir_hint)
        open_row.addStretch(1)
        self._open_dir_btn = PushButton('打开目录', report_card)
        self._open_dir_btn.setEnabled(False)
        open_row.addWidget(self._open_dir_btn)
        report_layout.addLayout(open_row)
        root.addWidget(report_card)

    def _build_backup_card(self, root: QVBoxLayout) -> None:
        """卡片3：备份与恢复（无主操作按钮：备份/恢复并列，均为低频操作）。"""
        backup_card, backup_layout = make_card('备份与恢复', parent=self)
        self._restore_name_edit = LineEdit(backup_card)
        self._restore_name_edit.setPlaceholderText('恢复后的新项目目录名（留空自动命名，不覆盖当前项目）')
        backup_layout.addLayout(make_form_row('恢复目录名', self._restore_name_edit, parent=backup_card))
        backup_row = QHBoxLayout()
        backup_row.setSpacing(constants.CARD_SPACING)
        self._backup_btn = PushButton('备份当前项目', backup_card)
        self._backup_btn.setToolTip(
            '创建备份时会完整恢复并校验项目，需要足够容纳完整项目的临时磁盘空间。')
        self._restore_btn = PushButton('从备份恢复', backup_card)
        backup_row.addWidget(self._backup_btn)
        backup_row.addWidget(self._restore_btn)
        backup_row.addStretch(1)
        backup_layout.addLayout(backup_row)
        options_row = QHBoxLayout()
        options_row.setSpacing(constants.CARD_SPACING)
        self._incremental_check = CheckBox('增量备份（仅打包相对上次备份的变化）', backup_card)
        self._incremental_check.setChecked(True)
        options_row.addWidget(self._incremental_check)
        options_row.addWidget(CaptionLabel('保留最近'))
        self._retention_spin = SpinBox(backup_card)
        self._retention_spin.setRange(1, 100)
        self._retention_spin.setValue(10)
        self._retention_spin.setToolTip(
            '保留最近指定数量的恢复点及其必需增量祖先；实际 ZIP 档案数量可能超过该数量。')
        options_row.addWidget(self._retention_spin)
        options_row.addWidget(CaptionLabel('个恢复点'))
        options_row.addStretch(1)
        backup_layout.addLayout(options_row)
        root.addWidget(backup_card)

    # ============================================================ 内部接线
    def _connect_internal(self) -> None:
        self._spatial_context = ObjectMenuBinding(self, self._spatial_table, self._spatial_result_menu)
        self._report_contexts = [ObjectMenuBinding(self, label,
            lambda _pos, key=key: self._report_path_menu(key))
            for key, label in self._report_path_labels.items()]
        self._spatial_btn.clicked.connect(self._on_spatial_clicked)
        self._report_btn.clicked.connect(self._on_report_clicked)
        self._open_dir_btn.clicked.connect(self._on_open_dir)
        self._backup_btn.clicked.connect(self._on_backup_clicked)
        self._restore_btn.clicked.connect(self._on_restore_clicked)

    # ============================================================ 公共接口（供主窗口接线）
    def set_spatial_results(self, results: list) -> None:
        """空间成果列表 → 结果表格（名称/测线数/创建时间）。"""
        results = list(results or [])
        self._spatial_context.close()
        self._spatial_revision += 1
        self._spatial_results = results
        self._spatial_empty.setVisible(not results)
        self._spatial_table.setRowCount(len(results))
        for row, item in enumerate(results):
            name = str(_get(item, 'name', '') or _get(item, 'title', ''))
            line_ids = _get(item, 'line_ids', None)
            if line_ids is None:
                n_lines = _get(item, 'line_count', 0)
            else:
                n_lines = len(line_ids)
            created = str(_get(item, 'created_at', '')
                          or _get(item, 'created', '') or '--')
            self._spatial_table.setItem(row, 0, QTableWidgetItem(name))
            self._spatial_table.setItem(row, 1, QTableWidgetItem(str(n_lines)))
            self._spatial_table.setItem(row, 2, QTableWidgetItem(created))
            self._spatial_table.item(row, 0).setData(
                Qt.ItemDataRole.UserRole, str(_get(item, 'result_id', '') or _get(item, 'artifact_id', '')))

    def _spatial_result_menu(self, pos):
        menu = object_menu(self)
        row = self._spatial_table.indexAt(pos).row()
        if not 0 <= row < len(self._spatial_results):
            object_action(menu, 'copy-count', '复制成果数量',
                          lambda: QApplication.clipboard().setText(str(len(self._spatial_results))))
            return menu
        result, revision = self._spatial_results[row], self._spatial_revision
        result_id = str(_get(result, 'result_id', '') or _get(result, 'artifact_id', ''))

        def valid():
            return (revision == self._spatial_revision and row < len(self._spatial_results)
                    and self._spatial_results[row] is result)

        metadata = {}
        for key in ('result_id', 'name', 'revision', 'status', 'line_ids', 'created_at',
                    'coordinate_system', 'vertical_datum', 'stale', 'files'):
            value = _get(result, key, '')
            metadata[key] = dict(value) if isinstance(value, Mapping) else value
        text = json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
        object_action(menu, 'result-details', '参数与来源…',
                      lambda: show_details(self, '空间成果详情', text) if valid() else None)
        object_action(menu, 'copy-id', '复制成果标识',
                      lambda: QApplication.clipboard().setText(result_id) if valid() else None,
                      enabled=bool(result_id), reason='记录没有稳定成果标识')
        object_action(menu, 'copy-name', '复制成果名称',
                      lambda: QApplication.clipboard().setText(str(_get(result, 'name', ''))) if valid() else None)
        menu.addSeparator()
        object_action(menu, 'set-current-result', '设为当前空间成果',
                      lambda: self.spatial_current_requested.emit(result_id) if valid() else None,
                      enabled=bool(result_id))
        object_action(menu, 'open-result', '打开成果目录',
                      lambda: self._open_spatial_result(result_id) if valid() else None,
                      enabled=bool(result_id and self._project_root))
        object_action(menu, 'export-result', '导出成果…',
                      lambda: self._export_spatial_result(result_id) if valid() else None,
                      enabled=bool(result_id))
        object_action(menu, 'delete-result', '移入回收站…',
                      lambda: self._delete_spatial_result(result_id) if valid() else None,
                      enabled=bool(result_id and not self._busy))
        return menu

    def set_project_root(self, root):
        self._project_root = os.path.normpath(str(root or '')) if root else ''

    def _open_spatial_result(self, result_id):
        path = os.path.join(self._project_root, 'spatial', 'results', result_id)
        if os.path.isdir(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _export_spatial_result(self, result_id):
        from ui.dialogs import choose_spatial_export_path
        path, format_name = choose_spatial_export_path(self, result_id)
        if path:
            self.spatial_export_requested.emit({
                'result_id': result_id,
                'destination': path,
                'format_name': format_name,
            })

    def _delete_spatial_result(self, result_id):
        from ui.dialogs import ask_spatial_delete
        if ask_spatial_delete(self, result_id):
            self.spatial_delete_requested.emit(result_id)

    def _report_path_menu(self, key):
        menu = object_menu(self)
        path = self._report_path_labels[key].text()
        report_dir = self._report_dir
        def valid():
            return self._report_path_labels[key].text() == path and self._report_dir == report_dir
        object_action(menu, 'copy-path', '复制报告文件路径',
                      lambda: QApplication.clipboard().setText(path) if valid() else None,
                      enabled=bool(path and path != '--'), reason='报告文件尚未生成')
        object_action(menu, 'open-report-directory', '打开报告包目录',
                      lambda: self._on_open_dir() if valid() else None,
                      enabled=bool(report_dir), reason='报告包目录尚未生成')
        return menu

    def set_artifacts(self, artifacts):
        self._artifacts = list(artifacts or [])

    def set_report_packages(self, packages):
        current = self._report_history_combo.currentData()
        self._report_packages = list(packages or [])
        self._report_history_combo.blockSignals(True)
        self._report_history_combo.clear()
        for index, package in enumerate(self._report_packages):
            directory = str(_get(package, 'package_dir', ''))
            label = os.path.basename(directory.rstrip('/\\')) or str(_get(package, 'package_name', '')) or f'报告 {index + 1}'
            self._report_history_combo.addItem(label, userData=index)
        restored = self._report_history_combo.findData(current)
        self._report_history_combo.setCurrentIndex(restored if restored >= 0 else 0)
        self._report_history_combo.blockSignals(False)
        if self._report_packages and self._report_history_combo.currentIndex() >= 0:
            self._on_report_history_selected(self._report_history_combo.currentIndex())

    def _on_report_history_selected(self, index):
        if 0 <= index < len(self._report_packages):
            self.set_report_result(self._report_packages[index], notify=False)

    def set_report_result(self, result, *, notify=True) -> None:
        """报告包结果（dict 或 ReportPackage 对象）→ 结果区路径 + 打开目录。"""
        for binding in self._report_contexts:
            binding.close()
        has_path = False
        for key, _caption in _REPORT_FIELDS:
            path = _report_output_path(result, key)
            label = self._report_path_labels[key]
            label.setText(path if path else '--')
            has_path = has_path or bool(path)
        report_dir = ''
        for key in _REPORT_DIR_KEYS:
            value = _report_output_path(result, key)
            if value:
                report_dir = value
                break
        if not report_dir:
            for key, _caption in _REPORT_FIELDS:
                path = _report_output_path(result, key)
                if path:
                    report_dir = os.path.dirname(path)
                    break
        self._report_dir = report_dir
        self._open_dir_btn.setEnabled(bool(report_dir))
        # 未就绪 hint：有报告包后隐藏（按钮同时点亮）
        self._open_dir_hint.setVisible(not report_dir)
        if has_path and notify:
            InfoBar.success(title='项目报告', content='报告包已生成',
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=2000,
                            parent=self)

    def set_busy(self, busy: bool) -> None:
        """忙态：禁用全部操作按钮。"""
        self._busy = bool(busy)
        for btn in (self._report_btn, self._backup_btn, self._restore_btn):
            btn.setEnabled(not self._busy)
        self._update_spatial_btn_enabled()

    def _update_spatial_btn_enabled(self) -> None:
        """「生成空间成果」双门控：非忙态且有可选测线（P2-3 未就绪禁用）。"""
        self._spatial_btn.setEnabled(
            not self._busy and self._lines_list.count() > 0)

    def selected_line_ids(self) -> list:
        """当前勾选的测线 id 列表。"""
        ids = []
        for i in range(self._lines_list.count()):
            item = self._lines_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                ids.append(item.data(Qt.ItemDataRole.UserRole))
        return ids

    def spatial_name(self) -> str:
        return self._spatial_name_edit.text().strip()

    def report_name(self) -> str:
        return self._report_name_edit.text().strip()

    def set_lines(self, lines: list) -> None:
        """可选测线列表（主窗口注入；dict/对象鸭子类型，取 line_id 与 name）。

        重建列表时保持已有勾选状态（rebuild_check_list 同 spatial_page
        模式），避免刷新测线集合后丢掉用户已勾选的测线；新出现的测线维持
        默认不勾选。
        """
        self._line_records = list(lines or [])
        rebuild_check_list(
            self._lines_list, lines or [],
            key_fn=lambda line: str(
                _get(line, 'line_id', '') or _get(line, 'id', '')),
            text_fn=_line_display,
            default_checked=False)
        # 空态显隐：有测线显示勾选列表，无测线显示引导文案
        has_lines = self._lines_list.count() > 0
        self._lines_list.setVisible(has_lines)
        self._lines_empty_hint.setVisible(not has_lines)
        self._update_spatial_btn_enabled()

    # ============================================================ 内部逻辑
    def _on_spatial_clicked(self) -> None:
        if self._busy:
            return
        ok, msg = validate_non_empty(self._spatial_name_edit.text(),
                                     '成果名称')
        if not ok:
            mark_invalid(self._spatial_name_edit, msg)
            InfoBar.warning(title='空间成果', content=msg,
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=3000,
                            parent=self)
            return
        clear_invalid(self._spatial_name_edit)
        line_ids = self.selected_line_ids()
        if not line_ids:
            InfoBar.warning(title='空间成果', content='请至少勾选一条测线',
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=3000,
                            parent=self)
            return
        self.spatial_requested.emit(
            {'name': self.spatial_name(), 'line_ids': line_ids})

    def _on_report_clicked(self) -> None:
        if self._busy:
            return
        from ui.pages.report_config import ReportConfigDialog
        dialog = ReportConfigDialog(
            self._line_records, self._artifacts, self._spatial_results, self)
        dialog._package_name.setText(self.report_name())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            payload = dialog.payload()
            self._report_name_edit.setText(payload['package_name'])
            self.report_requested.emit(payload)

    def _on_open_dir(self) -> None:
        if self._report_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._report_dir))

    def _on_backup_clicked(self) -> None:
        if self._busy:
            return
        dest = file_dialogs.getExistingDirectory(
            self, '选择备份目录', constants.DEFAULT_PROJECT_ROOT)
        if dest:
            self.backup_requested.emit({
                'destination_dir': dest,
                'incremental': self._incremental_check.isChecked(),
                'retention_keep': int(self._retention_spin.value()),
            })

    def _on_restore_clicked(self) -> None:
        if self._busy:
            return
        path, _selected_filter = file_dialogs.getOpenFileName(
            self, '选择备份归档', constants.DEFAULT_PROJECT_ROOT,
            '备份归档 (*.zip *.tar *.tar.gz *.tgz);;所有文件 (*)')
        if not path:
            return
        destination = file_dialogs.getExistingDirectory(
            self, '选择恢复后的新项目父目录（不会覆盖当前项目）', constants.DEFAULT_PROJECT_ROOT)
        if not destination:
            return
        project_dir_name = self._restore_name_edit.text().strip()
        box = MessageBox(
            '从备份恢复为新项目',
            f'备份归档：\n{path}\n\n恢复位置：\n{destination}\n\n'
            f'项目目录名：{project_dir_name or "自动命名"}\n'
            '此操作恢复为新项目，不覆盖当前项目。确认继续？',
            self,
        )
        box.yesButton.setText('恢复为新项目')
        box.cancelButton.setText('取消')
        if box.exec() == QDialog.DialogCode.Accepted:
            self.restore_requested.emit({
                'archive_path': path,
                'destination_root': destination,
                'project_dir_name': project_dir_name,
                'open_after_restore': True,
            })
