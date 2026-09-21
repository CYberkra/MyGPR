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

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QHeaderView, QListWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    CaptionLabel, CheckBox, InfoBar, InfoBarPosition, LineEdit,
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


class DeliveryPage(QWidget):
    """成果与交付页面。"""

    spatial_requested = pyqtSignal(dict)   # {'name': str, 'line_ids': list[str]}
    report_requested = pyqtSignal(dict)    # {'package_name': str}
    backup_requested = pyqtSignal(dict)    # {'destination_dir': str, 'incremental': bool, 'retention_keep': int|None}
    restore_requested = pyqtSignal(str)    # 备份归档路径

    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = False
        self._report_dir = ''
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

        # ---------------- 卡片1：空间成果（主操作进卡头行，P2-3）
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

        # ---------------- 卡片2：项目报告（主操作进卡头行，P2-3）
        self._report_btn = PrimaryPushButton('生成报告包', self)
        report_card, report_layout = make_card(
            '项目报告', parent=self, header_action=self._report_btn)
        self._report_name_edit = LineEdit(report_card)
        self._report_name_edit.setPlaceholderText('可空，留空使用默认包名')
        report_layout.addLayout(make_form_row(
            '报告包名:', self._report_name_edit, parent=report_card,
            trailing_stretch=False))
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

        # ---------------- 卡片3：备份与恢复
        backup_card, backup_layout = make_card('备份与恢复', parent=self)
        backup_row = QHBoxLayout()
        backup_row.setSpacing(constants.CARD_SPACING)
        self._backup_btn = PushButton('备份当前项目', backup_card)
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
        options_row.addWidget(self._retention_spin)
        options_row.addWidget(CaptionLabel('个备份'))
        options_row.addStretch(1)
        backup_layout.addLayout(options_row)
        root.addWidget(backup_card)
        root.addStretch(1)

    # ============================================================ 内部接线
    def _connect_internal(self) -> None:
        self._spatial_btn.clicked.connect(self._on_spatial_clicked)
        self._report_btn.clicked.connect(self._on_report_clicked)
        self._open_dir_btn.clicked.connect(self._on_open_dir)
        self._backup_btn.clicked.connect(self._on_backup_clicked)
        self._restore_btn.clicked.connect(self._on_restore_clicked)

    # ============================================================ 公共接口（供主窗口接线）
    def set_spatial_results(self, results: list) -> None:
        """空间成果列表 → 结果表格（名称/测线数/创建时间）。"""
        results = list(results or [])
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

    def set_report_result(self, result) -> None:
        """报告包结果（dict 或 ReportPackage 对象）→ 结果区路径 + 打开目录。"""
        has_path = False
        for key, _caption in _REPORT_FIELDS:
            path = str(_get(result, key, '') or '')
            label = self._report_path_labels[key]
            label.setText(path if path else '--')
            has_path = has_path or bool(path)
        report_dir = ''
        for key in _REPORT_DIR_KEYS:
            value = str(_get(result, key, '') or '')
            if value:
                report_dir = value
                break
        if not report_dir:
            for key, _caption in _REPORT_FIELDS:
                path = str(_get(result, key, '') or '')
                if path:
                    report_dir = os.path.dirname(path)
                    break
        self._report_dir = report_dir
        self._open_dir_btn.setEnabled(bool(report_dir))
        # 未就绪 hint：有报告包后隐藏（按钮同时点亮）
        self._open_dir_hint.setVisible(not report_dir)
        if has_path:
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
        self.report_requested.emit({'package_name': self.report_name()})

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
        # P0-3：恢复会覆盖当前项目，需显式确认
        box = MessageBox(
            '从备份恢复',
            f'将用以下备份归档覆盖当前项目：\n{path}\n\n'
            '恢复会替换当前项目的测线、成果与配置，此操作不可撤销。确认继续？',
            self,
        )
        box.yesButton.setText('恢复')
        box.cancelButton.setText('取消')
        if box.exec() == QDialog.DialogCode.Accepted:
            self.restore_requested.emit(path)
