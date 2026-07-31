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
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QHeaderView, QListWidget,
    QListWidgetItem, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    CaptionLabel, CardWidget, InfoBar, InfoBarPosition, LineEdit,
    PrimaryPushButton, PushButton, ScrollArea, SubtitleLabel,
)

from ui import constants
from ui.widgets import clear_invalid, mark_invalid, validate_non_empty

# 报告结果字段（鸭子类型：dict 键或对象属性）
_REPORT_FIELDS = (
    ('pdf_path', 'PDF:'),
    ('html_path', 'HTML:'),
    ('excel_path', 'Excel:'),
    ('zip_path', 'ZIP:'),
)
_REPORT_DIR_KEYS = ('package_dir', 'output_dir', 'root_dir', 'dir')


def _page_title(text: str) -> SubtitleLabel:
    """页面标题：SubtitleLabel 微软雅黑 12pt Bold 居中（SPEC §1）。"""
    label = SubtitleLabel(text)
    label.setFont(QFont(constants.FONT_FAMILY, 12, QFont.Weight.Bold))
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


def _card_title(text: str) -> SubtitleLabel:
    """卡片标题：SubtitleLabel 微软雅黑 10pt Bold（SPEC §1）。"""
    label = SubtitleLabel(text)
    label.setFont(QFont(constants.FONT_FAMILY, 10, QFont.Weight.Bold))
    return label


def _create_separator() -> QFrame:
    """分隔线工厂：QFrame.HLine + Sunken + color:#e0e0e0（SPEC §1）。"""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet('color: #e0e0e0;')
    return line


def _get(obj, key, default=''):
    """鸭子类型取值：dict 键优先，其次对象属性。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class DeliveryPage(QWidget):
    """成果与交付页面。"""

    spatial_requested = pyqtSignal(dict)   # {'name': str, 'line_ids': list[str]}
    report_requested = pyqtSignal(dict)    # {'package_name': str}
    backup_requested = pyqtSignal(str)     # 备份目标目录
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
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            'QScrollArea { background-color: transparent; border: none; }')
        content = QWidget(scroll)
        content.setObjectName('deliveryScrollContent')
        content.setStyleSheet(
            'QWidget#deliveryScrollContent { background-color: transparent; }')
        root = QVBoxLayout(content)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        root.addWidget(_page_title('成果与交付'))

        # ---------------- 卡片1：空间成果
        spatial_card, spatial_layout = self._make_card('空间成果')
        name_row = QHBoxLayout()
        name_row.setSpacing(constants.CARD_SPACING)
        name_label = CaptionLabel('成果名称:', spatial_card)
        name_label.setMinimumWidth(100)
        name_row.addWidget(name_label)
        self._spatial_name_edit = LineEdit(spatial_card)
        self._spatial_name_edit.setPlaceholderText('例如: 全场剖面拼接成果')
        name_row.addWidget(self._spatial_name_edit, 1)
        spatial_layout.addLayout(name_row)

        lines_label = CaptionLabel('选择测线（可多选）:', spatial_card)
        lines_label.setStyleSheet(
            'font-family: "Microsoft YaHei"; font-size: 9pt; font-weight: bold;')
        spatial_layout.addWidget(lines_label)
        self._lines_list = QListWidget(spatial_card)
        self._lines_list.setMinimumHeight(120)
        self._lines_list.setMaximumHeight(180)
        spatial_layout.addWidget(self._lines_list)

        self._spatial_btn = PrimaryPushButton('生成空间成果', spatial_card)
        spatial_layout.addWidget(self._spatial_btn)
        spatial_layout.addWidget(_create_separator())

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
        root.addWidget(spatial_card)

        # ---------------- 卡片2：项目报告
        report_card, report_layout = self._make_card('项目报告')
        pkg_row = QHBoxLayout()
        pkg_row.setSpacing(constants.CARD_SPACING)
        pkg_label = CaptionLabel('报告包名:', report_card)
        pkg_label.setMinimumWidth(100)
        pkg_row.addWidget(pkg_label)
        self._report_name_edit = LineEdit(report_card)
        self._report_name_edit.setPlaceholderText('可空，留空使用默认包名')
        pkg_row.addWidget(self._report_name_edit, 1)
        report_layout.addLayout(pkg_row)
        self._report_btn = PrimaryPushButton('生成报告包', report_card)
        report_layout.addWidget(self._report_btn)
        report_layout.addWidget(_create_separator())

        self._report_path_labels = {}
        for key, caption in _REPORT_FIELDS:
            row = QHBoxLayout()
            row.setSpacing(constants.CARD_SPACING)
            cap = CaptionLabel(caption, report_card)
            cap.setMinimumWidth(100)
            row.addWidget(cap)
            value = CaptionLabel('--', report_card)
            value.setStyleSheet(
                'color: %s; font-size: 11px;' % constants.COLOR_DISABLED)
            row.addWidget(value, 1)
            report_layout.addLayout(row)
            self._report_path_labels[key] = value
        open_row = QHBoxLayout()
        open_row.addStretch(1)
        self._open_dir_btn = PushButton('打开目录', report_card)
        self._open_dir_btn.setEnabled(False)
        open_row.addWidget(self._open_dir_btn)
        report_layout.addLayout(open_row)
        root.addWidget(report_card)

        # ---------------- 卡片3：备份与恢复
        backup_card, backup_layout = self._make_card('备份与恢复')
        backup_row = QHBoxLayout()
        backup_row.setSpacing(constants.CARD_SPACING)
        self._backup_btn = PushButton('备份当前项目', backup_card)
        self._restore_btn = PushButton('从备份恢复', backup_card)
        backup_row.addWidget(self._backup_btn)
        backup_row.addWidget(self._restore_btn)
        backup_row.addStretch(1)
        backup_layout.addLayout(backup_row)
        root.addWidget(backup_card)
        root.addStretch(1)

    def _make_card(self, title: str) -> tuple:
        """卡片范式：CardWidget + QVBoxLayout spacing=10 margins=(15,15,15,15)。"""
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(*constants.CARD_MARGINS)
        layout.setSpacing(constants.CARD_SPACING)
        layout.addWidget(_card_title(title))
        return card, layout

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
        if has_path:
            InfoBar.success(title='项目报告', content='报告包已生成',
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=2000,
                            parent=self)

    def set_busy(self, busy: bool) -> None:
        """忙态：禁用全部操作按钮。"""
        self._busy = bool(busy)
        for btn in (self._spatial_btn, self._report_btn,
                    self._backup_btn, self._restore_btn):
            btn.setEnabled(not self._busy)

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
        """可选测线列表（主窗口注入；dict/对象鸭子类型，取 line_id 与 name）。"""
        self._lines_list.clear()
        for line in (lines or []):
            line_id = str(_get(line, 'line_id', '') or _get(line, 'id', ''))
            name = str(_get(line, 'name', '') or line_id)
            item = QListWidgetItem('%s (%s)' % (name, line_id)
                                   if name != line_id else line_id)
            item.setData(Qt.ItemDataRole.UserRole, line_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self._lines_list.addItem(item)

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
        dest = QFileDialog.getExistingDirectory(
            self, '选择备份目录', constants.DEFAULT_PROJECT_ROOT)
        if dest:
            self.backup_requested.emit(dest)

    def _on_restore_clicked(self) -> None:
        if self._busy:
            return
        path, _selected_filter = QFileDialog.getOpenFileName(
            self, '选择备份归档', constants.DEFAULT_PROJECT_ROOT,
            '备份归档 (*.zip *.tar *.tar.gz *.tgz);;所有文件 (*)')
        if path:
            self.restore_requested.emit(path)
