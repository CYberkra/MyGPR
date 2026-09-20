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

from PyQt6.QtCore import Qt, QSize, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QHeaderView, QListWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    CaptionLabel, CheckBox, InfoBar, InfoBarPosition, LineEdit,
    MessageBox, PrimaryPushButton, PushButton, ScrollArea, SpinBox,
    StrongBodyLabel, TransparentToolButton,
)
from qfluentwidgets import FluentIcon as FIF

from ui import constants, file_dialogs
from ui.page_scaffold import (make_card, make_form_row, rebuild_check_list,
                              style_transparent_scroll, wrap_centered)
from ui.theme_helpers import status_color
from ui.widgets import (clear_invalid, make_separator, mark_invalid,
                        validate_non_empty)
from ui.widgets.context_menus import add_action, make_menu

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
    set_current_spatial_requested = pyqtSignal(str)  # 空间成果 result_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._busy = False
        self._report_dir = ''
        self._report_paths = {}     # key -> 生成产物路径（打开文件按钮用）
        self._spatial_results = []  # 最近一次 set_spatial_results 的原始数据
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

        # ---------------- 卡片1：空间成果
        spatial_card, spatial_layout = make_card('空间成果', parent=self)
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
        # 空态引导：无测线时列表藏起、提示占位
        self._lines_empty_hint = CaptionLabel(
            '暂无测线，请先在项目管理页导入', spatial_card)
        self._lines_empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lines_empty_hint.setStyleSheet(
            'color: %s; font-size: 11px;' % status_color('disabled'))
        spatial_layout.addWidget(self._lines_empty_hint)
        self._lines_list.setVisible(False)

        self._spatial_btn = PrimaryPushButton('生成空间成果', spatial_card)
        spatial_btn_row = QHBoxLayout()
        spatial_btn_row.addStretch(1)
        spatial_btn_row.addWidget(self._spatial_btn)
        spatial_layout.addLayout(spatial_btn_row)
        spatial_layout.addWidget(make_separator())

        self._spatial_table = QTableWidget(0, 3, spatial_card)
        self._spatial_table.setHorizontalHeaderLabels(
            ['名称', '测线数', '创建时间'])
        self._spatial_table.verticalHeader().setVisible(False)
        self._spatial_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._spatial_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._spatial_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection)
        self._spatial_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._spatial_table.setMinimumHeight(140)
        # 双击/Enter = 设为当前成果；右键 = 更多操作（复制名称/ID）
        self._spatial_table.itemDoubleClicked.connect(
            lambda _item: self._activate_spatial_row(
                self._spatial_table.currentRow()))
        self._spatial_table.activated.connect(
            lambda _index: self._activate_spatial_row(
                self._spatial_table.currentRow()))
        self._spatial_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._spatial_table.customContextMenuRequested.connect(
            self._on_spatial_context_menu)
        spatial_layout.addWidget(self._spatial_table)
        root.addWidget(spatial_card)

        # ---------------- 卡片2：项目报告
        report_card, report_layout = make_card('项目报告', parent=self)
        self._report_name_edit = LineEdit(report_card)
        self._report_name_edit.setPlaceholderText('可空，留空使用默认包名')
        report_layout.addLayout(make_form_row(
            '报告包名:', self._report_name_edit, parent=report_card,
            trailing_stretch=False))
        self._report_btn = PrimaryPushButton('生成报告包', report_card)
        report_btn_row = QHBoxLayout()
        report_btn_row.addStretch(1)
        report_btn_row.addWidget(self._report_btn)
        report_layout.addLayout(report_btn_row)
        report_layout.addWidget(make_separator())

        self._report_path_labels = {}
        self._report_open_btns = {}
        for key, caption in _REPORT_FIELDS:
            value = CaptionLabel('--', report_card)
            value.setStyleSheet(
                'color: %s; font-size: 11px;' % status_color('disabled'))
            open_btn = TransparentToolButton(FIF.DOCUMENT, report_card)
            open_btn.setIconSize(QSize(14, 14))
            open_btn.setFixedSize(24, 24)
            open_btn.setToolTip('打开该文件')
            open_btn.setEnabled(False)
            open_btn.clicked.connect(
                lambda _checked=False, k=key: self._on_open_file(k))
            report_layout.addLayout(make_form_row(
                caption, value, open_btn, parent=report_card,
                trailing_stretch=False))
            # 路径标签右键 = 打开文件 / 打开所在目录 / 复制路径
            value.setContextMenuPolicy(
                Qt.ContextMenuPolicy.CustomContextMenu)
            value.customContextMenuRequested.connect(
                lambda _pos, k=key: self._on_report_label_context_menu(k, _pos))
            self._report_path_labels[key] = value
            self._report_open_btns[key] = open_btn
        open_row = QHBoxLayout()
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
        """空间成果列表 → 结果表格（名称/测线数/创建时间）；原始数据留存供行操作。"""
        self._spatial_results = list(results or [])
        self._spatial_table.setRowCount(len(self._spatial_results))
        for row, item in enumerate(self._spatial_results):
            name = str(_get(item, 'name', '') or _get(item, 'title', ''))
            line_ids = _get(item, 'line_ids', None)
            if line_ids is None:
                n_lines = _get(item, 'line_count', 0)
            else:
                n_lines = len(line_ids)
            created = str(_get(item, 'created_at', '')
                          or _get(item, 'created', '') or '--')
            name_item = QTableWidgetItem(name)
            name_item.setToolTip('双击设为当前成果，右键更多操作')
            self._spatial_table.setItem(row, 0, name_item)
            self._spatial_table.setItem(row, 1, QTableWidgetItem(str(n_lines)))
            self._spatial_table.setItem(row, 2, QTableWidgetItem(created))

    def set_report_result(self, result) -> None:
        """报告包结果（dict 或 ReportPackage 对象）→ 结果区路径 + 打开目录。"""
        has_path = False
        for key, _caption in _REPORT_FIELDS:
            path = str(_get(result, key, '') or '')
            label = self._report_path_labels[key]
            label.setText(path if path else '--')
            label.setToolTip(
                '%s\n右键：打开文件 / 复制路径' % path if path else '')
            self._report_paths[key] = path
            self._report_open_btns[key].setEnabled(bool(path))
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

    def _on_open_file(self, key: str) -> None:
        """单个产物路径的"打开文件"按钮。"""
        path = self._report_paths.get(key, '')
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _open_containing_dir(self, key: str) -> None:
        """在文件管理器中打开产物所在目录。"""
        path = self._report_paths.get(key, '')
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(
                os.path.dirname(path) or '.'))

    def _on_report_label_context_menu(self, key: str, pos) -> None:
        """报告产物路径标签右键；无产物时不弹全禁用菜单。"""
        if not self._report_paths.get(key, ''):
            return
        menu = self._build_report_menu(key)
        menu.exec(self._report_path_labels[key].mapToGlobal(pos))

    def _build_report_menu(self, key: str):
        """构造产物路径右键菜单（与 exec 分离，便于测试检查动作）。"""
        path = self._report_paths.get(key, '')
        menu = make_menu(parent=self._report_path_labels[key])
        add_action(menu, FIF.DOCUMENT, '打开文件',
                   lambda: self._on_open_file(key),
                   enabled=bool(path))
        add_action(menu, FIF.FOLDER, '打开所在目录',
                   lambda: self._open_containing_dir(key),
                   enabled=bool(path))
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制路径',
                   lambda: QApplication.clipboard().setText(path),
                   enabled=bool(path))
        return menu

    # ---------------- 空间成果表：行操作（双击/Enter = 设为当前成果）
    def _spatial_row_data(self, row: int):
        if 0 <= row < len(self._spatial_results):
            return self._spatial_results[row]
        return None

    def _activate_spatial_row(self, row: int) -> None:
        item = self._spatial_row_data(row)
        if item is None or self._busy:
            return
        result_id = str(_get(item, 'result_id', '') or _get(item, 'id', ''))
        if result_id:
            self.set_current_spatial_requested.emit(result_id)

    def _on_spatial_context_menu(self, pos) -> None:
        """空间成果行右键：设为当前成果 / 复制名称 / 复制成果 ID。"""
        row = self._spatial_table.rowAt(pos.y())
        item = self._spatial_row_data(row)
        if item is None:
            return
        menu = self._build_spatial_menu(item, row)
        menu.exec(self._spatial_table.viewport().mapToGlobal(pos))

    def _build_spatial_menu(self, item, row: int):
        """构造空间成果右键菜单（与 exec 分离，便于测试检查动作）。"""
        name = str(_get(item, 'name', '') or _get(item, 'title', ''))
        result_id = str(_get(item, 'result_id', '') or _get(item, 'id', ''))
        menu = make_menu(parent=self._spatial_table)
        add_action(menu, FIF.ACCEPT, '设为当前成果',
                   lambda: self._activate_spatial_row(row),
                   enabled=bool(result_id) and not self._busy)
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制名称',
                   lambda: QApplication.clipboard().setText(name),
                   enabled=bool(name))
        add_action(menu, FIF.COPY, '复制成果 ID',
                   lambda: QApplication.clipboard().setText(result_id),
                   enabled=bool(result_id))
        return menu

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
