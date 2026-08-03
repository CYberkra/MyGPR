"""ProjectPage — 项目管理页（SPEC §6.3）。

纯展示 + 发信号：不直接调 controller/backend。
公共接口（供主窗口接线喂数据）：
- set_project_info(summary|None)：刷新"项目信息"卡片；None → 空态（操作按钮禁用 + 顶部提示）
- set_lines(list[ProjectLine])：刷新测线表
- set_artifacts(list[ProjectArtifact])：刷新成果表
- set_preflight_result(text, ok)：预检结果区
- set_preview_bundle(bundle|None)：数据预览
- set_busy(bool)：禁用操作按钮

信号：import_requested(dict) / sync_requested(dict) /
line_selected(str) / line_process_requested(str) /
artifact_preview_requested(str, str)。

右键菜单（RoundMenu）：测线表 = 处理该测线（跳转处理页，双击同效）/
复制数据文件路径 / 打开数据所在文件夹 /
复制测线号（路径经 set_source_path_resolver 注入的回调查询）；
成果表 = 预览所选（双击同效）。

import_requested payload：{'preflight': bool, 'source', 'line_id', 'name', 'dielectric'}
（'预检' 按钮 preflight=True → ProjectController.preflight_import；
 '导入' 按钮 preflight=False → ProjectController.import_line）
sync_requested payload：{'line_id', 'paths': {'rtk', 'imu', 'altimeter', 'trace_timestamps'}}
"""

import os

from PyQt6.QtCore import Qt, QSettings, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    BodyLabel, CaptionLabel, CardWidget, DoubleSpinBox, InfoBar,
    InfoBarPosition, LineEdit, MessageBox, PrimaryPushButton, PushButton,
    ScrollArea, SubtitleLabel,
)
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.widgets import BScanView, clear_invalid, mark_invalid, validate_non_empty
from ui.widgets.context_menus import add_action, make_menu

# 导入文件对话框过滤器：通过 DesktopBackendFacade 获取（SPEC §6.3）
try:
    from ui.desktop_backend_facade import supported_file_dialog_filter
    _GPR_FILE_FILTER = supported_file_dialog_filter()
except Exception:  # noqa: BLE001 - 后端模块不可用时回退
    _GPR_FILE_FILTER = ('GPR 数据文件 (*.csv *.txt *.dat *.sgy *.segy *.rd3 '
                        '*.rd7 *.out *.npy *.npz);;所有文件 (*)')

_SENSOR_FILE_FILTER = '数据文件 (*.csv *.txt *.dat *.json *.gpx);;所有文件 (*)'

# 传感器同步四个文件行：键 → 行标签
_SENSOR_ROWS = (
    ('rtk', 'RTK 文件:'),
    ('imu', 'IMU 文件:'),
    ('altimeter', '高度计文件:'),
    ('trace_timestamps', '道时间戳文件:'),
)

# 项目信息只读字段：标签 → summary 属性名（ProjectSummary 缺失字段显示 '--'）
_INFO_FIELDS = (
    ('名称:', 'name'),
    ('编号:', 'project_no'),
    ('位置:', 'location'),
    ('操作员:', 'operator'),
    ('设备型号:', 'device_model'),
    ('坐标系:', 'coordinate_system'),
    ('高程基准:', 'vertical_datum'),
)


# ------------------------------------------------------------ 小工厂（卡片范式逐字 SPEC §1）
def _create_separator() -> QFrame:
    """分隔线工厂：QFrame.HLine + Sunken + 'color: #e0e0e0;'。"""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet('color: #e0e0e0;')
    return line


def _create_card(title: str) -> tuple:
    """卡片范式：CardWidget + QVBoxLayout(spacing=10, margins=15)，首行 SubtitleLabel 10pt Bold。"""
    card = CardWidget()
    layout = QVBoxLayout(card)
    layout.setContentsMargins(*constants.CARD_MARGINS)
    layout.setSpacing(constants.CARD_SPACING)
    header = SubtitleLabel(title, card)
    header.setFont(QFont(constants.FONT_FAMILY, 10, QFont.Weight.Bold))
    layout.addWidget(header)
    return card, layout


class ProjectPage(QWidget):
    """项目管理：项目信息 / 导入测线 / 传感器同步 / 测线列表 / 成果 / 预览。"""

    import_requested = pyqtSignal(dict)
    sync_requested = pyqtSignal(dict)
    line_selected = pyqtSignal(str)
    line_process_requested = pyqtSignal(str)   # 双击/右键 → 跳转处理页处理该测线
    line_delete_requested = pyqtSignal(list)   # 批量删除所选测线
    artifact_preview_requested = pyqtSignal(str, str)
    close_project_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._summary = None
        self._busy = False
        self._lines = []            # list[ProjectLine]（鸭子类型）
        self._artifacts = []        # list[ProjectArtifact]
        self._current_line_id = ''
        self._filling_table = False
        self._source_path_resolver = None   # 主窗口注入：line_id → 源文件路径

        root = QVBoxLayout(self)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)

        # 页面标题：SubtitleLabel 微软雅黑 12pt Bold 居中（SPEC §1）
        title = SubtitleLabel('项目管理', self)
        title.setFont(QFont(constants.FONT_FAMILY, 12, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        # 无项目提示（SPEC §7：未打开项目时操作按钮禁用并提示）
        self._no_project_hint = CaptionLabel(
            '尚未打开项目 —— 请先在主页打开或新建项目', self)
        self._no_project_hint.setStyleSheet(
            'color: %s; font-size: 11px;' % constants.COLOR_WARNING)
        self._no_project_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._no_project_hint)

        body = QHBoxLayout()
        body.setSpacing(constants.PAGE_SPACING)
        body.addWidget(self._build_left_column(), 0)
        body.addLayout(self._build_right_column(), 1)
        root.addLayout(body, 1)

        self._update_action_state()

    # ============================================================ 左列（固定 460px ScrollArea）
    def _build_left_column(self) -> ScrollArea:
        scroll = ScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(460)
        scroll.setStyleSheet(
            'QScrollArea { background-color: transparent; border: none; }')
        container = QWidget(scroll)
        container.setStyleSheet('background-color: transparent;')
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(constants.PAGE_SPACING)

        layout.addWidget(self._build_info_card(container))
        layout.addWidget(self._build_import_card(container))
        layout.addWidget(self._build_sync_card(container))
        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _build_info_card(self, parent) -> CardWidget:
        """卡片1"项目信息"：名称/编号/位置/操作员/设备型号/坐标系/高程基准 只读 + 关闭项目。"""
        card, layout = _create_card('项目信息')
        self._info_values = {}
        for label_text, attr in _INFO_FIELDS:
            row = QHBoxLayout()
            label = CaptionLabel(label_text, card)
            label.setMinimumWidth(100)
            value = BodyLabel('--', card)
            value.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            row.addWidget(label)
            row.addWidget(value, 1)
            layout.addLayout(row)
            self._info_values[attr] = value
        layout.addWidget(_create_separator())
        btn_row = QHBoxLayout()
        self.close_btn = PushButton('关闭项目', card)
        self.close_btn.clicked.connect(self.close_project_requested)
        btn_row.addStretch(1)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)
        return card

    def _build_import_card(self, parent) -> CardWidget:
        """卡片2"导入测线"：文件+浏览 / 测线号(默认 L01) / 名称 / 介电常数(1-81, 默认 9.0) / 预检+导入 / 预检结果区。"""
        card, layout = _create_card('导入测线')

        # 数据文件行
        file_row = QHBoxLayout()
        file_label = CaptionLabel('数据文件:', card)
        file_label.setMinimumWidth(100)
        self.file_edit = LineEdit(card)
        self.file_edit.setPlaceholderText('选择 GPR 数据文件…')
        self.file_edit.setMinimumWidth(180)
        browse_btn = PushButton('浏览', card)
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_import_file)
        file_row.addWidget(file_label)
        file_row.addWidget(self.file_edit, 1)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        # 测线号 / 名称
        self.line_id_edit = self._add_labeled_edit(
            layout, card, '测线号:', default='L01', placeholder='例如 L01')
        self.line_name_edit = self._add_labeled_edit(
            layout, card, '名称:', placeholder='留空则使用测线号')

        # 介电常数
        diel_row = QHBoxLayout()
        diel_label = CaptionLabel('介电常数:', card)
        diel_label.setMinimumWidth(100)
        self.dielectric_spin = DoubleSpinBox(card)
        self.dielectric_spin.setRange(1.0, 81.0)
        self.dielectric_spin.setDecimals(2)
        self.dielectric_spin.setSingleStep(0.5)
        self.dielectric_spin.setValue(constants.DEFAULT_DIELECTRIC)
        self.dielectric_spin.setMinimumWidth(120)
        diel_row.addWidget(diel_label)
        diel_row.addWidget(self.dielectric_spin)
        diel_row.addStretch(1)
        layout.addLayout(diel_row)

        # 按钮行：预检 + 导入
        btn_row = QHBoxLayout()
        self.preflight_btn = PushButton('预检', card)
        self.import_btn = PrimaryPushButton('导入', card)
        self.preflight_btn.clicked.connect(
            lambda: self._emit_import_request(preflight=True))
        self.import_btn.clicked.connect(
            lambda: self._emit_import_request(preflight=False))
        btn_row.addStretch(1)
        btn_row.addWidget(self.preflight_btn)
        btn_row.addWidget(self.import_btn)
        layout.addLayout(btn_row)

        # 预检结果 CaptionLabel 区
        layout.addWidget(_create_separator())
        self._preflight_label = CaptionLabel('预检结果将显示在此处', card)
        self._preflight_label.setWordWrap(True)
        self._preflight_label.setStyleSheet(
            'color: %s; font-size: 11px;' % constants.COLOR_DISABLED)
        self._preflight_label.setMinimumHeight(34)
        layout.addWidget(self._preflight_label)
        return card

    def _build_sync_card(self, parent) -> CardWidget:
        """卡片3"传感器同步"：rtk/imu/altimeter/trace_timestamps 四个文件行 + 提交同步。"""
        card, layout = _create_card('传感器同步')
        self._sensor_edits = {}
        for key, label_text in _SENSOR_ROWS:
            row = QHBoxLayout()
            label = CaptionLabel(label_text, card)
            label.setMinimumWidth(100)
            edit = LineEdit(card)
            edit.setPlaceholderText('可选' if key != 'rtk' else '选择 RTK 文件（必填）…')
            edit.setMinimumWidth(180)
            browse_btn = PushButton('浏览', card)
            browse_btn.setFixedWidth(70)
            browse_btn.clicked.connect(
                lambda _checked=False, k=key: self._browse_sensor_file(k))
            row.addWidget(label)
            row.addWidget(edit, 1)
            row.addWidget(browse_btn)
            layout.addLayout(row)
            self._sensor_edits[key] = edit

        hint = CaptionLabel('同步目标测线 = 右侧测线列表当前选中行（未选中时使用上方测线号）', card)
        hint.setStyleSheet('color: %s; font-size: 11px;' % constants.COLOR_DISABLED)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        self.sync_btn = PrimaryPushButton('提交同步', card)
        self.sync_btn.clicked.connect(self._emit_sync_request)
        btn_row.addStretch(1)
        btn_row.addWidget(self.sync_btn)
        layout.addLayout(btn_row)
        return card

    def _add_labeled_edit(self, layout, card, label_text, *, default='',
                          placeholder='') -> LineEdit:
        row = QHBoxLayout()
        label = CaptionLabel(label_text, card)
        label.setMinimumWidth(100)
        edit = LineEdit(card)
        edit.setText(default)
        if placeholder:
            edit.setPlaceholderText(placeholder)
        edit.setMinimumWidth(180)
        row.addWidget(label)
        row.addWidget(edit, 1)
        layout.addLayout(row)
        return edit

    # ============================================================ 右侧（stretch）
    def _build_right_column(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(constants.PAGE_SPACING)

        # 卡片"测线列表"
        lines_card, lines_layout = _create_card('测线列表')
        self._lines_table = QTableWidget(0, 7, lines_card)
        self._lines_table.setHorizontalHeaderLabels(
            ['测线号', '名称', '道数', '采样数', '长度m', '质量', '处理状态'])
        self._lines_table.verticalHeader().setVisible(False)
        self._lines_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._lines_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._lines_table.setSelectionMode(
            QTableWidget.SelectionMode.ExtendedSelection)
        header = self._lines_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.sectionResized.connect(self._save_lines_column_widths)
        self._lines_table.setMinimumHeight(180)
        self._lines_table.itemSelectionChanged.connect(
            self._on_line_selection_changed)
        self._lines_table.itemDoubleClicked.connect(
            self._emit_line_process_request)
        self._delete_lines_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Delete), self._lines_table,
            context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_lines_shortcut.activated.connect(self._on_delete_selected_lines)
        self._lines_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._lines_table.customContextMenuRequested.connect(
            self._on_lines_context_menu)
        lines_layout.addWidget(self._lines_table)
        layout.addWidget(lines_card)

        # 卡片"处理成果(Artifact)"
        art_card, art_layout = _create_card('处理成果(Artifact)')
        self._artifacts_table = QTableWidget(0, 5, art_card)
        self._artifacts_table.setHorizontalHeaderLabels(
            ['名称', '方法', '形状', '创建时间', 'SHA前8位'])
        self._artifacts_table.verticalHeader().setVisible(False)
        self._artifacts_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._artifacts_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._artifacts_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection)
        art_header = self._artifacts_table.horizontalHeader()
        art_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        art_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        art_header.sectionResized.connect(self._save_artifacts_column_widths)
        self._artifacts_table.setMinimumHeight(150)
        self._artifacts_table.itemDoubleClicked.connect(
            lambda _item: self._emit_artifact_preview())
        self._artifacts_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._artifacts_table.customContextMenuRequested.connect(
            self._on_artifacts_context_menu)
        art_layout.addWidget(self._artifacts_table)
        art_btn_row = QHBoxLayout()
        self.preview_artifact_btn = PushButton('预览所选', art_card)
        self.preview_artifact_btn.clicked.connect(self._emit_artifact_preview)
        art_btn_row.addStretch(1)
        art_btn_row.addWidget(self.preview_artifact_btn)
        art_layout.addLayout(art_btn_row)
        layout.addWidget(art_card)

        # 卡片"数据预览"
        preview_card, preview_layout = _create_card('数据预览')
        self._bscan = BScanView(preview_card)
        self._bscan.setMinimumHeight(320)
        preview_layout.addWidget(self._bscan, 1)
        layout.addWidget(preview_card, 1)
        return layout

    # ============================================================ 公共接口（主窗口喂数据）
    def set_project_info(self, summary) -> None:
        """ProjectSummary（鸭子类型）或 None（空态）。"""
        self._summary = summary
        if summary is None:
            for value in self._info_values.values():
                value.setText('--')
        else:
            for _label, attr in _INFO_FIELDS:
                text = str(getattr(summary, attr, '') or '').strip()
                self._info_values[attr].setText(text or '--')
        self._update_action_state()

    def set_lines(self, lines: list) -> None:
        """刷新测线表；自动选中首行（触发 line_selected）。"""
        self._lines = list(lines or [])
        self._filling_table = True
        try:
            self._lines_table.setRowCount(0)
            for line in self._lines:
                row = self._lines_table.rowCount()
                self._lines_table.insertRow(row)
                values = (
                    str(getattr(line, 'line_id', '') or '--'),
                    str(getattr(line, 'name', '') or '--'),
                    str(getattr(line, 'trace_count', 0)),
                    str(getattr(line, 'sample_count', 0)),
                    ('%.2f' % float(getattr(line, 'length_m', 0.0) or 0.0)),
                    str(getattr(line, 'data_quality', '') or '--'),
                    str(getattr(line, 'processing_status', '') or '--'),
                )
                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    if col in (2, 3, 4):
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self._lines_table.setItem(row, col, item)
        finally:
            self._filling_table = False
        self._restore_column_widths(self._lines_table, 'lines')
        # 不再自动选中首行：由主窗口在 _on_lines_updated 中按需恢复/设置当前测线，
        # 避免刷新时先跳到首行再跳回，导致结果预览和 InfoBar 闪烁错位。
        if not self._lines:
            self._current_line_id = ''
            self.set_artifacts([])

    def select_line(self, line_id: str) -> bool:
        """按 line_id 选中并触发预览；未找到返回 False。"""
        line_id = str(line_id or '')
        for idx, line in enumerate(self._lines):
            if str(getattr(line, 'line_id', '') or '') == line_id:
                self._lines_table.selectRow(idx)
                return True
        return False

    def set_artifacts(self, artifacts: list) -> None:
        """刷新成果表。"""
        self._artifacts = list(artifacts or [])
        self._artifacts_table.setRowCount(0)
        for artifact in self._artifacts:
            row = self._artifacts_table.rowCount()
            self._artifacts_table.insertRow(row)
            shape = getattr(artifact, 'shape', ()) or ()
            shape_text = '×'.join(str(int(v)) for v in shape) if shape else '--'
            method = (str(getattr(artifact, 'method_name', '') or '')
                      or str(getattr(artifact, 'method_id', '') or '') or '--')
            sha = str(getattr(artifact, 'sha256', '') or '')
            values = (
                str(getattr(artifact, 'name', '') or '--'),
                method,
                shape_text,
                str(getattr(artifact, 'created_at', '') or '--'),
                sha[:8] if sha else '--',
            )
            for col, text in enumerate(values):
                self._artifacts_table.setItem(row, col, QTableWidgetItem(text))
        self._restore_column_widths(self._artifacts_table, 'artifacts')

    def set_preflight_result(self, text: str, ok: bool) -> None:
        """预检结果区：ok 绿色 / 失败红色。"""
        self._preflight_label.setText(str(text or ''))
        color = constants.COLOR_SUCCESS if ok else constants.COLOR_ERROR
        self._preflight_label.setStyleSheet(
            'color: %s; font-size: 11px;' % color)

    def set_preview_bundle(self, bundle) -> None:
        """PreviewBundle（鸭子类型）或 None（清空）。"""
        if bundle is None:
            self._bscan.clear()
        else:
            self._bscan.set_bundle(bundle)

    def set_busy(self, busy: bool) -> None:
        """控制器 busy 状态 → 禁用操作按钮。"""
        self._busy = bool(busy)
        self._update_action_state()

    def current_line_id(self) -> str:
        """当前选中测线号（供主窗口接线使用）。"""
        return self._current_line_id

    # ============================================================ 内部
    def _has_project(self) -> bool:
        return self._summary is not None

    def _update_action_state(self) -> None:
        """无项目时操作按钮禁用 + 顶部提示 CaptionLabel（SPEC §6.3/§7）。"""
        enabled = self._has_project() and not self._busy
        for btn in (self.close_btn, self.preflight_btn, self.import_btn,
                    self.sync_btn, self.preview_artifact_btn):
            btn.setEnabled(enabled)
        self._no_project_hint.setVisible(not self._has_project())

    def _browse_import_file(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, '选择 GPR 数据文件', '', _GPR_FILE_FILTER)
        if path:
            self.file_edit.setText(path)
            clear_invalid(self.file_edit)

    def _browse_sensor_file(self, key: str) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, '选择%s' % dict(_SENSOR_ROWS)[key].rstrip(':'), '',
            _SENSOR_FILE_FILTER)
        if path:
            self._sensor_edits[key].setText(path)
            clear_invalid(self._sensor_edits[key])

    def _emit_import_request(self, *, preflight: bool) -> None:
        source = self.file_edit.text().strip()
        ok, msg = validate_non_empty(source, '数据文件')
        if not ok:
            mark_invalid(self.file_edit, msg)
            InfoBar.warning(
                title='导入测线', content=msg,
                orient=Qt.Orientation.Horizontal, isClosable=True,
                position=InfoBarPosition.TOP, duration=3000, parent=self)
            return
        clear_invalid(self.file_edit)
        line_id = self.line_id_edit.text().strip() or 'L01'
        payload = {
            'preflight': bool(preflight),
            'source': source,
            'line_id': line_id,
            'name': self.line_name_edit.text().strip() or line_id,
            'dielectric': float(self.dielectric_spin.value()),
        }
        if preflight:
            self._preflight_label.setText('预检中…')
            self._preflight_label.setStyleSheet(
                'color: %s; font-size: 11px;' % constants.COLOR_INFO)
        self.import_requested.emit(payload)

    def _emit_sync_request(self) -> None:
        rtk_edit = self._sensor_edits['rtk']
        rtk_path = rtk_edit.text().strip()
        ok, msg = validate_non_empty(rtk_path, 'RTK 文件')
        if not ok:
            mark_invalid(rtk_edit, msg)
            InfoBar.warning(
                title='传感器同步', content=msg,
                orient=Qt.Orientation.Horizontal, isClosable=True,
                position=InfoBarPosition.TOP, duration=3000, parent=self)
            return
        clear_invalid(rtk_edit)
        paths = {key: edit.text().strip()
                 for key, edit in self._sensor_edits.items()}
        line_id = (self._current_line_id
                   or self.line_id_edit.text().strip() or 'L01')
        self.sync_requested.emit({'line_id': line_id, 'paths': paths})

    def set_source_path_resolver(self, resolver) -> None:
        """主窗口注入：line_id → 源数据文件路径（str|None）的查询回调。

        右键菜单"复制路径/打开所在文件夹"的可用性依赖它；未注入或
        查询返回 None 时对应菜单项禁用。
        """
        self._source_path_resolver = resolver

    # ------------------------------------------------------------ 右键菜单
    def _on_lines_context_menu(self, pos) -> None:
        row = self._lines_table.rowAt(pos.y())
        if row < 0 or row >= len(self._lines):
            return
        self._lines_table.selectRow(row)
        line_id = str(getattr(self._lines[row], 'line_id', '') or '')
        if not line_id:
            return
        source = None
        if self._source_path_resolver is not None:
            try:
                source = self._source_path_resolver(line_id)
            except Exception:  # noqa: BLE001 - 查询失败按无路径处理
                source = None
        menu = make_menu(self)
        add_action(menu, FIF.DEVELOPER_TOOLS, '处理该测线（跳转处理页）',
                   lambda: self.line_process_requested.emit(line_id))
        add_action(menu, FIF.DELETE, '删除所选测线',
                   self._on_delete_selected_lines)
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制数据文件路径',
                   lambda: QApplication.clipboard().setText(source),
                   enabled=bool(source))
        add_action(menu, FIF.FOLDER, '打开数据所在文件夹',
                   lambda: self._open_source_folder(source),
                   enabled=bool(source))
        menu.addSeparator()
        add_action(menu, None, '复制测线号',
                   lambda: QApplication.clipboard().setText(line_id))
        menu.exec(self._lines_table.viewport().mapToGlobal(pos))

    @staticmethod
    def _open_source_folder(source: str) -> None:
        folder = os.path.dirname(str(source))
        if folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _on_artifacts_context_menu(self, pos) -> None:
        row = self._artifacts_table.rowAt(pos.y())
        if row < 0 or row >= len(self._artifacts):
            return
        self._artifacts_table.selectRow(row)
        menu = make_menu(self)
        add_action(menu, FIF.VIEW, '预览所选', self._emit_artifact_preview)
        menu.exec(self._artifacts_table.viewport().mapToGlobal(pos))

    def _on_line_selection_changed(self) -> None:
        if self._filling_table:
            return
        row = self._lines_table.currentRow()
        if row < 0 or row >= len(self._lines):
            self._current_line_id = ''
            return
        line_id = str(getattr(self._lines[row], 'line_id', '') or '')
        if not line_id:
            return
        self._current_line_id = line_id
        self.line_selected.emit(line_id)

    def _emit_line_process_request(self, item) -> None:
        """双击测线行 → 请主窗口跳转处理页处理该测线（单击已保证选中）。"""
        row = self._lines_table.row(item)
        if row < 0 or row >= len(self._lines):
            return
        line_id = str(getattr(self._lines[row], 'line_id', '') or '')
        if line_id:
            self.line_process_requested.emit(line_id)

    def _emit_artifact_preview(self) -> None:
        row = self._artifacts_table.currentRow()
        if row < 0 or row >= len(self._artifacts):
            InfoBar.info(
                title='成果预览', content='请先在成果列表中选择一行',
                orient=Qt.Orientation.Horizontal, isClosable=True,
                position=InfoBarPosition.TOP, duration=2000, parent=self)
            return
        artifact_id = str(getattr(self._artifacts[row], 'artifact_id', '') or '')
        line_id = (self._current_line_id
                   or str(getattr(self._artifacts[row], 'line_id', '') or ''))
        if artifact_id and line_id:
            self.artifact_preview_requested.emit(line_id, artifact_id)

    # ------------------------------------------------------------- 批量删除 / 列宽记忆
    def _on_delete_selected_lines(self) -> None:
        """Delete 键 / 右键：确认后批量删除所选测线。"""
        rows = sorted({idx.row() for idx in self._lines_table.selectionModel().selectedRows()})
        if not rows:
            return
        line_ids = []
        for row in rows:
            if 0 <= row < len(self._lines):
                lid = str(getattr(self._lines[row], 'line_id', '') or '')
                if lid:
                    line_ids.append(lid)
        if not line_ids:
            return
        box = MessageBox(
            '确认删除所选测线？',
            f'将删除 {len(line_ids)} 条测线（数据会移入项目 .trash 回收站，可恢复）：\n'
            + '\n'.join(f'  • {lid}' for lid in line_ids),
            self,
        )
        box.yesButton.setText('删除')
        box.cancelButton.setText('取消')
        if box.exec() == QDialog.DialogCode.Accepted:
            self.line_delete_requested.emit(line_ids)

    def _column_widths_key(self, table_name: str) -> str:
        return f'ui/project_page/{table_name}_column_widths'

    def _save_lines_column_widths(self) -> None:
        self._save_column_widths(self._lines_table, 'lines')

    def _save_artifacts_column_widths(self) -> None:
        self._save_column_widths(self._artifacts_table, 'artifacts')

    def _save_column_widths(self, table: QTableWidget, table_name: str) -> None:
        header = table.horizontalHeader()
        widths = [header.sectionSize(i) for i in range(table.columnCount())]
        settings = QSettings('MyGPR', 'MyGPR')
        settings.setValue(self._column_widths_key(table_name), widths)

    def _restore_column_widths(self, table: QTableWidget, table_name: str) -> None:
        settings = QSettings('MyGPR', 'MyGPR')
        widths = settings.value(self._column_widths_key(table_name))
        if not isinstance(widths, list):
            return
        header = table.horizontalHeader()
        for i, w in enumerate(widths):
            if isinstance(w, int) and 0 <= i < table.columnCount():
                header.resizeSection(i, w)

    def _settings(self) -> QSettings:
        """统一 QSettings 根，避免各页用不同组织名。"""
        return QSettings('MyGPR', 'MyGPR')


__all__ = ['ProjectPage']
