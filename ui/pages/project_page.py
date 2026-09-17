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
复制测线号（路径为 controller 异步查询回包缓存，set_line_source_path 喂入）；
成果表 = 预览所选（双击同效）。

import_requested payload：{'preflight': bool, 'source', 'line_id', 'name', 'dielectric'}
（'预检' 按钮 preflight=True → ProjectController.preflight_import；
 '导入' 按钮 preflight=False → ProjectController.import_line）
sync_requested payload：{'line_id', 'paths': {'rtk', 'imu', 'altimeter', 'trace_timestamps'}}
"""

import os

from PyQt6.QtCore import Qt, QSettings, QUrl, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QHeaderView, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    BodyLabel, CaptionLabel, CardWidget, DoubleSpinBox, InfoBar,
    InfoBarPosition, LineEdit, MessageBox, PrimaryPushButton, PushButton,
    ToolButton,
)
from qfluentwidgets import FluentIcon as FIF

from ui import constants, file_dialogs
from ui.page_scaffold import make_card, make_form_row, make_scroll_column
from ui.theme_helpers import status_color
from ui.widgets import (BScanView, CollapsiblePanel, clear_invalid,
                        make_separator, mark_invalid, validate_non_empty)
from ui.widgets.context_menus import add_action, make_menu

# 导入文件对话框过滤器：通过 DesktopBackendFacade 获取（SPEC §6.3）
try:
    from ui.desktop_backend_facade import file_dialog_filter
    _GPR_FILE_FILTER = file_dialog_filter()
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


class ProjectPage(QWidget):
    """项目管理：项目信息 / 导入测线 / 传感器同步 / 测线列表 / 成果 / 预览。"""

    import_requested = pyqtSignal(dict)
    sync_requested = pyqtSignal(dict)
    line_selected = pyqtSignal(str)
    line_process_requested = pyqtSignal(str)   # 双击/右键 → 跳转处理页处理该测线
    line_delete_requested = pyqtSignal(list)   # 批量删除所选测线
    artifact_preview_requested = pyqtSignal(str, str)
    artifact_delete_requested = pyqtSignal(str, list)  # 删除成果（单/批）
    close_project_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._summary = None
        self._busy = False
        self._lines = []            # list[ProjectLine]（鸭子类型）
        self._artifacts = []        # list[ProjectArtifact]
        self._current_line_id = ''
        self._filling_table = False
        self._syncing_selection = False  # select_line 程序化选中时抑止回发 line_selected
        self._source_path_cache: dict[str, str | None] = {}  # line_id → 源文件路径（异步回包缓存）
        self._sm = None                     # 共享 SettingsManager（主窗口注入）

        root = QVBoxLayout(self)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)

        # 页头：无项目提示居右（页内大标题已由顶部页签条承担，不再重复）
        header_row = QHBoxLayout()
        header_row.setSpacing(constants.CARD_SPACING)
        header_row.addStretch(1)

        # 无项目提示（SPEC §7：未打开项目时操作按钮禁用并提示）
        self._no_project_hint = CaptionLabel(
            '尚未打开项目 —— 请先在主页打开或新建项目', self)
        self._no_project_hint.setStyleSheet(
            'color: %s; font-size: 11px;' % status_color('warning'))
        self._no_project_hint.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_row.addWidget(self._no_project_hint)
        root.addLayout(header_row)

        body = QHBoxLayout()
        body.setSpacing(constants.PAGE_SPACING)
        body.addWidget(self._build_left_column(), 0)
        body.addLayout(self._build_right_column(), 1)
        root.addLayout(body, 1)

        self._update_action_state()

    # ============================================================ 左列（CollapsiblePanel + 固定宽滚动栏）
    def _build_left_column(self) -> CollapsiblePanel:
        """左栏：项目信息 / 导入测线 / 传感器同步，可折叠（展开宽 SIDE_FORM_WIDTH）。"""
        scroll, layout = make_scroll_column(constants.SIDE_FORM_WIDTH)
        panel = CollapsiblePanel(
            'left', expand_width=constants.SIDE_FORM_WIDTH,
            collapse_width=40, parent=self)
        panel.set_content_widget(scroll)

        layout.addWidget(self._build_info_card(panel))
        layout.addWidget(self._build_import_card(panel))
        layout.addWidget(self._build_sync_card(panel))
        layout.addStretch(1)
        return panel

    def _build_info_card(self, parent) -> CardWidget:
        """卡片1"项目信息"：名称/编号/位置/操作员/设备型号/坐标系/高程基准 只读 + 关闭项目。"""
        card, layout = make_card('项目信息')
        self._info_values = {}
        for label_text, attr in _INFO_FIELDS:
            value = BodyLabel('--', card)
            value.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addLayout(make_form_row(label_text, value, parent=card,
                                           trailing_stretch=False))
            self._info_values[attr] = value
        layout.addWidget(make_separator())
        btn_row = QHBoxLayout()
        self.close_btn = PushButton('关闭项目', card)
        self.close_btn.clicked.connect(self.close_project_requested)
        btn_row.addStretch(1)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)
        return card

    def _build_import_card(self, parent) -> CardWidget:
        """卡片2"导入测线"：文件+浏览 / 测线号(默认 L01) / 名称 / 介电常数(1-81, 默认 9.0) / 预检+导入 / 预检结果区。"""
        card, layout = make_card('导入测线')

        # 数据文件行
        self.file_edit = LineEdit(card)
        self.file_edit.setPlaceholderText('选择 GPR 数据文件…')
        self.file_edit.setMinimumWidth(180)
        browse_btn = PushButton('浏览', card)
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_import_file)
        layout.addLayout(make_form_row('数据文件:', self.file_edit, browse_btn,
                                       parent=card, trailing_stretch=False))

        # 测线号 / 名称
        self.line_id_edit = self._add_labeled_edit(
            layout, card, '测线号:', default='L01', placeholder='例如 L01')
        self.line_name_edit = self._add_labeled_edit(
            layout, card, '名称:', placeholder='留空则使用测线号')

        # 介电常数
        self.dielectric_spin = DoubleSpinBox(card)
        self.dielectric_spin.setRange(1.0, 81.0)
        self.dielectric_spin.setDecimals(2)
        self.dielectric_spin.setSingleStep(0.5)
        self.dielectric_spin.setValue(constants.DEFAULT_DIELECTRIC)
        self.dielectric_spin.setMinimumWidth(120)
        layout.addLayout(make_form_row('介电常数:', self.dielectric_spin,
                                       parent=card))

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
        layout.addWidget(make_separator())
        self._preflight_label = CaptionLabel('预检结果将显示在此处', card)
        self._preflight_label.setWordWrap(True)
        self._preflight_label.setStyleSheet(
            'color: %s; font-size: 11px;' % status_color('disabled'))
        self._preflight_label.setMinimumHeight(34)
        layout.addWidget(self._preflight_label)
        return card

    def _build_sync_card(self, parent) -> CardWidget:
        """卡片3"传感器同步"：rtk/imu/altimeter/trace_timestamps 四个文件行 + 提交同步。"""
        card, layout = make_card('传感器同步')
        self._sensor_edits = {}
        for key, label_text in _SENSOR_ROWS:
            edit = LineEdit(card)
            edit.setPlaceholderText('可选' if key != 'rtk' else '选择 RTK 文件（必填）…')
            edit.setMinimumWidth(180)
            browse_btn = PushButton('浏览', card)
            browse_btn.setFixedWidth(70)
            browse_btn.clicked.connect(
                lambda _checked=False, k=key: self._browse_sensor_file(k))
            layout.addLayout(make_form_row(label_text, edit, browse_btn,
                                           parent=card,
                                           trailing_stretch=False))
            self._sensor_edits[key] = edit

        hint = CaptionLabel('同步目标测线 = 右侧测线列表当前选中行（未选中时使用上方测线号）', card)
        hint.setStyleSheet('color: %s; font-size: 11px;'
                           % status_color('disabled'))
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
        """标签+输入框表单行（收敛实现见 ui.page_scaffold.make_form_row）。"""
        edit = LineEdit(card)
        edit.setText(default)
        if placeholder:
            edit.setPlaceholderText(placeholder)
        edit.setMinimumWidth(180)
        layout.addLayout(make_form_row(label_text, edit, parent=card,
                                       trailing_stretch=False))
        return edit

    # ============================================================ 右侧（QSplitter 纵向三卡，stretch）
    def _build_right_column(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(constants.PAGE_SPACING)
        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setChildrenCollapsible(False)

        # 卡片"测线列表"
        lines_card, lines_layout = make_card('测线列表')
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
        self._lines_table.setMinimumHeight(120)
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
        splitter.addWidget(lines_card)

        # 卡片"处理成果(Artifact)"
        art_card, art_layout = make_card('处理成果(Artifact)')
        self._artifacts_table = QTableWidget(0, 6, art_card)
        self._artifacts_table.setHorizontalHeaderLabels(
            ['名称', '方法', '形状', '创建时间', 'SHA前8位', '操作'])
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
        art_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self._artifacts_table.setColumnWidth(5, 56)
        art_header.sectionResized.connect(self._save_artifacts_column_widths)
        self._artifacts_table.setMinimumHeight(120)
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
        splitter.addWidget(art_card)

        # 卡片"数据预览"（吃掉剩余空间，B-Scan 完整显示）
        preview_card, preview_layout = make_card('数据预览')
        self._bscan = BScanView(preview_card)
        self._bscan.setMinimumHeight(constants.PREVIEW_MIN_HEIGHT)
        preview_layout.addWidget(self._bscan, 1)
        splitter.addWidget(preview_card)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([160, 140, 400])
        layout.addWidget(splitter, 1)
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
        """刷新测线表；自动建议下一测线号（UX P2-1）。"""
        self._lines = list(lines or [])
        # P2-1：如果导入表单的测线号仍是默认 L01，按现有线数自动建议 L02/L03…
        if self.line_id_edit.text().strip() in ('', 'L01'):
            suggest = self._suggest_line_id()
            if suggest != 'L01':
                self.line_id_edit.setText(suggest)
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
        """按 line_id 选中并触发预览；未找到返回 False。

        程序化选中（文件树/空间页反向同步）不再次回发 line_selected——
        选中来源已经走过完整的选择链路，回发会重复预览刷新。
        """
        line_id = str(line_id or '')
        for idx, line in enumerate(self._lines):
            if str(getattr(line, 'line_id', '') or '') == line_id:
                self._syncing_selection = True
                try:
                    self._lines_table.selectRow(idx)
                finally:
                    self._syncing_selection = False
                self._current_line_id = line_id
                return True
        return False

    def set_artifacts(self, artifacts: list) -> None:
        """刷新成果表；空表用一行占位引导（不占选择、不响应双击）。"""
        self._artifacts = list(artifacts or [])
        self._artifacts_table.setRowCount(0)
        if not self._artifacts:
            self._artifacts_table.insertRow(0)
            placeholder = QTableWidgetItem('暂无成果，处理完成后在此显示')
            placeholder.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            placeholder.setForeground(QBrush(QColor(status_color('disabled'))))
            self._artifacts_table.setItem(0, 0, placeholder)
            self._artifacts_table.setSpan(0, 0, 1, 6)
            return
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
            del_btn = ToolButton(FIF.DELETE)
            del_btn.setToolTip('删除该成果（移入回收站）')
            aid = str(getattr(artifact, 'artifact_id', '') or '')
            del_btn.clicked.connect(
                lambda _=False, aid_=aid: self._on_artifact_delete_clicked(aid_))
            self._artifacts_table.setCellWidget(row, 5, del_btn)
            for col, text in enumerate(values):
                self._artifacts_table.setItem(row, col, QTableWidgetItem(text))
        self._restore_column_widths(self._artifacts_table, 'artifacts')

    def set_preflight_result(self, text: str, ok: bool) -> None:
        """预检结果区：ok 绿色 / 失败红色。"""
        self._preflight_label.setText(str(text or ''))
        color = status_color('success' if ok else 'error')
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

    def set_default_dielectric(self, value) -> None:
        """设置默认介电常数（导入表单默认值，来自设置页）。"""
        try:
            self.dielectric_spin.setValue(float(value))
        except (TypeError, ValueError):
            pass

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
        path, _selected = file_dialogs.getOpenFileName(
            self, '选择 GPR 数据文件', '', _GPR_FILE_FILTER)
        if path:
            self.file_edit.setText(path)
            clear_invalid(self.file_edit)

    def _suggest_line_id(self) -> str:
        """按现有测线号自动建议下一个（L01→L02→L03…，跳过已占用）。"""
        existing = {str(getattr(line, 'line_id', '') or '')
                    for line in self._lines}
        for i in range(1, 100):
            candidate = f'L{i:02d}'
            if candidate not in existing:
                return candidate
        return 'L99'

    def _browse_sensor_file(self, key: str) -> None:
        path, _selected = file_dialogs.getOpenFileName(
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
                'color: %s; font-size: 11px;' % status_color('info'))
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

    def set_line_source_path(self, line_id: str, path) -> None:
        """主窗口/controller 异步回包：缓存测线源文件路径（右键菜单素材）。

        由 page_coordinator 把 ProjectController.line_source_path_ready
        接到本方法；查询在 worker 线程完成，右键菜单构建时直接读缓存。
        """
        line_id = str(line_id or '')
        if line_id:
            self._source_path_cache[line_id] = str(path) if path else None

    # ------------------------------------------------------------ 右键菜单
    def _on_lines_context_menu(self, pos) -> None:
        row = self._lines_table.rowAt(pos.y())
        if row < 0 or row >= len(self._lines):
            return
        self._lines_table.selectRow(row)
        line_id = str(getattr(self._lines[row], 'line_id', '') or '')
        if not line_id:
            return
        source = self._source_path_cache.get(line_id)
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
        add_action(menu, FIF.DELETE, '删除所选成果',
                   self._on_artifacts_context_delete)
        menu.exec(self._artifacts_table.viewport().mapToGlobal(pos))

    def _on_line_selection_changed(self) -> None:
        if self._filling_table:
            self._current_line_id = ''
            return
        if self._syncing_selection:
            # 程序化选中（select_line）：不回发，当前线由 select_line 维护
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

    def _on_artifact_delete_clicked(self, artifact_id: str) -> None:
        """成果表操作列删除按钮 → 请求删除（级联确认在 coordinator）。"""
        artifact_id = str(artifact_id or '')
        if not artifact_id:
            return
        line_id = self._current_line_id
        if not line_id:
            for item in self._artifacts:
                if str(getattr(item, 'artifact_id', '') or '') == artifact_id:
                    line_id = str(getattr(item, 'line_id', '') or '')
                    break
        if not line_id:
            return
        self.artifact_delete_requested.emit(line_id, [artifact_id])

    def _on_artifacts_context_delete(self) -> None:
        """右键删除所选成果（级联确认在 coordinator）。"""
        row = self._artifacts_table.currentRow()
        if row < 0 or row >= len(self._artifacts):
            return
        artifact_id = str(getattr(self._artifacts[row], 'artifact_id', '') or '')
        if artifact_id:
            self._on_artifact_delete_clicked(artifact_id)

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

    def set_settings_manager(self, sm) -> None:
        """注入主窗口共享的 SettingsManager 并重放列宽恢复（含 QSettings 迁移）。"""
        self._sm = sm
        self._restore_column_widths(self._lines_table, 'lines')
        self._restore_column_widths(self._artifacts_table, 'artifacts')

    def _save_lines_column_widths(self) -> None:
        self._save_column_widths(self._lines_table, 'lines')

    def _save_artifacts_column_widths(self) -> None:
        self._save_column_widths(self._artifacts_table, 'artifacts')

    def _save_column_widths(self, table: QTableWidget, table_name: str) -> None:
        header = table.horizontalHeader()
        widths = [int(header.sectionSize(i)) for i in range(table.columnCount())]
        sm = self._sm
        if sm is None:
            return
        sm.set(self._column_widths_key(table_name), widths)
        sm.save()

    def _restore_column_widths(self, table: QTableWidget, table_name: str) -> None:
        """恢复列宽：优先读共享 SettingsManager；旧版 QSettings 读到即迁移。

        迁移语义：QSettings('MyGPR','MyGPR') 里残留的列宽写入 SettingsManager
        （JSON list）并清掉旧键；之后新写入只走 SettingsManager。
        """
        key = self._column_widths_key(table_name)
        widths = None
        legacy = QSettings('MyGPR', 'MyGPR')
        stored = legacy.value(key)
        if isinstance(stored, list):
            widths = [int(w) for w in stored
                      if isinstance(w, (int, float)) and int(w) > 0]
            sm = self._sm
            if sm is not None:
                sm.set(key, widths)
                sm.save()
                legacy.remove(key)
        elif self._sm is not None:
            stored = self._sm.get(key)
            if isinstance(stored, list):
                widths = [int(w) for w in stored
                          if isinstance(w, (int, float)) and int(w) > 0]
        if not widths:
            return
        header = table.horizontalHeader()
        for i, w in enumerate(widths):
            if 0 <= i < table.columnCount():
                header.resizeSection(i, w)


__all__ = ['ProjectPage']
