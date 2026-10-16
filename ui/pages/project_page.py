"""Single project workspace: startup, empty project, or line ledger.

Presentation and signals only; controllers retain all I/O and long-running work.
HomePage is embedded startup content and the legacy home-route facade. Import,
preflight, metadata and sensor forms live in tools; line details and processing
history share one selected-line area. B-scan previews belong to processing.

import_requested payload：{'preflight': bool, 'source', 'line_id', 'name', 'dielectric'}
（'预检' 按钮 preflight=True → ProjectController.preflight_import；
 '导入' 按钮 preflight=False → ProjectController.import_line）
sync_requested payload：{'line_id', 'paths': {'rtk', 'imu', 'altimeter', 'trace_timestamps'}}
"""

import os

from PyQt6.QtCore import QSize, Qt, QSettings, QUrl, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QSizePolicy, QSplitter,
    QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    BodyLabel, CardWidget, DoubleSpinBox, InfoBar,
    InfoBarPosition, LineEdit, MessageBox, PrimaryPushButton, PushButton,
    ToolButton, TransparentToolButton, isDarkTheme,
)
from qfluentwidgets import FluentIcon as FIF

from ui import constants, file_dialogs
from ui.page_scaffold import make_card, make_form_row, make_hint
from ui.pages.project_visuals import (
    apply_project_theme, configure_ledger_table, make_detail_scroll, style_ledger_items,
)
from ui.theme_helpers import status_color
from ui.widgets import (clear_invalid,
                        make_separator, mark_invalid, validate_non_empty)
from ui.widgets.context_menus import add_action, make_menu
from ui.pages.home_page import HomePage

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
    """A project identity, line ledger and a single selected-line context."""

    import_requested = pyqtSignal(dict)
    sync_requested = pyqtSignal(dict)
    line_selected = pyqtSignal(str)
    line_process_requested = pyqtSignal(str)   # 双击/右键 → 跳转处理页处理该测线
    line_delete_requested = pyqtSignal(list)   # 批量删除所选测线
    artifact_preview_requested = pyqtSignal(str, str)
    artifact_delete_requested = pyqtSignal(str, list)  # 删除成果（单/批）
    close_project_requested = pyqtSignal()
    workspace_state_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._summary = None
        self._busy = False
        self._lines = []            # list[ProjectLine]（鸭子类型）
        self._artifacts = []        # list[ProjectArtifact]
        self._current_line_id = ''
        self._suggested_line_id = 'L01'
        self._filling_table = False
        self._syncing_selection = False  # select_line 程序化选中时抑止回发 line_selected
        self._source_path_cache: dict[str, str | None] = {}  # line_id → 源文件路径（异步回包缓存）
        self._sm = None                     # 共享 SettingsManager（主窗口注入）
        self._resume_lines: set[str] = set()
        self._sensor_target_line_id = ''

        root = QVBoxLayout(self)
        root.setContentsMargins(*(constants.SPACE_4,) * 4)
        root.setSpacing(constants.SPACE_3)

        self._state_stack = QStackedWidget(self)
        self.startup = HomePage(self._state_stack)
        self._state_stack.addWidget(self.startup)
        self._work_widget = QWidget(self._state_stack)
        self._state_stack.addWidget(self._work_widget)
        root.addWidget(self._state_stack, 1)
        work = QVBoxLayout(self._work_widget)
        work.setContentsMargins(0, 0, 0, 0)
        work.setSpacing(constants.SPACE_3)
        header_row = QHBoxLayout()
        header_row.setSpacing(constants.SPACE_2)
        self._project_heading = BodyLabel('', self._work_widget)
        self._project_heading.setWordWrap(True)
        self._project_heading.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        header_row.addWidget(self._project_heading, 1)
        self.header_import_btn = PrimaryPushButton('导入数据', self._work_widget, FIF.DOWNLOAD)
        self.header_import_btn.clicked.connect(self.show_import_tool)
        header_row.addWidget(self.header_import_btn)
        self.project_more_btn = ToolButton(FIF.MORE, self._work_widget)
        self.project_more_btn.setToolTip('项目设置与工具')
        self.project_more_btn.setAccessibleName('项目设置与工具')
        self.project_more_btn.setFixedSize(*constants.TOOL_BTN_SIZE)
        self.project_more_btn.setIconSize(QSize(*constants.TOOL_BTN_ICON))
        self.project_more_btn.clicked.connect(self._show_project_tools)
        header_row.addWidget(self.project_more_btn)
        work.addLayout(header_row)
        self._no_project_hint = make_hint(
            '', parent=self._work_widget,
            key='warning')
        self._no_project_hint.setWordWrap(True)
        work.addWidget(self._no_project_hint)
        self._work_stack = QStackedWidget(self._work_widget)
        work.addWidget(self._work_stack, 1)
        self._empty_project = QWidget(self._work_stack)
        empty = QVBoxLayout(self._empty_project)
        empty.setSpacing(constants.SPACE_3)
        empty.addStretch(1)
        empty_label = BodyLabel('项目已就绪，导入第一条测线', self._empty_project)
        empty_label.setObjectName('projectEmptyLabel')
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.addWidget(empty_label)
        self.empty_import_btn = PrimaryPushButton('导入数据', self._empty_project, FIF.DOWNLOAD)
        self.empty_import_btn.clicked.connect(self.show_import_tool)
        empty.addWidget(self.empty_import_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        empty.addStretch(1)
        self._work_stack.addWidget(self._empty_project)
        self._ledger_widget = QWidget(self._work_stack)
        self._ledger_widget.setLayout(self._build_right_column())
        self._work_stack.addWidget(self._ledger_widget)
        self._import_dialog = self._tool_dialog('导入测线', self._build_import_card)
        self._sync_dialog = self._tool_dialog('传感器同步', self._build_sync_card)
        self._info_dialog = self._tool_dialog('项目设置', self._build_info_card)
        self.close_btn.hide()  # closing uses the shared project switch menu
        self._update_action_state()
        self.apply_theme(isDarkTheme())

    def _tool_dialog(self, title, builder):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(440)
        dialog.setModal(False)
        layout = QVBoxLayout(dialog)
        layout.addWidget(builder(dialog))
        return dialog

    def show_import_tool(self) -> None:
        if not self._has_project() or self._busy:
            return
        self._import_dialog.show()
        self._import_dialog.raise_()
        self.file_edit.setFocus()

    def show_sensor_tool(self) -> None:
        if not self._current_line_id or not self._has_project():
            return
        self._sync_dialog.setWindowTitle(f'传感器同步 · {self._current_line_id}')
        self._sync_dialog.show()
        self._sync_dialog.raise_()

    def _show_project_tools(self) -> None:
        menu = make_menu(self)
        add_action(menu, FIF.SETTING, '项目设置 / 元数据', self._info_dialog.show)
        add_action(menu, FIF.SEARCH, '导入与预检…', self.show_import_tool, enabled=not self._busy)
        root = str(getattr(self._summary, 'root_path', '') or '')
        add_action(menu, FIF.FOLDER, '打开项目位置',
                   lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(root)), enabled=bool(root))
        menu.exec(self.project_more_btn.mapToGlobal(self.project_more_btn.rect().bottomLeft()))

    def workspace_state(self) -> str:
        if self._summary is None:
            return 'startup'
        return 'ledger' if self._lines else 'empty'

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
        layout.addWidget(BodyLabel('数据文件:', card))
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_edit, 1)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)
        self.file_edit.textChanged.connect(self.file_edit.setToolTip)

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
        self._preflight_label = make_hint('预检结果将显示在此处', parent=card)
        self._preflight_label.setWordWrap(True)
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
            layout.addWidget(BodyLabel(label_text, card))
            file_row = QHBoxLayout()
            file_row.addWidget(edit, 1)
            file_row.addWidget(browse_btn)
            layout.addLayout(file_row)
            edit.textChanged.connect(edit.setToolTip)
            self._sensor_edits[key] = edit

        hint = make_hint(
            '请先在右侧选择测线，再提交传感器同步',
            parent=card)
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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(constants.SPACE_2)
        self.line_search = LineEdit(self)
        self.line_search.setPlaceholderText('搜索测线号或名称')
        self.line_search.setClearButtonEnabled(True)
        self.line_search.textChanged.connect(self._filter_lines)
        layout.addWidget(self.line_search)
        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setChildrenCollapsible(False)

        lines_card = QWidget(self)
        lines_layout = QVBoxLayout(lines_card)
        lines_layout.setContentsMargins(0, 0, 0, 0)
        self._lines_table = QTableWidget(0, 7, lines_card)
        configure_ledger_table(self._lines_table)
        self._lines_table.setHorizontalHeaderLabels(
            ['测线号', '名称', '道数', '采样数', '长度 / m', '质量', '处理状态'])
        self._lines_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._lines_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._lines_table.setSelectionMode(
            QTableWidget.SelectionMode.ExtendedSelection)
        header = self._lines_table.horizontalHeader()
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

        detail = QWidget(self)
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(constants.PANEL_SPACING)
        detail_header = QHBoxLayout()
        detail_header.setSpacing(constants.SPACE_2)
        self._selected_heading = BodyLabel('选择一条测线以查看详情', detail)
        self._selected_heading.setWordWrap(True)
        self._selected_heading.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        detail_header.addWidget(self._selected_heading, 1)
        self.process_line_btn = PrimaryPushButton('打开处理', detail, FIF.DEVELOPER_TOOLS)
        self.process_line_btn.clicked.connect(self._process_selected_line)
        detail_header.addWidget(self.process_line_btn)
        self.line_more_btn = ToolButton(FIF.MORE, detail)
        self.line_more_btn.setToolTip('所选测线操作')
        self.line_more_btn.setAccessibleName('所选测线操作')
        self.line_more_btn.setFixedSize(*constants.TOOL_BTN_SIZE)
        self.line_more_btn.setIconSize(QSize(*constants.TOOL_BTN_ICON))
        self.line_more_btn.clicked.connect(self._show_selected_line_tools)
        detail_header.addWidget(self.line_more_btn)
        detail_layout.addLayout(detail_header)
        self._detail_tabs = QTabWidget(detail)
        self._detail_tabs.setObjectName('projectDetailTabs')
        self._detail_tabs.setDocumentMode(True)
        self._detail_tabs.setUsesScrollButtons(True)
        detail_layout.addWidget(self._detail_tabs, 1)
        info_tab = QWidget(self._detail_tabs)
        info_layout = QVBoxLayout(info_tab)
        info_layout.setContentsMargins(*(constants.SPACE_2,) * 4)
        self._line_detail = BodyLabel('选择一条测线', info_tab)
        self._line_detail.setWordWrap(True)
        self._line_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info_layout.addWidget(make_detail_scroll(self._line_detail, info_tab))
        self._detail_tabs.addTab(info_tab, '详情')
        sensors_tab = QWidget(self._detail_tabs)
        sensors_layout = QVBoxLayout(sensors_tab)
        sensors_layout.setContentsMargins(*(constants.SPACE_2,) * 4)
        sensors_layout.setSpacing(constants.SPACE_2)
        self._sensor_detail = make_hint('选择测线后可查看和同步 RTK / IMU / 高度计数据。', parent=sensors_tab)
        self._sensor_detail.setWordWrap(True)
        sensors_layout.addWidget(make_detail_scroll(self._sensor_detail, sensors_tab), 1)
        self.sensor_tool_btn = PushButton('传感器同步…', sensors_tab, FIF.SYNC)
        self.sensor_tool_btn.clicked.connect(self.show_sensor_tool)
        sensors_layout.addWidget(self.sensor_tool_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self._detail_tabs.addTab(sensors_tab, '传感器')
        art_card = QWidget(self._detail_tabs)
        art_layout = QVBoxLayout(art_card)
        art_layout.setContentsMargins(0, 0, 0, 0)
        art_layout.setSpacing(constants.SPACE_2)
        self._artifacts_table = QTableWidget(0, 6, art_card)
        configure_ledger_table(self._artifacts_table, artifacts=True)
        self._artifacts_table.setHorizontalHeaderLabels(
            ['名称', '方法', '形状', '创建时间', 'SHA前8位', '操作'])
        self._artifacts_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._artifacts_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._artifacts_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection)
        art_header = self._artifacts_table.horizontalHeader()
        art_header.sectionResized.connect(self._save_artifacts_column_widths)
        self._artifacts_table.setMinimumHeight(120)
        self._artifacts_table.itemDoubleClicked.connect(
            lambda _item: self._emit_artifact_preview())
        self._artifacts_table.itemSelectionChanged.connect(self._update_action_state)
        self._artifacts_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._artifacts_table.customContextMenuRequested.connect(
            self._on_artifacts_context_menu)
        art_layout.addWidget(self._artifacts_table)
        art_btn_row = QHBoxLayout()
        # 本页已无 B-Scan 预览（见模块文档），按钮实为"跳到处理页去看"，
        # 故文案明确去处，避免用户以为就地会弹图。
        self.preview_artifact_btn = PushButton('在处理页查看', art_card)
        self.preview_artifact_btn.setToolTip('跳到处理页并载入该成果的 B-Scan 预览')
        self.preview_artifact_btn.clicked.connect(self._emit_artifact_preview)
        art_btn_row.addStretch(1)
        art_btn_row.addWidget(self.preview_artifact_btn)
        art_layout.addLayout(art_btn_row)
        self._detail_tabs.addTab(art_card, '处理历史')
        splitter.addWidget(detail)

        # 台账优先分配空间，选中测线详情可拖动调整；B-scan 留在处理工作台。
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 220])
        layout.addWidget(splitter, 1)
        return layout

    def apply_theme(self, dark: bool) -> None:
        """Compact project surfaces use the same semantic palette as the workbench."""
        apply_project_theme(self, dark)

    def _style_ledger_items(self) -> None:
        style_ledger_items(self._lines_table, self._artifacts_table,
                           getattr(self, '_dark', isDarkTheme()))

    def _filter_lines(self, text: str) -> None:
        text = str(text).casefold().strip()
        for row, line in enumerate(self._lines):
            haystack = f'{getattr(line, "line_id", "")} {getattr(line, "name", "")}'.casefold()
            self._lines_table.setRowHidden(row, bool(text and text not in haystack))
        # A filtered-out row cannot remain an invisible processing target.
        if self._current_line_id and any(
                self._lines_table.isRowHidden(index.row())
                for index in self._lines_table.selectionModel().selectedRows()):
            self._lines_table.clearSelection()

    def _process_selected_line(self) -> None:
        if self._current_line_id and self.process_line_btn.isEnabled():
            self.line_process_requested.emit(self._current_line_id)

    def _show_selected_line_tools(self) -> None:
        menu = make_menu(self)
        add_action(menu, FIF.SYNC, '传感器同步…', self.show_sensor_tool,
                   enabled=bool(self._current_line_id) and self._can_write())
        add_action(menu, FIF.DELETE, '删除所选测线', self._on_delete_selected_lines,
                   enabled=self._can_write() and bool(self._lines_table.selectionModel().selectedRows()))
        menu.exec(self.line_more_btn.mapToGlobal(self.line_more_btn.rect().bottomLeft()))

    # ============================================================ 公共接口（主窗口喂数据）
    def set_project_info(self, summary) -> None:
        """ProjectSummary（鸭子类型）或 None（空态）。"""
        def identity(value):
            return (str(getattr(value, 'project_id', '') or ''),
                    str(getattr(value, 'root_path', '') or ''))
        changed = self._summary is None or summary is None or identity(self._summary) != identity(summary)
        self._summary = summary
        self._last_workspace_state = None
        if changed:
            for dialog in (self._import_dialog, self._sync_dialog, self._info_dialog):
                dialog.close()
            self._current_line_id = ''
            self._source_path_cache.clear()
            self._resume_lines.clear()
            self.file_edit.clear()
            self.line_name_edit.clear()
            self.line_id_edit.setText('L01')
            self._suggested_line_id = 'L01'
            for edit in self._sensor_edits.values():
                edit.clear()
            self.set_lines([])
        self.startup.set_current_project(summary)
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
        # 自动建议随连续导入推进；用户改过的测线号保留。
        if self.line_id_edit.text().strip() in ('', self._suggested_line_id):
            suggest = self._suggest_line_id()
            self.line_id_edit.setText(suggest)
            self._suggested_line_id = suggest
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
        self._style_ledger_items()
        self._restore_column_widths(self._lines_table, 'lines')
        # 不再自动选中首行：由主窗口在 _on_lines_updated 中按需恢复/设置当前测线，
        # 避免刷新时先跳到首行再跳回，导致结果预览和 InfoBar 闪烁错位。
        if not self._lines:
            self._current_line_id = ''
            self.set_artifacts([])
        self._filter_lines(self.line_search.text())
        self._update_action_state()

    def select_line(self, line_id: str) -> bool:
        """按 line_id 选中并触发预览；未找到返回 False。

        程序化选中（文件树/空间页反向同步）不再次回发 line_selected——
        选中来源已经走过完整的选择链路，回发会重复预览刷新。
        """
        line_id = str(line_id or '')
        for idx, line in enumerate(self._lines):
            if str(getattr(line, 'line_id', '') or '') == line_id:
                if self._lines_table.isRowHidden(idx):
                    self.line_search.clear()
                self._syncing_selection = True
                try:
                    self._lines_table.selectRow(idx)
                finally:
                    self._syncing_selection = False
                self._current_line_id = line_id
                self._update_action_state()
                return True
        return False

    def set_artifacts(self, artifacts: list) -> None:
        """刷新成果表；空表用一行占位引导（不占选择、不响应双击）。"""
        self._artifacts = list(artifacts or []) if self._current_line_id else []
        self._artifacts_table.setRowCount(0)
        if not self._artifacts:
            self._artifacts_table.insertRow(0)
            placeholder = QTableWidgetItem('暂无成果，处理完成后在此显示')
            placeholder.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            placeholder.setForeground(QBrush(QColor(status_color('disabled'))))
            self._artifacts_table.setItem(0, 0, placeholder)
            self._artifacts_table.setSpan(0, 0, 1, 6)
            self._style_ledger_items()
            self._update_action_state()
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
            del_btn = TransparentToolButton(FIF.DELETE)
            del_btn.setIconSize(QSize(*constants.TOOL_BTN_ICON))
            del_btn.setAccessibleName('删除该成果')
            del_btn.setToolTip('删除该成果（移入回收站）')
            aid = str(getattr(artifact, 'artifact_id', '') or '')
            del_btn.clicked.connect(
                lambda _=False, aid_=aid: self._on_artifact_delete_clicked(aid_))
            self._artifacts_table.setCellWidget(row, 5, del_btn)
            for col, text in enumerate(values):
                self._artifacts_table.setItem(row, col, QTableWidgetItem(text))
        self._restore_column_widths(self._artifacts_table, 'artifacts')
        self._style_ledger_items()
        self._update_action_state()

    def set_preflight_result(self, text: str, ok: bool) -> None:
        """预检结果区：ok 绿色 / 失败红色。"""
        self._preflight_label.setText(str(text or ''))
        self._preflight_label.set_hint_key('success' if ok else 'error')

    def set_busy(self, busy: bool) -> None:
        """控制器 busy 状态 → 禁用操作按钮。"""
        self._busy = bool(busy)
        self._update_action_state()

    def current_line_id(self) -> str:
        """当前选中测线号（供主窗口接线使用）。"""
        return self._current_line_id

    def set_line_resume_available(self, line_id: str, available: bool) -> None:
        line_id = str(line_id or '')
        if available:
            self._resume_lines.add(line_id)
        else:
            self._resume_lines.discard(line_id)
        self._update_action_state()

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
        """操作随项目权限、后台忙状态和当前选择同步。"""
        if self._sensor_target_line_id != self._current_line_id:
            self._sensor_target_line_id = self._current_line_id
            self._sync_dialog.close()
            for edit in self._sensor_edits.values():
                edit.clear()
        state = self.workspace_state()
        self._state_stack.setCurrentWidget(self.startup if state == 'startup' else self._work_widget)
        self._work_stack.setCurrentWidget(self._ledger_widget if state == 'ledger' else self._empty_project)
        self.header_import_btn.setVisible(state == 'ledger')
        self.empty_import_btn.setVisible(state == 'empty')
        name = str(getattr(self._summary, 'name', '') or '项目')
        self._project_heading.setText(f'{name} · {len(self._lines)} 条测线')
        enabled = self._has_project() and not self._busy
        for btn in (self.close_btn, self.preflight_btn):
            btn.setEnabled(enabled)
        writable = self._can_write()
        self.import_btn.setEnabled(writable)
        self.header_import_btn.setEnabled(writable)
        self.empty_import_btn.setEnabled(writable)
        self.sync_btn.setEnabled(writable and bool(self._current_line_id))
        self.sensor_tool_btn.setEnabled(writable and bool(self._current_line_id))
        self.process_line_btn.setEnabled(enabled and bool(self._current_line_id))
        self.process_line_btn.setText('继续处理' if self._current_line_id in self._resume_lines else '打开处理')
        self.preview_artifact_btn.setEnabled(
            enabled and bool(self._current_line_id)
            and 0 <= self._artifacts_table.currentRow() < len(self._artifacts))
        for row in range(self._artifacts_table.rowCount()):
            button = self._artifacts_table.cellWidget(row, 5)
            if button is not None:
                button.setEnabled(writable and bool(self._current_line_id))
        read_only = bool(getattr(self._summary, 'read_only', False))
        self._no_project_hint.setText(
            '当前项目为只读：可预检和查看，导入、同步与删除已停用' if read_only
            else '')
        self._no_project_hint.setVisible(read_only)
        self.sync_btn.setToolTip('同步传感器到当前选中测线' if self._current_line_id
                                 else '请先在测线台账中选择一条测线')
        rows = self._lines_table.selectionModel().selectedRows()
        selected = next((line for line in self._lines
                         if str(getattr(line, 'line_id', '') or '') == self._current_line_id), None)
        if selected is None:
            hint = f'已选 {len(rows)} 条测线 · 可从更多菜单批量删除' if len(rows) > 1 else '选择一条测线以查看详情'
            self._selected_heading.setText(hint)
            self._line_detail.setText(hint)
            self._sensor_detail.setText('请先选择一条测线。')
        else:
            lid = self._current_line_id
            self._selected_heading.setText(str(getattr(selected, 'name', '') or lid))
            source = self._source_path_cache.get(lid) or '正在查询 / 未记录'
            quality = str(getattr(selected, 'data_quality', '') or '尚未检查')
            self._line_detail.setText(
                f'测线号：{lid}\n采样 × 道：{getattr(selected, "sample_count", 0)} × {getattr(selected, "trace_count", 0)}\n'
                f'数据检查：{quality}\n数据来源：{source}')
            self._sensor_detail.setText(
                f'当前测线：{lid}\n传感器文件由同步工具选择。预检和同步状态不代表正式成果验收。')
        if getattr(self, '_last_workspace_state', None) != state:
            self._last_workspace_state = state
            self.workspace_state_changed.emit(state)

    def _can_write(self) -> bool:
        return (self._has_project() and not self._busy
                and not bool(getattr(self._summary, 'read_only', False)))

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
        for i in range(1, len(existing) + 2):
            candidate = f'L{i:02d}'
            if candidate not in existing:
                return candidate
        raise AssertionError('a free line id must exist')

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
            self._preflight_label.set_hint_key('info')
        self.import_requested.emit(payload)

    def _emit_sync_request(self) -> None:
        if not self._can_write() or not self._current_line_id:
            return
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
        line_id = self._current_line_id
        self.sync_requested.emit({'line_id': line_id, 'paths': paths})

    def set_line_source_path(self, line_id: str, path) -> None:
        """主窗口/controller 异步回包：缓存测线源文件路径（右键菜单素材）。

        由 page_coordinator 把 ProjectController.line_source_path_ready
        接到本方法；查询在 worker 线程完成，右键菜单构建时直接读缓存。
        """
        line_id = str(line_id or '')
        if line_id:
            self._source_path_cache[line_id] = str(path) if path else None
            if line_id == self._current_line_id:
                self._update_action_state()

    # ------------------------------------------------------------ 右键菜单
    def _on_lines_context_menu(self, pos) -> None:
        row = self._lines_table.rowAt(pos.y())
        if row < 0 or row >= len(self._lines):
            return
        if not self._lines_table.selectionModel().isRowSelected(row, self._lines_table.rootIndex()):
            self._lines_table.selectRow(row)
        line_id = str(getattr(self._lines[row], 'line_id', '') or '')
        if not line_id:
            return
        source = self._source_path_cache.get(line_id)
        menu = make_menu(self)
        single = len(self._lines_table.selectionModel().selectedRows()) == 1
        add_action(menu, FIF.DEVELOPER_TOOLS, '处理该测线（跳转处理页）',
                   lambda: self.line_process_requested.emit(line_id), enabled=single and not self._busy)
        add_action(menu, FIF.SYNC, '传感器同步…', self.show_sensor_tool,
                   enabled=single and self._can_write())
        add_action(menu, FIF.DELETE, '删除所选测线',
                   self._on_delete_selected_lines, enabled=self._can_write())
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
        add_action(menu, FIF.VIEW, '在处理页查看',
                   self._emit_artifact_preview)
        add_action(menu, FIF.DELETE, '删除所选成果',
                   self._on_artifacts_context_delete, enabled=self._can_write())
        menu.exec(self._artifacts_table.viewport().mapToGlobal(pos))

    def _on_line_selection_changed(self) -> None:
        if self._filling_table:
            self._current_line_id = ''
            self._update_action_state()
            return
        if self._syncing_selection:
            # 程序化选中（select_line）：不回发，当前线由 select_line 维护
            return
        rows = self._lines_table.selectionModel().selectedRows()
        row = rows[0].row() if len(rows) == 1 else -1
        if row < 0 or row >= len(self._lines):
            self._current_line_id = ''
            if self._artifacts:
                self.set_artifacts([])
            self._update_action_state()
            return
        line_id = str(getattr(self._lines[row], 'line_id', '') or '')
        if not line_id:
            return
        self._current_line_id = line_id
        self._update_action_state()
        self.line_selected.emit(line_id)

    def _emit_line_process_request(self, item) -> None:
        """双击测线行 → 请主窗口跳转处理页处理该测线（单击已保证选中）。"""
        row = self._lines_table.row(item)
        if row < 0 or row >= len(self._lines):
            return
        line_id = str(getattr(self._lines[row], 'line_id', '') or '')
        if line_id and line_id == self._current_line_id and self.process_line_btn.isEnabled():
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
        if not self._can_write():
            return
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
        if not self._can_write():
            return
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
        self.startup.set_settings_manager(sm)
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
