# -*- coding: utf-8 -*-
"""ProcessingPage — 处理工作台（SPEC §6.5）。

三栏 QHBoxLayout：
- 左栏 ScrollArea 固定 320px：卡片"方法库"（MethodBrowser）
- 中栏 stretch：卡片"数据预览"（SegmentedWidget 原始数据/处理结果 + BScanView
  + colormap ComboBox + p_low/p_high + 刷新色阶 + 加载测线数据）+ 进度条（初始隐藏）
- 右栏 ScrollArea 固定 340px：卡片"处理链"（PipelineList + 添加所选方法）、
  卡片"参数设置"（ParamForm + 应用到选中步骤）、卡片"执行"、卡片"AutoTune 自动调参"

页面纯展示 + 发信号，不直接调 controller/backend。
内部联动：PipelineList.sig_step_selected → ParamForm 载入该步骤参数；
"应用到选中步骤"按钮 → 表单值写回选中步骤。
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel, CardWidget, ComboBox, DoubleSpinBox, InfoBar,
    InfoBarPosition, LineEdit, PrimaryPushButton, ProgressBar, PushButton,
    ScrollArea, SegmentedWidget, SubtitleLabel,
)
from qfluentwidgets import FluentIcon as FIF

from ui.desktop_backend_facade import compute_display_levels
from ui import constants
from ui.widgets import (BScanView, CollapsiblePanel, MethodBrowser, ParamForm,
                        PipelineList, mark_invalid, clear_invalid,
                        validate_non_empty)

# 预览分段（SegmentedWidget routeKey）
_SEG_ORIGINAL = 'originalData'
_SEG_RESULT = 'processResult'


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


def _make_card(title: str) -> tuple:
    """卡片范式：CardWidget + QVBoxLayout spacing=10 margins=(15,15,15,15)，
    首行 SubtitleLabel 卡片标题。返回 (card, layout)。"""
    card = CardWidget()
    layout = QVBoxLayout(card)
    layout.setContentsMargins(*constants.CARD_MARGINS)
    layout.setSpacing(constants.CARD_SPACING)
    layout.addWidget(_card_title(title))
    return card, layout


def _create_separator() -> QFrame:
    """分隔线工厂：QFrame.HLine + Sunken + color:#e0e0e0（SPEC §1）。"""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet('color: #e0e0e0;')
    return line


def _make_scroll_column(width: int) -> tuple:
    """固定宽滚动栏：ScrollArea(固定 width，透明、隐藏横向滚动条) +
    内容 widget(固定 width-16，透明) + 内容 QVBoxLayout(spacing=15)。
    返回 (scroll_area, content_layout)。"""
    scroll = ScrollArea()
    scroll.setFixedWidth(width)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setStyleSheet(
        'QScrollArea { background-color: transparent; border: none; }')
    content = QWidget(scroll)
    content.setFixedWidth(width - 16)
    content.setObjectName('pageScrollContent')
    content.setStyleSheet(
        'QWidget#pageScrollContent { background-color: transparent; }')
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(constants.PAGE_SPACING)
    scroll.setWidget(content)
    return scroll, layout


class ProcessingPage(QWidget):
    """处理工作台页面。"""

    run_requested = pyqtSignal(dict)            # current_pipeline() 载荷（含 steps）
    cancel_requested = pyqtSignal()
    autotune_requested = pyqtSignal(str, dict)  # method_id, params_hint
    line_load_requested = pyqtSignal()
    line_changed = pyqtSignal(str)              # 处理页测线选择变化
    artifact_selected = pyqtSignal(str)         # 处理页成果选择变化

    def __init__(self, parent=None):
        super().__init__(parent)
        self._methods = []
        self._methods_by_id = {}
        self._original_bundle = None
        self._result_bundle = None
        self._running = False
        self._job_id = ''
        self._selected_step = -1
        self._autotune_result = None    # (method_id, dict)
        self._selected_method_id = ''   # 方法库当前选中方法
        self._line_ids: list[str] = []  # 与 _line_combo 逐项对应的 line_id
        self._artifact_ids: list[str] = []  # 与 _artifact_combo 逐项对应的 artifact_id

        self._build_ui()
        self._connect_internal()
        self._restore_panel_state()

    # ============================================================ 面板状态
    def panel_states(self) -> dict:
        """当前左右面板折叠状态。"""
        return {
            'left': self._left_panel.is_collapsed(),
            'right': self._right_panel.is_collapsed(),
        }

    def set_panel_collapsed(self, *, left: bool = None, right: bool = None,
                            animate: bool = True) -> None:
        """设置左右面板折叠状态。"""
        if left is not None:
            self._left_panel.set_collapsed(bool(left), animate=animate)
        if right is not None:
            self._right_panel.set_collapsed(bool(right), animate=animate)

    def _restore_panel_state(self) -> None:
        """从 SettingsManager 恢复折叠状态。"""
        from ui.settings_manager import SettingsManager
        sm = SettingsManager()
        self._left_panel.set_collapsed(
            bool(sm.get('processing_left_collapsed', False)), animate=False)
        self._right_panel.set_collapsed(
            bool(sm.get('processing_right_collapsed', False)), animate=False)

    def _save_panel_state(self) -> None:
        """把当前折叠状态写回 SettingsManager。"""
        from ui.settings_manager import SettingsManager
        sm = SettingsManager()
        sm.set('processing_left_collapsed', self._left_panel.is_collapsed())
        sm.set('processing_right_collapsed', self._right_panel.is_collapsed())
        sm.save()

    # ============================================================ UI 构建
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)
        root.addWidget(_page_title('处理工作台'))

        columns = QHBoxLayout()
        columns.setSpacing(constants.PAGE_SPACING)
        root.addLayout(columns, 1)

        # ---------------- 左栏（展开 320px，可折叠；滚动栏宽须与面板展开宽一致）
        left_scroll, left_layout = _make_scroll_column(320)
        left_panel = CollapsiblePanel(
            'left', expand_width=320, collapse_width=40, parent=self)
        left_panel.set_content_widget(left_scroll)
        columns.addWidget(left_panel)
        self._left_panel = left_panel

        methods_card, methods_layout = _make_card('方法库')
        self._method_browser = MethodBrowser(methods_card)
        self._method_browser.setMinimumHeight(320)
        methods_layout.addWidget(self._method_browser, 1)
        left_layout.addWidget(methods_card, 1)
        left_layout.addStretch(1)

        # ---------------- 中栏（stretch）
        middle = QWidget(self)
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(constants.PAGE_SPACING)
        columns.addWidget(middle, 1)

        preview_card, preview_layout = _make_card('数据预览')
        seg_row = QHBoxLayout()
        seg_row.setSpacing(constants.CARD_SPACING)
        self._preview_segment = SegmentedWidget(preview_card)
        self._preview_segment.addItem(
            _SEG_ORIGINAL, '原始数据',
            onClick=lambda: self._show_bundle(_SEG_ORIGINAL))
        self._preview_segment.addItem(
            _SEG_RESULT, '处理结果',
            onClick=lambda: self._show_bundle(_SEG_RESULT))
        self._preview_segment.setCurrentItem(_SEG_ORIGINAL)
        seg_row.addWidget(self._preview_segment)
        seg_row.addStretch(1)

        line_label = CaptionLabel('当前测线:', preview_card)
        seg_row.addWidget(line_label)
        self._line_combo = ComboBox(preview_card)
        self._line_combo.setMinimumWidth(130)
        self._line_combo.setToolTip('在处理页直接切换当前测线')
        seg_row.addWidget(self._line_combo)

        art_label = CaptionLabel('成果:', preview_card)
        seg_row.addWidget(art_label)
        self._artifact_combo = ComboBox(preview_card)
        self._artifact_combo.setMinimumWidth(150)
        self._artifact_combo.setToolTip('选择该测线历次处理结果进行预览')
        seg_row.addWidget(self._artifact_combo)

        self._load_line_btn = PushButton('加载测线数据', preview_card)
        self._load_line_btn.setToolTip('加载当前测线的原始数据到预览区（Ctrl+L）')
        seg_row.addWidget(self._load_line_btn)
        preview_layout.addLayout(seg_row)

        self._bscan = BScanView(preview_card)
        self._bscan.setMinimumHeight(300)
        preview_layout.addWidget(self._bscan, 1)

        # 色阶工具行：控件用前缀代替独立标签、收窄最小宽，保证窄屏
        # （左右栏均展开）时整行不被裁切。
        tool_row = QHBoxLayout()
        tool_row.setSpacing(constants.CARD_SPACING)
        cmap_label = CaptionLabel('色阶:', preview_card)
        tool_row.addWidget(cmap_label)
        self._cmap_combo = ComboBox(preview_card)
        self._cmap_combo.addItems(constants.COLORMAPS)
        self._cmap_combo.setCurrentText(constants.DEFAULT_COLORMAP)
        self._cmap_combo.setMinimumWidth(90)
        self._cmap_combo.setToolTip('选择 B-scan 色标')
        tool_row.addWidget(self._cmap_combo)
        tool_row.addStretch(1)
        self._p_low_spin = DoubleSpinBox(preview_card)
        self._p_low_spin.setRange(0.0, 100.0)
        self._p_low_spin.setDecimals(1)
        self._p_low_spin.setSingleStep(0.5)
        self._p_low_spin.setValue(0.0)
        self._p_low_spin.setPrefix('低% ')
        self._p_low_spin.setMinimumWidth(76)
        self._p_low_spin.setToolTip('色阶下限百分位')
        tool_row.addWidget(self._p_low_spin)
        self._p_high_spin = DoubleSpinBox(preview_card)
        self._p_high_spin.setRange(0.0, 100.0)
        self._p_high_spin.setDecimals(1)
        self._p_high_spin.setSingleStep(0.5)
        self._p_high_spin.setValue(100.0)
        self._p_high_spin.setPrefix('高% ')
        self._p_high_spin.setMinimumWidth(76)
        self._p_high_spin.setToolTip('色阶上限百分位')
        tool_row.addWidget(self._p_high_spin)
        self._refresh_levels_btn = PushButton('刷新色阶', preview_card)
        self._refresh_levels_btn.setToolTip('按当前百分位重新计算色阶')
        tool_row.addWidget(self._refresh_levels_btn)
        preview_layout.addLayout(tool_row)
        middle_layout.addWidget(preview_card, 1)

        # 进度条（初始隐藏）
        self._progress_bar = ProgressBar(middle)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        middle_layout.addWidget(self._progress_bar)

        # ---------------- 右栏（展开 340px，可折叠；滚动栏宽须与面板展开宽一致）
        right_scroll, right_layout = _make_scroll_column(340)
        right_panel = CollapsiblePanel(
            'right', expand_width=340, collapse_width=40, parent=self)
        right_panel.set_content_widget(right_scroll)
        columns.addWidget(right_panel)
        self._right_panel = right_panel

        pipeline_card, pipeline_layout = _make_card('处理链')
        self._pipeline_list = PipelineList(pipeline_card)
        self._pipeline_list.setMinimumHeight(200)
        pipeline_layout.addWidget(self._pipeline_list, 1)
        self._add_method_btn = PushButton('添加所选方法', pipeline_card)
        self._add_method_btn.setToolTip('将左侧方法库当前选中的方法加入处理链')
        pipeline_layout.addWidget(self._add_method_btn)
        right_layout.addWidget(pipeline_card, 1)

        param_card, param_layout = _make_card('参数设置')
        self._param_form = ParamForm(param_card)
        param_layout.addWidget(self._param_form, 1)
        self._apply_params_btn = PushButton('应用到选中步骤', param_card)
        self._apply_params_btn.setToolTip('把当前参数表单内容写入处理链当前选中步骤')
        self._apply_params_btn.setEnabled(False)
        param_layout.addWidget(self._apply_params_btn)
        right_layout.addWidget(param_card, 1)

        exec_card, exec_layout = _make_card('执行')
        name_row = QHBoxLayout()
        name_row.setSpacing(constants.CARD_SPACING)
        name_label = CaptionLabel('结果名称:', exec_card)
        name_label.setMinimumWidth(100)
        name_row.addWidget(name_label)
        self._result_name_edit = LineEdit(exec_card)
        self._result_name_edit.setPlaceholderText('例如: 增益处理后结果')
        self._result_name_edit.setToolTip('处理成果保存名称')
        name_row.addWidget(self._result_name_edit, 1)
        exec_layout.addLayout(name_row)
        run_row = QHBoxLayout()
        run_row.setSpacing(constants.CARD_SPACING)
        self._run_btn = PrimaryPushButton('运行处理链', exec_card, FIF.PLAY)
        self._run_btn.setToolTip('执行右侧处理链（Ctrl+R）')
        self._cancel_btn = PushButton('取消', exec_card)
        self._cancel_btn.setToolTip('取消正在运行的处理任务')
        self._cancel_btn.setEnabled(False)
        run_row.addWidget(self._run_btn, 1)
        run_row.addWidget(self._cancel_btn)
        exec_layout.addLayout(run_row)
        right_layout.addWidget(exec_card)

        autotune_card, autotune_layout = _make_card('AutoTune 自动调参')
        method_row = QHBoxLayout()
        method_row.setSpacing(constants.CARD_SPACING)
        method_label = CaptionLabel('当前方法:', autotune_card)
        method_label.setMinimumWidth(100)
        method_row.addWidget(method_label)
        self._autotune_method_label = CaptionLabel('--', autotune_card)
        method_row.addWidget(self._autotune_method_label, 1)
        autotune_layout.addLayout(method_row)
        self._autotune_btn = PushButton('开始调参', autotune_card)
        self._autotune_btn.setToolTip('对左侧选中的方法自动搜索最优参数')
        self._autotune_btn.setEnabled(False)
        autotune_layout.addWidget(self._autotune_btn)
        autotune_layout.addWidget(_create_separator())
        self._autotune_result_label = CaptionLabel('暂无调参结果', autotune_card)
        self._autotune_result_label.setWordWrap(True)
        autotune_layout.addWidget(self._autotune_result_label)
        self._adopt_params_btn = PushButton('采用最优参数', autotune_card)
        self._adopt_params_btn.setToolTip('把调参结果写入处理链当前步骤')
        self._adopt_params_btn.setEnabled(False)
        autotune_layout.addWidget(self._adopt_params_btn)
        right_layout.addWidget(autotune_card)
        right_layout.addStretch(1)

    # ============================================================ 内部接线
    def _connect_internal(self) -> None:
        # 方法库 → 选中方法（AutoTune 目标）/ 双击添加 / 按钮添加
        self._method_browser.sig_method_selected.connect(self._on_method_selected)
        self._method_browser.sig_add_requested.connect(self._add_method_to_pipeline)
        self._add_method_btn.clicked.connect(self._on_add_selected_method)

        # 处理链 ↔ 参数表单
        self._pipeline_list.sig_step_selected.connect(self._on_step_selected)
        self._apply_params_btn.clicked.connect(self._apply_params_to_selected)

        # 预览
        self._cmap_combo.currentTextChanged.connect(self._bscan.set_colormap)
        # 反向同步：B-scan 右键菜单改色标 → ComboBox 跟随（防两处状态不一致）
        self._bscan.sig_colormap_changed.connect(
            self._cmap_combo.setCurrentText)
        self._refresh_levels_btn.clicked.connect(self._refresh_levels)
        self._load_line_btn.clicked.connect(self.line_load_requested)
        self._line_combo.currentIndexChanged.connect(self._on_line_combo_changed)
        self._artifact_combo.currentIndexChanged.connect(self._on_artifact_combo_changed)

        # 执行
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._cancel_btn.clicked.connect(self.cancel_requested)

        # AutoTune
        self._autotune_btn.clicked.connect(self._on_autotune_clicked)
        self._adopt_params_btn.clicked.connect(self._on_adopt_params)

        # 快捷键：运行 / 加载测线
        self._run_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        self._run_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._run_shortcut.activated.connect(self._on_run_clicked)
        self._load_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        self._load_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._load_shortcut.activated.connect(self.line_load_requested)

        # 面板折叠状态持久化
        self._left_panel.sig_collapsed.connect(self._save_panel_state)
        self._right_panel.sig_collapsed.connect(self._save_panel_state)

    # ============================================================ 公共接口（供主窗口接线）
    def set_methods(self, methods: list) -> None:
        """方法列表 → MethodBrowser（结构见 ProcessingController.methods_loaded）。"""
        self._methods = [dict(m) for m in (methods or [])]
        self._methods_by_id = {m.get('method_id', ''): m for m in self._methods}
        self._method_browser.set_methods(self._methods)

    def set_original_bundle(self, bundle) -> None:
        """原始数据预览 bundle。"""
        self._original_bundle = bundle
        if self._current_segment() == _SEG_ORIGINAL:
            self._show_bundle(_SEG_ORIGINAL)

    def set_result_bundle(self, bundle) -> None:
        """处理结果预览 bundle。"""
        self._result_bundle = bundle
        if self._current_segment() == _SEG_RESULT:
            self._show_bundle(_SEG_RESULT)

    def set_running(self, running: bool, job_id: str = '') -> None:
        """运行态切换：运行按钮/取消按钮互斥 + 进度条显隐。"""
        self._running = bool(running)
        self._job_id = job_id or ''
        self._run_btn.setEnabled(not self._running)
        self._cancel_btn.setEnabled(self._running)
        self._progress_bar.setVisible(self._running)
        if self._running:
            self._progress_bar.setValue(0)

    def set_progress(self, completed: int, total: int, message: str) -> None:
        """进度更新：total>0 按比例，否则按百分数；message 写入 tooltip。"""
        if total and total > 0:
            self._progress_bar.setRange(0, int(total))
            self._progress_bar.setValue(min(int(completed), int(total)))
        else:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(max(0, min(int(completed), 100)))
        self._progress_bar.setToolTip(message or '')

    def set_autotune_result(self, method_id: str, result: dict) -> None:
        """AutoTune 结果 {best_params, ...} → CaptionLabel 区 + 暂存最优参数。"""
        result = dict(result or {})
        self._autotune_result = (method_id, result)
        best = result.get('best_params') or {}
        method = self._methods_by_id.get(method_id, {})
        display = method.get('display_name') or method_id
        lines = ['方法: %s' % display]
        if 'score' in result:
            lines.append('评分: %s' % result.get('score'))
        if 'metric' in result:
            lines.append('指标: %s' % result.get('metric'))
        if best:
            params_text = ', '.join('%s=%s' % (k, v) for k, v in best.items())
            lines.append('最优参数: %s' % params_text)
        else:
            lines.append('最优参数: (无)')
        self._autotune_result_label.setText('\n'.join(lines))
        self._adopt_params_btn.setEnabled(bool(best))

    def set_line_label(self, text: str) -> None:
        """当前测线标签（同步到测线选择下拉，不触发信号）。"""
        self._set_line_combo_without_emit(str(text or ''))

    def set_lines(self, lines: list) -> None:
        """测线列表 → 处理页测线选择下拉。"""
        previous = str(self._line_combo.currentText() or '')
        # 从显示文本解析 line_id（兼容 qfluentwidgets ComboBox 不保存 userData）
        previous_id = self._line_ids[self._line_combo.currentIndex()] if self._line_combo.currentIndex() >= 0 else ''
        self._line_combo.blockSignals(True)
        self._line_combo.clear()
        self._line_ids = []
        for line in lines or []:
            line_id = str(getattr(line, 'line_id', '') or '')
            name = str(getattr(line, 'name', '') or '')
            display = f"{line_id} {name}".strip()
            self._line_combo.addItem(display or line_id)
            self._line_ids.append(line_id)
        self._line_combo.blockSignals(False)
        self._set_line_combo_without_emit(previous_id)

    def set_artifacts(self, artifacts: list) -> None:
        """成果列表 → 处理页成果选择下拉；默认选中最新一条。"""
        self._artifact_combo.blockSignals(True)
        self._artifact_combo.clear()
        self._artifact_ids = []
        for art in artifacts or []:
            artifact_id = str(getattr(art, 'artifact_id', '') or '')
            name = str(getattr(art, 'name', '') or '')
            created = str(getattr(art, 'created_at', '') or '')
            display = f"{name}  {created}".strip()
            self._artifact_combo.addItem(display or artifact_id)
            self._artifact_ids.append(artifact_id)
        self._artifact_combo.blockSignals(False)
        if self._artifact_combo.count():
            self._artifact_combo.setCurrentIndex(self._artifact_combo.count() - 1)

    def _set_line_combo_without_emit(self, line_id: str) -> None:
        line_id = str(line_id or '')
        try:
            idx = self._line_ids.index(line_id)
        except ValueError:
            idx = 0 if self._line_ids else -1
        if idx >= 0:
            self._line_combo.blockSignals(True)
            self._line_combo.setCurrentIndex(idx)
            self._line_combo.blockSignals(False)

    def _on_line_combo_changed(self, index: int) -> None:
        if 0 <= index < len(self._line_ids):
            self.line_changed.emit(self._line_ids[index])

    def _on_artifact_combo_changed(self, index: int) -> None:
        if 0 <= index < len(self._artifact_ids):
            self.artifact_selected.emit(self._artifact_ids[index])

    def current_pipeline(self) -> dict:
        """当前处理链定义：{"steps": [...], "result_name": str}。"""
        return {
            'steps': self._pipeline_list.steps(),
            'result_name': self._result_name_edit.text().strip(),
        }

    # ============================================================ 内部逻辑
    def _current_segment(self) -> str:
        item = self._preview_segment.currentItem()
        return item.property('routeKey') if item is not None else _SEG_ORIGINAL

    def _show_bundle(self, which: str) -> None:
        bundle = (self._original_bundle if which == _SEG_ORIGINAL
                  else self._result_bundle)
        if bundle is None:
            self._bscan.clear()
            return
        self._bscan.set_bundle(bundle)

    def _refresh_levels(self) -> None:
        """按 p_low/p_high 百分比重算当前 bundle 的显示色阶。"""
        bundle = (self._original_bundle
                  if self._current_segment() == _SEG_ORIGINAL
                  else self._result_bundle)
        if bundle is None:
            InfoBar.info(title='数据预览', content='当前没有可刷新的预览数据',
                         orient=Qt.Orientation.Horizontal, isClosable=True,
                         position=InfoBarPosition.TOP, duration=2000,
                         parent=self)
            return
        p_low = float(self._p_low_spin.value())
        p_high = float(self._p_high_spin.value())
        if p_low >= p_high:
            InfoBar.warning(title='数据预览', content='低百分比必须小于高百分比',
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=3000,
                            parent=self)
            return
        vmin, vmax = compute_display_levels(bundle.matrix, p_low=p_low, p_high=p_high)
        self._bscan.set_matrix(
            bundle.matrix, vmin, vmax,
            title=getattr(bundle, 'title', ''),
            x_label=getattr(bundle, 'x_label', '道数'),
            y_label=getattr(bundle, 'y_label', '采样点'))

    # ---------------- 方法库
    def _on_method_selected(self, method_id: str) -> None:
        self._selected_method_id = method_id
        method = self._methods_by_id.get(method_id, {})
        self._autotune_method_label.setText(
            method.get('display_name') or method_id or '--')
        self._autotune_btn.setEnabled(bool(method_id))

    def _on_add_selected_method(self) -> None:
        method_id = self._method_browser.current_method_id()
        if not method_id:
            InfoBar.warning(title='处理链', content='请先在方法库中选择方法',
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=3000,
                            parent=self)
            return
        self._add_method_to_pipeline(method_id)

    def _add_method_to_pipeline(self, method_id: str) -> None:
        method = self._methods_by_id.get(method_id, {})
        label = method.get('display_name') or method.get('name') or method_id
        params = {item.get('name'): item.get('default')
                  for item in (method.get('parameter_schema') or [])
                  if item.get('name') is not None}
        self._pipeline_list.add_step(method_id, label, params)

    # ---------------- 处理链 ↔ 参数表单
    def _on_step_selected(self, index: int) -> None:
        self._selected_step = index if index >= 0 else -1
        steps = self._pipeline_list.steps()
        if not (0 <= self._selected_step < len(steps)):
            self._param_form.clear()
            self._apply_params_btn.setEnabled(False)
            return
        step = steps[self._selected_step]
        method = self._methods_by_id.get(step.get('method_id', ''), {})
        schema = method.get('parameter_schema') or []
        if schema:
            self._param_form.set_schema(schema)
            self._param_form.set_values(step.get('params') or {})
        else:
            # 无 schema：保持表单为空
            self._param_form.clear()
        self._apply_params_btn.setEnabled(bool(schema))

    def _apply_params_to_selected(self) -> None:
        """"应用到选中步骤"按钮：表单值写回选中步骤。"""
        if self._selected_step < 0:
            InfoBar.warning(title='参数设置', content='请先选中处理链中的步骤',
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=3000,
                            parent=self)
            return
        values = self._param_form.values()
        if not self._pipeline_list.update_step_params(self._selected_step, values):
            InfoBar.warning(title='参数设置', content='处理链步骤已失效，请重新选择',
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=3000,
                            parent=self)
            return
        InfoBar.success(title='参数设置', content='参数已应用到选中步骤',
                        orient=Qt.Orientation.Horizontal, isClosable=True,
                        position=InfoBarPosition.TOP, duration=2000,
                        parent=self)

    # ---------------- 执行
    def _on_run_clicked(self) -> None:
        if self._running:
            return
        steps = self._pipeline_list.steps()
        if not steps:
            InfoBar.warning(title='处理链', content='处理链为空，请先添加处理步骤',
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=3000,
                            parent=self)
            return
        ok, msg = validate_non_empty(self._result_name_edit.text(), '结果名称')
        if not ok:
            mark_invalid(self._result_name_edit, msg)
            InfoBar.warning(title='执行', content=msg,
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=3000,
                            parent=self)
            return
        clear_invalid(self._result_name_edit)
        self.run_requested.emit(self.current_pipeline())

    # ---------------- AutoTune
    def _on_autotune_clicked(self) -> None:
        method_id = self._selected_method_id
        if not method_id:
            InfoBar.warning(title='AutoTune 自动调参',
                            content='请先在方法库中选择方法',
                            orient=Qt.Orientation.Horizontal, isClosable=True,
                            position=InfoBarPosition.TOP, duration=3000,
                            parent=self)
            return
        params_hint = self._param_form.values()
        self.autotune_requested.emit(method_id, params_hint)

    def _on_adopt_params(self) -> None:
        if self._autotune_result is None:
            return
        method_id, result = self._autotune_result
        best = dict(result.get('best_params') or {})
        if not best:
            return
        # 选中步骤与方法一致 → 写回选中步骤；否则写回处理链中第一个同方法
        # 步骤并选中；处理链中没有同方法步骤时仅载入参数表单
        steps = self._pipeline_list.steps()
        target = -1
        if (0 <= self._selected_step < len(steps)
                and steps[self._selected_step].get('method_id') == method_id):
            target = self._selected_step
        else:
            for i, step in enumerate(steps):
                if step.get('method_id') == method_id:
                    target = i
                    break
        if target >= 0:
            merged = dict(steps[target].get('params') or {})
            merged.update(best)
            steps[target]['params'] = merged
            self._pipeline_list.set_steps(steps)
            self._pipeline_list._list.setCurrentRow(target)
            self._pipeline_list.sig_changed.emit()
        else:
            self._param_form.set_values(best)
        InfoBar.success(title='AutoTune 自动调参', content='已采用最优参数',
                        orient=Qt.Orientation.Horizontal, isClosable=True,
                        position=InfoBarPosition.TOP, duration=2000,
                        parent=self)
