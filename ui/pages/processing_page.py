# -*- coding: utf-8 -*-
"""ProcessingPage — 处理工作台（SPEC §6.5）。

三栏 QHBoxLayout：
- 左栏 ScrollArea 固定 320px：卡片"方法库"（MethodBrowser）
- 中栏 stretch：卡片"数据预览"（标题与 SlimSegment 原始数据/处理结果同行
  header + BScanContainer 多视图 + colormap ComboBox + p_low/p_high +
  刷新色阶）+ 进度条（初始隐藏）
- 右栏 ScrollArea 固定 340px：卡片"处理链"（PipelineList + 添加所选方法）、
  卡片"参数设置"（ParamForm + 应用到选中步骤）、卡片"执行"（输入数据选择
  支持从某个成果继续处理 + 结果名 + 运行/取消）、卡片"AutoTune 自动调参"

预览布局（BScanContainer，Phase 2）：
- auto（默认）：面板数自动跟随数据——只有原始数据→单视图（分段控件
  切换显示），原始+成果齐→自动变 0 号位原始 | 1 号位成果同屏对比；
- single：固定单视图，分段控件切换显示原始数据 / 处理结果（历史行为）；
- dual：固定左右并排双视图，分段控件此时决定色阶刷新焦点；
- quad：固定 2×2 四宫格，0/1 号位与 dual 相同，2/3 号位留空占位；
- free：自由窗口（Windows 视窗式）——两个可拖动/缩放/最大化的子窗口
  （0 号位原始数据、1 号位处理结果），右键空白处平铺/层叠/重置。

页面纯展示 + 发信号，不直接调 controller/backend。
内部联动：PipelineList.sig_step_selected → ParamForm 载入该步骤参数；
"应用到选中步骤"按钮 → 表单值写回选中步骤。
"""

from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QHBoxLayout, QVBoxLayout, QWidget)
from qfluentwidgets import (
    CaptionLabel, ComboBox, InfoBar,
    InfoBarPosition, LineEdit, PrimaryPushButton, ProgressBar, PushButton,
)
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.motion import animate_progress
from ui.page_scaffold import (PanelStateMixin, make_card, make_form_row,
                              make_scroll_column, make_segment_card,
                              refill_combo)
from ui.widgets import (BScanContainer, BScanView, CollapsiblePanel,
                        MethodBrowser, ParamForm, PipelineList, SlimSegment,
                        clear_invalid, make_separator)
from ui.widgets.bscan_container import LAYOUT_AUTO, LAYOUT_SINGLE

# 预览分段（SlimSegment routeKey）
_SEG_ORIGINAL = 'originalData'
_SEG_RESULT = 'processResult'


def _short_timestamp(raw: str) -> str:
    """'2026-09-22T14:36:50' → '09-22 14:36'；空串/解析失败回落原文。

    成果下拉宽度有限，ISO 全时间戳必然截断（截断的时间不可读），
    短格式保住「同测线多次处理」的区分信息。
    """
    text = str(raw or '').strip()
    if not text:
        return ''
    try:
        return datetime.fromisoformat(text).strftime('%m-%d %H:%M')
    except ValueError:
        return text


class ProcessingPage(PanelStateMixin, QWidget):
    """处理工作台页面。"""

    run_requested = pyqtSignal(dict)            # current_pipeline() 载荷（含 steps）
    cancel_requested = pyqtSignal()
    autotune_requested = pyqtSignal(str, dict, str)  # method_id, params_hint, input_artifact_id
    line_changed = pyqtSignal(str)              # 处理页测线选择变化
    artifact_selected = pyqtSignal(str)         # 处理页成果选择变化
    # 批量处理（B4）UI 已按用户决策暂时屏蔽（2026-09-02）：卡片、信号与
    # 接线整体撤下；后端 run_pipeline_batch 契约保留，恢复时重建本页卡片
    # 并回接 page_coordinator._on_batch_run_requested 即可。

    _PANEL_STATE_PREFIX = 'processing'

    def __init__(self, parent=None):
        super().__init__(parent)
        self._methods = []
        self._methods_by_id = {}
        self._original_bundle = None
        self._result_bundle = None
        self._auto_sticky_dual = False    # 粘性 auto：见 _sync_auto_layout
        self._running = False
        self._job_id = ''
        self._selected_step = -1
        self._autotune_result = None    # (method_id, dict)
        self._autotune_running = False  # AutoTune 运行中（防重复提交）
        self._selected_method_id = ''   # 方法库当前选中方法
        self._sm = None                 # 共享 SettingsManager（主窗口注入，唯一写者）

        self._build_ui()
        self._connect_internal()
        self._restore_panel_state()

    # ============================================================ 设置注入
    def set_settings_manager(self, sm) -> None:
        """注入主窗口共享的 SettingsManager（唯一写者）并恢复折叠状态。"""
        self._sm = sm
        self._restore_panel_state()

    # ============================================================ UI 构建
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)

        columns = QHBoxLayout()
        columns.setSpacing(constants.PAGE_SPACING)
        root.addLayout(columns, 1)

        # ---------------- 左栏（展开 SIDE_TOOL_WIDTH px，可折叠；滚动栏宽须与面板展开宽一致）
        left_scroll, left_layout = make_scroll_column(constants.SIDE_TOOL_WIDTH)
        left_panel = CollapsiblePanel(
            'left', expand_width=constants.SIDE_TOOL_WIDTH, collapse_width=40, parent=self)
        left_panel.set_content_widget(left_scroll)
        columns.addWidget(left_panel)
        self._left_panel = left_panel

        methods_card, methods_layout = make_card('方法库')
        self._method_browser = MethodBrowser(methods_card)
        self._method_browser.setMinimumHeight(320)
        methods_layout.addWidget(self._method_browser, 1)
        # 卡片占满左栏全部可用高度，不再在底部留空白
        left_layout.addWidget(methods_card, 1)

        # ---------------- 中栏（stretch）
        middle = QWidget(self)
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(constants.PAGE_SPACING)
        columns.addWidget(middle, 1)

        # header 单行化：标题居左 + 原始/结果瘦页签居右（make_segment_card
        # 范式）；选择器行（测线 / 成果 下拉）保持第二行。
        self._preview_segment = SlimSegment(self)
        self._preview_segment.addItem(
            _SEG_ORIGINAL, '原始数据',
            onClick=lambda: self._show_bundle(_SEG_ORIGINAL))
        self._preview_segment.addItem(
            _SEG_RESULT, '处理结果',
            onClick=lambda: self._show_bundle(_SEG_RESULT))
        self._preview_segment.setCurrentItem(_SEG_ORIGINAL)
        preview_card, preview_layout = make_segment_card(
            '数据预览', self._preview_segment, parent=self)

        sel_row = QHBoxLayout()
        sel_row.setSpacing(constants.CARD_SPACING)
        # 无前置标签：下拉内容自明（测线短号 / 「处理结果_」前缀成果名），
        # tooltip 承担语义说明；stretch 因子按文本长度分配。
        self._line_combo = ComboBox(preview_card)
        self._line_combo.setMinimumWidth(130)
        self._line_combo.setToolTip('当前测线：在处理页直接切换')
        sel_row.addWidget(self._line_combo, 1)
        self._artifact_combo = ComboBox(preview_card)
        self._artifact_combo.setMinimumWidth(150)
        self._artifact_combo.setToolTip('选择该测线历次处理结果进行预览')
        sel_row.addWidget(self._artifact_combo, 2)
        preview_layout.addLayout(sel_row)

        self._bscan_container = BScanContainer(preview_card)
        self._bscan_container.setMinimumHeight(constants.PREVIEW_MIN_HEIGHT)
        preview_layout.addWidget(self._bscan_container, 1)

        # 色阶工具行已退役（2026-09-23）：色标映射与色阶百分位收容进设置页
        # 「B-Scan 视图」卡（改值全量下发并持久化），单视图微调走 B-Scan
        # 右键菜单（色标子菜单 / 色阶设置…）——页面不再持有第二份状态源。
        middle_layout.addWidget(preview_card, 1)

        # 进度条 + 进度消息（初始隐藏）
        progress_row = QHBoxLayout()
        progress_row.setSpacing(constants.CARD_SPACING)
        self._progress_bar = ProgressBar(middle)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_row.addWidget(self._progress_bar, 1)
        self._progress_label = CaptionLabel('', middle)
        self._progress_label.setMinimumWidth(0)
        progress_row.addWidget(self._progress_label, 1)
        self._progress_row_widget = QWidget(middle)
        self._progress_row_widget.setLayout(progress_row)
        self._progress_row_widget.setVisible(False)
        middle_layout.addWidget(self._progress_row_widget)

        # ---------------- 右栏（展开 SIDE_FORM_WIDTH px，可折叠；滚动栏宽须与面板展开宽一致）
        right_scroll, right_layout = make_scroll_column(constants.SIDE_FORM_WIDTH)
        right_panel = CollapsiblePanel(
            'right', expand_width=constants.SIDE_FORM_WIDTH, collapse_width=40, parent=self)
        right_panel.set_content_widget(right_scroll)
        columns.addWidget(right_panel)
        self._right_panel = right_panel

        pipeline_card, pipeline_layout = make_card('处理链')
        self._pipeline_list = PipelineList(pipeline_card)
        self._pipeline_list.setMinimumHeight(200)
        pipeline_layout.addWidget(self._pipeline_list, 1)
        self._add_method_btn = PushButton('添加所选方法', pipeline_card)
        self._add_method_btn.setToolTip('将左侧方法库当前选中的方法加入处理链')
        pipeline_layout.addWidget(self._add_method_btn)
        right_layout.addWidget(pipeline_card, 1)

        param_card, param_layout = make_card('参数设置')
        self._param_form = ParamForm(param_card)
        param_layout.addWidget(self._param_form, 1)
        right_layout.addWidget(param_card, 1)

        exec_card, exec_layout = make_card('执行')
        self._input_combo = ComboBox(exec_card)
        self._input_combo.addItem('原始数据')
        self._input_combo.setToolTip(
            '处理链的输入：默认从原始数据开始；选择某个成果则在该成果基础上继续处理')
        exec_layout.addLayout(make_form_row(
            '输入数据:', self._input_combo, parent=exec_card))
        self._result_name_edit = LineEdit(exec_card)
        self._result_name_edit.setPlaceholderText('例如: 增益处理后结果')
        self._result_name_edit.setToolTip('处理成果保存名称')
        exec_layout.addLayout(make_form_row(
            '结果名称:', self._result_name_edit, parent=exec_card))
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

        autotune_card, autotune_layout = make_card('AutoTune 自动调参')
        self._autotune_method_label = CaptionLabel('--', autotune_card)
        autotune_layout.addLayout(make_form_row(
            '当前方法:', self._autotune_method_label, parent=autotune_card))
        self._autotune_btn = PushButton('开始调参', autotune_card)
        self._autotune_btn.setToolTip('对左侧选中的方法自动搜索最优参数')
        self._autotune_btn.setEnabled(False)
        autotune_layout.addWidget(self._autotune_btn)
        autotune_layout.addWidget(make_separator())
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
        # 参数表单值变化 → 自动写入选中步骤（任务 F 候选 4 需求确认书 B1：
        # 删除"应用到选中步骤"按钮，改值即生效，无需额外点击）
        self._param_form.sig_changed.connect(self._auto_write_params_to_selected)

        # 预览
        # 色标/色阶控件已收容进设置页（B-Scan 视图卡）与右键菜单；页面
        # 不再持有副本，视图偏好由主窗启动期统一恢复、用户操作镜像写回。
        self._line_combo.currentIndexChanged.connect(self._on_line_combo_changed)
        self._artifact_combo.currentIndexChanged.connect(self._on_artifact_combo_changed)
        # 布局切换：新面板是空白实例，重广播色标并重新分发 bundle
        self._bscan_container.sig_layout_changed.connect(self._on_layout_changed)

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

        # 面板折叠：状态持久化 + 窄窗自动折叠痕迹清理（尊重手动选择）
        self._left_panel.sig_collapsed.connect(
            lambda collapsed: self._on_side_panel_collapsed('left', collapsed))
        self._right_panel.sig_collapsed.connect(
            lambda collapsed: self._on_side_panel_collapsed('right', collapsed))

    # ============================================================ 公共接口（供主窗口接线）
    def toggle_side_panels(self) -> None:
        """B-Scan 工具条「⤢ 铺满」：两栏都收起则展开、否则全部收起。

        语义按「用户此刻想干什么」定：只要还有一栏占着宽度，就继续收；
        已全部收起时点一次恢复原状（此时按钮实际是个开关，符合直觉）。
        """
        collapse = not (self._left_panel.is_collapsed()
                        and self._right_panel.is_collapsed())
        self._left_panel.set_collapsed(collapse)
        self._right_panel.set_collapsed(collapse)

    def set_methods(self, methods: list) -> None:
        """方法列表 → MethodBrowser（结构见 ProcessingController.methods_loaded）。"""
        self._methods = [dict(m) for m in (methods or [])]
        self._methods_by_id = {m.get('method_id', ''): m for m in self._methods}
        self._method_browser.set_methods(self._methods)

    def set_original_bundle(self, bundle) -> None:
        """原始数据预览 bundle。

        auto 模式下先走粘性布局解析（见 _sync_auto_layout：原始数据是
        对比锚，锚没了粘性重置）；dual/quad/free 下 0 号位固定显示原始
        数据，无论分段停在哪一侧都要重发；single 下仅当分段选中
        "原始数据"时刷新。
        """
        self._original_bundle = bundle
        self._sync_auto_layout()
        if self._shows_both_panels():
            self._distribute_bundles()
        elif self._current_segment() == _SEG_ORIGINAL:
            self._show_bundle(_SEG_ORIGINAL)

    def set_result_bundle(self, bundle) -> None:
        """处理结果预览 bundle（分发语义同 set_original_bundle）。

        auto 模式下成果到达长出对比布局并粘住；bundle=None（删除/换线）
        只清成果位数据，**不**收回布局——看图时布局不跳变。
        """
        self._result_bundle = bundle
        self._sync_auto_layout()
        if self._shows_both_panels():
            self._distribute_bundles()
        elif self._current_segment() == _SEG_RESULT:
            self._show_bundle(_SEG_RESULT)

    def show_result_segment(self) -> None:
        """切换到"处理结果"预览分段（运行完成后自动展示新成果）。"""
        self._preview_segment.setCurrentItem(_SEG_RESULT)
        if self._current_segment() == _SEG_RESULT:
            self._show_bundle(_SEG_RESULT)

    def set_running(self, running: bool, job_id: str = '') -> None:
        """运行态切换：运行按钮/取消按钮互斥 + 进度条显隐。"""
        self._running = bool(running)
        self._job_id = job_id or ''
        self._run_btn.setEnabled(not self._running)
        self._cancel_btn.setEnabled(self._running)
        self._progress_row_widget.setVisible(self._running)
        if self._running:
            self._progress_bar.setValue(0)
            self._progress_label.setText('')

    def set_progress(self, completed: int, total: int, message: str) -> None:
        """进度更新：total>0 按比例，否则按百分数；message 显示在进度条右侧。"""
        if total and total > 0:
            self._progress_bar.setRange(0, int(total))
            animate_progress(self._progress_bar,
                             min(int(completed), int(total)))
        else:
            self._progress_bar.setRange(0, 100)
            animate_progress(self._progress_bar,
                             max(0, min(int(completed), 100)))
        self._progress_label.setText(str(message or ''))
        self._progress_label.setToolTip(str(message or ''))

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
        """测线列表 → 处理页测线选择下拉。

        显示去重：name 缺失或与 line_id 相同时只显示 line_id
        （否则出现「L01 L01」式拼接重复）。
        """

        def _line_display(line) -> str:
            line_id = str(getattr(line, 'line_id', '') or '').strip()
            name = str(getattr(line, 'name', '') or '').strip()
            if not name or name == line_id:
                return line_id
            return f"{line_id} {name}"

        refill_combo(
            self._line_combo, lines or [], _line_display,
            lambda line: str(getattr(line, 'line_id', '') or ''),
            previous_data=self._line_combo.currentData())

    def set_artifacts(self, artifacts: list) -> None:
        """成果列表 → 处理页成果选择下拉与执行卡输入数据下拉。

        保持用户当前选择（仍在列表中则不改动、不发射信号）；
        无有效选择时静默落到最新一条。自动预览由主窗口
        _preview_newest_artifact 路径统一负责，避免双重预览。
        """
        def _artifact_id(art) -> str:
            return str(getattr(art, 'artifact_id', '') or '')

        def _display(art) -> str:
            name = str(getattr(art, 'name', '') or '')
            created = _short_timestamp(
                str(getattr(art, 'created_at', '') or ''))
            text = f"{name} · {created}" if created else name
            return text or _artifact_id(art)

        refill_combo(
            self._artifact_combo, artifacts or [],
            _display, _artifact_id,
            previous_data=self._artifact_combo.currentData())
        refill_combo(
            self._input_combo, artifacts or [],
            lambda art: f'成果: {_display(art)}', _artifact_id,
            previous_data=self._input_combo.currentData(),
            prepend=(('原始数据', ''),))   # 索引 0 = 从原始数据开始

    def select_artifact(self, artifact_id: str) -> bool:
        """静默选中指定成果（不发射 artifact_selected，供主窗口自动预览时同步）。"""
        index = self._artifact_combo.findData(str(artifact_id or ''))
        if index < 0:
            return False
        self._artifact_combo.blockSignals(True)
        self._artifact_combo.setCurrentIndex(index)
        self._artifact_combo.blockSignals(False)
        return True

    def _set_line_combo_without_emit(self, line_id: str) -> None:
        index = self._line_combo.findData(str(line_id or ''))
        if index < 0:
            index = 0 if self._line_combo.count() else -1
        if index >= 0:
            self._line_combo.blockSignals(True)
            self._line_combo.setCurrentIndex(index)
            self._line_combo.blockSignals(False)

    def _on_line_combo_changed(self, index: int) -> None:
        if index >= 0:
            self.line_changed.emit(str(self._line_combo.itemData(index) or ''))

    def _on_artifact_combo_changed(self, index: int) -> None:
        if index >= 0:
            self.artifact_selected.emit(
                str(self._artifact_combo.itemData(index) or ''))

    def current_pipeline(self) -> dict:
        """当前处理链定义：{"steps", "result_name", "input_artifact_id"}。"""
        return {
            'steps': self._pipeline_list.steps(),
            'result_name': self._result_name_edit.text().strip(),
            'input_artifact_id': self._input_artifact_id(),
        }

    # ============================================================ 内部逻辑
    def _current_segment(self) -> str:
        item = self._preview_segment.currentItem()
        return item.property('routeKey') if item is not None else _SEG_ORIGINAL

    # ---------------- 预览分发（BScanContainer 多视图）
    def _shows_both_panels(self) -> bool:
        """当前实际布局是否同屏展示原始与成果两侧（dual/quad/free）。"""
        return self._bscan_container.effective_mode() != LAYOUT_SINGLE

    def _sync_auto_layout(self) -> None:
        """auto 模式：布局稳定优先（粘性），面板只"长出"不"收回"。

        - 对比条件首次满足（原始+成果同时在场）→ 长出左右对比并粘住；
        - 成果被删 / 换测线清空 → **保持对比布局**，成果位显示空态——
          用户正在看图时布局绝不跳变；随后新成果到达直接填入；
        - 原始数据也没了（切项目等上下文清空）→ 重置粘性回单视图。

        旧版按 bundle 数量实时解析：成果清空即缩回、自动预览到达再长出，
        一次换测线布局抖动两次，观察被打断——已废弃。
        实体模式（手动固定）下是空操作。解析换了页后 bundle 分发由调用方
        随后的 _show_bundle/_distribute_bundles 完成（各面板显示偏好由
        主窗启动期统一恢复，页面无需重广播）。
        """
        if self._bscan_container.layout_mode() != LAYOUT_AUTO:
            return
        if self._original_bundle is None:
            self._auto_sticky_dual = False
        elif self._result_bundle is not None:
            self._auto_sticky_dual = True
        self._bscan_container.resolve_auto(2 if self._auto_sticky_dual else 1)

    def _show_bundle(self, which: str) -> None:
        """分段切换 / bundle 到达的统一入口（按布局分发）。"""
        self._sync_auto_layout()
        if self._shows_both_panels():
            self._distribute_bundles()
            return
        bundle = (self._original_bundle if which == _SEG_ORIGINAL
                  else self._result_bundle)
        self._set_panel_data(self._bscan_container.primary_view(), bundle)

    def _distribute_bundles(self) -> None:
        """dual/quad/free：0 号位固定原始数据、1 号位固定处理结果，其余留空。

        按 views() 实际数量分发——不许用 view_at(2/3) 凑数：dual 模式下
        view_at 越界会回落面板 0，随后 clear() 把刚填的原始数据清掉。
        """
        views = self._bscan_container.views()
        bundles = [self._original_bundle, self._result_bundle]
        bundles += [None] * (len(views) - len(bundles))
        for view, bundle in zip(views, bundles):
            self._set_panel_data(view, bundle)

    def _set_panel_data(self, view: BScanView, bundle) -> None:
        """单面板数据写入：None → 清空空态；有数据 → 交给视图偏好。

        色阶只由视图偏好决定（构造默认 2/98，启动期主窗从设置恢复）：
        view.set_matrix 收到的 vmin/vmax 只是默认裁切，会被视图自身
        _p_low/_p_high 覆盖——页面不再另持一份色阶状态源。
        """
        if bundle is None:
            view.clear()
            return
        view.set_bundle(bundle)

    def _on_layout_changed(self, _mode: str) -> None:
        """布局切换：重新分发 bundle（各面板显示偏好启动期已统一恢复、
        后续由设置页/右键各自维护，页面无需重广播）。"""
        self._show_bundle(self._current_segment())

    # ---------------- 方法库
    def _on_method_selected(self, method_id: str) -> None:
        self._selected_method_id = method_id
        method = self._methods_by_id.get(method_id, {})
        self._autotune_method_label.setText(
            method.get('display_name') or method_id or '--')
        self._update_autotune_enabled()

    def _update_autotune_enabled(self) -> None:
        self._autotune_btn.setEnabled(
            bool(self._selected_method_id) and not self._autotune_running)

    def set_autotune_running(self, running: bool) -> None:
        """AutoTune 运行期间禁用「开始调参」，防重复提交（P2-7）。"""
        self._autotune_running = bool(running)
        self._update_autotune_enabled()

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

    def _auto_write_params_to_selected(self) -> None:
        """参数表单值变化 → 自动写入选中步骤（B1：改值即生效，无按钮）。

        静默写回：表单由本页驱动（set_values 不触发 sig_changed 循环），此处
        仅处理用户编辑；步骤失效时静默丢弃（选中态变化会重新载入表单）。
        """
        if not (0 <= self._selected_step < len(self._pipeline_list.steps())):
            return
        self._pipeline_list.update_step_params(self._selected_step,
                                               self._param_form.values())

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
        clear_invalid(self._result_name_edit)
        # 结果名可留空：窗口层有默认名回退（处理结果_{line_id}），此处不强填
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
        self.autotune_requested.emit(method_id, params_hint, self._input_artifact_id())

    def _input_artifact_id(self) -> str:
        """输入数据下拉当前选中的 artifact_id（''=原始数据）。"""
        if self._input_combo.currentIndex() <= 0:
            return ''
        return str(self._input_combo.currentData() or '')

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
            self._pipeline_list.select_step(target)
            self._pipeline_list.sig_changed.emit()
        else:
            self._param_form.set_values(best)
        InfoBar.success(title='AutoTune 自动调参', content='已采用最优参数',
                        orient=Qt.Orientation.Horizontal, isClosable=True,
                        position=InfoBarPosition.TOP, duration=2000,
                        parent=self)
