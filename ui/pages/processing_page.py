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

预览区（tab 模型，2026-09-24）：胶囊 TabBar = 打开的数据源清单——
- 原始数据固定首 tab 不可关；运行链的每步中间成果（B7 落盘的
  intermediate artifact）与最终成果跑完自动开 tab，标题 = 算法名
  （method_id），末位挂 ✓；tab 可关、可拖排序；
- **tab 即窗口**：窗口数 = min(tab 数, 4) 自动排布（1→单窗 2→左右
  3-4→2×2），>4 主区留前 4、全部进总览墙；选中的 tab 总在主区；
- bundle 懒加载：可见面板缺 bundle 时发 artifact_preview_requested，
  协调器接 project_controller.preview_artifact 异步回填
  set_artifact_bundle。

页面纯展示 + 发信号，不直接调 controller/backend。
内部联动：PipelineList.sig_step_selected → ParamForm 载入该步骤参数；
"应用到选中步骤"按钮 → 表单值写回选中步骤。
"""

from datetime import datetime

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QHBoxLayout, QVBoxLayout, QWidget)
from qfluentwidgets import (
    CaptionLabel, ComboBox, InfoBar, TabBar,
    InfoBarPosition, LineEdit, PrimaryPushButton, ProgressBar, PushButton,
)
from qfluentwidgets.components.widgets.tab_view import TabItem
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.motion import animate_progress
from ui.page_scaffold import (PanelStateMixin, make_card, make_form_row,
                              make_scroll_column, refill_combo)
from ui.widgets.bscan_result_grid import ResultGrid
from ui.widgets.chain_strip import ChainStrip
from ui.widgets import (BScanContainer, CollapsiblePanel, LAYOUT_FOCUS,
                        MethodBrowser, ParamForm, PipelineList, MAX_PANELS,
                        clear_invalid, make_separator)


_INPUT_KEY = 'input'         # 结果网格的「输入」槽位 key
_MAX_THUMBS = 3              # 缩略列容量（= MAX_PANELS - 1，含主窗共 4 面板）
_READABILITY_MIN_RATIO = 0.45  # 可读性红线：绘图区高 / 采样数（px/采样）


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
    # tab 模型懒加载：可见面板缺 bundle → 请求预览该成果（协调器接
    # project_controller.preview_artifact，异步回填 set_artifact_bundle）
    artifact_preview_requested = pyqtSignal(str)
    # 批量处理（B4）UI 已按用户决策暂时屏蔽（2026-09-02）：卡片、信号与
    # 接线整体撤下；后端 run_pipeline_batch 契约保留，恢复时重建本页卡片
    # 并回接 page_coordinator._on_batch_run_requested 即可。

    _PANEL_STATE_PREFIX = 'processing'

    def __init__(self, parent=None):
        super().__init__(parent)
        self._methods = []
        self._methods_by_id = {}
        # tab 模型（2026-09-24）：打开的数据源 = 画布窗口，tab 数决定面板
        # 数（1→单窗 2→左右 3-4→2×2，>4 主区留前 4、其余总览墙）。
        self._preview_sources = []        # [{key,title,bundle,artifact_id,closable,is_final}]
        self._selected_source_key = 'original'
        self._opened_run_groups = set()   # 已自动展开的 run_group_id（每组一次）
        self._artifacts_by_id = {}        # artifact_id -> ProjectArtifact
        self._gallery = None              # 总览墙（懒创建，随主页销毁）
        self._thumb_views_bound = []      # 已装点击提升的缩略面板
        self._step_artifact_ids = {}      # v2：步骤序号 → 该步 intermediate 成果 id
        self._results_stale = False       # v2：链/参数已改但结果未重算
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

        # tab 模型：胶囊 TabBar = 打开的数据源清单（原始数据固定首 tab
        # 不可关；步骤/成果 tab 可关、可拖排序）；tab 数决定画布窗口数。
        self._source_tabs = TabBar(self)
        self._source_tabs.setTabsClosable(True)
        self._source_tabs.setMovable(True)
        # qfw 自带的「+」加页按钮：本页 tab 只随数据源增减，按钮无功能，
        # 留着只会诱导误点（死按钮）
        self._source_tabs.setAddButtonVisible(False)
        # ---------------- v2 主区：上链条 / 下结果网格（旧预览卡隐藏，代码保留）
        self._line_combo = ComboBox(middle)
        self._line_combo.setMinimumWidth(130)
        self._line_combo.setToolTip('当前测线：在处理页直接切换')
        self._artifact_combo = ComboBox(middle)
        self._artifact_combo.setMinimumWidth(150)
        self._artifact_combo.setToolTip('选择该测线历次处理结果作为输入')
        input_row = QWidget(middle)
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(constants.CARD_SPACING)
        input_layout.addWidget(self._line_combo)
        input_layout.addWidget(self._artifact_combo)

        self._chain_strip = ChainStrip(middle)
        self._chain_strip.set_input_widget(input_row)
        middle_layout.addWidget(self._chain_strip)

        self._result_grid = ResultGrid(middle)
        middle_layout.addWidget(self._result_grid, 1)

        preview_card, preview_layout = make_card('数据预览')
        self._preview_card = preview_card
        preview_card.setVisible(False)      # v2 隐藏（tab 模型待确认后退役）
        tab_row = QHBoxLayout()
        tab_row.setSpacing(constants.CARD_SPACING)
        tab_row.addWidget(self._source_tabs, 1)
        self._gallery_btn = PushButton('总览墙', self)
        self._gallery_btn.setMinimumWidth(60)   # 溢出徽标 +N 需自适应宽
        self._gallery_btn.setToolTip(
            '弹出总览墙：网格展示全部打开的数据源，点格子的标题把该源'
            '送回主区（tab 数超过 4 时跑完链会自动弹一次）。')
        self._gallery_btn.clicked.connect(self._open_gallery)
        tab_row.addWidget(self._gallery_btn)
        self._readability_label = CaptionLabel('', self)
        self._readability_label.setToolTip(
            '画布纵向像素不足以呈现全部采样点（低于 0.45px/采样）：'
            '收起两侧栏（⤢ 铺满）或全屏浏览可获得更高纵向密度。')
        self._readability_label.setVisible(False)
        tab_row.addWidget(self._readability_label)
        preview_layout.addLayout(tab_row)

        self._bscan_container = BScanContainer(preview_card)
        self._bscan_container.setMinimumHeight(constants.PREVIEW_MIN_HEIGHT)
        preview_layout.addWidget(self._bscan_container, 1)
        # 原始数据锚点 tab（须在容器就位后建：_sync_tabs 会走 resolve_auto）
        self._ensure_original_source()
        self._sync_tabs()

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
        self._result_name_edit.setPlaceholderText('例如：增益处理后结果…')
        self._result_name_edit.setToolTip('处理成果保存名称')
        exec_layout.addLayout(make_form_row(
            '结果名称:', self._result_name_edit, parent=exec_card))
        run_row = QHBoxLayout()
        run_row.setSpacing(constants.CARD_SPACING)
        self._run_btn = PrimaryPushButton('运行处理链', exec_card, FIF.PLAY)
        self._run_btn.setToolTip('执行右侧处理链（Ctrl+R）')
        # v2：运行入口统一到顶部链条（此按钮保留对象供 set_running 驱动，
        # 不再显示，消除「左栏运行 / 顶部运行」双入口）
        self._run_btn.setVisible(False)
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
        # v2：参数 / 执行 / 自动调参移入左栏（处理链改由顶部 chip 条承担，
        # 右栏整体隐藏——PipelineList 仍作为步骤数据源留在右栏内，代码不删）
        left_layout.addWidget(param_card)
        left_layout.addWidget(exec_card)
        left_layout.addWidget(autotune_card)
        self._right_panel.setVisible(False)
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
        # tab 模型：关闭 / 选中 → 源清单变更 → 重排画布面板（v2 暂隐藏）
        self._source_tabs.tabCloseRequested.connect(self._on_tab_close)
        self._source_tabs.currentChanged.connect(self._on_tab_selected)
        # v2：顶部 chip 条 ↔ 处理链（PipelineList 仍是步骤数据源）
        self._chain_strip.sig_step_selected.connect(self._on_chain_step_selected)
        self._chain_strip.sig_step_toggled.connect(self._on_chain_step_toggled)
        self._chain_strip.sig_step_removed.connect(self._on_chain_step_removed)
        self._chain_strip.sig_step_moved.connect(self._on_chain_step_moved)
        self._chain_strip.sig_add_requested.connect(self._on_add_selected_method)
        # 反向：处理链选中（程序化）也同步 chip 条高亮
        self._pipeline_list.sig_step_selected.connect(self._chain_strip.select_step)
        self._chain_strip.run_button().clicked.connect(self._on_run_clicked)
        self._pipeline_list.sig_changed.connect(self._refresh_chain_and_results)
        self._pipeline_list.sig_changed.connect(self._mark_results_stale)
        self._refresh_chain_and_results()

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
        """原始数据预览 bundle → 固定首 tab（不可关；换测线即换内容）。"""
        # v2：输入数据到位才建输入卡（未运行不摆各步的空画布）
        self._ensure_original_source()
        for source in self._preview_sources:
            if source['key'] == 'original':
                source['bundle'] = bundle
        self._result_grid.set_bundle(_INPUT_KEY, bundle)
        if not self._result_grid.cards():
            self._refresh_chain_and_results()
        self._redistribute()

    def set_artifact_bundle(self, artifact_id: str, bundle) -> None:
        """成果/步骤预览 bundle → 打开（或更新）对应的 artifact tab。

        tab 已开 → 换内容；未开 → 新建（标题=算法名，取本页成果登记表
        的 method_id，缺省回落 bundle.title）。bundle 异步到达时对应
        面板已在 _redistribute 里绑定/清空过，这里补齐后重分发。
        """
        key = f'artifact:{artifact_id}'
        # v2：同 bundle 回填到结果网格的对应步骤格（旧 tab 模型并存）
        for index, step_aid in (self._step_artifact_ids or {}).items():
            if step_aid == artifact_id:
                self._result_grid.set_bundle(f'step:{index}', bundle)
        for source in self._preview_sources:
            if source['key'] == key:
                source['bundle'] = bundle
                if bundle is not None and getattr(bundle, 'title', ''):
                    source['title'] = str(bundle.title)
                self._redistribute()
                return
        title = self._artifact_title(artifact_id, bundle)
        self._preview_sources.append({
            'key': key, 'title': title, 'bundle': bundle,
            'artifact_id': str(artifact_id), 'closable': True,
            'is_final': False})
        self._selected_source_key = key
        self._sync_tabs()

    def close_artifact_tab(self, artifact_id: str) -> None:
        """外部删除成果 → 关闭其 tab（原始 tab 永不受影响）。"""
        key = f'artifact:{artifact_id}'
        for index, source in enumerate(self._preview_sources):
            if source['key'] == key:
                self._on_tab_close(index)
                return

    def close_all_artifact_tabs(self) -> None:
        """换测线/清上下文：关闭全部成果/步骤 tab（原始锚点保留换内容）。"""
        self._preview_sources = [s for s in self._preview_sources
                                 if s['key'] == 'original']
        self._selected_source_key = 'original'
        self._sync_tabs()

    def show_latest_result(self) -> None:
        """跑完链后自动选中末位 tab（= 最终结果）；>4 源自动弹总览墙。"""
        if self._preview_sources:
            self._selected_source_key = self._preview_sources[-1]['key']
            self._sync_tabs()
            if len(self._preview_sources) > MAX_PANELS:
                self._open_gallery()

    def _open_gallery(self) -> None:
        """总览墙（非模态）：每次打开重建内容（源清单 ≤16，成本可忽略）。"""
        from ui.widgets.bscan_gallery import BScanGallery
        self._gallery = BScanGallery(self)
        self._gallery.show()

    # ------------------------------------------------ v2：链条 ↔ 处理链
    def _on_chain_step_selected(self, index: int) -> None:
        """选 chip → 参数区跟随该步骤（参数表单已接 sig_step_selected）。

        双向同步：PipelineList 选中也回写 chip 条（select_step 内部
        blockSignals，不会循环）。
        """
        if index >= 0:
            self._pipeline_list.select_step(index)
        self._chain_strip.select_step(index)
        # 选中节点与结果图同步高亮
        self._result_grid.set_selected(
            _INPUT_KEY if index < 0 else f'step:{index}')

    def _on_chain_step_toggled(self, index: int, enabled: bool) -> None:
        """启用/禁用：与当前状态不同才翻转（PipelineList 内置翻转语义）。"""
        steps = self._pipeline_list.steps()
        if 0 <= index < len(steps) and steps[index]['enabled'] != enabled:
            self._pipeline_list._toggle_enabled(index)

    def _on_chain_step_removed(self, index: int) -> None:
        self._pipeline_list._remove_step(index)

    def _on_chain_step_moved(self, source: int, target: int) -> None:
        """chip 拖拽落点 → 步骤换位（target 为插入位语义）。"""
        self._pipeline_list._move_step_to(source, target)

    def _refresh_chain_and_results(self) -> None:
        """上面怎么排，下面就按同序铺格。

        网格数据源优先级：已运行 → run_group 成员（运行事实，标题=算法名）；
        未运行 → 当前链步骤占位（空态「点运行后生成」）。禁用的步骤留
        置灰占位卡（1:1 对应，不占画布）。
        """
        steps = self._pipeline_list.steps()
        self._chain_strip.set_steps(steps)
        members = self._newest_run_members()
        if members:
            slots = [{'key': _INPUT_KEY, 'title': '输入', 'enabled': True}]
            for i, (_s, kind, _c, artifact_id, art) in enumerate(members):
                method = (str(getattr(art, 'method_id', '') or '')
                          or str(getattr(art, 'name', '') or ''))
                slots.append({'key': f'step:{i}', 'title': f'{i + 1} {method}',
                              'enabled': True})
            self._step_artifact_ids = {
                i: artifact_id
                for i, (_s, _k, _c, artifact_id, _a) in enumerate(members)}
        else:
            # 未运行：不摆各步的空画布——只有输入数据真的到位才建输入卡
            original = next((src['bundle'] for src in self._preview_sources
                             if src['key'] == 'original'), None)
            slots = ([{'key': _INPUT_KEY, 'title': '输入', 'enabled': True}]
                     if original is not None else [])
            self._step_artifact_ids = {}
        self._result_grid.set_slots(slots)
        self._result_grid.sig_card_selected.connect(self._on_card_selected)
        # set_slots 会重建卡片 → 输入卡的 bundle 需重喂（原始 bundle
        # 存在源清单的 original 槽位里）
        original = next((src['bundle'] for src in self._preview_sources
                         if src['key'] == 'original'), None)
        self._result_grid.set_bundle(_INPUT_KEY, original)
        self._result_grid.set_selected(
            _INPUT_KEY if self._selected_step_index() < 0
            else f'step:{self._selected_step_index()}')
        self._chain_strip.set_dirty(
            bool(self._step_artifact_ids) and self._results_stale)

    def _mark_results_stale(self) -> None:
        """链/参数变更且已有运行结果 → 结果过期（琥珀提示，运行后清除）。"""
        if self._step_artifact_ids:
            self._results_stale = True
            self._chain_strip.set_dirty(True)

    def _selected_step_index(self) -> int:
        """当前选中的步骤索引（-1 = 输入 / 未选）。"""
        row = self._chain_strip._list.currentRow()
        return row - 1

    def _on_card_selected(self, key: str) -> None:
        """点结果卡 → 选中对应 chip（与链式条双向同步）。"""
        if key == _INPUT_KEY:
            self._chain_strip.select_step(-1)
            self._result_grid.set_selected(_INPUT_KEY)
            return
        index = int(key.split(':', 1)[1])
        self._chain_strip.select_step(index)
        self._pipeline_list.select_step(index)

    def _request_step_previews(self) -> None:
        """按步骤顺序请求各步结果（异步回填，generation 守卫防串线）。"""
        for _index, artifact_id in sorted(self._step_artifact_ids.items()):
            self.artifact_preview_requested.emit(artifact_id)

    def on_gallery_pick(self, key: str) -> None:
        """总览墙点格子：选中该源（可见性规则把它换入主面板）。"""
        self._selected_source_key = key
        self._ensure_selected_visible()

    # -------------------------------------------------- tab 模型（源清单）
    def _ensure_original_source(self) -> None:
        """原始数据是 tab 条的锚点：固定首 tab、不可关。"""
        if not any(s['key'] == 'original' for s in self._preview_sources):
            self._preview_sources.insert(0, {
                'key': 'original', 'title': '原始数据', 'bundle': None,
                'artifact_id': '', 'closable': False, 'is_final': False})

    def _artifact_title(self, artifact_id: str, bundle) -> str:
        """tab 标题 = 算法名（成果登记表的 method_id），缺省回落 bundle 标题。"""
        art = self._artifacts_by_id.get(str(artifact_id or ''))
        method = str(getattr(art, 'method_id', '') or '') if art else ''
        if method:
            return method
        return str(getattr(bundle, 'title', '') or '') or '处理结果'

    def _selected_source_index(self) -> int:
        for index, source in enumerate(self._preview_sources):
            if source['key'] == self._selected_source_key:
                return index
        return 0 if self._preview_sources else -1

    def _sync_tabs(self) -> None:
        """TabBar 与源清单整体对齐（条目少，重建成本可忽略）。

        blockSignals 包住重建：addTab/removeTab 触发的 currentChanged
        不能中途跑 _on_tab_selected（会改选中态引发重入）。
        """
        bar = self._source_tabs
        bar.blockSignals(True)
        while bar.count():
            bar.removeTab(bar.count() - 1)
        for source in self._preview_sources:
            suffix = ' ✓' if source.get('is_final') else ''
            bar.addTab(source['key'], source['title'] + suffix)
        index = self._selected_source_index()
        if index >= 0:
            bar.setCurrentIndex(index)
        bar.blockSignals(False)
        self._purge_orphan_tab_items()
        self._redistribute()

    def _purge_orphan_tab_items(self) -> None:
        """清扫 qfw TabBar 的孤儿 TabItem（视觉验收发现的真 bug）。

        qfw ``removeTab`` 依赖 ``deleteLater`` 销毁旧 TabItem，但在本页的
        blockSignals+closable 重建路径下孤儿会逃逸销毁——每次重建残留一个
        首位同名 item，累积后渲染成"tab 重复/碎片墙"（截图实证：4 源渲染
        出 7+ 个 tab）。按对象身份比对：不在 ``bar.items`` 里的即为孤儿 →
        setParent(None) 立即脱离显示 + deleteLater 兜底销毁。
        """
        bar = self._source_tabs
        live = {id(item) for item in bar.items}
        for item in bar.findChildren(TabItem):
            if id(item) not in live:
                item.setParent(None)
                item.deleteLater()

    def _on_tab_selected(self, index: int) -> None:
        """选中 tab：更新焦点 key；越出主区（>4）的换入末位主面板。"""
        if not (0 <= index < len(self._preview_sources)):
            return
        self._selected_source_key = self._preview_sources[index]['key']
        self._ensure_selected_visible()

    def _ensure_selected_visible(self) -> None:
        """可见性规则（锁定稿）：选中的 tab 总在主区——超出前 4 的换入
        末位主面板位置（其余 tab 留在总览墙）。"""
        index = self._selected_source_index()
        visible = min(len(self._preview_sources), 4)
        if visible and 0 <= index >= visible:
            source = self._preview_sources.pop(index)
            self._preview_sources.insert(visible - 1, source)
            self._sync_tabs()

    def _on_tab_close(self, index: int) -> None:
        """关闭 tab = 从会话移除该源（原始数据是锚点，不可关）。"""
        if not (0 <= index < len(self._preview_sources)):
            return
        source = self._preview_sources[index]
        if not source['closable']:
            return
        del self._preview_sources[index]
        if self._selected_source_key == source['key']:
            self._selected_source_key = 'original'
        self._sync_tabs()

    def _redistribute(self) -> None:
        """tab 序/选中态 → 面板绑定：面板数 = min(源数, 4)。

        - single/dual：按 tab 序依次绑定；
        - focus（≥3 源）：**主窗 = 选中源**，其余源按 tab 序进缩略列
          （最多 3 个；再多的进总览墙，tab 行挂「+N」徽标）；
        - 缺 bundle 的可见面板（懒加载）清空后发 artifact_preview_requested。
        """
        container = self._bscan_container
        count = min(len(self._preview_sources), MAX_PANELS)
        container.resolve_auto(count)
        focus = container.effective_mode() == LAYOUT_FOCUS
        if focus:
            selected = self._preview_sources[self._selected_source_index()]
            self._bind_source(container.primary_view(), selected, thumb=False)
            others = [s for s in self._preview_sources
                      if s['key'] != selected['key']][:_MAX_THUMBS]
            for view, source in zip(container.thumb_views(), others):
                self._bind_source(view, source, thumb=True)
        else:
            for index in range(count):
                self._bind_source(container.view_at(index),
                                  self._preview_sources[index], thumb=False)
        self._sync_thumb_activation()
        self._update_readability_hint()
        extra = len(self._preview_sources) - MAX_PANELS
        self._gallery_btn.setText(f'总览墙 +{extra}' if extra > 0
                                  else '总览墙')

    def _bind_source(self, view, source, *, thumb: bool = False) -> None:
        """单面板绑定：有 bundle 直接送，缺则清空并发懒加载请求。

        ``thumb`` 决定视图形态（缩略隐藏轴/标题/全屏钮）；升主窗/降缩略
        都经本函数重设形态，轴与全屏钮随角色还原。
        """
        view.set_thumbnail_mode(thumb)
        if source['bundle'] is not None:
            view.set_bundle(source['bundle'])
            view.set_thumbnail_mode(thumb)
            return
        view.clear()
        view.set_thumbnail_mode(thumb)
        artifact_id = str(source.get('artifact_id') or '')
        if artifact_id and self._preview_card is not None \
                and self._preview_card.isVisibleTo(self):
            self.artifact_preview_requested.emit(artifact_id)

    def _sync_thumb_activation(self) -> None:
        """缩略升主窗：给当前缩略面板装上点击提升（同一批只装一次）。"""
        thumbs = self._bscan_container.thumb_views()
        if self._thumb_views_bound == thumbs:
            return
        for view in self._thumb_views_bound:
            try:
                view.removeEventFilter(self)
            except RuntimeError:      # 视图已被 Qt 销毁（C++ 侧已删）
                pass
        for view in thumbs:
            view.installEventFilter(self)
        self._thumb_views_bound = list(thumbs)

    def _update_readability_hint(self) -> None:
        """可读性守护：绘图区高/采样数 < 0.45px 时提示（挂 tab 行右侧，
        不占纵向预算——画布本来就是最缺高度的那一块）。"""
        view = self._bscan_container.primary_view()
        samples = int(getattr(view, '_sample_count', 0) or 0)
        height = self._bscan_container.height()
        ratio = (height / samples) if samples > 0 else 0.0
        show = bool(samples) and ratio < _READABILITY_MIN_RATIO
        self._readability_label.setVisible(show)
        if show:
            self._readability_label.setText(
                f'画布偏矮（{ratio:.2f}px/采样）· 建议 ⤢ 铺满或全屏浏览')

    def eventFilter(self, obj, event) -> bool:
        """缩略面板点击 → 把该源升为主窗（选中态即主窗，见 _redistribute）。"""
        if (event.type() == QEvent.Type.MouseButtonRelease
                and obj in self._thumb_views_bound):
            selected = self._preview_sources[self._selected_source_index()]
            others = [s for s in self._preview_sources
                      if s['key'] != selected['key']][:_MAX_THUMBS]
            for index, view in enumerate(self._thumb_views_bound):
                if view is obj and index < len(others):
                    self._selected_source_key = others[index]['key']
                    self._sync_tabs()
                    return True
        return super().eventFilter(obj, event)

    def set_running(self, running: bool, job_id: str = '',
                    success: bool | None = None) -> None:
        """运行态切换：运行按钮/取消按钮互斥 + 进度条显隐。

        ``success=True``（运行正常结束）时顶部链条的运行钮闪一次 ✓。
        """
        self._running = bool(running)
        self._job_id = job_id or ''
        self._run_btn.setEnabled(not self._running)
        self._cancel_btn.setEnabled(self._running)
        self._progress_row_widget.setVisible(self._running)
        # v2：顶部链条的运行钮同步进入 spinner 态（结束回到「运行」）
        self._chain_strip.set_running(self._running)
        if not self._running and success:
            self._chain_strip.flash_success()
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
        # tab 模型：登记表（标题解析用）+ 最新 run_group 自动展开步骤 tab
        self._artifacts_by_id = {
            _artifact_id(art): art for art in (artifacts or [])
            if _artifact_id(art)}
        self._auto_open_newest_run_group()

    def _run_group_params(self, art) -> dict:
        """成果的归组参数，兼容两种 manifest 形态。

        - catalog/sqlite 路径：manifest.params.{run_group_id,...}（嵌套）；
        - field/文件系统路径：ProjectArtifact.manifest = 索引记录 to_dict()，
          run_group_id / artifact_kind 在**顶层**且没有 run_step_index
          （2026-09-24 视觉验收发现的真因：按嵌套格式读取永远为空，
          步骤 tab 不展开）。
        """
        manifest = getattr(art, 'manifest', None) or {}
        if not isinstance(manifest, dict):
            return {}
        nested = manifest.get('params')
        if isinstance(nested, dict):
            return nested
        return manifest

    def _newest_run_members(self) -> list:
        """最新 run_group 的成员（已按步序排序；无则空表）。

        元素：(kind, created, artifact_id, art)。结果网格与步骤 tab 共用
        ——「上面怎么排，下面就按同序看各步结果」以运行事实（B7 落盘）
        为准，而非当前链定义（用户可能已改链）。
        """
        groups = {}
        for artifact_id, art in self._artifacts_by_id.items():
            params = self._run_group_params(art)
            group = str(params.get('run_group_id') or '')
            if not group:
                continue
            info = groups.setdefault(group, {'created': '', 'members': []})
            info['members'].append((
                int(params.get('run_step_index') or 0),
                str(params.get('artifact_kind') or ''),
                str(getattr(art, 'created_at', '') or ''),
                artifact_id, art))
            created = str(getattr(art, 'created_at', '') or '')
            info['created'] = max(info['created'], created)
        if not groups:
            return []
        newest = max(groups, key=lambda g: groups[g]['created'])
        return sorted(groups[newest]['members'],
                      key=lambda m: (m[1] == 'processing', m[0], m[2]))

    def _auto_open_newest_run_group(self) -> None:
        """最新 run_group 的步骤/最终成果自动开 tab（每组只开一次）。

        B7：链的每步中间成果已落盘（artifact_kind=intermediate，
        run_group_id 归组）。步骤 tab 直接复用 artifact 预览链路——
        懒加载（可见面板触发 artifact_preview_requested），内存不持有
        全尺寸矩阵。标题 = 算法名（method_id），末位（最终成果）挂 ✓。
        排序：优先 run_step_index（catalog 路径），缺失（field 路径）
        按 created_at 升序——B7 逐步落盘天然按步序递增，最终成果最后。
        """
        groups = {}
        for artifact_id, art in self._artifacts_by_id.items():
            params = self._run_group_params(art)
            group = str(params.get('run_group_id') or '')
            if not group:
                continue
            info = groups.setdefault(group, {'created': '', 'members': []})
            info['members'].append((
                int(params.get('run_step_index') or 0),
                str(params.get('artifact_kind') or ''),
                str(getattr(art, 'created_at', '') or ''),
                artifact_id, art))
            created = str(getattr(art, 'created_at', '') or '')
            info['created'] = max(info['created'], created)
        if not groups:
            return
        newest = max(groups, key=lambda g: groups[g]['created'])
        if newest in self._opened_run_groups:
            return
        self._opened_run_groups.add(newest)
        self._ensure_original_source()
        # catalog 路径有 run_step_index；field 路径缺失（全 0）时按落盘
        # 时间升序回退（B7 逐步保存天然按步序递增）。最终成果恒排末位。
        members = sorted(groups[newest]['members'],
                         key=lambda m: (m[1] == 'processing', m[0], m[2]))
        self._step_artifact_ids = {
            index: artifact_id
            for index, (_s, _k, _c, artifact_id, _a) in enumerate(members)}
        for _, kind, _created, artifact_id, art in members:
            key = f'artifact:{artifact_id}'
            if any(s['key'] == key for s in self._preview_sources):
                continue
            title = (str(getattr(art, 'method_id', '') or '')
                     or str(getattr(art, 'name', '') or '') or '处理结果')
            self._preview_sources.append({
                'key': key, 'title': title, 'bundle': None,
                'artifact_id': artifact_id, 'closable': True,
                'is_final': kind == 'processing'})
        # 跑完自动选中末位 tab（= 最终结果）
        if self._preview_sources:
            self._selected_source_key = self._preview_sources[-1]['key']
        self._sync_tabs()
        self._results_stale = False
        self._refresh_chain_and_results()
        self._request_step_previews()

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
    # ---------------- 预览分发（tab 模型：见 set_original_bundle / _redistribute）

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
