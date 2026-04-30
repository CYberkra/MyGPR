#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPR 方法调试工作台 - 主工作台页面

这是主要的调试界面，整合方法树、参数编辑、预览、日志
"""

import logging
import time
import numpy as np
from datetime import datetime
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QSplitter,
    QGroupBox,
    QButtonGroup,
    QRadioButton,
    QApplication,
    QFileDialog,
    QMessageBox,
    QMenu,
)
from qfluentwidgets import PushButton, SplitPushButton, FluentIcon

from core.methods_registry import PROCESSING_METHODS
from core.workflow_data import WorkflowMethod
from core.workflow_executor import WorkflowExecutor, ExecutionError
from core.processing_engine import (
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from ui.gui_method_browser import MethodBrowserTree
from ui.gui_param_editor import ParamEditorPanel
from core.favorites_manager import FavoritesManager
from core.workflow_template_manager import WorkflowTemplateManager
from core.theme_manager import get_theme_manager

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

logger = logging.getLogger(__name__)


class PreviewWorker(QObject):
    """后台预览计算线程。"""

    finished = pyqtSignal(object)
    error = pyqtSignal(object)

    def __init__(self, request: dict):
        super().__init__()
        self.request = request
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested)

    def run(self):
        if self.is_cancel_requested():
            self.finished.emit({"seq": self.request["seq"], "cancelled": True})
            return

        start_ts = time.perf_counter()
        try:
            result, meta = run_processing_method(
                self.request["input_data"],
                self.request["method_id"],
                prepare_runtime_params(
                    self.request["method_id"],
                    self.request["params"],
                    self.request.get("header_info"),
                    self.request.get("trace_metadata"),
                    self.request["input_data"].shape,
                ),
                cancel_checker=self.is_cancel_requested,
            )
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            resolved_header_info = merge_result_header_info(
                self.request.get("header_info"), meta, result.shape
            )
            resolved_trace_metadata = merge_result_trace_metadata(
                self.request.get("trace_metadata"), meta
            )
            self.finished.emit(
                {
                    "seq": self.request["seq"],
                    "cancelled": self.is_cancel_requested(),
                    "data": result,
                    "meta": meta,
                    "header_info": resolved_header_info,
                    "trace_metadata": resolved_trace_metadata,
                    "title": self.request["title"],
                    "elapsed_ms": elapsed_ms,
                    "announce": self.request.get("announce", False),
                    "method_name": self.request.get(
                        "method_name", self.request["method_id"]
                    ),
                    "source_text": self.request.get("source_text", ""),
                    "params": self.request.get("params", {}),
                    "context": self.request.get("context"),
                }
            )
        except Exception as e:
            self.error.emit(
                {
                    "seq": self.request["seq"],
                    "error": str(e),
                    "announce": self.request.get("announce", False),
                    "method_name": self.request.get(
                        "method_name", self.request["method_id"]
                    ),
                }
            )


class WorkbenchPage(QWidget):
    """GPR 方法调试工作台主页面"""

    # 信号
    data_import_requested = pyqtSignal()  # 请求导入数据
    save_result_requested = pyqtSignal()

    def __init__(self, parent=None, data_state=None):
        super().__init__(parent)
        self.parent_window = parent
        self.data_state = data_state

        # 数据状态
        self.raw_data = None  # 原始数据
        self.current_result = None  # 当前处理结果
        self.raw_header_info = None
        self.raw_trace_metadata = None
        self.current_result_header_info = None
        self.current_result_trace_metadata = None
        self.has_processed_result = False  # 当前结果是否来自处理步骤
        self.all_results = []  # 结果历史 (name, data)
        self.all_result_entries = []  # 结果历史及其配套 header/trace metadata
        self.preview_data = None  # 参数变化时的临时预览结果
        self.preview_header_info = None
        self.preview_trace_metadata = None
        self.preview_commit_data = None
        self.preview_commit_header_info = None
        self.preview_commit_trace_metadata = None
        self.preview_title = "预览"
        self.preview_source_data = None
        self.preview_source_title = "预览前"
        self.selected_history_index = 0
        self._syncing_compare_combo = False
        self._syncing_view_combo = False
        self._slider_compare_ratio = 0.5
        self._slider_dragging = False
        self._preview_thread = None
        self._preview_worker = None
        self._preview_running = False
        self._preview_seq = 0
        self._pending_preview_request = None
        self._apply_after_preview = False
        self.preview_request_context = None

        # 收藏管理器
        self.favorites_manager = FavoritesManager()

        # 流程模板管理器
        self.workflow_manager = WorkflowTemplateManager()

        self.setup_ui()

        # 加载收藏列表
        self._update_favorites_list()

        # 更新流程模板列表
        self._update_workflow_list()

        if self.data_state is not None:
            self.sync_from_shared_state({"reason": "init"})

    def setup_ui(self):
        self.setObjectName("workbenchRoot")
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(12, 8, 12, 8)

        # ========== 顶部工具栏 ==========
        self._create_toolbar(main_layout)

        # ========== 主体区域（用 QSplitter 实现可调宽度）=========
        # 先创建水平 splitter
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧方法树
        self.method_browser = MethodBrowserTree()
        self.method_browser.setFixedWidth(220)
        self.method_browser.method_selected.connect(self._on_method_selected)
        self.method_browser.action_triggered.connect(self._on_action_triggered)
        self.method_browser.template_execute_requested.connect(
            self._on_template_execute
        )
        h_splitter.addWidget(self.method_browser)

        # 中间参数区
        self.param_editor = ParamEditorPanel()
        setattr(self.param_editor, "parent_window", self)
        self.param_editor.setMinimumWidth(280)
        self.param_editor.setMaximumWidth(360)
        self.param_editor.run_requested.connect(self._run_current_method)
        self.param_editor.params_changed.connect(self._on_params_changed)
        h_splitter.addWidget(self.param_editor)

        # 右侧预览区
        preview_container = QWidget()
        preview_container.setObjectName("workbenchPreviewPanel")
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        preview_layout.setSpacing(8)

        # 预览工具条容器（两行布局）
        preview_toolbar = QWidget()
        preview_toolbar.setObjectName("workbenchPreviewToolbar")
        toolbar_layout = QVBoxLayout(preview_toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(4)

        # 第一行：预览模式 + 数据信息
        first_row = QWidget()
        first_row_layout = QHBoxLayout(first_row)
        first_row_layout.setContentsMargins(0, 0, 0, 0)
        first_row_layout.setSpacing(8)

        mode_label = QLabel("预览:")
        mode_label.setProperty("class", "titleSmall")
        first_row_layout.addWidget(mode_label)

        self.mode_group = QButtonGroup(self)

        self.radio_original = QRadioButton("单图")
        self.radio_original.setChecked(True)
        self.mode_group.addButton(self.radio_original, 0)
        self.radio_original.toggled.connect(self._refresh_preview)
        first_row_layout.addWidget(self.radio_original)

        self.radio_result = QRadioButton("结果")
        self.mode_group.addButton(self.radio_result, 1)
        self.radio_result.toggled.connect(self._refresh_preview)
        first_row_layout.addWidget(self.radio_result)

        self.radio_compare = QRadioButton("对比")
        self.mode_group.addButton(self.radio_compare, 2)
        self.radio_compare.toggled.connect(self._refresh_preview)
        first_row_layout.addWidget(self.radio_compare)

        self.radio_slider = QRadioButton("滑动对比")
        self.mode_group.addButton(self.radio_slider, 3)
        self.radio_slider.toggled.connect(self._refresh_preview)
        first_row_layout.addWidget(self.radio_slider)

        display_label = QLabel("显示:")
        display_label.setProperty("class", "titleSmall")
        first_row_layout.addWidget(display_label)

        self.display_group = QButtonGroup(self)

        self.radio_raster = QRadioButton("栅格")
        self.radio_raster.setChecked(True)
        self.display_group.addButton(self.radio_raster, 0)
        self.radio_raster.toggled.connect(self._refresh_preview)
        first_row_layout.addWidget(self.radio_raster)

        self.radio_wiggle = QRadioButton("摆动")
        self.display_group.addButton(self.radio_wiggle, 1)
        self.radio_wiggle.toggled.connect(self._refresh_preview)
        first_row_layout.addWidget(self.radio_wiggle)

        view_label = QLabel("查看:")
        view_label.setProperty("class", "titleSmall")
        first_row_layout.addWidget(view_label)

        from qfluentwidgets import ComboBox

        self.view_combo = ComboBox()
        self.view_combo.setFixedHeight(24)
        self.view_combo.setFixedWidth(190)
        self.view_combo.addItem("原始数据")
        self.view_combo.currentIndexChanged.connect(self._on_view_selection_changed)
        first_row_layout.addWidget(self.view_combo)

        self.preview_group = QButtonGroup(self)

        self.preview_before_radio = QRadioButton("预览前")
        self.preview_before_radio.setAutoExclusive(False)
        self.preview_before_radio.setChecked(True)
        self.preview_group.addButton(self.preview_before_radio, 0)
        self.preview_before_radio.toggled.connect(self._refresh_preview)
        first_row_layout.addWidget(self.preview_before_radio)

        self.preview_after_radio = QRadioButton("预览后")
        self.preview_after_radio.setAutoExclusive(False)
        self.preview_group.addButton(self.preview_after_radio, 1)
        self.preview_after_radio.toggled.connect(self._refresh_preview)
        first_row_layout.addWidget(self.preview_after_radio)

        # 添加对比基准选择下拉框
        self.compare_label = QLabel("基准:")
        self.compare_label.setProperty("class", "titleSmall")
        self.compare_label.setVisible(False)
        first_row_layout.addWidget(self.compare_label)

        self.compare_combo = ComboBox()
        self.compare_combo.setFixedHeight(24)
        self.compare_combo.setFixedWidth(170)
        self.compare_combo.addItem("原始数据")
        self.compare_combo.currentIndexChanged.connect(
            self._on_compare_selection_changed
        )
        self.compare_combo.setVisible(False)  # 默认隐藏，只在对比/滑动对比模式显示
        first_row_layout.addWidget(self.compare_combo)

        first_row_layout.addStretch()

        # 预览信息
        self.preview_info = QLabel("未加载数据")
        self.preview_info.setProperty("class", "hintText")
        first_row_layout.addWidget(self.preview_info)

        toolbar_layout.addWidget(first_row)

        # 第二行：质量指标
        second_row = QWidget()
        second_row_layout = QHBoxLayout(second_row)
        second_row_layout.setContentsMargins(0, 0, 0, 0)
        second_row_layout.setSpacing(12)

        metrics_label = QLabel("指标:")
        metrics_label.setProperty("class", "titleSmall")
        second_row_layout.addWidget(metrics_label)

        # 聚焦比
        self.focus_ratio_label = QLabel("聚焦比: --")
        self.focus_ratio_label.setToolTip("聚焦比，越高越好")
        self.focus_ratio_label.setProperty("class", "textSecondary")
        second_row_layout.addWidget(self.focus_ratio_label)

        # 热像素
        self.hot_pixels_label = QLabel("热像素: --")
        self.hot_pixels_label.setToolTip("热像素数量，越低越好")
        self.hot_pixels_label.setProperty("class", "textSecondary")
        second_row_layout.addWidget(self.hot_pixels_label)

        # 信噪比
        self.snr_label = QLabel("SNR: --")
        self.snr_label.setToolTip("信噪比，越高越好")
        self.snr_label.setProperty("class", "textSecondary")
        second_row_layout.addWidget(self.snr_label)

        # 数据范围
        self.data_range_label = QLabel("范围: --")
        self.data_range_label.setToolTip("数据范围")
        self.data_range_label.setProperty("class", "textSecondary")
        second_row_layout.addWidget(self.data_range_label)

        second_row_layout.addStretch()

        toolbar_layout.addWidget(second_row)

        preview_layout.addWidget(preview_toolbar)

        # 预览画布
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._apply_plot_theme()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_move)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        preview_layout.addWidget(self.canvas, stretch=1)

        h_splitter.addWidget(preview_container)

        # 设置默认比例 左(固定):中:右 ≈ 0:1.5:4.5
        h_splitter.setStretchFactor(0, 0)   # 方法树固定宽度
        h_splitter.setStretchFactor(1, 15)  # 参数区
        h_splitter.setStretchFactor(2, 45)  # 预览区
        h_splitter.setSizes([220, 300, 660])  # 初始像素宽度

        main_layout.addWidget(h_splitter, stretch=1)

        # ========== 底部日志区 ==========
        self._create_log_panel(main_layout)

        # 初始状态
        self._refresh_preview()

    def _create_toolbar(self, parent_layout):
        """创建顶部工具栏"""
        toolbar = QWidget()
        toolbar.setObjectName("workbenchTopBar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 2, 0, 2)
        toolbar_layout.setSpacing(10)

        # 标题
        title = QLabel("工作台总控中心")
        title.setProperty("class", "titleMedium")
        toolbar_layout.addWidget(title)

        toolbar_layout.addStretch()

        # 返回日常处理按钮
        btn_back_main = PushButton("返回日常处理")
        btn_back_main.setMinimumHeight(36)
        btn_back_main.clicked.connect(self._on_back_to_main)
        toolbar_layout.addWidget(btn_back_main)

        # 分隔
        toolbar_layout.addSpacing(20)

        # 数据状态
        self.data_status_label = QLabel("未加载数据")
        self.data_status_label.setProperty("class", "hintText")
        toolbar_layout.addWidget(self.data_status_label)

        # 导入入口与日常处理页保持一致
        self.btn_import = SplitPushButton(self)
        self.btn_import.setText("导入数据")
        self.btn_import.setMinimumHeight(36)
        self.import_menu = QMenu(self)
        self.action_import_csv = QAction("导入 CSV", self)
        self.action_import_folder = QAction("导入 A-scan 文件夹", self)
        self.action_import_out = QAction("导入 gprMax .out", self)
        self.import_menu.addAction(self.action_import_csv)
        self.import_menu.addAction(self.action_import_folder)
        self.import_menu.addAction(self.action_import_out)
        self.btn_import.setToolTip(
            "点击主按钮默认导入 CSV，点击右侧箭头可选择其它导入方式"
        )
        self.btn_import.setFlyout(self.import_menu)
        self.action_import_csv.triggered.connect(self._on_import_csv)
        self.action_import_folder.triggered.connect(self._on_import_folder)
        self.action_import_out.triggered.connect(self._on_import_out)
        self.btn_import.clicked.connect(self._on_import_csv)
        toolbar_layout.addWidget(self.btn_import)

        toolbar_layout.addSpacing(10)

        # 导航按钮组
        btn_basic = PushButton("日常处理")
        btn_basic.setMinimumHeight(36)
        btn_basic.clicked.connect(self._on_basic_mode)
        toolbar_layout.addWidget(btn_basic)

        btn_advanced = PushButton("显示与对比")
        btn_advanced.setMinimumHeight(36)
        btn_advanced.clicked.connect(self._on_advanced_mode)
        toolbar_layout.addWidget(btn_advanced)

        btn_quality = PushButton("质量与导出")
        btn_quality.setMinimumHeight(36)
        btn_quality.clicked.connect(self._on_quality_mode)
        toolbar_layout.addWidget(btn_quality)

        toolbar_layout.addSpacing(10)

        # 主题切换按钮
        self.theme_manager = get_theme_manager()
        self.btn_theme = PushButton()
        self.btn_theme.setMinimumHeight(36)
        self.btn_theme.clicked.connect(self._toggle_theme)
        self._update_theme_button()
        toolbar_layout.addWidget(self.btn_theme)

        parent_layout.addWidget(toolbar)

    def _create_log_panel(self, parent_layout):
        """创建底部日志区"""
        # 折叠状态标签
        self.log_status_label = QLabel("准备就绪")
        self.log_status_label.setObjectName("workbenchLogStatus")
        self.log_status_label.setVisible(False)

        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(136)
        self.log_text.setObjectName("workbenchLogText")

        # 折叠按钮
        fold_btn = PushButton("▼ 折叠")
        fold_btn.setFixedHeight(24)
        fold_btn.clicked.connect(self._toggle_log)

        log_header = QWidget()
        log_header_layout = QHBoxLayout(log_header)
        log_header_layout.setContentsMargins(0, 0, 0, 0)

        log_label = QLabel("运行日志")
        log_label.setProperty("class", "titleSmall")
        log_header_layout.addWidget(log_label)
        log_header_layout.addStretch()
        log_header_layout.addWidget(fold_btn)

        # 日志容器
        log_container = QWidget()
        log_container_layout = QVBoxLayout(log_container)
        log_container_layout.setContentsMargins(0, 0, 0, 0)
        log_container_layout.setSpacing(4)
        log_container_layout.addWidget(log_header)
        log_container_layout.addWidget(self.log_text)

        self.log_container = log_container
        self.fold_btn = fold_btn

        parent_layout.addWidget(log_container)
        parent_layout.addWidget(self.log_status_label)

    def _toggle_log(self):
        """折叠/展开日志区"""
        is_visible = self.log_text.isVisible()
        self.log_text.setVisible(not is_visible)
        self.fold_btn.setText("▲ 展开" if not is_visible else "▼ 折叠")

        if not is_visible:
            # 折叠时显示状态摘要
            self.log_status_label.setVisible(True)
            self.log_status_label.setText(self._get_log_summary())
        else:
            self.log_status_label.setVisible(False)

    def _get_log_summary(self) -> str:
        """获取日志摘要"""
        text = self.log_text.toPlainText().strip()
        if not text:
            return "日志为空"
        lines = text.split("\n")
        last_line = lines[-1] if lines else ""
        return f"共 {len(lines)} 行 | 最后: {last_line[:80]}..."

    def _log(self, msg: str, level: str = "INFO"):
        """写入日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {msg}"
        self.log_text.append(log_line)

        if (
            self.parent_window is not None
            and hasattr(self.parent_window, "page_quality")
            and self.parent_window.page_quality is not None
        ):
            self.parent_window.page_quality.append_record(log_line)

        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    # ========== 事件处理 ==========

    def _on_import_data(self):
        """导入数据（通用）"""
        self.data_import_requested.emit()

    def _on_import_csv(self):
        """导入 CSV 文件"""
        if self.parent_window and hasattr(self.parent_window, "import_csv_file"):
            self.parent_window.import_csv_file()
        else:
            self.data_import_requested.emit()

    def _on_import_folder(self):
        """导入 A-scan 文件夹"""
        if self.parent_window and hasattr(self.parent_window, "import_ascans_folder"):
            self.parent_window.import_ascans_folder()
        else:
            self.data_import_requested.emit()

    def _on_import_out(self):
        """导入 gprMax .out 文件"""
        if self.parent_window and hasattr(self.parent_window, "import_gprmax_out_file"):
            self.parent_window.import_gprmax_out_file()
        else:
            self.data_import_requested.emit()

    def _on_back_to_main(self):
        """返回日常处理界面"""
        if self.parent_window and hasattr(self.parent_window, "switch_to_main_mode"):
            self.parent_window.switch_to_main_mode("basic")
            self._log("已返回日常处理界面")

    def _on_basic_mode(self):
        """切换到日常处理页"""
        if self.parent_window and hasattr(self.parent_window, "switch_to_main_mode"):
            self.parent_window.switch_to_main_mode("basic")
            self._log("已切换到日常处理页")

    def _on_advanced_mode(self):
        """切换到显示与对比页"""
        if self.parent_window and hasattr(self.parent_window, "switch_to_main_mode"):
            self.parent_window.switch_to_main_mode("advanced")
            self._log("已切换到显示与对比页")

    def _on_quality_mode(self):
        """切换到质量与导出页"""
        if self.parent_window and hasattr(self.parent_window, "switch_to_main_mode"):
            self.parent_window.switch_to_main_mode("quality")
            self._log("已切换到质量与导出页")

    def _update_theme_button(self):
        """更新主题切换按钮的图标和文字。"""
        current_theme = self.theme_manager.get_current_theme()
        theme_info = self.theme_manager.get_theme_info(current_theme)
        icon_key = theme_info.get("icon")
        if icon_key == "sun":
            icon = FluentIcon.BRIGHTNESS
        else:
            icon = FluentIcon.CONSTRACT
        self.btn_theme.setIcon(icon)
        self.btn_theme.setText(f"  {theme_info['name']}")

    def _toggle_theme(self):
        """切换主题"""
        self.theme_manager.toggle_theme()
        current_theme = self.theme_manager.get_current_theme()
        theme_info = self.theme_manager.get_theme_info(current_theme)

        # 更新按钮图标和文字
        self._update_theme_button()

        app = QApplication.instance()
        if app is not None:
            self.theme_manager.apply_app_theme(app, current_theme)

        # 重新抛光主窗口局部样式
        if self.parent_window and hasattr(self.parent_window, "_apply_style"):
            self.parent_window._apply_style()
        if self.parent_window and hasattr(self.parent_window, "_refresh_plot"):
            self.parent_window._refresh_plot()

        self._apply_plot_theme()
        self._refresh_preview()

        self._log(f"已切换到{theme_info['name']}")

    def _on_method_selected(self, method_id: str):
        """方法被选中"""
        self._log(f"选择方法: {method_id}")
        self.param_editor.load_method(method_id)
        self._update_action_buttons()
        if self.raw_data is not None:
            self._on_params_changed()

    def _on_action_triggered(self, action_id: str):
        """特殊操作被触发"""
        if action_id == "_import_csv":
            self._on_import_data()
        elif action_id == "_data_info":
            self._show_data_info()
        elif (
            action_id.startswith("_robust_")
            or action_id.startswith("_high_")
        ):
            profile_map = {
                "_robust_imaging": "robust_imaging",
                "_high_focus": "high_focus",
            }
            profile_key = profile_map.get(action_id)
            if (
                profile_key
                and self.parent_window
                and hasattr(self.parent_window, "run_recommended_pipeline")
            ):
                self._log(f"运行推荐流程: {action_id}")
                self.parent_window.run_recommended_pipeline(profile_key)
            else:
                self._log(f"预设流程不可用: {action_id}", "WARN")

    def _on_template_execute(self, template_name: str):
        """执行流程模板"""
        # 检查是否加载了数据
        if self.raw_data is None:
            self._log("未加载数据，请先导入数据", "ERROR")
            return

        # 获取模板
        template = self.workflow_manager.get_template(template_name)
        if not template:
            self._log(f"模板不存在: {template_name}", "ERROR")
            return

        # 获取模板中的方法
        methods = self.workflow_manager.get_template_methods(template_name)
        if not methods:
            self._log(f"模板 {template_name} 没有方法", "WARN")
            return

        workflow_methods = [WorkflowMethod.from_dict(m) for m in methods]
        executor = WorkflowExecutor(
            header_info=self.resolve_input_header_info(
                self.param_editor.get_input_source()
            ),
            trace_metadata=self.resolve_input_trace_metadata(
                self.param_editor.get_input_source()
            ),
        )

        self._log(f"{'=' * 40}")
        self._log(f"开始执行流程模板: {template_name}")
        self._log(f"包含 {len(workflow_methods)} 个方法")

        try:
            current_data = self.resolve_input_data(
                self.param_editor.get_input_source()
            )[0]
            if current_data is None:
                current_data = self.raw_data
            result = executor.execute_all(current_data, workflow_methods)
            self.update_current_result(
                result,
                header_info=executor.current_header_info,
                trace_metadata=executor.current_trace_metadata,
            )
            self._log("流程执行完成")
        except ExecutionError as e:
            self._log(f"流程执行失败: {e}", "ERROR")
        finally:
            self._log(f"{'=' * 40}")

    def _show_data_info(self):
        """显示数据信息"""
        if self.raw_data is not None:
            lines = [
                f"数据形状: {self.raw_data.shape}",
                f"数据范围: [{self.raw_data.min():.4f}, {self.raw_data.max():.4f}]",
                f"数据类型: {self.raw_data.dtype}",
            ]
            if self.data_state is not None:
                header = self.data_state.header_info or {}
                if header.get("has_airborne_metadata"):
                    lines.extend(
                        [
                            "",
                            "航空元数据摘要:",
                            f"- 测线长度: {float(header.get('track_length_m', 0.0)):.2f} m",
                            "- 道间距范围: {:.3f} ~ {:.3f} m (均值 {:.3f} m)".format(
                                float(header.get("trace_interval_min_m", 0.0)),
                                float(header.get("trace_interval_max_m", 0.0)),
                                float(header.get("trace_interval_m", 0.0)),
                            ),
                            "- 地表高程范围: {:.2f} ~ {:.2f} m".format(
                                float(header.get("ground_elevation_min_m", 0.0)),
                                float(header.get("ground_elevation_max_m", 0.0)),
                            ),
                            "- 飞行高度范围: {:.2f} ~ {:.2f} m".format(
                                float(header.get("flight_height_min_m", 0.0)),
                                float(header.get("flight_height_max_m", 0.0)),
                            ),
                        ]
                    )
            info = "\n".join(lines)
            self._log(info)
            QMessageBox.information(self, "数据信息", info)
        else:
            self._log("未加载数据", "WARN")

    def add_to_favorites(self, method_id: str, params: dict):
        """添加收藏"""
        self.favorites_manager.add_favorite(method_id, params)
        self._update_favorites_list()
        self._log(f"已收藏 {method_id} 的参数组")

    def select_method(self, method_id: str):
        """选择方法"""
        self.method_browser.select_method(method_id)
        self.param_editor.load_method(method_id)

    def _update_favorites_list(self):
        """更新收藏列表"""
        favorites = self.favorites_manager.get_favorites()
        self.param_editor.update_favorites_list(favorites)

    def sync_from_shared_state(self, payload: dict | None = None):
        """从共享数据状态同步工作台视图。"""
        if self.data_state is None:
            return

        current_data = self.data_state.current_data
        original_data = self.data_state.original_data

        if current_data is None or original_data is None:
            self.raw_data = None
            self.current_result = None
            self.raw_header_info = None
            self.raw_trace_metadata = None
            self.current_result_header_info = None
            self.current_result_trace_metadata = None
            self.has_processed_result = False
            self.all_results = []
            self.all_result_entries = []
            self.selected_history_index = 0
            self._clear_preview_data()
            self._slider_compare_ratio = 0.5
            self.data_status_label.setText("未加载数据")
            self.param_editor.update_result_list([])
            self._update_action_buttons()
            self._refresh_preview()
            return

        reason = (payload or {}).get("reason", "sync")

        self.raw_data = np.array(original_data, copy=True)
        self.raw_header_info = getattr(self.data_state, "original_header_info", None)
        self.raw_trace_metadata = getattr(
            self.data_state, "original_trace_metadata", None
        )
        self.current_result = np.array(current_data, copy=True)
        self.current_result_header_info = self.data_state.header_info
        self.current_result_trace_metadata = self.data_state.current_trace_metadata
        self.has_processed_result = not np.array_equal(
            self.current_result, self.raw_data
        )

        self._rebuild_result_history_cache()

        self.selected_history_index = max(0, len(self.all_results) - 1)
        self.data_status_label.setText(
            f"原始: {self.raw_data.shape} | 当前: {self.current_result.shape}"
        )
        self.param_editor.set_input_source_enabled(self.has_processed_result)
        if reason in {"loaded", "reset", "undo", "current_updated", "manual_sync"}:
            self._clear_preview_data()
        self.param_editor.update_result_list(self.all_results)
        if self.has_processed_result:
            self.param_editor.set_input_source("current")
        else:
            self.param_editor.set_input_source("raw")
        self._update_action_buttons()
        self._refresh_preview()

    def _rebuild_result_history_cache(self) -> None:
        """Rebuild formal result history with data and matching metadata."""
        if self.data_state is not None:
            self.all_result_entries = self.data_state.build_result_history_entries()
        elif self.raw_data is not None:
            self.all_result_entries = [
                {
                    "label": "原始数据",
                    "data": np.array(self.raw_data, copy=True),
                    "header_info": self.raw_header_info,
                    "trace_metadata": self.raw_trace_metadata,
                }
            ]
        else:
            self.all_result_entries = []

        self.all_results = [
            (str(entry["label"]), np.array(entry["data"], copy=True))
            for entry in self.all_result_entries
        ]

    def _selected_history_entry(self):
        """Return the selected formal history entry, including metadata."""
        entries = getattr(self, "all_result_entries", [])
        if 0 <= self.selected_history_index < len(entries):
            return entries[self.selected_history_index]
        return None

    def _update_workflow_list(self):
        """更新流程模板列表"""
        templates = self.workflow_manager.get_all_templates()
        self.log_workflow_templates(templates)

    def log_workflow_templates(self, templates: list):
        """记录流程模板信息"""
        if templates:
            self._log(f"已加载 {len(templates)} 个流程模板")
            for template in templates:
                method_count = template.get("method_count", 0)
                self._log(f"  - {template['name']}: {method_count} 个方法")
        else:
            self._log("未找到自定义流程模板，使用预设模板")

        # 更新方法树中的流程模板列表
        self.method_browser.update_workflow_templates(templates)

    # ========== 核心方法 ==========

    def _on_params_changed(self):
        """参数改变时触发预览"""
        # 检查是否有数据和方法
        if self.raw_data is None:
            return

        method_id = self.param_editor.current_method_id
        if not method_id:
            return

        # 获取当前参数
        try:
            params = self.param_editor.get_current_params()
        except Exception:
            return

        # 获取输入数据
        source = self.param_editor.get_input_source()
        input_data, _ = self.resolve_input_data(source)

        if input_data is None:
            return

        method_info = PROCESSING_METHODS.get(method_id, {})
        method_name = method_info.get("name", method_id)
        self._request_preview(
            method_id=method_id,
            params=params,
            input_data=input_data,
            source_text=source,
            title=f"预览: {method_name}",
            method_name=method_name,
            announce=False,
        )

    def _refresh_preview_with_data(self, data: np.ndarray, title: str = "预览"):
        """用指定数据刷新预览（不更新 current_result）"""
        self.set_preview_result(data, title)

    def _clear_preview_data(self):
        """清除临时预览结果。"""
        self.preview_data = None
        self.preview_header_info = None
        self.preview_trace_metadata = None
        self.preview_commit_data = None
        self.preview_commit_header_info = None
        self.preview_commit_trace_metadata = None
        self.preview_title = "预览"
        self.preview_source_data = None
        self.preview_source_title = "预览前"
        self.preview_request_context = None

    def _update_action_buttons(self):
        """根据当前状态统一刷新按钮。"""
        if self.raw_data is None:
            self.param_editor.set_buttons_for_no_data()
            return
        if not self.param_editor.current_method_id:
            self.param_editor.set_buttons_for_no_method()
            return
        if self.preview_data is not None:
            self.param_editor.set_buttons_for_preview()
            return
        if self.has_processed_result:
            self.param_editor.set_buttons_for_committed_result()
            return
        self.param_editor.set_buttons_for_ready()

    def set_preview_result(
        self,
        data: np.ndarray,
        title: str = "预览",
        header_info: dict | None = None,
        trace_metadata: dict | None = None,
        commit_data: np.ndarray | None = None,
        commit_header_info: dict | None = None,
        commit_trace_metadata: dict | None = None,
    ):
        """设置当前预览结果，不修改正式结果历史。"""
        self.preview_data = np.array(data, copy=True)
        self.preview_header_info = header_info
        self.preview_trace_metadata = trace_metadata
        self.preview_commit_data = (
            np.array(commit_data, copy=True) if commit_data is not None else None
        )
        self.preview_commit_header_info = commit_header_info
        self.preview_commit_trace_metadata = commit_trace_metadata
        self.preview_title = title
        self.radio_result.setChecked(True)
        self._refresh_preview()
        self._update_action_buttons()

    def _build_request_context(self, method_id: str, params: dict, source: str) -> dict:
        return {
            "method_id": method_id,
            "params": dict(params),
            "source": source,
            "history_index": self.selected_history_index
            if source == "history"
            else None,
        }

    def _is_preview_current(self, context: dict) -> bool:
        return self.preview_data is not None and self.preview_request_context == context

    def _request_preview(
        self,
        method_id: str,
        params: dict,
        input_data: np.ndarray,
        source_text: str,
        title: str,
        method_name: str,
        announce: bool,
    ):
        """请求后台计算预览，只保留最新一条。"""
        self._preview_seq += 1
        request = {
            "seq": self._preview_seq,
            "method_id": method_id,
            "params": dict(params),
            "input_data": np.array(input_data, copy=True),
            "header_info": self.resolve_input_header_info(source_text),
            "trace_metadata": self.resolve_input_trace_metadata(source_text),
            "source_text": source_text,
            "title": title,
            "method_name": method_name,
            "announce": announce,
            "context": self._build_request_context(method_id, params, source_text),
        }
        self._pending_preview_request = request

        if announce:
            self.preview_info.setText("正在计算预览...")
        self.preview_data = None
        self.preview_title = title
        self.preview_source_data = np.array(input_data, copy=True)
        self.preview_source_title = f"预览前: {source_text}"
        self._update_action_buttons()

        if self._preview_running:
            if self._preview_worker is not None:
                self._preview_worker.request_cancel()
            return

        self._start_pending_preview_request()

    def _start_pending_preview_request(self):
        """启动等待中的最新预览请求。"""
        if self._pending_preview_request is None:
            return

        request = self._pending_preview_request
        self._pending_preview_request = None
        self._preview_running = True

        self._preview_thread = QThread(self)
        self._preview_worker = PreviewWorker(request)
        self._preview_worker.moveToThread(self._preview_thread)

        self._preview_thread.started.connect(self._preview_worker.run)
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.error.connect(self._on_preview_error)
        self._preview_worker.finished.connect(self._preview_thread.quit)
        self._preview_worker.error.connect(self._preview_thread.quit)
        self._preview_thread.finished.connect(self._cleanup_preview_worker)
        self._preview_thread.start()

    def _cleanup_preview_worker(self):
        """清理预览线程。"""
        if self._preview_worker is not None:
            self._preview_worker.deleteLater()
            self._preview_worker = None
        if self._preview_thread is not None:
            self._preview_thread.deleteLater()
            self._preview_thread = None
        self._preview_running = False
        if self._pending_preview_request is not None:
            self._start_pending_preview_request()
        else:
            self._update_action_buttons()

    def _invalidate_preview_requests(self):
        """使正在进行/排队的预览失效。"""
        self._preview_seq += 1
        self._pending_preview_request = None
        self._apply_after_preview = False
        if self._preview_worker is not None:
            self._preview_worker.request_cancel()

    def _on_preview_finished(self, payload: dict):
        """后台预览完成。"""
        seq = payload.get("seq", -1)
        if payload.get("cancelled"):
            return
        if seq != self._preview_seq:
            return

        result = payload.get("data")
        if result is None:
            return

        self.preview_request_context = payload.get("context")
        self.set_preview_result(
            result,
            payload.get("title", "预览"),
            header_info=payload.get("header_info"),
            trace_metadata=payload.get("trace_metadata"),
        )
        if payload.get("announce"):
            elapsed = payload.get("elapsed_ms", 0.0)
            if self._apply_after_preview:
                self._log(f"完成！耗时 {elapsed:.1f}ms")
                self._log(
                    f"结果: {result.shape} | 范围 [{result.min():.3f}, {result.max():.3f}]"
                )
            else:
                self._log(f"预览完成！耗时 {elapsed:.1f}ms")
                self._log(
                    f"预览: {result.shape} | 范围 [{result.min():.3f}, {result.max():.3f}]"
                )
            self._log(f"{'=' * 40}")

        if self._apply_after_preview:
            self._apply_after_preview = False
            self._save_result()

    def _on_preview_error(self, payload: dict):
        """后台预览失败。"""
        if payload.get("seq", -1) != self._preview_seq:
            return
        self._apply_after_preview = False
        error_msg = payload.get("error", "未知错误")
        if payload.get("announce"):
            self._log(f"执行失败: {error_msg}", "ERROR")
        else:
            self._log(f"预览计算失败: {error_msg}", "WARN")
        self._update_action_buttons()

    def _safe_remove_colorbar(self):
        """安全移除当前颜色条。"""
        cbar = getattr(self, "_cbar", None)
        if cbar is None:
            return
        try:
            cbar.remove()
        except Exception as e:
            logger.debug("Failed to remove workbench colorbar: %s", e)
        finally:
            self._cbar = None

    def _apply_plot_theme(self):
        """让 Matplotlib 画布颜色跟随当前主题。"""
        theme = (
            self.theme_manager.get_current_theme()
            if hasattr(self, "theme_manager")
            else "light"
        )
        if theme == "dark":
            fig_face = "#1f2125"
            ax_face = "#23252a"
            text_color = "#e8e8e8"
            spine_color = "#5a606b"
            grid_color = "#4b515c"
        else:
            fig_face = "#ffffff"
            ax_face = "#f8f8f8"
            text_color = "#333333"
            spine_color = "#bbbbbb"
            grid_color = "#cccccc"

        self.fig.patch.set_facecolor(fig_face)
        if hasattr(self, "ax") and self.ax is not None:
            self.ax.set_facecolor(ax_face)
            self.ax.tick_params(colors=text_color)
            self.ax.xaxis.label.set_color(text_color)
            self.ax.yaxis.label.set_color(text_color)
            self.ax.title.set_color(text_color)
            for spine in self.ax.spines.values():
                spine.set_color(spine_color)
            self.ax.grid(color=grid_color, alpha=0.35)

    def _get_effective_result_data(self):
        """获取当前应显示/比较/统计的结果数据。"""
        return (
            self.preview_data if self.preview_data is not None else self.current_result
        )

    def _get_effective_header_info(self):
        """获取当前应使用的结果头信息。"""
        return (
            self.preview_header_info
            if self.preview_data is not None
            else self.current_result_header_info
        )

    def _build_view_entries(self) -> list[dict]:
        """Build selectable single-view entries for raw/current/history/preview data."""
        entries: list[dict] = []

        if self.all_result_entries:
            for idx, entry in enumerate(self.all_result_entries):
                data = entry.get("data")
                if data is None:
                    continue
                label = str(entry.get("label") or f"步骤{idx}")
                if idx == 0 and label == "原始数据":
                    display_label = "原始数据"
                elif idx == len(self.all_result_entries) - 1:
                    display_label = f"当前结果: {label}"
                else:
                    display_label = f"步骤{idx}: {label}"
                entries.append(
                    {
                        "label": display_label,
                        "data": data,
                        "header_info": entry.get("header_info"),
                        "trace_metadata": entry.get("trace_metadata"),
                    }
                )
        elif self.raw_data is not None:
            entries.append(
                {
                    "label": "原始数据",
                    "data": self.raw_data,
                    "header_info": self.raw_header_info,
                    "trace_metadata": self.raw_trace_metadata,
                }
            )

        if self.preview_source_data is not None:
            entries.append(
                {
                    "label": self.preview_source_title or "预览前",
                    "data": self.preview_source_data,
                    "header_info": self.current_result_header_info,
                    "trace_metadata": self.current_result_trace_metadata,
                }
            )

        if self.preview_data is not None:
            entries.append(
                {
                    "label": self.preview_title or "预览后",
                    "data": self.preview_data,
                    "header_info": self.preview_header_info,
                    "trace_metadata": self.preview_trace_metadata,
                }
            )

        return entries

    def _update_view_combo(self) -> None:
        """Refresh the single-view selector without triggering redraw loops."""
        if not hasattr(self, "view_combo") or self.view_combo is None:
            return

        entries = self._build_view_entries()
        previous_text = self.view_combo.currentText() if self.view_combo.count() else ""
        self._syncing_view_combo = True
        old_block = self.view_combo.blockSignals(True)
        try:
            self.view_combo.clear()
            for entry in entries:
                self.view_combo.addItem(str(entry.get("label") or "未命名结果"))

            if not entries:
                return

            target_index = len(entries) - 1 if self.has_processed_result else 0
            labels = [str(entry.get("label") or "") for entry in entries]
            if previous_text in labels:
                target_index = labels.index(previous_text)
            self.view_combo.setCurrentIndex(target_index)
        finally:
            self.view_combo.blockSignals(old_block)
            self._syncing_view_combo = False

    def _get_selected_view_entry(self):
        """Return the data entry currently selected by the single-view selector."""
        entries = self._build_view_entries()
        if not entries:
            return None
        if not hasattr(self, "view_combo") or self.view_combo is None:
            return entries[-1]
        index = self.view_combo.currentIndex()
        if index < 0 or index >= len(entries):
            index = len(entries) - 1 if self.has_processed_result else 0
        return entries[index]

    def resolve_input_header_info(self, source: str):
        """根据输入源解析对应头信息。"""
        if source == "raw":
            return self.raw_header_info or (
                self.data_state.header_info if self.data_state is not None else None
            )
        if source == "history":
            entry = self._selected_history_entry()
            if entry is not None:
                return entry.get("header_info")
            return self.current_result_header_info
        return self.current_result_header_info or (
            self.data_state.header_info if self.data_state is not None else None
        )

    def resolve_input_trace_metadata(self, source: str):
        """根据输入源解析对应道元数据。"""
        if self.data_state is None:
            return None
        if source == "raw":
            return self.raw_trace_metadata or self.data_state.original_trace_metadata
        if source == "history":
            entry = self._selected_history_entry()
            if entry is not None:
                return entry.get("trace_metadata")
        return (
            self.current_result_trace_metadata or self.data_state.current_trace_metadata
        )

    def resolve_input_data(self, source: str):
        """根据输入源解析当前要处理的数据。"""
        if source == "raw":
            return self.raw_data, "原始数据"
        if source == "history":
            entry = self._selected_history_entry()
            if entry is not None:
                return entry.get("data"), f"历史结果: {entry.get('label')}"
            return self.current_result, "当前结果（历史回退）"
        if self.current_result is not None:
            return self.current_result, "当前结果"
        return self.raw_data, "原始数据（fallback）"

    def _run_current_method(self):
        """应用当前方法。若已有当前预览则直接提交，否则先计算再提交。"""
        method_id = self.param_editor.current_method_id
        if not method_id:
            self._log("请先选择一个方法", "WARN")
            return

        self.param_editor.cancel_pending_preview()

        # 获取输入数据
        source = self.param_editor.get_input_source()
        input_data, _ = self.resolve_input_data(source)

        if input_data is None:
            self._log("未加载数据，请先导入数据", "ERROR")
            return

        # 获取参数
        params = self.param_editor.get_current_params()
        method_info = PROCESSING_METHODS.get(method_id, {})
        method_name = method_info.get("name", method_id)
        context = self._build_request_context(method_id, params, source)

        self._log(f"应用方法: {method_id}, 输入源: {source}, 参数: {params}")

        if self._is_preview_current(context):
            self._save_result()
            return

        self.param_editor.set_buttons_for_running()
        self._apply_after_preview = True
        self._request_preview(
            method_id=method_id,
            params=params,
            input_data=input_data,
            source_text=source,
            title=f"预览: {method_name}",
            method_name=method_name,
            announce=True,
        )

    def _save_result(self):
        """将当前预览或当前正式结果提交到主界面。"""
        if self.preview_data is None:
            if self.current_result is None or not self.has_processed_result:
                self._log("没有可应用的结果", "WARN")
                return
            self.save_result_requested.emit()
            self._log("当前结果已同步到主界面")
            self._update_action_buttons()
            return

        self.save_result_requested.emit()
        self._clear_preview_data()
        self._update_action_buttons()

    def _export_csv(self):
        """导出当前处理数据为 CSV。"""
        target = (
            self.preview_data if self.preview_data is not None else self.current_result
        )
        if target is None:
            target = self.raw_data
        if target is None:
            self._log("没有可导出的数据", "WARN")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前处理数据为 CSV",
            "workbench_data.csv",
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not path:
            return

        try:
            from read_file_data import savecsv

            savecsv(target, path)
            self._log(f"CSV 已导出: {path}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"无法导出 CSV：\n{e}")
            self._log(f"导出 CSV 失败: {e}", "ERROR")

    def _export_image(self):
        """导出图像"""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出工作台图像",
            "workbench_preview.png",
            "PNG 文件 (*.png);;JPEG 文件 (*.jpg *.jpeg);;所有文件 (*)",
        )
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"图像已导出: {path}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"无法导出图像：\n{e}")
            self._log(f"导出图像失败: {e}", "ERROR")

    def _on_compare_selection_changed(self, index):
        """对比选择改变"""
        if getattr(self, "_syncing_compare_combo", False):
            return
        if self.radio_compare.isChecked() or self.radio_slider.isChecked():
            self._refresh_preview()

    def _on_view_selection_changed(self, index):
        """单图/目标结果选择改变。"""
        if getattr(self, "_syncing_view_combo", False):
            return
        self._refresh_preview()

    def _on_canvas_press(self, event):
        """开始拖动滑动对比分割线。"""
        if not self.radio_slider.isChecked() or event.inaxes != self.ax:
            return
        self._slider_dragging = True
        self._update_slider_ratio_from_event(event)

    def _on_canvas_move(self, event):
        """拖动中更新滑动对比分割线。"""
        if not self._slider_dragging or not self.radio_slider.isChecked():
            return
        self._update_slider_ratio_from_event(event)

    def _on_canvas_release(self, event):
        """结束拖动滑动对比分割线。"""
        if not self._slider_dragging:
            return
        self._slider_dragging = False
        if self.radio_slider.isChecked():
            self._update_slider_ratio_from_event(event)

    def _update_slider_ratio_from_event(self, event):
        """根据鼠标事件更新滑动对比位置。"""
        target_entry = self._get_selected_view_entry()
        effective_result = (
            self.preview_data
            if self.preview_data is not None
            else (target_entry.get("data") if target_entry is not None else None)
        )
        compare_data = self._get_compare_data()
        if effective_result is None or compare_data is None:
            return

        compare_arr = np.asarray(compare_data)
        result_arr = np.asarray(effective_result)
        if compare_arr.ndim != 2 or result_arr.ndim != 2:
            return
        if compare_arr.shape != result_arr.shape or compare_arr.shape[1] <= 1:
            return
        if event is None or event.xdata is None:
            return

        width = compare_arr.shape[1]
        x = max(0.0, min(float(event.xdata), float(width - 1)))
        new_ratio = x / float(width - 1)
        if abs(new_ratio - self._slider_compare_ratio) < 1e-4:
            return
        self._slider_compare_ratio = new_ratio
        self._refresh_preview()

    def _update_compare_combo(self):
        """更新对比下拉框内容"""
        if not hasattr(self, "compare_combo") or self.compare_combo is None:
            return

        current_index = self.compare_combo.currentIndex()
        self._syncing_compare_combo = True
        old_block = self.compare_combo.blockSignals(True)
        try:
            self.compare_combo.clear()

            for entry in self._build_view_entries():
                self.compare_combo.addItem(str(entry.get("label") or "未命名结果"))

            # 恢复之前的选择；如果无效则默认原始数据
            target_index = 0
            if 0 <= current_index < self.compare_combo.count():
                target_index = current_index
            if self.compare_combo.count() > 0:
                self.compare_combo.setCurrentIndex(target_index)
        finally:
            self.compare_combo.blockSignals(old_block)
            self._syncing_compare_combo = False

    def _get_compare_data(self):
        """获取对比数据"""
        try:
            if not hasattr(self, "compare_combo") or self.compare_combo is None:
                return self.raw_data

            if self.compare_combo.count() == 0:
                return self.raw_data

            index = self.compare_combo.currentIndex()
            if index < 0:
                return self.raw_data

            entries = self._build_view_entries()
            if 0 <= index < len(entries):
                return entries[index].get("data")
            return self.raw_data
        except Exception as e:
            logger.warning("Failed to resolve compare data: %s", e)
            return self.raw_data

    def _get_compare_label(self) -> str:
        """Return the label selected as comparison baseline."""
        if hasattr(self, "compare_combo") and self.compare_combo is not None:
            label = self.compare_combo.currentText()
            if label:
                return label
        return "原始数据"

    # ========== 数据更新接口 ==========

    def update_raw_data(self, data: np.ndarray):
        """更新原始数据"""
        self._invalidate_preview_requests()
        self.raw_data = np.array(data, copy=True)
        self.current_result = np.array(data, copy=True)
        self.raw_header_info = (
            self.data_state.original_header_info
            if self.data_state is not None
            else None
        )
        self.raw_trace_metadata = (
            self.data_state.original_trace_metadata
            if self.data_state is not None
            else None
        )
        self.current_result_header_info = (
            self.data_state.header_info if self.data_state is not None else None
        )
        self.current_result_trace_metadata = (
            self.data_state.current_trace_metadata if self.data_state is not None else None
        )
        self.has_processed_result = False
        self.selected_history_index = 0
        self._slider_compare_ratio = 0.5
        self._clear_preview_data()
        self._rebuild_result_history_cache()
        self.param_editor.set_input_source("raw")

        # 更新状态
        self.data_status_label.setText(f"数据: {data.shape[0]}×{data.shape[1]}")
        self._log(f"已加载数据: {data.shape}")

        # 更新预览
        self._refresh_preview()

        # 更新历史结果选择
        self.param_editor.update_result_list(self.all_results)

        self._update_action_buttons()

    def update_current_result(
        self,
        data: np.ndarray,
        result_name: str | None = None,
        header_info: dict | None = None,
        trace_metadata: dict | None = None,
    ):
        """更新当前处理结果预览，不直接修改正式共享状态。"""
        self._invalidate_preview_requests()
        result_data = np.array(data, copy=True)
        source_data = (
            self.current_result if self.current_result is not None else self.raw_data
        )
        self.preview_source_data = (
            None if source_data is None else np.array(source_data, copy=True)
        )
        self.preview_source_title = (
            "预览前: 当前结果"
            if self.current_result is not None
            else "预览前: 原始数据"
        )
        self.set_preview_result(
            result_data,
            result_name or "流程结果预览",
            header_info=header_info,
            trace_metadata=trace_metadata,
        )
        self._log(f"结果已更新为预览，待确认保存: {result_name or '流程结果预览'}")

    def _undo(self):
        """撤回上一步操作"""
        if self.preview_data is not None:
            self._invalidate_preview_requests()
            self._clear_preview_data()
            self._refresh_preview()
            self._update_action_buttons()
            self._log("已取消当前预览")
            return

        if self.data_state is not None and self.data_state.history:
            self._invalidate_preview_requests()
            self.data_state.undo()
            self._log("已撤回到上一个正式结果")
            return

        self._log("没有可撤回的操作", "WARN")

    def select_history_result_by_index(self, index: int):
        """根据索引选择历史结果"""
        if 0 <= index < len(self.all_results):
            name, data = self.all_results[index]
            self.selected_history_index = index
            if self.param_editor.radio_history.isChecked():
                self._invalidate_preview_requests()
                self._clear_preview_data()
                self._refresh_preview_with_data(data, f"历史结果: {name}")
                self._update_action_buttons()
            self._log(f"已选中历史结果: {name}")

    def _refresh_preview(self):
        """刷新预览区"""
        try:
            self._safe_remove_colorbar()

            self.ax.clear()
            self.ax.set_axis_off()
            self._apply_plot_theme()

            # 判断状态
            has_data = self.raw_data is not None

            effective_result = self._get_effective_result_data()
            has_result = (
                effective_result is not None
                and self.raw_data is not None
                and (self.has_processed_result or self.preview_data is not None)
            )

            # 控制单图/对比选择下拉框的显示
            mode = self.mode_group.checkedId()
            use_wiggle = self.radio_wiggle.isChecked() and mode in {0, 1}
            if hasattr(self, "view_combo"):
                self.view_combo.setVisible(mode in {0, 1, 2, 3})
                self._update_view_combo()
            if hasattr(self, "compare_label"):
                self.compare_label.setVisible(mode in {2, 3})
            if hasattr(self, "compare_combo"):
                self.compare_combo.setVisible(mode in {2, 3})
                if mode in {2, 3}:
                    self._update_compare_combo()

            if not has_data:
                # 状态1：未加载数据
                placeholder_primary = "#7ab8ff" if self.theme_manager.get_current_theme() == "dark" else "#1976d2"
                placeholder_secondary = "#b7bcc6" if self.theme_manager.get_current_theme() == "dark" else "#888"
                self.ax.text(
                    0.5,
                    0.6,
                    "请先导入数据",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color=placeholder_primary,
                )
                self.ax.text(
                    0.5,
                    0.4,
                    "点击左上角「打开数据」按钮",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=placeholder_secondary,
                )
                self.preview_info.setText("未加载数据")
            else:
                raw_data = self.raw_data
                assert raw_data is not None
                # 有数据时，根据模式显示
                if mode == 0:  # 单图
                    entry = self._get_selected_view_entry()
                    if entry is not None:
                        self._plot_data(
                            entry["data"],
                            str(entry.get("label") or "单图"),
                            wiggle=use_wiggle,
                            header_info=entry.get("header_info"),
                        )
                    else:
                        self._plot_data(raw_data, "原始数据", wiggle=use_wiggle)
                elif mode == 1:  # 结果
                    if self.preview_data is not None:
                        if (
                            self.preview_after_radio.isChecked()
                            or self.preview_source_data is None
                        ):
                            self._plot_data(
                                self.preview_data,
                                self.preview_title,
                                wiggle=use_wiggle,
                            )
                        else:
                            self._plot_data(
                                self.preview_source_data,
                                self.preview_source_title,
                                wiggle=use_wiggle,
                            )
                    elif has_result:
                        assert effective_result is not None
                        current_label = str(
                            getattr(self.data_state, "current_label", "") or ""
                        ).strip()
                        result_title = (
                            f"处理结果 - {current_label}"
                            if current_label and current_label != "原始数据"
                            else "处理结果"
                        )
                        self._plot_data(effective_result, result_title, wiggle=use_wiggle)
                    else:
                        self._plot_data(
                            raw_data,
                            "原始数据 (未处理)",
                            wiggle=use_wiggle,
                        )
                elif mode == 2:  # 对比
                    if has_result:
                        self._plot_comparison()
                    else:
                        self._plot_data(raw_data, "原始数据 (无结果可对比)")
                elif mode == 3:  # 滑动对比
                    if has_result:
                        self._plot_slider_comparison()
                    else:
                        self._plot_data(raw_data, "原始数据 (无结果可对比)")

            # 更新指标
            self._update_metrics()

            self.canvas.draw()
        except Exception as e:
            self._log(f"预览刷新失败: {e}", "ERROR")
            logger.exception("Workbench preview refresh failed")

    def _build_plot_axes(self, data: np.ndarray, header_info: dict | None = None):
        """根据头信息构建坐标轴。"""
        header = header_info or self._get_effective_header_info() or {}
        trace_interval = float(header.get("trace_interval_m", 0.0) or 0.0)
        if trace_interval > 0:
            x_axis = np.arange(data.shape[1], dtype=float) * trace_interval
            x_label = "距离 (m)"
        else:
            x_axis = np.arange(data.shape[1], dtype=float)
            x_label = "距离"

        if header.get("is_elevation"):
            elev_top = header.get("elevation_top_m")
            elev_bottom = header.get("elevation_bottom_m")
            if elev_top is not None and elev_bottom is not None:
                y_axis = np.linspace(float(elev_top), float(elev_bottom), data.shape[0])
                y_label = "高程 (m)"
            else:
                y_axis = np.arange(data.shape[0], dtype=float)
                y_label = "高程"
        elif header.get("is_depth"):
            depth_step = float(header.get("depth_step_m", 0.0) or 0.0)
            if depth_step > 0:
                y_axis = np.arange(data.shape[0], dtype=float) * depth_step
                y_label = "深度 (m)"
            else:
                y_axis = np.arange(data.shape[0], dtype=float)
                y_label = "深度"
        elif header.get("total_time_ns"):
            total_time_ns = float(header.get("total_time_ns", 0.0) or 0.0)
            y_axis = np.linspace(0.0, total_time_ns, data.shape[0])
            y_label = "时间 (ns)"
        else:
            y_axis = np.arange(data.shape[0], dtype=float)
            y_label = "时间"

        return header, x_axis, y_axis, x_label, y_label

    def _plot_data(
        self,
        data: np.ndarray,
        title: str,
        wiggle: bool = False,
        header_info: dict | None = None,
    ):
        """绘制单个数据。"""
        if wiggle:
            self._plot_wiggle_data(data, title, header_info=header_info)
            return

        self.ax.clear()
        header, x_axis, y_axis, x_label, y_label = self._build_plot_axes(data, header_info)
        extent = None
        if x_axis.size > 0 and y_axis.size > 0:
            x_max = float(x_axis[-1]) if x_axis.size > 1 else 1.0
            if header.get("is_elevation"):
                extent = (0.0, x_max, float(y_axis[-1]), float(y_axis[0]))
            elif header.get("is_depth"):
                extent = (0.0, x_max, float(y_axis[-1]), float(y_axis[0]))
            elif y_label == "时间 (ns)":
                extent = (0.0, x_max, float(y_axis[-1]), float(y_axis[0]))
        im = self.ax.imshow(
            data,
            aspect="auto",
            cmap="gray",
            interpolation="bilinear",
            extent=extent,
        )
        self.ax.set_title(title, fontsize=10)
        self.ax.set_xlabel(x_label, fontsize=8)
        self.ax.set_ylabel(y_label, fontsize=8)

        self._safe_remove_colorbar()

        # 创建新的颜色条
        try:
            self._cbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
        except Exception as e:
            logger.warning("Workbench colorbar creation failed: %s", e)

        self.preview_info.setText(
            f"{data.shape[0]}×{data.shape[1]} | [{data.min():.3f}, {data.max():.3f}]"
        )

    def _plot_wiggle_data(
        self,
        data: np.ndarray,
        title: str,
        header_info: dict | None = None,
    ):
        """绘制摆动图。"""
        self.ax.clear()
        self.ax.set_axis_on()

        if data.ndim != 2 or data.size == 0:
            placeholder_secondary = "#b7bcc6" if self.theme_manager.get_current_theme() == "dark" else "#888"
            self.ax.text(
                0.5,
                0.5,
                "摆动图需要二维数据",
                ha="center",
                va="center",
                fontsize=12,
                color=placeholder_secondary,
            )
            self.preview_info.setText("摆动图不可用")
            return

        _, x_axis, y_axis, x_label, y_label = self._build_plot_axes(data, header_info)

        n_samples, n_traces = data.shape
        max_traces = 80
        step = max(1, int(np.ceil(n_traces / max_traces)))
        trace_indices = np.arange(0, n_traces, step, dtype=int)

        finite_values = np.asarray(data[np.isfinite(data)], dtype=float)
        amp_ref = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
        if amp_ref <= 0:
            amp_ref = 1.0

        spacing = float(np.median(np.diff(x_axis))) if x_axis.size > 1 else 1.0
        spacing = spacing if spacing > 0 else 1.0
        wiggle_scale = spacing * 0.45

        theme = self.theme_manager.get_current_theme()
        line_color = "#f5f5f5" if theme == "dark" else "#111111"
        fill_color = "#8fb7ff" if theme == "dark" else "#4a4a4a"

        for trace_idx in trace_indices:
            trace = np.asarray(data[:, trace_idx], dtype=float)
            trace = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)
            wiggle = x_axis[trace_idx] + (trace / amp_ref) * wiggle_scale
            self.ax.plot(wiggle, y_axis, color=line_color, linewidth=0.8)
            self.ax.fill_betweenx(
                y_axis,
                x_axis[trace_idx],
                wiggle,
                where=(wiggle >= x_axis[trace_idx]).tolist(),
                color=fill_color,
                alpha=0.25,
                interpolate=True,
            )

        self.ax.set_title(f"{title} - 摆动图", fontsize=10)
        self.ax.set_xlabel(x_label, fontsize=8)
        self.ax.set_ylabel(y_label, fontsize=8)

        if x_axis.size > 0:
            self.ax.set_xlim(x_axis[0] - spacing * 0.5, x_axis[-1] + spacing * 0.5)
        if y_axis.size > 0:
            if y_axis[0] <= y_axis[-1]:
                self.ax.set_ylim(y_axis[-1], y_axis[0])
            else:
                self.ax.set_ylim(y_axis[-1], y_axis[0])

        self.ax.grid(True, alpha=0.2)
        self._safe_remove_colorbar()
        self.preview_info.setText(
            f"摆动图 | 道数: {n_traces} | 显示: {len(trace_indices)} | 采样: {n_samples}"
        )

    def _plot_comparison(self):
        """绘制差异图 - 显示处理前后的差异"""
        try:
            target_entry = self._get_selected_view_entry()
            result_data = (
                self.preview_data
                if self.preview_data is not None
                else (target_entry.get("data") if target_entry is not None else None)
            )
            compare_data = self._get_compare_data()
            compare_label = self._get_compare_label()
            result_label = (
                self.preview_title
                if self.preview_data is not None
                else str(target_entry.get("label") if target_entry else "当前结果")
            )
            if compare_data is None or result_data is None:
                placeholder_secondary = "#b7bcc6" if self.theme_manager.get_current_theme() == "dark" else "#888"
                self.ax.text(
                    0.5,
                    0.5,
                    "需要原始数据和处理结果\n才能显示差异图",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color=placeholder_secondary,
                )
                self.ax.set_axis_off()
                self.preview_info.setText("无数据可对比")
                return

            # 检查形状是否匹配
            if compare_data.shape != result_data.shape:
                self.ax.clear()
                placeholder_secondary = "#b7bcc6" if self.theme_manager.get_current_theme() == "dark" else "#888"
                self.ax.text(
                    0.5,
                    0.5,
                    f"数据形状不匹配\n基准: {compare_data.shape}\n目标: {result_data.shape}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=placeholder_secondary,
                )
                self.ax.set_axis_off()
                self.preview_info.setText("形状不匹配")
                return

            self.ax.clear()

            # 计算差异：目标图像 - 基准图像
            diff = result_data - compare_data

            # 绘制差异图
            im = self.ax.imshow(
                diff,
                aspect="auto",
                cmap="seismic",
                interpolation="bilinear",
            )

            self.ax.set_title(f"差异图 ({result_label} - {compare_label})", fontsize=10)
            self.ax.set_xlabel("距离", fontsize=8)
            self.ax.set_ylabel("时间", fontsize=8)

            # 添加颜色条
            self._safe_remove_colorbar()

            try:
                self._cbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
            except Exception as e:
                logger.warning("Workbench comparison colorbar failed: %s", e)

            # 更新预览信息
            self.preview_info.setText(
                f"基准: {compare_label} | 目标: {result_label} | 差异范围: [{diff.min():.3f}, {diff.max():.3f}]"
            )

        except Exception as e:
            self.ax.clear()
            error_color = "#ff8f8f" if self.theme_manager.get_current_theme() == "dark" else "#f44336"
            self.ax.text(
                0.5,
                0.5,
                f"对比图绘制失败: {e}",
                ha="center",
                va="center",
                fontsize=12,
                color=error_color,
            )
            self.preview_info.setText("对比失败")
            logger.exception("Workbench comparison plot failed")

    def _plot_slider_comparison(self):
        """绘制真正可拖动的滑动对比图。"""
        if self.raw_data is None:
            self._plot_data(np.zeros((1, 1), dtype=np.float32), "未加载数据")
            self.preview_info.setText("未加载数据")
            return

        target_entry = self._get_selected_view_entry()
        effective_result = (
            self.preview_data
            if self.preview_data is not None
            else (target_entry.get("data") if target_entry is not None else None)
        )
        if effective_result is None:
            self._plot_data(self.raw_data, "原始数据 (无结果)")
            return

        compare_data = self._get_compare_data()
        if compare_data is None:
            compare_data = self.raw_data

        try:
            compare_data = np.asarray(compare_data, dtype=np.float32)
            current_result = np.asarray(effective_result, dtype=np.float32)
        except Exception as e:
            self._log(f"滑动对比数据转换失败: {e}", "ERROR")
            return

        if compare_data.ndim != 2 or current_result.ndim != 2:
            self.preview_info.setText("滑动对比数据维度错误")
            return
        if compare_data.shape != current_result.shape:
            self.preview_info.setText("滑动对比数据形状不匹配")
            return

        height, width = compare_data.shape
        if height <= 0 or width <= 0:
            self.preview_info.setText("滑动对比数据为空")
            return

        split_idx = int(round(self._slider_compare_ratio * max(width - 1, 1)))
        split_idx = max(0, min(split_idx, width - 1))

        merged = np.array(current_result, copy=True)
        merged[:, : split_idx + 1] = compare_data[:, : split_idx + 1]

        self.ax.clear()
        self.ax.set_axis_on()
        im = self.ax.imshow(
            merged,
            aspect="auto",
            cmap="gray",
            interpolation="bilinear",
            origin="upper",
        )
        self._safe_remove_colorbar()
        try:
            self._cbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
        except Exception as e:
            logger.warning("Workbench slider colorbar creation failed: %s", e)

        left_name = self._get_compare_label()
        right_name = (
            self.preview_title
            if self.preview_data is not None
            else str(target_entry.get("label") if target_entry else "当前结果")
        )

        is_dark = self.theme_manager.get_current_theme() == "dark"
        divider_color = "#d9e6ff" if is_dark else "#ffffff"
        label_text_color = "#f5f5f5" if is_dark else "#ffffff"
        label_bg_color = "#111318" if is_dark else "#000000"

        self.ax.axvline(x=split_idx - 0.5, color=divider_color, linewidth=1.6, alpha=0.85)
        self.ax.text(
            max(1, width * 0.15),
            max(1, height * 0.08),
            left_name,
            color=label_text_color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=label_bg_color, edgecolor="none", alpha=0.58),
        )
        self.ax.text(
            max(1, width * 0.85),
            max(1, height * 0.08),
            right_name,
            color=label_text_color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=label_bg_color, edgecolor="none", alpha=0.58),
        )

        self.ax.set_title(f"滑动对比 ({left_name} | {right_name})", fontsize=10)
        self.ax.set_xlabel("距离", fontsize=8)
        self.ax.set_ylabel("时间", fontsize=8)
        self.preview_info.setText(
            f"左侧: {left_name} | 右侧: {right_name} | 位置: {self._slider_compare_ratio * 100:.1f}%"
        )

    def _update_metrics(self):
        """更新质量指标"""
        data = self._get_effective_result_data()
        if data is None:
            self.focus_ratio_label.setText("聚焦比: --")
            self.hot_pixels_label.setText("热像素: --")
            self.snr_label.setText("SNR: --")
            self.data_range_label.setText("范围: --")
            return

        # 计算聚焦比
        # 聚焦比 = 信号能量 / 总能量
        signal_energy = np.sum(data**2)
        total_energy = data.size * np.mean(data**2)
        focus_ratio = signal_energy / total_energy if total_energy > 0 else 0

        # 计算热像素
        # 热像素定义为超过均值3个标准差的像素
        mean_val = np.mean(data)
        std_val = np.std(data)
        hot_pixels = np.sum(np.abs(data - mean_val) > 3 * std_val)

        # 计算信噪比 (SNR)
        # SNR = 10 * log10(信号功率 / 噪声功率)
        # 这里使用简化计算：SNR = 20 * log10(mean / std)
        if std_val > 0 and np.abs(mean_val) > 1e-12:
            snr = 20 * np.log10(np.abs(mean_val) / std_val)
        else:
            snr = 0

        # 更新显示
        self.focus_ratio_label.setText(f"聚焦比: {focus_ratio:.3f}")
        self.hot_pixels_label.setText(f"热像素: {hot_pixels}")
        self.snr_label.setText(f"SNR: {snr:.1f}dB")
        self.data_range_label.setText(f"范围: [{data.min():.2f}, {data.max():.2f}]")

        # 根据指标质量设置颜色（使用主题 class）
        from core.theme_manager import get_theme_manager
        tm = get_theme_manager()

        # focus_ratio: 越大越好 (>0.5 good, <0.2 bad)
        focus_cls = tm.get_metric_color_class(focus_ratio, (0.5, 0.2))
        self.focus_ratio_label.setProperty("class", focus_cls)
        focus_style = self.focus_ratio_label.style()
        if focus_style is not None:
            focus_style.unpolish(self.focus_ratio_label)
            focus_style.polish(self.focus_ratio_label)

        # hot_pixels: 越小越好 (<100 good, >500 bad) → 反向判断
        if hot_pixels < 100:
            hot_cls = "metricGood"
        elif hot_pixels > 500:
            hot_cls = "metricBad"
        else:
            hot_cls = "metricWarning"
        self.hot_pixels_label.setProperty("class", hot_cls)
        hot_style = self.hot_pixels_label.style()
        if hot_style is not None:
            hot_style.unpolish(self.hot_pixels_label)
            hot_style.polish(self.hot_pixels_label)

        # snr: 越大越好 (>10 good, <5 bad)
        snr_cls = tm.get_metric_color_class(snr, (10.0, 5.0))
        self.snr_label.setProperty("class", snr_cls)
        snr_style = self.snr_label.style()
        if snr_style is not None:
            snr_style.unpolish(self.snr_label)
            snr_style.polish(self.snr_label)
