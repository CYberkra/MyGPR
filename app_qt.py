#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MyGPR (PyQt6 + themed) - 主入口模块

模块化重构版本：
- gui_base.py: 基础工具和函数
- methods_registry.py: 统一方法注册表
- gui_basic_flow.py: 基础流程页面
- gui_auto_tune_page.py: 调参与实验页面
- gui_advanced_settings.py: 显示与对比页面
- gui_quality_log.py: 质量与导出页面
"""

import os
import sys
import time
import json
import csv
import html
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

# In normal desktop use, force QtAgg so Matplotlib embeds in the PyQt window.
# In headless/offscreen CI, calling ``matplotlib.use('QtAgg')`` before a
# QApplication exists raises ``ImportError: headless is currently running`` even
# though ``FigureCanvasQTAgg`` can still be imported for Qt widget tests.  Keep
# the desktop path strict, but let tests/import smoke use the default backend.
if str(os.environ.get("QT_QPA_PLATFORM", "")).lower() not in {"offscreen", "minimal"}:
    matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtCore import Qt, QThread, QTimer, QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QTabWidget,
    QLabel,
    QGroupBox,
    QFrame,
    QStackedLayout,
    QFileDialog,
    QMessageBox,
    QPushButton,
    QToolButton,
    QTextEdit,
    QScrollArea,
    QSizePolicy,
    QProgressBar,
)

from core.app_paths import get_logs_dir, get_output_dir
from core.app_errors import (
    InputDataError,
    error_info_from_exception,
)
from core.log_events import LogEvent, LogEventBuffer, classify_log_event
from core.perf_monitor import PerfMonitor

# 确保本地目录在路径中
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logger = logging.getLogger(__name__)










# 导入基础模块
from ui.gui_base import (
    detect_csv_header,
    _detect_skiprows,
    build_csv_load_error_message,
    build_processing_error_message,
    ProcessingCancelled,
    load_quality_dashboard_thresholds,
    _configure_qt_cjk_font,
    build_version_string,
)
from core.methods_registry import (
    PROCESSING_METHODS,
    get_auto_tune_stage,
    get_public_method_keys,
)
from core.preset_profiles import (
    GUI_PRESETS_V1,
    DEFAULT_STARTUP_PRESET_KEY,
    RECOMMENDED_RUN_PROFILES,
    build_profile_workflow_summary,
    compute_quality_metrics,
)
from core.gpr_io import (
    extract_airborne_csv_payload,
    read_ascans_folder,
)
from core.data_context import recommended_profile_for_header
from core.processing_engine import (
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.runtime_warnings import (
    build_runtime_warning,
    format_runtime_warning_text,
    merge_runtime_warnings,
)
from core.auto_tune import (
    auto_select_method_group,
    auto_tune_method,
    AutoTuneCancelled,
)
from core.auto_tune_comparison import (
    run_auto_tune_comparison,
    to_summary_dict,
)
from core.auto_tune_comparison_export import (
    export_auto_tune_comparison_artifacts as export_auto_tune_comparison_bundle,
)
from core.evidence_export import (
    export_replay_evidence_bundle as export_replay_evidence_zip,
)
from core.uav_georeference_3d import (
    build_airborne_georeference_3d_payload,
    export_airborne_georeference_3d_bundle,
)
from core.shared_data_state import SharedDataState
from PythonModule.kirchhoff_migration import load_cagpr_kir_parameter_file
from qfluentwidgets import FluentIcon

# 导入页面模块
from ui.gui_basic_flow import BasicFlowPage
from ui.autotune_tuning_page import AutoTuneTuningPage
from ui.gui_advanced_settings import AdvancedSettingsPage
from ui.gui_quality_log import QualityLogPage
from ui.georef3d_results_page import Terrain3DResultsPage

# 导入新的工作台页面
from ui.loading_dialog import LoadingProgressDialog
from ui.auto_tune_result_dialog import AutoTuneResultDialog
from ui.report_export_controller import ReportExportController
from ui.processing_lineage_controller import ProcessingLineageController
from ui.bscan_interaction_controller import BscanInteractionController
from ui.autotune_sync_controller import AutoTuneSyncController
from ui.processing_worker_controller import ProcessingWorkerController
from ui.airborne_payload_controller import AirbornePayloadController
from ui.sidecar_controller import SidecarController
from ui.main_window_display_mixin import MainWindowDisplayMixin
from ui.main_window_quality_mixin import MainWindowQualityMixin
from ui.main_window_worker_mixin import MainWindowWorkerMixin
from ui.main_window_data_loading_mixin import MainWindowDataLoadingMixin
from ui.main_window_autotune_start_mixin import MainWindowAutoTuneStartMixin
from ui.main_window_export_mixin import MainWindowExportMixin
from ui.app_branding import HiResNavigationToolbar, MyGPRMark, make_mygpr_brand_pixmap
from core.gpr_format_registry import supported_file_dialog_filter

from ui.worker_threads import (
    AutoTuneComparisonWorker,
    AutoTuneStageWorker,
    AutoTuneWorker,
    ProcessingWorker,
)
from core.app_runtime import (
    configure_logging,
    _load_app_settings_dict,
    _load_last_data_path,
    _save_app_settings_dict,
    _save_last_data_path,
    _sanitize_qss,
)






















class GPRGuiQt(MainWindowDataLoadingMixin, MainWindowAutoTuneStartMixin, MainWindowExportMixin, MainWindowDisplayMixin, MainWindowQualityMixin, MainWindowWorkerMixin, QMainWindow):
    """MyGPR 主窗口"""

    @property
    def data(self):
        return self.shared_data.current_data

    @data.setter
    def data(self, value):
        self.shared_data.current_data = (
            None if value is None else np.array(value, copy=True)
        )

    @property
    def original_data(self):
        return self.shared_data.original_data

    @original_data.setter
    def original_data(self, value):
        self.shared_data.original_data = (
            None if value is None else np.array(value, copy=True)
        )

    @property
    def history(self):
        return self.shared_data.history

    @property
    def data_path(self):
        return self.shared_data.data_path

    @data_path.setter
    def data_path(self, value):
        self.shared_data.data_path = value

    @property
    def header_info(self):
        return self.shared_data.header_info

    @header_info.setter
    def header_info(self, value):
        self.shared_data.header_info = value

    @property
    def trace_metadata(self):
        return self.shared_data.current_trace_metadata

    @trace_metadata.setter
    def trace_metadata(self, value):
        self.shared_data.current_trace_metadata = value

    def __init__(self, version_text: str = ""):
        super().__init__()
        self.version_text = version_text.strip() or "MyGPR"
        self.setWindowTitle(self.version_text)
        try:
            self.setWindowIcon(QIcon(make_mygpr_brand_pixmap(64)))
        except Exception:
            pass
        self.resize(1280, 800)
        self.setMinimumSize(1120, 720)

        self.shared_data = SharedDataState(self)
        self.shared_data.changed.connect(self._on_shared_data_changed)
        self.report_export_controller = ReportExportController(self)
        self.processing_lineage_controller = ProcessingLineageController(self)
        self.bscan_interaction_controller = BscanInteractionController(self)
        self.autotune_sync_controller = AutoTuneSyncController(self)
        self.processing_worker_controller = ProcessingWorkerController(self)
        self.airborne_payload_controller = AirbornePayloadController(self)
        self.sidecar_controller = SidecarController(self)

        # 数据状态
        self.data = None
        self.data_path = None
        self.header_info = None
        self.original_data = None
        self.cbar = None

        # 工作线程
        self._worker_thread = None
        self._worker = None
        self._auto_tune_thread = None
        self._auto_tune_worker = None
        self._auto_tune_stage_thread = None
        self._auto_tune_stage_worker = None
        self._auto_tune_comparison_thread = None
        self._auto_tune_comparison_worker = None
        self._pending_apply_after_auto_tune = False
        self._current_run_context = None
        self._cancel_in_flight = False
        self._last_auto_tune_result = None
        self._last_auto_tune_group_result = None
        self._last_auto_tune_comparison_result = None

        # 缓存和状态
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.timeout.connect(self._do_refresh_plot)
        self._canvas_draw_timer = QTimer(self)
        self._canvas_draw_timer.setSingleShot(True)
        self._canvas_draw_timer.timeout.connect(self._flush_pending_canvas_draw)
        self._pending_canvas_draw = False
        self._last_canvas_draw_flush_ts = 0.0
        self._canvas_draw_min_interval_s = 1.0 / 60.0
        self._ds_cache = {}
        self._view_cache = {}
        self._vmin_vmax_cache = {}
        self._perf_monitor = PerfMonitor()
        self._auto_tune_progress_timer = QTimer(self)
        self._auto_tune_progress_timer.setSingleShot(True)
        self._auto_tune_progress_timer.timeout.connect(self._flush_auto_tune_progress_update)
        self._pending_auto_tune_progress = None
        self._last_auto_tune_progress_flush_ts = 0.0
        self._auto_tune_progress_min_interval_s = 0.12
        self._last_view_cache_hit = False
        self._last_vmin_vmax_cache_hit = False
        self.compare_snapshots = []
        self._transient_compare_snapshots = []
        self._lineage_compare_source_index = None
        self._compare_syncing = False
        self._last_compare_combo_labels = ()
        self._data_revision = 0
        self._last_plot_signature = None
        self._plot_debug_metrics = os.getenv(
            "MYGPR_PLOT_DEBUG", os.getenv("GPR_GUI_PLOT_DEBUG", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._plot_skip_count = 0
        self._plot_draw_count = 0
        self._last_plot_ms = 0.0
        self._last_prepare_ms = None
        self._last_compare_ms = None
        self._last_n_panels = 1
        self._last_compare_combo_labels = ()

        # 参数覆盖
        self._method_param_overrides = {}
        self._selected_preset_key = None
        self._last_stolt_adaptive_stats = None
        self._last_stolt_adaptive_reason = ""
        self._last_quality_metrics = None
        self._last_run_summary = None
        self._last_no_prior_qc_policy = None
        self._no_prior_guard_events = []
        self._runtime_warnings = []
        self._runtime_log_events = LogEventBuffer(max_events=2000)
        self._pending_plain_log_lines = []
        self._pending_runtime_log_html = []
        self._pending_quality_log_lines = []
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setSingleShot(True)
        self._log_flush_timer.timeout.connect(self._flush_pending_log_ui)
        self._last_structured_errors = []
        self._sidecar_files = {"rtk": None, "imu": None, "altimeter": None}
        self._trace_timestamps_s = None
        self._quality_thresholds = load_quality_dashboard_thresholds()
        self._ui_busy = False
        self._display_data_override = None
        self._display_header_info_override = None
        self._display_trace_metadata_override = None
        self._selected_trace_index = None
        self._manual_roi_values = None
        self._manual_roi_pick_enabled = False
        self._drag_roi_preview_patch = None
        self._main_view_limits = None
        self._main_press_state = None
        self._main_drag_threshold_px = 8
        self._main_motion_draw_interval_s = 1.0 / 60.0
        self._roi_preview_draw_interval_s = 1.0 / 60.0
        # Slider compare dragging should feel immediate.  The initial v0.8.12
        # path throttled to 30 Hz but still rebuilt the whole Matplotlib figure,
        # which was visibly choppy on larger B-scans.  v0.8.13 uses a
        # lightweight artist update during drag and only falls back to full
        # refresh when the cache is unavailable.
        self._slider_compare_draw_interval_s = 1.0 / 120.0
        self._main_coord_update_interval_s = 1.0 / 40.0
        self._last_main_motion_draw_ts = 0.0
        self._last_roi_preview_draw_ts = 0.0
        self._last_slider_compare_draw_ts = 0.0
        self._last_main_coord_update_ts = 0.0
        self._last_display_trace_axis = np.array([], dtype=np.float32)
        self._last_display_trace_indices = np.array([], dtype=np.int32)
        self._last_display_time_axis = np.array([], dtype=np.float32)
        self._last_display_data = None
        self._last_plot_extent = None
        self._main_plot_axes = []
        self._main_slider_compare_ratio = 0.5
        self._slider_compare_render_cache = {}
        self._selected_trace_marker_artists = []
        self._hover_crosshair_artists = []
        self._hover_crosshair_axes = None
        self._hover_crosshair_last_key = None
        self._hover_crosshair_last_draw_ts = 0.0
        self._hover_crosshair_draw_interval_s = 1.0 / 45.0
        self._selected_trace_payload_timer = QTimer(self)
        self._selected_trace_payload_timer.setSingleShot(True)
        self._selected_trace_payload_timer.timeout.connect(self._refresh_selected_trace_payload)
        self._slider_hit_tolerance_px = 10
        self._last_coord_label_key = None
        self._display_override_revision = 0
        self._lineage_step_buttons = []
        self._lineage_view_index = None
        self._lineage_silent_update = False

        # 布局/容器状态
        self._main_content_widget = None
        self._content_stack = None
        self._main_splitter = None
        self._left_scroll = None
        self._left_panel = None
        self._right_panel = None
        self._progress_panel = None
        self._progress_bar = None
        self._progress_stage_label = None
        self._main_toolbar = None
        self._plot_coord_label = None
        self._runtime_panel_bar = None
        self._runtime_panel_container = None
        self._runtime_panel_stack = None
        self._runtime_panel_buttons = {}
        self._active_runtime_panel = None

        self._setup_ui()
        self._apply_style()
        self._sync_history_action_state()

    def closeEvent(self, event):
        """关闭窗口时释放嵌入式 Matplotlib 资源，避免批量 GUI 测试累积占用。"""
        page_quality = getattr(self, "page_quality", None)
        if page_quality is not None and hasattr(page_quality, "release_plot_resources"):
            page_quality.release_plot_resources()
        page_terrain = getattr(self, "page_terrain3d", None)
        if page_terrain is not None and hasattr(page_terrain, "release_plot_resources"):
            page_terrain.release_plot_resources()
        fig = getattr(self, "fig", None)
        canvas = getattr(self, "canvas", None)
        try:
            if fig is not None:
                fig.clear()
        except Exception:
            pass
        try:
            if canvas is not None:
                canvas.close()
                canvas.deleteLater()
        except Exception:
            pass
        super().closeEvent(event)

    def _setup_ui(self):
        """设置UI"""
        central = QWidget()
        central.setObjectName("appCentralRoot")
        self.setCentralWidget(central)

        self._content_stack = QStackedLayout(central)
        self._content_stack.setContentsMargins(0, 0, 0, 0)
        self._content_stack.setStackingMode(QStackedLayout.StackingMode.StackOne)

        self._main_content_widget = QWidget()
        root_layout = QHBoxLayout(self._main_content_widget)
        # Keep the chrome tight: the B-scan card should receive most of the
        # available horizontal and vertical space.
        root_layout.setContentsMargins(8, 6, 8, 6)
        root_layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("mainSplitter")
        self._main_splitter = splitter
        root_layout.addWidget(splitter)

        # 右侧面板（绘图区）
        right_panel = QWidget()
        right_panel.setObjectName("mainWorkspacePanel")
        self._right_panel = right_panel
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(3, 3, 3, 3)
        right_layout.setSpacing(7)

        # 左侧折叠导航抽屉：竖向工作区栏常驻，参数/导航面板按需弹出。
        # 主工作区默认保留给 B-scan / 质量报告 / 空间成果，导航位置固定在最左侧。
        side_shell = QWidget()
        side_shell.setObjectName("SideDrawerShell")
        self._side_shell = side_shell
        side_shell_layout = QHBoxLayout(side_shell)
        side_shell_layout.setContentsMargins(0, 0, 0, 0)
        side_shell_layout.setSpacing(6)

        self._side_nav_rail = QFrame()
        self._side_nav_rail.setObjectName("SideNavRail")
        self._side_nav_rail.setMinimumWidth(54)
        self._side_nav_rail.setMaximumWidth(56)
        self._side_nav_layout = QVBoxLayout(self._side_nav_rail)
        self._side_nav_layout.setContentsMargins(3, 6, 3, 6)
        self._side_nav_layout.setSpacing(4)

        self._side_drawer_toggle = QToolButton()
        self._side_drawer_toggle.setObjectName("SideDrawerToggle")
        self._side_drawer_toggle.setText("☰")
        self._side_drawer_toggle.setToolTip("展开/收起左侧工具抽屉")
        self._side_drawer_toggle.clicked.connect(self._toggle_side_drawer)
        self._side_nav_layout.addWidget(self._side_drawer_toggle)

        self._side_nav_buttons = []
        self._side_nav_layout.addStretch(1)

        left_shell = QWidget()
        left_shell.setObjectName("controlPanelShell")
        self._left_shell = left_shell
        left_shell_layout = QVBoxLayout(left_shell)
        left_shell_layout.setContentsMargins(0, 0, 0, 0)
        left_shell_layout.setSpacing(8)

        left_panel = QWidget()
        left_panel.setObjectName("controlPanel")
        self._left_panel = left_panel
        # The processing drawer is the default startup surface.  Keep it compact
        # but wide enough for the two-column action buttons and parameter editor.
        # V0.8.33 removes the over-corrected blank splitter gutter introduced
        # while fixing scrollbars: the shell width now matches the rail + drawer
        # width instead of reserving an unused spacer between drawer and B-scan.
        self._drawer_content_width = 348
        self._rail_width = 56
        self._rail_drawer_spacing = 6
        left_shell.setMinimumWidth(self._drawer_content_width)
        left_shell.setMaximumWidth(self._drawer_content_width)
        left_panel.setMinimumWidth(self._drawer_content_width)
        left_panel.setMaximumWidth(self._drawer_content_width)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(7)

        # The drawer pages already own their vertical scrolling. Wrapping the
        # whole drawer in another QScrollArea created a second scrollbar on the
        # drawer edge, which visually covered the control cards in compact
        # windows.  Keep the drawer content direct and let each page reserve its
        # own scrollbar gutter.
        self._left_scroll = None
        left_shell_layout.addWidget(left_panel, 1)

        # Drawer content opens to the right of the rail; the vertical workspace rail
        # remains fixed at the far left edge, so navigation no longer shifts when
        # the drawer expands/collapses.
        side_shell_layout.addWidget(self._side_nav_rail, 0)
        side_shell_layout.addWidget(left_shell, 1)

        # Default startup state: processing tools are open.  Users should see
        # the daily-processing controls immediately, without first expanding a
        # collapsed drawer.
        self._side_drawer_expanded = True
        self._set_side_drawer_expanded(True, resize=False)

        splitter.addWidget(side_shell)
        splitter.addWidget(right_panel)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 8)
        splitter.setSizes([620, 1080])

        # 创建多页控制面板。TabBar 隐藏，改由右侧竖向图标栏负责导航；
        # 面板本体作为可折叠 Inspector / 工具抽屉使用。
        self.control_tabs = QTabWidget()
        self.control_tabs.setDocumentMode(True)
        self.control_tabs.setUsesScrollButtons(False)
        self.control_tabs.setMovable(False)
        self.control_tabs.tabBar().hide()
        left_layout.addWidget(self.control_tabs)

        self._content_stack.addWidget(self._main_content_widget)

        # ===== 原有页面（保留作为日常处理界面）=====
        # 页面1: 日常处理
        self.page_basic = BasicFlowPage(self)
        idx_basic = self.control_tabs.addTab(
            self.page_basic, FluentIcon.HOME.icon(), "日常处理"
        )
        self.control_tabs.setTabToolTip(idx_basic, "日常连续处理操作")

        # 页面2: 调参与实验
        self.page_auto_tune = AutoTuneTuningPage(self)
        idx_auto_tune = self.control_tabs.addTab(
            self.page_auto_tune, FluentIcon.SETTING.icon(), "自动选参"
        )
        self.control_tabs.setTabToolTip(idx_auto_tune, "自动选参、候选评估与推荐报告")

        # 页面3: 显示与对比
        self.page_advanced = AdvancedSettingsPage(self)
        idx_advanced = self.control_tabs.addTab(
            self.page_advanced, FluentIcon.VIEW.icon(), "显示与对比"
        )
        self.control_tabs.setTabToolTip(
            idx_advanced, "主图显示、双图对比、裁剪与预览设置"
        )

        # 页面4/5: 质量与空间成果现在是“主工作区”，右侧标签只作为轻量导航。
        # 这样避免把复杂结果页挤在侧栏里，符合工程软件“主视图优先”的布局。
        self.page_quality = QualityLogPage(self)
        self.page_quality.set_trace_selected_callback(
            self._on_trajectory_trace_selected
        )
        self._page_quality_nav = self._create_workspace_nav_placeholder(
            title="质量与报告工作区",
            body="质量概览、航迹质控、报告导出和处理记录将在左侧主工作区显示。",
            action="点击本标签后，主视图切换到质量与报告。",
        )
        idx_quality = self.control_tabs.addTab(
            self._page_quality_nav, FluentIcon.PIE_SINGLE.icon(), "质量与导出"
        )
        self.control_tabs.setTabToolTip(idx_quality, "质量概览、处理记录与报告导出入口")

        self.page_terrain3d = Terrain3DResultsPage(self)
        self._page_terrain_nav = self._create_workspace_nav_placeholder(
            title="空间成果工作区",
            body="地形剖面、三维地理参考、C-scan 和基覆界面解释将在左侧主工作区显示。",
            action="点击本标签后，主视图切换到地形/三维成果。",
        )
        idx_terrain3d = self.control_tabs.addTab(
            self._page_terrain_nav, FluentIcon.VIEW.icon(), "地形/三维成果"
        )
        self.control_tabs.setTabToolTip(
            idx_terrain3d, "UAV-GPR 航迹、三维剖面带与地理参考成果"
        )

        self._build_side_workspace_rail()
        self._sync_side_nav_buttons()

        # 旧工作台已彻底退役；主流程统一使用五个主标签页。

        # 默认显示主内容区（日常处理界面）
        self._content_stack.setCurrentWidget(self._main_content_widget)
        self.control_tabs.setCurrentWidget(self.page_basic)
        self.control_tabs.currentChanged.connect(self._on_control_tab_changed_for_workspace)
        self._reorder_basic_groups_for_flow()

        # 右侧面板 - 状态栏
        status_bar = QWidget()
        status_bar.setObjectName("topInfoBar")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)

        self.status_label = QLabel("未加载文件")
        self.status_label.setProperty("class", "topInfoText")
        self.status_label.setToolTip("当前应用状态和数据文件信息")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)

        # 顶部内联进度条：与状态文字同一行，避免重复显示两行说明。
        self._progress_panel = self._create_progress_panel()
        status_layout.addWidget(self._progress_panel)

        self.version_label = QLabel(self.version_text)
        self.version_label.setProperty("class", "topInfoMeta")
        status_layout.addWidget(self.version_label)
        right_layout.addWidget(status_bar)

        # 主工作区绘图卡片：B-scan 是核心视觉区域，统一放入产品化卡片中。
        self.main_plot_card = QFrame()
        self.main_plot_card.setObjectName("mainPlotCard")
        main_plot_layout = QVBoxLayout(self.main_plot_card)
        main_plot_layout.setContentsMargins(12, 8, 12, 7)
        main_plot_layout.setSpacing(4)

        plot_header = self._create_plot_card_header()
        main_plot_layout.addWidget(plot_header)

        self.fig = Figure(figsize=(11.0, 6.4), dpi=100)
        self.fig.subplots_adjust(left=0.055, right=0.994, top=0.920, bottom=0.108, wspace=0.18)
        self._main_ax = self.fig.add_subplot(111)
        self._main_ax.set_title("B-scan")
        self._main_ax.set_xlabel("距离（道索引）")
        self._main_ax.set_ylabel("时间（采样索引）")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._main_toolbar = HiResNavigationToolbar(self.canvas, self)
        self._main_toolbar.setObjectName("mainPlotToolbar")
        for action in self._main_toolbar.actions():
            action_haystack = " ".join(
                part
                for part in [
                    action.text() or "",
                    action.toolTip() or "",
                    action.statusTip() or "",
                    action.iconText() or "",
                ]
                if part
            ).lower()
            if "home" in action_haystack or "reset original view" in action_haystack:
                action.triggered.connect(
                    lambda checked=False: QTimer.singleShot(
                        0, self._reset_main_plot_view_to_default
                    )
                )
            else:
                action.triggered.connect(
                    lambda checked=False: QTimer.singleShot(
                        0, self._capture_main_view_limits_from_axes
                    )
                )
        self.canvas.mpl_connect("button_press_event", self._on_main_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_main_canvas_motion)
        self.canvas.mpl_connect("button_release_event", self._on_main_canvas_release)
        self.canvas.mpl_connect("scroll_event", self._on_main_canvas_scroll)
        self.canvas.mpl_connect("key_press_event", self._on_main_canvas_key_press)
        self.canvas.mpl_connect("figure_leave_event", self._on_main_canvas_leave)
        self._last_n_panels = 1

        plot_toolbar_row = QWidget()
        self._plot_toolbar_row = plot_toolbar_row
        plot_toolbar_row.setObjectName("PlotToolbarRow")
        plot_toolbar_layout = QHBoxLayout(plot_toolbar_row)
        plot_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        plot_toolbar_layout.setSpacing(4)
        plot_toolbar_layout.addWidget(self._main_toolbar)
        plot_toolbar_layout.addStretch(1)
        # The top row is reserved for Matplotlib navigation only.  Display mode,
        # colormap and stretch controls live in the right-side Display/Compare page,
        # so the main B-scan card does not duplicate those chips.
        self._plot_display_mode_chip = None
        self._plot_colormap_chip = None
        self._plot_range_chip = None
        main_plot_layout.addWidget(plot_toolbar_row)

        # 空状态卡片 / 绘图区堆叠
        self.plot_stack_host = QWidget()
        self.plot_stack_host.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_stack_layout = QStackedLayout(self.plot_stack_host)
        plot_stack_layout.setContentsMargins(0, 0, 0, 0)

        self.empty_state_card = self._create_empty_state_card()
        self.empty_state_card.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_stack_layout.addWidget(self.empty_state_card)
        plot_stack_layout.addWidget(self.canvas)
        main_plot_layout.addWidget(self.plot_stack_host, 1)

        self._plot_stepper_bar = self._create_processing_stepper_bar()
        main_plot_layout.addWidget(self._plot_stepper_bar)

        self._plot_bottom_status_bar = QFrame()
        self._plot_bottom_status_bar.setObjectName("PlotBottomStatusBar")
        self._plot_bottom_status_bar.setMinimumHeight(28)
        self._plot_bottom_status_bar.setMaximumHeight(34)
        bottom_status_layout = QHBoxLayout(self._plot_bottom_status_bar)
        bottom_status_layout.setContentsMargins(5, 2, 5, 2)
        bottom_status_layout.setSpacing(4)
        self._plot_lineage_label = QLabel("链路：Raw")
        self._plot_lineage_label.setObjectName("PlotInfoChip")
        self._plot_lineage_label.setMaximumWidth(560)
        self._plot_lineage_label.setToolTip("当前显示数据的处理链路")
        self._plot_coord_label = QLabel("坐标 --")
        self._plot_coord_label.setObjectName("PlotInfoChip")
        self._plot_coord_label.setMinimumWidth(128)
        self._plot_coord_label.setToolTip("鼠标悬停时显示道号、采样点和振幅；完整坐标不再常驻占位。")
        self._interaction_mode_chip = QLabel("查看")
        self._interaction_mode_chip.setObjectName("PlotModeChip")
        self._interaction_mode_chip.setMaximumWidth(78)
        self._interaction_mode_chip.setProperty("tone", "neutral")
        self._interaction_mode_chip.setToolTip("当前 B-scan 鼠标交互模式")
        bottom_status_layout.addWidget(self._plot_lineage_label, 2)
        bottom_status_layout.addWidget(self._plot_coord_label, 1)
        bottom_status_layout.addWidget(self._interaction_mode_chip)
        main_plot_layout.addWidget(self._plot_bottom_status_bar)

        # 主工作区栈：B-scan、质量与报告、空间成果三类工作区在同一主视图位置切换。
        # 右侧标签页只承担导航/参数作用，避免结果页被压缩在窄侧栏中。
        self._workspace_host = QFrame()
        self._workspace_host.setObjectName("WorkspaceHost")
        self._workspace_layout = QStackedLayout(self._workspace_host)
        self._workspace_layout.setContentsMargins(0, 0, 0, 0)
        self._workspace_layout.addWidget(self.main_plot_card)
        self._workspace_layout.addWidget(self.page_quality)
        self._workspace_layout.addWidget(self.page_terrain3d)
        self._workspace_layout.setCurrentWidget(self.main_plot_card)
        right_layout.addWidget(self._workspace_host, 1)

        # 运行信息抽屉：默认收起，避免长期压缩主绘图区。
        self.global_log_box = self._create_global_log_box()
        self._runtime_panel_bar, self._runtime_panel_container = (
            self._create_runtime_panel_drawer()
        )
        right_layout.addWidget(self._runtime_panel_bar)
        right_layout.addWidget(self._runtime_panel_container)

        self._sync_runtime_panels_visibility()

        # 连接信号
        self._connect_signals()

        # 初始化
        self._restore_view_style_from_settings()
        self._apply_startup_preset_defaults()
        self._reset_auto_tune_state()
        self._update_manual_roi_status()
        self._update_interaction_mode_status()
        self._refresh_observability_panel()
        self._sync_runtime_panels_visibility()
        self._update_empty_state_and_brief()
        self._update_processing_lineage_display()
        self._log(f"版本: {self.version_text}")
        self._log("欢迎使用。请导入数据开始处理。")

        # Final startup layout pass: open the Processing drawer by default and
        # apply the processing splitter profile after all child widgets exist.
        self._set_side_drawer_expanded(True, resize=False)
        self.control_tabs.setCurrentWidget(self.page_basic)
        self._workspace_layout.setCurrentWidget(self.main_plot_card)
        self._refresh_processing_workspace_status()
        QTimer.singleShot(0, lambda: self._set_workspace_splitter_profile("processing"))

    def _create_workspace_nav_placeholder(self, title: str, body: str, action: str) -> QWidget:
        """Create a compact drawer placeholder for full-width workspaces."""
        card = QFrame()
        card.setObjectName("WorkspaceNavPlaceholder")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setObjectName("WorkspaceNavTitle")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        body_label = QLabel(body)
        body_label.setObjectName("WorkspaceNavBody")
        body_label.setWordWrap(True)
        layout.addWidget(body_label)

        action_label = QLabel("主视图已切换。右侧抽屉仅作为导航/高级信息入口。")
        action_label.setObjectName("WorkspaceNavAction")
        action_label.setWordWrap(True)
        layout.addWidget(action_label)
        layout.addStretch(1)
        return card

    def _build_side_workspace_rail(self) -> None:
        """Build the vertical workspace rail shown at the far left."""
        layout = getattr(self, "_side_nav_layout", None)
        if layout is None or getattr(self, "control_tabs", None) is None:
            return
        # Remove existing dynamic buttons while keeping the top drawer toggle and trailing stretch.
        for btn in list(getattr(self, "_side_nav_buttons", [])):
            try:
                layout.removeWidget(btn)
                btn.deleteLater()
            except Exception:
                pass
        self._side_nav_buttons = []

        items = [
            ("处理", "日常处理", 0),
            ("选参", "自动选参", 1),
            ("显示", "显示与对比", 2),
            ("质量", "质量与导出", 3),
            ("空间", "地形/三维成果", 4),
        ]
        insert_at = max(1, layout.count() - 1)
        for text, tip, index in items:
            btn = QToolButton()
            btn.setObjectName("SideWorkspaceButton")
            btn.setText(text)
            btn.setToolTip(tip)
            btn.setCheckable(True)
            btn.setFixedSize(50, 40)
            btn.clicked.connect(lambda checked=False, i=index: self._switch_side_workspace(i))
            layout.insertWidget(insert_at, btn)
            insert_at += 1
            self._side_nav_buttons.append(btn)

    def _switch_side_workspace(self, index: int) -> None:
        """Switch workspace from the vertical rail and expand/collapse as appropriate."""
        if getattr(self, "control_tabs", None) is None:
            return
        if index < 0 or index >= self.control_tabs.count():
            return
        self.control_tabs.setCurrentIndex(index)
        current = self.control_tabs.widget(index)
        if current in (getattr(self, "_page_quality_nav", None), getattr(self, "_page_terrain_nav", None)):
            self._set_side_drawer_expanded(False)
        else:
            self._set_side_drawer_expanded(True)
        self._sync_side_nav_buttons()

    def _toggle_side_drawer(self) -> None:
        """Toggle the right inspector/tool drawer."""
        self._set_side_drawer_expanded(not bool(getattr(self, "_side_drawer_expanded", False)))

    def _set_side_drawer_expanded(self, expanded: bool, resize: bool = True) -> None:
        """Show or hide the side drawer content while keeping the vertical rail visible."""
        self._side_drawer_expanded = bool(expanded)
        shell = getattr(self, "_left_shell", None)
        if shell is not None:
            shell.setVisible(self._side_drawer_expanded)
        toggle = getattr(self, "_side_drawer_toggle", None)
        if toggle is not None:
            toggle.setText("×" if self._side_drawer_expanded else "☰")
            toggle.setToolTip("收起左侧工具抽屉" if self._side_drawer_expanded else "展开左侧工具抽屉")
        if resize:
            current = self.control_tabs.currentWidget() if getattr(self, "control_tabs", None) is not None else None
            profile = "results" if current in (getattr(self, "_page_quality_nav", None), getattr(self, "_page_terrain_nav", None)) else "processing"
            self._set_workspace_splitter_profile(profile)

    def _sync_side_nav_buttons(self) -> None:
        """Reflect the current tab in the vertical rail buttons."""
        buttons = getattr(self, "_side_nav_buttons", [])
        current_index = self.control_tabs.currentIndex() if getattr(self, "control_tabs", None) is not None else -1
        for idx, btn in enumerate(buttons):
            try:
                btn.setChecked(idx == current_index)
            except Exception:
                pass

    def _on_control_tab_changed_for_workspace(self, _index: int) -> None:
        """Switch the central workspace according to the selected navigation tab."""
        self._sync_workspace_shell_for_current_tab()
        self._sync_side_nav_buttons()

    def _sync_workspace_shell_for_current_tab(self) -> None:
        """Keep the central workspace and splitter ratio aligned with the active tab."""
        if not hasattr(self, "_workspace_layout") or self._workspace_layout is None:
            return
        current = self.control_tabs.currentWidget() if getattr(self, "control_tabs", None) is not None else None
        if current is getattr(self, "_page_quality_nav", None):
            self._workspace_layout.setCurrentWidget(self.page_quality)
            if getattr(self, "status_label", None) is not None:
                self.status_label.setText("质量与报告工作区")
            self._set_side_drawer_expanded(False, resize=False)
            self._set_workspace_splitter_profile("results")
        elif current is getattr(self, "_page_terrain_nav", None):
            self._workspace_layout.setCurrentWidget(self.page_terrain3d)
            if getattr(self, "status_label", None) is not None:
                self.status_label.setText("地形/三维成果工作区")
            self._set_side_drawer_expanded(False, resize=False)
            self._set_workspace_splitter_profile("results")
        else:
            self._workspace_layout.setCurrentWidget(self.main_plot_card)
            self._refresh_processing_workspace_status()
            self._set_workspace_splitter_profile("processing")

    def _refresh_processing_workspace_status(self) -> None:
        """Restore the top status text when the central view is the B-scan workspace.

        The quality and spatial-result workspaces write their own title into the
        top status row.  When users return to processing, keep that row tied to
        the loaded file/data state instead of leaving a stale workspace title.
        """
        label = getattr(self, "status_label", None)
        if label is None:
            return
        try:
            if getattr(self, "data", None) is not None:
                if getattr(self, "header_info", None):
                    label.setText(self._build_status_text())
                else:
                    name = os.path.basename(getattr(self, "data_path", "") or "data")
                    shape = getattr(getattr(self, "data", None), "shape", None)
                    label.setText(f"{name} | shape={shape}" if shape else name)
            else:
                label.setText("未加载文件")
        except Exception:
            label.setText("日常处理工作区")

    def _set_workspace_splitter_profile(self, profile: str) -> None:
        """Resize main workspace and side drawer for processing vs result browsing."""
        splitter = getattr(self, "_main_splitter", None)
        if splitter is None:
            return
        total = max(1, splitter.size().width())
        rail_width = int(getattr(self, "_rail_width", 56))
        spacing = int(getattr(self, "_rail_drawer_spacing", 6))
        drawer_width = int(getattr(self, "_drawer_content_width", 348))
        expanded_width = rail_width + spacing + drawer_width
        if not bool(getattr(self, "_side_drawer_expanded", False)):
            side = rail_width
        elif profile == "results":
            # Result workspaces normally collapse the drawer, but if it is opened
            # manually keep it as a real tool drawer instead of creating a large
            # no-op gutter.
            side = max(rail_width + spacing, min(expanded_width, max(rail_width, total - 620)))
        else:
            # Match the side shell to its visible content: fixed rail + spacing +
            # drawer.  Previous versions requested ~446--462 px while the drawer
            # content was only ~348 px wide, leaving a wide blank band between the
            # controls and the B-scan canvas.
            min_main = 560
            side = min(expanded_width, max(rail_width, total - min_main))
        side_shell = getattr(self, "_side_shell", None)
        if side_shell is not None:
            side_shell.setMinimumWidth(side)
            side_shell.setMaximumWidth(side)
        left_shell = getattr(self, "_left_shell", None)
        if left_shell is not None:
            if side <= rail_width + spacing:
                left_shell.setMinimumWidth(0)
                left_shell.setMaximumWidth(0)
            else:
                usable_drawer = max(0, side - rail_width - spacing)
                left_shell.setMinimumWidth(usable_drawer)
                left_shell.setMaximumWidth(usable_drawer)
        main = max(1, total - side)
        splitter.setSizes([side, main])

    def _create_plot_card_header(self):
        """Create the productized B-scan workspace header."""
        header = QFrame()
        header.setObjectName("PlotCardHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(0)
        self._plot_title_label = QLabel("B-scan / 等待导入")
        self._plot_title_label.setObjectName("PlotTitle")
        # Kept as an attribute for compatibility, but intentionally not shown:
        # the figure title now carries the current processing-stage wording.
        self._plot_meta_label = QLabel("")
        self._plot_meta_label.setObjectName("PlotSubtitle")
        self._plot_meta_label.setWordWrap(True)
        self._plot_meta_label.setVisible(False)
        title_box.addWidget(self._plot_title_label)
        layout.addLayout(title_box, 1)

        self._plot_data_chip = QLabel("未载入")
        self._plot_data_chip.setObjectName("PlotStatusChip")
        self._plot_stage_chip = QLabel("原始数据")
        self._plot_stage_chip.setObjectName("PlotStatusChip")
        self._plot_shape_chip = QLabel("尺寸: --")
        self._plot_shape_chip.setObjectName("PlotStatusChip")
        layout.addWidget(self._plot_data_chip)
        layout.addWidget(self._plot_stage_chip)
        layout.addWidget(self._plot_shape_chip)
        return header

    def _set_plot_chip_tone(self, label, text, tone="neutral"):
        if label is None:
            return
        changed = label.text() != text or label.property("tone") != tone
        if not changed:
            return
        label.setText(text)
        try:
            from ui.theme import set_dynamic_property, repolish

            if set_dynamic_property(label, "tone", tone):
                repolish(label)
            else:
                label.update()
        except Exception:
            label.setProperty("tone", tone)
            label.style().unpolish(label)
            label.style().polish(label)
            label.update()

    def _update_main_workspace_summary(self):
        """Refresh the B-scan workspace title, metadata and status chips."""
        if not hasattr(self, "_plot_title_label"):
            return
        has_data = self.data is not None
        if not has_data:
            self._plot_title_label.setText("B-scan / 等待导入")
            self._plot_meta_label.setText("")
            self._plot_meta_label.setVisible(False)
            self._set_plot_chip_tone(self._plot_data_chip, "未载入", "neutral")
            self._set_plot_chip_tone(self._plot_stage_chip, "等待导入", "neutral")
            self._set_plot_chip_tone(self._plot_shape_chip, "尺寸: --", "neutral")
            if getattr(self, "runtime_summary_chip", None) is not None:
                self.runtime_summary_chip.setText("状态：等待数据导入")
            return

        file_name = os.path.basename(self.data_path) if self.data_path else "当前数据"
        stage = getattr(self.shared_data, "current_label", None) or "原始数据"
        try:
            shape_text = " × ".join(str(int(v)) for v in self.data.shape[:2])
        except Exception:
            shape_text = "--"
        self._plot_title_label.setText(f"B-scan / {stage}")
        self._plot_meta_label.setText(file_name)
        self._plot_meta_label.setToolTip(file_name)
        self._plot_meta_label.setVisible(False)
        self._set_plot_chip_tone(self._plot_data_chip, "已载入", "good")
        self._set_plot_chip_tone(self._plot_stage_chip, str(stage), "neutral")
        self._set_plot_chip_tone(self._plot_shape_chip, f"尺寸: {shape_text}", "neutral")
        if getattr(self, "runtime_summary_chip", None) is not None:
            self._set_runtime_summary(f"状态：{stage} · {shape_text}", "good")


    def _polish_main_figure(self):
        """Apply theme-aware visual polish to the Matplotlib B-scan canvas."""
        try:
            from core.theme_manager import get_theme_manager
            from ui.theme import get_effective_theme_key

            theme_key = get_effective_theme_key(
                get_theme_manager().get_current_theme(), widget=self
            )
        except Exception:
            theme_key = "light"

        if theme_key == "dark":
            fig_bg = "#151A21"
            ax_bg = "#111820"
            text = "#EAF0F8"
            muted = "#A6B1C2"
            grid = "#334155"
            spine = "#3B4654"
        else:
            fig_bg = "#FBFDFF"
            ax_bg = "#FFFFFF"
            text = "#172033"
            muted = "#52627A"
            grid = "#D6DEE9"
            spine = "#DDE5EF"

        try:
            self.fig.set_facecolor(fig_bg)
            self.fig.subplots_adjust(
                left=0.055, right=0.994, top=0.920, bottom=0.108, wspace=0.18
            )
            for ax in self.fig.axes:
                ax.set_facecolor(ax_bg)
                ax.title.set_color(text)
                ax.title.set_fontsize(12)
                ax.title.set_fontweight("semibold")
                ax.title.set_pad(7)
                ax.xaxis.label.set_color(muted)
                ax.yaxis.label.set_color(muted)
                ax.tick_params(colors=muted, labelsize=9)
                for side in ("top", "right", "left", "bottom"):
                    if side in ax.spines:
                        ax.spines[side].set_color(spine)
                        ax.spines[side].set_linewidth(0.8)
                try:
                    ax.grid(False)
                except Exception:
                    pass
        except Exception:
            return

    def _create_empty_state_card(self):
        """创建更克制的产品化空状态卡片。"""
        card = QFrame()
        card.setObjectName("emptyStateCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(36, 28, 36, 28)
        layout.setSpacing(12)

        empty_icon = MyGPRMark()

        empty_badge = QLabel("GPR / UAV-GPR")
        empty_badge.setObjectName("EmptyBadge")
        empty_badge.setProperty("class", "emptyBadge")
        empty_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        empty_title = QLabel("导入一条 GPR 测线开始处理")
        empty_title.setObjectName("EmptyTitle")
        empty_title.setProperty("class", "emptyTitle")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        empty_tip = QLabel("B-scan、坐标读数、处理链路和报告导出将在这里集中显示。")
        empty_tip.setObjectName("EmptySubtitle")
        empty_tip.setProperty("class", "emptySubtitle")
        empty_tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_tip.setWordWrap(True)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.btn_empty_import_csv = QPushButton("导入 CSV")
        self.btn_empty_import_csv.setObjectName("PrimaryButton")
        self.btn_empty_import_csv.setMinimumWidth(110)
        self.btn_empty_import_csv.clicked.connect(self.import_csv_file)
        self.btn_empty_import_folder = QPushButton("导入 A-scan 文件夹")
        self.btn_empty_import_folder.setObjectName("SecondaryButton")
        self.btn_empty_import_folder.setMinimumWidth(140)
        self.btn_empty_import_folder.clicked.connect(self.import_ascans_folder)
        action_row.addStretch(1)
        action_row.addWidget(self.btn_empty_import_csv)
        action_row.addWidget(self.btn_empty_import_folder)
        action_row.addStretch(1)

        empty_steps = QLabel("Raw → 校正 → 抑制 → 增强 → 导出")
        empty_steps.setObjectName("EmptySteps")
        empty_steps.setProperty("class", "emptySteps")
        empty_steps.setAlignment(Qt.AlignmentFlag.AlignCenter)

        empty_hint = QLabel("默认查看模式保持安静；滚轮缩放，左键选道，中/右键拖动平移。")
        empty_hint.setObjectName("EmptyHint")
        empty_hint.setProperty("class", "emptyHint")
        empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_hint.setWordWrap(True)

        layout.addStretch(1)
        layout.addWidget(empty_icon, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(empty_badge, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(empty_title)
        layout.addWidget(empty_tip)
        layout.addLayout(action_row)
        layout.addWidget(empty_steps)
        layout.addWidget(empty_hint)
        layout.addStretch(1)

        return card

    def _create_progress_panel(self):
        """创建顶部内联进度反馈条。"""
        panel = QFrame()
        panel.setObjectName("progressPanel")
        panel.setVisible(False)
        panel.setMaximumWidth(300)

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("等待开始")
        self._progress_bar.setMinimumHeight(18)

        layout.addWidget(self._progress_bar)
        return panel

    def _create_runtime_panel_drawer(self):
        """创建主图下方的抽屉式运行信息区。"""
        bar = QWidget()
        bar.setMaximumHeight(34)
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(4)

        title = QLabel("运行")
        title.setProperty("class", "topInfoMeta")
        title.setMaximumWidth(38)
        title.setToolTip("运行信息")
        bar_layout.addWidget(title)

        self.runtime_summary_chip = QLabel("状态：等待数据导入")
        self.runtime_summary_chip.setObjectName("RuntimeSummaryChip")
        self.runtime_summary_chip.setMinimumWidth(138)
        self.runtime_summary_chip.setMaximumWidth(420)
        bar_layout.addWidget(self.runtime_summary_chip, 1)

        self.btn_toggle_global_log = QPushButton("日志")
        self.btn_toggle_global_log.setCheckable(True)
        self.btn_toggle_global_log.clicked.connect(
            lambda checked: self._show_runtime_panel("global_log" if checked else None)
        )
        bar_layout.addWidget(self.btn_toggle_global_log)

        btn_collapse = QPushButton("收")
        btn_collapse.clicked.connect(lambda: self._show_runtime_panel(None))
        bar_layout.addWidget(btn_collapse)
        bar_layout.addStretch(1)

        container = QFrame()
        container.setObjectName("runtimeDrawer")
        container.setVisible(False)
        container.setMaximumHeight(128)
        drawer_layout = QStackedLayout(container)
        drawer_layout.setContentsMargins(0, 0, 0, 0)

        drawer_layout.addWidget(self.global_log_box)

        self._runtime_panel_stack = drawer_layout
        self._runtime_panel_buttons = {
            "global_log": self.btn_toggle_global_log,
        }
        return bar, container

    def _show_runtime_panel(self, panel_key: str | None):
        """控制主图下方抽屉式运行信息区。"""
        self._active_runtime_panel = panel_key
        if not self._runtime_panel_buttons:
            return

        for key, btn in self._runtime_panel_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(key == panel_key)
            btn.blockSignals(False)

        has_data = self.data is not None
        if not has_data or panel_key is None:
            if self._runtime_panel_container is not None:
                self._runtime_panel_container.setVisible(False)
            return

        if (
            self._runtime_panel_container is not None
            and self._runtime_panel_stack is not None
        ):
            self._runtime_panel_stack.setCurrentIndex(0)
            self._runtime_panel_container.setVisible(True)


    def _create_processing_stepper_bar(self) -> QFrame:
        """Create the processing-lineage stepper bar.

        Compatibility wrapper: implementation lives in
        ``ui.processing_lineage_controller.ProcessingLineageController``.
        """
        return self.processing_lineage_controller.create_stepper_bar()

    def _compact_step_label(self, label: str, index: int = 0) -> str:
        return self.processing_lineage_controller.compact_step_label(label, index)

    def _lineage_step_tooltip(self, entry: dict, index: int, total: int) -> str:
        return self.processing_lineage_controller.step_tooltip(entry, index, total)

    def _sync_processing_stepper(self) -> None:
        return self.processing_lineage_controller.sync_stepper()

    def _on_processing_step_clicked(self, index: int) -> None:
        return self.processing_lineage_controller.on_step_clicked(index)

    def _create_global_log_box(self):
        """创建全局日志面板。"""
        box = QGroupBox("全局日志")
        box.setToolTip("集中查看导入、处理、告警和导出等全局事件")

        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.runtime_log_view = QTextEdit()
        self.runtime_log_view.setObjectName("RuntimeEventLog")
        self.runtime_log_view.setReadOnly(True)
        self.runtime_log_view.setPlaceholderText("暂无全局日志")
        self.runtime_log_view.setMinimumHeight(78)
        self.runtime_log_view.setMaximumHeight(124)
        self.runtime_log_view.setToolTip("显示当前会话的全局运行日志")
        self.runtime_log_view.setPlainText(self.page_basic.info.toPlainText().strip())
        layout.addWidget(self.runtime_log_view)

        # Keep observability widgets alive for internal diagnostics, but do not
        # embed them in the user-facing global log drawer. The drawer should show
        # only the actual session log.
        self.performance_diag_box = self._create_observability_box()
        self.performance_diag_box.hide()

        return box

    def _create_observability_box(self):
        """创建低频性能诊断面板。"""
        box = QGroupBox("性能诊断（低频）")
        box.setCheckable(True)
        box.setChecked(False)
        box.setProperty("class", "lowProfileBox")
        box.setToolTip("仅在排查绘图卡顿、重绘频率或预处理耗时时展开")

        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        hint = QLabel("默认隐藏；仅在排查性能问题时查看这些统计。")
        hint.setWordWrap(True)
        hint.setProperty("class", "hintText")
        layout.addWidget(hint)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(1)

        self.obs_last_plot_label = QLabel("最近绘制耗时：--")
        self.obs_draw_count_label = QLabel("累计绘制次数：0")
        self.obs_skip_count_label = QLabel("累计跳过重绘：0")
        self.obs_last_prepare_label = QLabel("最近预处理耗时：--")

        for label in [
            self.obs_last_plot_label,
            self.obs_draw_count_label,
            self.obs_skip_count_label,
            self.obs_last_prepare_label,
        ]:
            label.setProperty("class", "metricLabel")
            body_layout.addWidget(label)

        layout.addWidget(body)
        box.toggled.connect(body.setVisible)
        body.setVisible(False)

        return box

    def _find_groupbox_by_title(self, root: QWidget, title: str):
        """在页面中按标题查找分组框。"""
        for box in root.findChildren(QGroupBox):
            if box.title().strip() == title:
                return box
        return None

    def _compress_status_group(self, status_group: QGroupBox):
        """压缩“当前状态”分组，减少它对主操作区的打断。"""
        if status_group is None:
            return
        status_group.setProperty("class", "compactStatusGroup")
        status_group.setMaximumHeight(180)
        editors = status_group.findChildren(QTextEdit)
        for editor in editors:
            editor.setMinimumHeight(72)
            editor.setMaximumHeight(96)
            editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _reorder_basic_groups_for_flow(self):
        """把“当前状态”移到“方法与常用参数”下方，避免打断主操作流程。"""
        page = getattr(self, "page_basic", None)
        if page is None:
            return

        method_group = self._find_groupbox_by_title(page, "方法与常用参数")
        status_group = self._find_groupbox_by_title(page, "当前状态")
        if status_group is None or method_group is None:
            return

        self._compress_status_group(status_group)

        parent_widget = status_group.parentWidget()
        if parent_widget is None or parent_widget is not method_group.parentWidget():
            return

        layout = parent_widget.layout()
        if layout is None:
            return

        status_idx = layout.indexOf(status_group)
        method_idx = layout.indexOf(method_group)
        if status_idx < 0 or method_idx < 0:
            return

        target_idx = method_idx + 1
        if status_idx == target_idx:
            return

        layout.removeWidget(status_group)
        if status_idx < target_idx:
            target_idx -= 1
        layout.insertWidget(target_idx, status_group)

    def _connect_signals(self):
        """连接信号和槽"""
        # 旧工作台信号连接已移除。

        # 基础流程页面 - 日常处理界面
        self.page_basic.btn_import.clicked.connect(self.import_csv_file)
        self.page_basic.action_import_csv.triggered.connect(self.import_csv_file)
        self.page_basic.action_import_folder.triggered.connect(
            self.import_ascans_folder
        )
        self.page_basic.action_import_out.triggered.connect(self.import_gprmax_out_file)
        self.page_basic.btn_apply.clicked.connect(
            self.apply_method_from_selected_source
        )
        self.page_basic.btn_quick.clicked.connect(self.run_default_pipeline)
        self.page_basic.btn_cancel.clicked.connect(self.cancel_processing)
        self.page_basic.btn_undo.clicked.connect(self.undo_last)
        self.page_basic.btn_reset.clicked.connect(self.reset_original)
        self.page_basic.method_combo.currentIndexChanged.connect(self._on_method_change)
        if hasattr(self.page_basic, "parametersChanged"):
            self.page_basic.parametersChanged.connect(self._on_basic_params_changed)

        # 显示与对比页面
        self.page_advanced.cmap_combo.currentIndexChanged.connect(self._refresh_plot)
        self.page_advanced.view_style_combo.currentIndexChanged.connect(
            self._refresh_plot
        )
        self.page_advanced.view_style_combo.currentIndexChanged.connect(
            self._on_view_style_changed
        )
        self.page_advanced.single_view_combo.currentIndexChanged.connect(
            self._refresh_plot
        )
        self.page_advanced.compare_left_combo.currentIndexChanged.connect(
            self._refresh_plot
        )
        self.page_advanced.compare_right_combo.currentIndexChanged.connect(
            self._refresh_plot
        )
        self.page_advanced.diff_var.stateChanged.connect(self._refresh_plot)
        self.page_advanced.slider_compare_var.stateChanged.connect(self._refresh_plot)
        self.page_advanced.slider_compare_var.stateChanged.connect(self._update_interaction_mode_status)
        self.page_advanced.diff_var.stateChanged.connect(self._on_compare_mode_state_changed)
        self.page_advanced.slider_compare_var.stateChanged.connect(self._on_compare_mode_state_changed)
        self.page_advanced.btn_apply_crop.clicked.connect(self._refresh_plot)
        self.page_advanced.btn_reset_crop.clicked.connect(self._reset_crop)
        self.page_advanced.btn_toggle_theme.clicked.connect(
            self._toggle_theme_from_main_ui
        )
        self.page_terrain3d.rtk_sidecar_button.clicked.connect(
            lambda: self._pick_sidecar_file("rtk")
        )
        self.page_terrain3d.rtk_sidecar_clear_button.clicked.connect(
            lambda: self._clear_sidecar_file("rtk")
        )
        self.page_terrain3d.imu_sidecar_button.clicked.connect(
            lambda: self._pick_sidecar_file("imu")
        )
        self.page_terrain3d.imu_sidecar_clear_button.clicked.connect(
            lambda: self._clear_sidecar_file("imu")
        )
        self.page_terrain3d.altimeter_sidecar_button.clicked.connect(
            lambda: self._pick_sidecar_file("altimeter")
        )
        self.page_terrain3d.altimeter_sidecar_clear_button.clicked.connect(
            lambda: self._clear_sidecar_file("altimeter")
        )
        self.page_auto_tune.btn_auto_tune.clicked.connect(
            self.start_auto_tune_current_method
        )
        self.page_auto_tune.btn_compare_stage.clicked.connect(
            self.start_auto_select_current_stage
        )
        self.page_auto_tune.btn_compare_manual_auto.clicked.connect(
            self.start_auto_tune_comparison
        )
        self.page_auto_tune.btn_export_comparison.clicked.connect(
            self.export_auto_tune_comparison_artifacts
        )
        self.page_auto_tune.btn_view_auto_tune.clicked.connect(
            self.show_auto_tune_details
        )
        self.page_auto_tune.btn_apply_stage_choice.clicked.connect(
            self.apply_stage_compare_choice
        )
        if hasattr(self.page_auto_tune, "recipe_run_requested"):
            self.page_auto_tune.recipe_run_requested.connect(
                self.apply_autotune_recipe_choice
            )
        self.page_auto_tune.btn_open_workbench.clicked.connect(
            self.switch_to_workbench_mode
        )
        self.page_advanced.btn_clear_manual_roi.clicked.connect(self._clear_manual_roi)

        # 显示选项
        for cb in [
            self.page_advanced.symmetric_var,
            self.page_advanced.auto_contrast_var,
            self.page_advanced.compare_var,
            self.page_advanced.cmap_invert_var,
            self.page_advanced.show_cbar_var,
            self.page_advanced.show_grid_var,
            self.page_advanced.show_physical_x_axis_var,
            self.page_advanced.show_physical_y_axis_var,
            self.page_advanced.percentile_var,
            self.page_advanced.normalize_var,
            self.page_advanced.demean_var,
            self.page_advanced.crop_enable_var,
        ]:
            cb.stateChanged.connect(self._refresh_plot)

        self.page_advanced.compare_var.toggled.connect(self._on_compare_toggled)
        self.page_advanced.compare_var.stateChanged.connect(self._on_compare_mode_state_changed)

        # 质量/日志页面
        self.page_quality.btn_generate_report.clicked.connect(self.generate_report)
        self.page_quality.btn_export_quality_snapshot.clicked.connect(
            self.export_quality_snapshot
        )
        self.page_quality.btn_export_replay_evidence.clicked.connect(
            self.export_replay_evidence_bundle
        )
        self.page_terrain3d.btn_export_georeference_3d.clicked.connect(
            self.export_airborne_georeference_3d_bundle
        )
        self.page_quality.btn_record_clear.clicked.connect(
            self.page_quality.record.clear
        )
        self.page_quality.btn_record_export.clicked.connect(self.export_record)
        self.page_quality.btn_open_log_dir.clicked.connect(self.open_log_directory)
        self.page_quality.btn_copy_diagnostics.clicked.connect(self.copy_diagnostics)
        self.page_terrain3d.btn_open_log_dir.clicked.connect(self.open_log_directory)

    def switch_to_legacy_mode(self):
        """切换到日常处理界面"""
        self.switch_to_main_mode("basic")

    def switch_to_main_mode(self, tab_key: str | None = None):
        """切换到日常处理界面

        Args:
            tab_key: 可选，指定要切换到的标签页，可选值：
                'basic' - 日常处理页
                'auto_tune' - 调参与实验页
                'advanced' - 显示与对比页
                'quality' - 质量与导出页
        """
        if self._content_stack is not None and self._main_content_widget is not None:
            self._content_stack.setCurrentWidget(self._main_content_widget)

            # 根据 tab_key 切换到指定标签页
            if tab_key == "basic" and self.page_basic is not None:
                self.control_tabs.setCurrentWidget(self.page_basic)
                self.status_label.setText("日常处理界面")
            elif tab_key == "auto_tune" and self.page_auto_tune is not None:
                self.control_tabs.setCurrentWidget(self.page_auto_tune)
                self.status_label.setText("自动选参")
            elif tab_key == "advanced" and self.page_advanced is not None:
                self.control_tabs.setCurrentWidget(self.page_advanced)
                self.status_label.setText("显示与对比")
            elif tab_key == "quality" and self.page_quality is not None:
                self.control_tabs.setCurrentWidget(self._page_quality_nav)
                self.status_label.setText("质量与导出")
            elif tab_key == "terrain3d" and getattr(self, "page_terrain3d", None) is not None:
                self.control_tabs.setCurrentWidget(self._page_terrain_nav)
                self.status_label.setText("地形/三维成果")
            else:
                # 默认切换到日常处理页
                if self.page_basic is not None:
                    self.control_tabs.setCurrentWidget(self.page_basic)
                self.status_label.setText("日常处理界面")

            tab_name = {
                "basic": "日常处理",
                "auto_tune": "自动选参",
                "advanced": "显示与对比",
                "quality": "质量与导出",
                "terrain3d": "地形/三维成果",
            }.get(tab_key, "日常处理")
            self._sync_workspace_shell_for_current_tab()
            self._log(f"切换到: {tab_name}")

    def switch_to_workbench_mode(self):
        """旧工作台已移除；保留兼容入口并回到日常处理页。"""
        self.switch_to_main_mode("basic")
        self._log("旧工作台已移除；请使用日常处理、AutoTune、显示与对比、质量与导出、地形/三维成果五个主标签页。")

    def _on_shared_data_changed(self, payload: dict):
        """共享数据状态变化后，同步相关视图。"""
        reason = (payload or {}).get("reason")
        self._store_trace_timestamps_from_metadata(self.trace_metadata)
        self._normalize_selected_trace_index()
        self._sync_history_action_state()
        self._refresh_compare_snapshots_from_state(
            clear_transient=reason in {"loaded", "current_updated", "undo", "reset"}
        )
        self._update_empty_state_and_brief()
        if reason == "loaded":
            self._manual_roi_values = None
            self._main_view_limits = None
            self._update_manual_roi_status()
        if reason in {"loaded", "undo", "reset"}:
            self._clear_runtime_warnings()
            if reason == "loaded":
                self._no_prior_guard_events = []
            self._reset_auto_tune_state("数据已更新，请重新自动选参。")
        if reason in {"loaded", "current_updated", "undo", "reset"}:
            self._lineage_view_index = None
        self._update_processing_lineage_display()
        self._sync_auto_tune_page_dataset_state(payload)

    def _sync_auto_tune_page_dataset_state(self, payload: dict | None = None) -> None:
        return self.autotune_sync_controller._sync_auto_tune_page_dataset_state(payload)

    def _normalize_selected_trace_index(self):
        """确保当前选中道号仍在有效范围内。"""
        if self.data is None or getattr(self.data, "ndim", 0) != 2:
            self._selected_trace_index = None
            return
        n_traces = int(self.data.shape[1])
        if self._selected_trace_index is not None and not (
            0 <= int(self._selected_trace_index) < n_traces
        ):
            self._selected_trace_index = None

    def _clear_manual_roi(self):
        return self.autotune_sync_controller._clear_manual_roi()

    def _set_manual_roi_pick_enabled(self, enabled: bool):
        return self.autotune_sync_controller._set_manual_roi_pick_enabled(enabled)

    def _sync_auto_tune_roi_picker_state(self):
        return self.autotune_sync_controller._sync_auto_tune_roi_picker_state()

    def _is_manual_roi_pick_enabled(self) -> bool:
        return self.autotune_sync_controller._is_manual_roi_pick_enabled()

    def _update_manual_roi_status(self):
        return self.autotune_sync_controller._update_manual_roi_status()

    def _set_selected_trace_index(self, trace_index: int | None):
        """设置当前选中的道号；主图优先轻量更新竖线，避免整张 B-scan 重绘。"""
        if self.data is None or getattr(self.data, "ndim", 0) != 2:
            normalized = None
        elif trace_index is None:
            normalized = None
        else:
            idx = int(trace_index)
            normalized = idx if 0 <= idx < int(self.data.shape[1]) else None

        if normalized == self._selected_trace_index:
            return

        self._selected_trace_index = normalized
        self._schedule_selected_trace_payload_refresh()

        if not self._refresh_selected_trace_marker_lightweight() and self.data is not None:
            self.plot_data(self.data)

    def _schedule_selected_trace_payload_refresh(self) -> None:
        """Delay heavy quality-page trajectory/3D payload refresh after trace selection."""
        timer = getattr(self, "_selected_trace_payload_timer", None)
        if timer is None:
            self._refresh_selected_trace_payload()
            return
        timer.start(160)

    def _refresh_selected_trace_payload(self) -> None:
        """Refresh expensive trace-dependent side panels after UI interaction settles."""
        if not (hasattr(self, "page_quality") and self.page_quality is not None):
            return
        try:
            self.page_quality.set_airborne_trajectory_visualization(
                self._build_airborne_trajectory_plot_payload()
            )
            if getattr(self, "page_terrain3d", None) is not None:
                self.page_terrain3d.set_airborne_georeference_3d_visualization(
                    self._build_airborne_georeference_3d_plot_payload()
                )
        except Exception:
            logger.debug("Delayed selected-trace payload refresh failed", exc_info=True)

    def _on_trajectory_trace_selected(self, trace_index: int):
        """响应航迹图中的 trace 选择。"""
        self._set_selected_trace_index(trace_index)

    def _on_main_canvas_press(self, event):
        return self.bscan_interaction_controller.on_main_canvas_press(event)

    def _on_main_canvas_motion(self, event):
        return self.bscan_interaction_controller.on_main_canvas_motion(event)

    def _on_main_canvas_release(self, event):
        return self.bscan_interaction_controller.on_main_canvas_release(event)

    def _event_button_number(self, event) -> int | None:
        return self.bscan_interaction_controller.event_button_number(event)

    def _is_main_slider_compare_active(self) -> bool:
        return self.bscan_interaction_controller.is_main_slider_compare_active()

    def _is_slider_compare_split_hit(self, event) -> bool:
        return self.bscan_interaction_controller.is_slider_compare_split_hit(event)

    def _update_main_slider_compare_ratio_from_event(self, event, force: bool = False):
        return self.bscan_interaction_controller.update_main_slider_compare_ratio_from_event(event, force=force)

    def _on_main_canvas_scroll(self, event):
        return self.bscan_interaction_controller.on_main_canvas_scroll(event)

    def _on_main_canvas_key_press(self, event):
        return self.bscan_interaction_controller.on_main_canvas_key_press(event)

    def _pan_main_plot_by_pixels(self, start: dict, dx_px: float, dy_px: float):
        return self.bscan_interaction_controller.pan_main_plot_by_pixels(start, dx_px, dy_px)

    def _set_clamped_axis_limits(self, ax, xlim, ylim):
        return self.bscan_interaction_controller.set_clamped_axis_limits(ax, xlim, ylim)

    def _clamp_main_view_limits(self, xlim, ylim):
        return self.bscan_interaction_controller.clamp_main_view_limits(xlim, ylim)

    def _update_plot_coord_label(self, event):
        return self.bscan_interaction_controller.update_plot_coord_label(event)

    def _on_main_canvas_leave(self, event):
        return self.bscan_interaction_controller.on_main_canvas_leave(event)

    def _toolbar_mode_active(self) -> bool:
        return self.bscan_interaction_controller.toolbar_mode_active()

    def _select_trace_from_x(self, x_value: float):
        return self.bscan_interaction_controller.select_trace_from_x(x_value)

    def _draw_drag_roi_preview(self, start: dict, event):
        return self.bscan_interaction_controller.draw_drag_roi_preview(start, event)

    def _remove_drag_roi_preview(self):
        return self.bscan_interaction_controller.remove_drag_roi_preview()

    def _capture_main_view_limits_from_axes(self):
        return self.bscan_interaction_controller.capture_main_view_limits_from_axes()

    def _reset_main_plot_view_to_default(self):
        return self.bscan_interaction_controller.reset_main_plot_view_to_default()

    def _sync_workbench_with_main_data(self):
        """旧工作台同步入口已移除；保留兼容空实现。"""
        self._update_empty_state_and_brief()

    def _ensure_processing_drawer_default_layout(self) -> None:
        """Keep the default processing drawer open and properly sized after show/import."""
        if (
            getattr(self, "control_tabs", None) is None
            or getattr(self, "_workspace_layout", None) is None
            or getattr(self, "_main_splitter", None) is None
        ):
            return
        current = self.control_tabs.currentWidget()
        if current in (getattr(self, "_page_quality_nav", None), getattr(self, "_page_terrain_nav", None)):
            return
        self._set_side_drawer_expanded(True, resize=False)
        self._workspace_layout.setCurrentWidget(self.main_plot_card)
        self._refresh_processing_workspace_status()
        self._set_workspace_splitter_profile("processing")

    def showEvent(self, event):
        """Stabilize the startup drawer after Qt has computed real window size."""
        super().showEvent(event)
        QTimer.singleShot(0, self._ensure_processing_drawer_default_layout)
        QTimer.singleShot(120, self._ensure_processing_drawer_default_layout)

    def resizeEvent(self, event):
        """窗口尺寸变化时，调整主图与右侧控制区比例。"""
        super().resizeEvent(event)

        # 只在主界面激活时调整，避免影响其它兼容页面。
        if (
            self._main_splitter is None
            or self._content_stack is None
            or self._content_stack.currentWidget() != self._main_content_widget
        ):
            return

        current = self.control_tabs.currentWidget() if getattr(self, "control_tabs", None) is not None else None
        if current in (getattr(self, "_page_quality_nav", None), getattr(self, "_page_terrain_nav", None)):
            self._set_workspace_splitter_profile("results")
        else:
            self._set_workspace_splitter_profile("processing")

    def _on_workbench_run_method(self, method_id: str, params: dict, source: str):
        """旧工作台运行入口已移除。"""
        self._log("旧工作台运行入口已移除；请使用日常处理页执行处理方法。")
        return

    def _apply_single_method(
        self,
        data: np.ndarray,
        method_id: str,
        params: dict,
        header_info: dict | None = None,
        trace_metadata: dict | None = None,
    ) -> dict:
        """执行单个方法"""
        input_header_info = header_info or self.header_info
        input_trace_metadata = trace_metadata or self.trace_metadata
        runtime_params = prepare_runtime_params(
            method_id,
            params,
            input_header_info,
            input_trace_metadata,
            data.shape,
        )
        result, result_meta = run_processing_method(data, method_id, runtime_params)
        result_header_info = merge_result_header_info(
            input_header_info, result_meta, result.shape
        )
        result_trace_metadata = merge_result_trace_metadata(
            input_trace_metadata, result_meta
        )

        display_data = result_meta.get("display_data")
        if display_data is not None:
            preview_data = np.asarray(display_data, dtype=np.float32)
            preview_header_info = merge_result_header_info(
                input_header_info,
                {"header_info_updates": result_meta.get("display_header_info_updates")},
                preview_data.shape,
            )
            display_trace_metadata = result_meta.get("display_trace_metadata")
            preview_trace_metadata = (
                display_trace_metadata
                if display_trace_metadata is not None
                else result_trace_metadata
            )
        else:
            preview_data = result
            preview_header_info = result_header_info
            preview_trace_metadata = result_trace_metadata

        return {
            "result_data": result,
            "result_header_info": result_header_info,
            "result_trace_metadata": result_trace_metadata,
            "preview_data": preview_data,
            "preview_header_info": preview_header_info,
            "preview_trace_metadata": preview_trace_metadata,
            "meta": result_meta,
        }

    def _on_workbench_save_result(self):
        """旧工作台保存入口已移除。"""
        self._log("旧工作台保存入口已移除；当前结果请通过质量与导出页处理。")
        return

    def _apply_style(self):
        """应用样式表 - 使用主题管理器 + 轻量视觉美化层。"""
        from core.theme_manager import get_theme_manager

        theme_manager = get_theme_manager()
        stylesheet = theme_manager.get_theme_stylesheet() or ""

        try:
            from ui.theme import get_app_polish_stylesheet

            stylesheet = stylesheet + "\n" + get_app_polish_stylesheet(theme_manager.get_current_theme(), widget=self)
        except Exception as exc:
            logger.debug("UI polish stylesheet unavailable: %s", exc)

        if stylesheet and stylesheet != getattr(self, "_last_applied_stylesheet", None):
            self.setStyleSheet(stylesheet)
            self._last_applied_stylesheet = stylesheet

        current_theme = theme_manager.get_current_theme()
        self._apply_main_workspace_direct_theme(current_theme)
        page_auto_tune = getattr(self, "page_auto_tune", None)
        if page_auto_tune is not None and hasattr(page_auto_tune, "refresh_theme"):
            try:
                page_auto_tune.refresh_theme(current_theme)
            except Exception as exc:
                logger.debug("AutoTune local theme refresh failed: %s", exc)

    def _apply_main_workspace_direct_theme(self, theme: str | None = None):
        """Apply local theme reinforcement to the B-scan workspace.

        Some qfluentwidgets and Matplotlib toolbar styles can override broad QSS
        selectors. This keeps the empty-state card and plot card readable in both
        light and dark themes without changing plotting logic.
        """
        try:
            from ui.theme import is_dark_ui

            is_dark = is_dark_ui(theme, widget=self)
        except Exception:
            is_dark = str(theme or "").lower() == "dark"
        effective_key = "dark" if is_dark else "light"
        if getattr(self, "_last_workspace_direct_theme", None) == effective_key:
            return
        self._last_workspace_direct_theme = effective_key
        if is_dark:
            card = "#1B2027"
            panel = "#151A21"
            border = "#303A48"
            dash = "#3B4654"
            text = "#EAF0F8"
            muted = "#A6B1C2"
            primary_bg = "#12323B"
            primary_text = "#8BE8DE"
            toolbar = "#14191F"
        else:
            card = "#FFFFFF"
            panel = "#FBFDFF"
            border = "#E3EAF3"
            dash = "#C8D3E1"
            text = "#172033"
            muted = "#64748B"
            primary_bg = "#E7FAF7"
            primary_text = "#08776F"
            toolbar = "#FFFFFF"

        if getattr(self, "main_plot_card", None) is not None:
            self.main_plot_card.setStyleSheet(
                f"QFrame#mainPlotCard {{ background: {card}; border: 1px solid {border}; border-radius: 22px; }}"
            )
        if getattr(self, "_workspace_host", None) is not None:
            self._workspace_host.setStyleSheet(
                f"QFrame#WorkspaceHost {{ background: transparent; border: none; }}"
            )
        if getattr(self, "_side_nav_rail", None) is not None:
            self._side_nav_rail.setStyleSheet(
                f"QFrame#SideNavRail {{ background: {panel}; border: 1px solid {border}; border-radius: 16px; }}"
                f"QToolButton#SideDrawerToggle, QToolButton#SideWorkspaceButton {{ background: transparent; border: 1px solid transparent; border-radius: 12px; color: {muted}; font-weight: 800; }}"
                f"QToolButton#SideDrawerToggle:hover, QToolButton#SideWorkspaceButton:hover {{ background: {primary_bg}; border-color: {border}; color: {primary_text}; }}"
                f"QToolButton#SideWorkspaceButton:checked {{ background: {primary_bg}; border-color: {primary_text}; color: {primary_text}; }}"
            )
        for placeholder_name in ("_page_quality_nav", "_page_terrain_nav"):
            placeholder = getattr(self, placeholder_name, None)
            if placeholder is not None:
                placeholder.setStyleSheet(
                    f"QFrame#WorkspaceNavPlaceholder {{ background: {panel}; border: 1px solid {border}; border-radius: 16px; }}"
                    f"QLabel#WorkspaceNavTitle {{ color: {text}; font-weight: 800; font-size: 14px; background: transparent; }}"
                    f"QLabel#WorkspaceNavBody {{ color: {muted}; background: transparent; line-height: 1.35; }}"
                    f"QLabel#WorkspaceNavAction {{ color: {primary_text}; background: {primary_bg}; border-radius: 10px; padding: 8px; font-weight: 700; }}"
                )
        if getattr(self, "empty_state_card", None) is not None:
            self.empty_state_card.setStyleSheet(
                f"QFrame#emptyStateCard {{ background: {panel}; border: 1px dashed {dash}; border-radius: 22px; }}"
            )
        for name in ("_plot_title_label",):
            label = getattr(self, name, None)
            if label is not None:
                label.setStyleSheet(f"color: {text}; background: transparent; font-weight: 800;")
        for name in ("_plot_meta_label",):
            label = getattr(self, name, None)
            if label is not None:
                label.setStyleSheet(f"color: {muted}; background: transparent;")
        for name in ("_main_toolbar",):
            widget = getattr(self, name, None)
            if widget is not None:
                if hasattr(widget, "apply_theme"):
                    try:
                        widget.apply_theme(effective_key)
                    except Exception:
                        pass
                widget.setStyleSheet(
                    f"QToolBar {{ background: {toolbar}; border: 1px solid {border}; border-radius: 13px; spacing: 3px; padding: 4px; }}"
                    f"QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 8px; padding: 4px; color: {text}; }}"
                    f"QToolButton:hover {{ background: {primary_bg}; border-color: {border}; color: {primary_text}; }}"
                    f"QToolButton:disabled {{ color: {muted}; background: transparent; }}"
                    f"QToolButton:pressed, QToolButton:checked {{ background: {primary_bg}; border-color: {border}; color: {primary_text}; }}"
                )

    def _toggle_theme_from_main_ui(self):
        """主界面显示页触发主题切换。"""
        from core.theme_manager import get_theme_manager

        theme_manager = get_theme_manager()
        current_theme = theme_manager.toggle_theme()
        app = QApplication.instance()
        if app is not None:
            theme_manager.apply_app_theme(app, current_theme)
        self._apply_style()
        self._refresh_plot()
        info = theme_manager.get_theme_info(current_theme)
        self._log(f"已切换到{info.get('name', current_theme)}")

    # ============ 日志和帮助方法 ============

    def _classify_log_event(self, msg: str) -> str:
        """Classify a log message into a compact event tag for the global log."""
        return classify_log_event(msg)

    def _format_runtime_log_html(self, timestamp: str, event_type: str, msg: str) -> str:
        """Return a single compact HTML row for the bottom event stream."""
        try:
            from core.theme_manager import get_theme_manager
            from ui.theme import get_effective_theme_key

            theme_key = get_effective_theme_key(get_theme_manager().get_current_theme(), widget=self)
        except Exception:
            theme_key = "light"
        if theme_key == "dark":
            palette = {
                "SYS": ("#CBD5E1", "#1F2937"),
                "INFO": ("#D1D5DB", "#1F2937"),
                "DATA": ("#99F6E4", "#0F2F2B"),
                "METHOD": ("#BFDBFE", "#172A46"),
                "WARN": ("#FCD34D", "#3A2A10"),
                "ERR": ("#FCA5A5", "#3B1517"),
                "EXPORT": ("#DDD6FE", "#2E2148"),
            }
        else:
            palette = {
                "SYS": ("#64748B", "#F1F5F9"),
                "INFO": ("#475569", "#F1F5F9"),
                "DATA": ("#0F766E", "#E7FAF7"),
                "METHOD": ("#2563EB", "#EFF6FF"),
                "WARN": ("#B45309", "#FFF7E8"),
                "ERR": ("#B91C1C", "#FEF2F2"),
                "EXPORT": ("#6D28D9", "#F5F3FF"),
            }
        fg, bg = palette.get(event_type, palette["INFO"])
        ts_color = "#94A3B8" if theme_key == "light" else "#9CA3AF"
        safe_msg = html.escape(str(msg))
        safe_ts = html.escape(str(timestamp))
        safe_type = html.escape(str(event_type))
        return (
            f'<div style="margin:2px 0; line-height:1.35;">'
            f'<span style="color:{ts_color}; font-family:Consolas, monospace;">[{safe_ts}]</span> '
            f'<span style="display:inline-block; min-width:46px; padding:1px 6px; '
            f'border-radius:7px; color:{fg}; background:{bg}; font-weight:700; '
            f'font-family:Consolas, monospace;">{safe_type}</span> '
            f'<span>{safe_msg}</span>'
            f'</div>'
        )

    def _flush_pending_log_ui(self):
        """Batch-flush visible log widgets to avoid QTextEdit churn during long tasks."""
        lines = list(getattr(self, "_pending_plain_log_lines", []) or [])
        html_lines = list(getattr(self, "_pending_runtime_log_html", []) or [])
        quality_lines = list(getattr(self, "_pending_quality_log_lines", []) or [])
        self._pending_plain_log_lines = []
        self._pending_runtime_log_html = []
        self._pending_quality_log_lines = []

        if lines and getattr(self, "page_basic", None) is not None:
            try:
                self.page_basic.info.append("\n".join(lines))
                self.page_basic.info.ensureCursorVisible()
            except Exception:
                logger.debug("Failed to flush basic log buffer", exc_info=True)
        if html_lines and getattr(self, "runtime_log_view", None) is not None:
            try:
                self.runtime_log_view.append("".join(html_lines))
                self.runtime_log_view.ensureCursorVisible()
            except Exception:
                logger.debug("Failed to flush runtime log buffer", exc_info=True)
        if quality_lines and getattr(self, "page_quality", None) is not None:
            try:
                for line in quality_lines:
                    self.page_quality.append_record(line)
            except Exception:
                logger.debug("Failed to flush quality log buffer", exc_info=True)

    def _log(
        self,
        msg: str,
        *,
        event_type: str | None = None,
        source: str = "ui",
        context: dict | None = None,
    ):
        """Record a user-visible event and its structured audit twin."""
        event = LogEvent.create(
            str(msg),
            event_type=event_type,
            source=source,
            context=context or {},
        )
        if not hasattr(self, "_runtime_log_events") or self._runtime_log_events is None:
            self._runtime_log_events = LogEventBuffer(max_events=2000)
        self._runtime_log_events.append(event)
        timestamp = datetime.now().strftime("%H:%M:%S")
        tag = event.event_type
        line = f"[{timestamp}] {msg}"
        self._pending_plain_log_lines.append(line)
        if hasattr(self, "runtime_log_view") and self.runtime_log_view is not None:
            self._pending_runtime_log_html.append(
                self._format_runtime_log_html(timestamp, tag, msg)
            )
        if hasattr(self, "page_quality") and self.page_quality is not None:
            self._pending_quality_log_lines.append(line)
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start(80)

    def _record_structured_error(
        self,
        exc: BaseException | str,
        *,
        category: str = "runtime",
        context: dict | None = None,
        log: bool = True,
    ) -> dict:
        """Normalize an exception to Evidence-friendly structured error metadata."""
        if isinstance(exc, BaseException):
            info = error_info_from_exception(exc, category=category, context=context or {})
        else:
            info = error_info_from_exception(
                RuntimeError(str(exc)),
                category=category,
                context=context or {},
            )
        payload = info.to_dict()
        if not hasattr(self, "_last_structured_errors") or self._last_structured_errors is None:
            self._last_structured_errors = []
        self._last_structured_errors.append(payload)
        self._last_structured_errors = self._last_structured_errors[-100:]
        if log:
            self._log(
                f"{info.error_code}: {info.user_message}",
                event_type="ERR",
                source=category,
                context=payload,
            )
        return payload

    def _get_structured_runtime_log_payload(self, timestamp: str) -> dict:
        """Return structured log sidecar payload for Evidence packages."""
        events = []
        if hasattr(self, "_runtime_log_events") and self._runtime_log_events is not None:
            events = self._runtime_log_events.to_list()
        return {
            "schema": "mygpr.runtime_events.v1",
            "timestamp": timestamp,
            "events": events,
            "structured_errors": self._json_safe(getattr(self, "_last_structured_errors", [])[-100:]),
        }

    def _default_output_dir(self) -> str:
        """默认输出目录"""
        return get_output_dir()

    def _build_error_hint(self, error_msg: str) -> str:
        """根据常见错误给出可操作提示"""
        lower = (error_msg or "").lower()
        if "no module named" in lower:
            return "建议：检查 PythonModule 路径和依赖是否完整。"
        if (
            "invalid parameter" in lower
            or "高于最大值" in error_msg
            or "低于最小值" in error_msg
        ):
            return "建议：降低窗口/阶数参数，先使用推荐预设再微调。"
        if "output csv not found" in lower:
            return "建议：确认输出目录可写，并检查磁盘权限/路径。"
        if "csv" in lower and ("format" in lower or "parse" in lower):
            return "建议：确认输入为二维数值 CSV（samples x traces），并去除非数值列。"
        return "建议：先尝试“重置原始”后用默认流程复跑；若仍失败请反馈日志末尾 20 行。"

    def _set_busy(self, busy: bool, text: str = "处理中..."):
        """设置忙碌状态"""
        self._ui_busy = bool(busy)
        controls = [
            self.page_basic.btn_import,
            self.page_basic.btn_apply,
            self.page_basic.btn_quick,
            self.page_basic.btn_undo,
            self.page_basic.btn_reset,
            self.page_advanced.btn_apply_crop,
            self.page_advanced.btn_reset_crop,
            self.page_auto_tune.btn_auto_tune,
            self.page_auto_tune.btn_compare_stage,
            self.page_auto_tune.btn_compare_manual_auto,
            self.page_auto_tune.btn_view_auto_tune,
            self.page_auto_tune.btn_apply_stage_choice,
            self.page_basic.method_combo,
            self.page_quality.btn_generate_report,
            self.page_quality.btn_export_quality_snapshot,
            self.page_quality.btn_export_replay_evidence,
            self.page_terrain3d.btn_export_georeference_3d,
        ]
        for w in controls:
            w.setEnabled(not busy)
        self.page_basic.btn_cancel.setEnabled(busy and (not self._cancel_in_flight))
        if busy:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.status_label.setText(text)
            try:
                self.page_basic.set_apply_button_state("busy", text)
            except Exception:
                pass
            self._set_runtime_summary(f"状态：{text}", "info")
            if self._progress_panel is not None:
                self._progress_panel.setVisible(True)
            if self._progress_bar is not None:
                self._progress_bar.setRange(0, 0)
                self._progress_bar.setValue(0)
                self._progress_bar.setFormat("准备处理中…")
        else:
            QApplication.restoreOverrideCursor()
            self.status_label.setText(text)
            if self._progress_bar is not None:
                self._progress_bar.setRange(0, 100)
                self._progress_bar.setValue(0)
                self._progress_bar.setFormat(text)
            if self._progress_panel is not None:
                self._progress_panel.setVisible(False)
        if hasattr(self, "page_auto_tune") and self.page_auto_tune is not None:
            self.page_auto_tune.btn_export_comparison.setEnabled(
                (not busy) and (self._last_auto_tune_comparison_result is not None)
            )
        self._sync_history_action_state()
        QApplication.processEvents()

    def _sync_history_action_state(self):
        """同步主界面撤回按钮状态。"""
        if not hasattr(self, "page_basic") or self.page_basic is None:
            return
        can_undo = (not self._ui_busy) and self.shared_data.can_undo()
        self.page_basic.btn_undo.setEnabled(can_undo)

    # ============ 历史管理 ============

    def _push_history(self):
        """保存当前状态到历史"""
        if self.data is None:
            return
        self.shared_data.push_history()

    def undo_last(self):
        """撤销上一步"""
        if not self.shared_data.can_undo():
            QMessageBox.information(self, "撤销", "无可恢复的历史状态。")
            return
        self.shared_data.undo()
        self._mark_data_changed()
        self._refresh_compare_snapshots_from_state(clear_transient=True)
        self._update_empty_state_and_brief()
        self.plot_data(self.data)
        self._log("撤销: restored previous state.")

    def reset_original(self):
        """重置为原始数据"""
        if self.original_data is None:
            QMessageBox.information(self, "重置", "未加载原始数据。")
            return
        self.shared_data.reset_to_original(push_history=False, clear_history=True)
        self._lineage_view_index = None
        try:
            self.processing_lineage_controller.clear_compare_selection()
        except Exception:
            pass
        try:
            self.processing_lineage_controller.clear_display_override()
        except Exception:
            pass
        self._mark_data_changed()
        self._clear_transient_compare_snapshots()
        self._update_empty_state_and_brief()
        self.plot_data(self.data)
        self._log("重置: restored original data.")

    # ============ UI回调 ============

    def _import_folder(self):
        """导入A-scan文件夹"""
        # 复用现有的导入逻辑
        if hasattr(self, "read_ascans_folder"):
            folder = QFileDialog.getExistingDirectory(self, "选择A-scan文件夹")
            if folder:
                try:
                    data = self.read_ascans_folder(folder)
                    self.shared_data.load_data(
                        data, path=folder, source="folder_import"
                    )
                    self._log_info(f"已导入文件夹: {folder}")
                    self._refresh_plot()
                except Exception as e:
                    QMessageBox.warning(self, "导入失败", f"无法导入文件夹:\n{str(e)}")

    def _on_basic_params_changed(self):
        """User changed method parameters; make the pending state visible."""
        method_key = self.page_basic.get_current_method_key() if self.page_basic is not None else None
        if method_key:
            try:
                self._method_param_overrides[method_key] = self.page_basic.get_current_params()
            except Exception:
                pass
        self._set_runtime_summary("状态：参数已修改，等待应用", "warning")

    def _set_runtime_summary(self, text: str, tone: str = "neutral") -> None:
        chip = getattr(self, "runtime_summary_chip", None)
        if chip is None:
            return
        chip.setText(str(text or ""))
        chip.setProperty("tone", str(tone or "neutral"))
        try:
            chip.style().unpolish(chip); chip.style().polish(chip); chip.update()
        except Exception:
            pass

    def _update_interaction_mode_status(self, *args):
        chip = getattr(self, "_interaction_mode_chip", None)
        if chip is None:
            return
        if self._is_manual_roi_pick_enabled():
            text, tone = "ROI", "warning"
            tip = "ROI 模式已开启：左键拖动框选 ROI，Esc 可取消临时框。"
        elif self._is_main_slider_compare_active():
            text, tone = "滑动", "info"
            tip = "滑动对比已开启：靠近分隔线后左键拖动分割位置。"
        else:
            text, tone = "查看", "neutral"
            tip = "默认查看：左键选道，滚轮缩放，中/右键拖动平移。"
        chip.setText(text)
        chip.setToolTip(tip)
        chip.setProperty("tone", tone)
        try:
            chip.style().unpolish(chip); chip.style().polish(chip); chip.update()
        except Exception:
            pass

    def _on_method_change(self, idx=None):
        """方法选择改变"""
        idx = self.page_basic.method_combo.currentIndex()
        if idx < 0:
            return
        key = self.page_basic.method_keys[idx]
        self.page_basic.render_method_params(key)
        self.page_basic.mark_params_applied("方法已切换；修改参数后点击“应用方法”。")
        self._reset_auto_tune_state()
        self._update_empty_state_and_brief()
        if self.data is not None:
            self._set_runtime_summary("状态：方法已切换，等待应用", "neutral")

    def _reset_auto_tune_state(self, message: str | None = None):
        return self.autotune_sync_controller._reset_auto_tune_state(message)

    def _clear_runtime_warnings(self):
        """清空当前运行告警。"""
        self._runtime_warnings = []

    def _append_runtime_warnings(
        self,
        warnings: list[dict] | None,
        *,
        source: str | None = None,
        log: bool = True,
    ):
        """追加结构化运行告警并按需写入日志。"""
        prepared = []
        for item in warnings or []:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            details = dict(normalized.get("details", {}) or {})
            if source and "source" not in details:
                details["source"] = source
            normalized["details"] = details
            prepared.append(normalized)

        previous = list(self._runtime_warnings)
        self._runtime_warnings = merge_runtime_warnings(
            self._runtime_warnings, prepared
        )
        if not log:
            return
        seen = {
            format_runtime_warning_text(item)
            for item in previous
            if format_runtime_warning_text(item)
        }
        for warning in self._runtime_warnings:
            text = format_runtime_warning_text(warning)
            if text and text not in seen:
                self._log(f"告警: {text}")
                seen.add(text)

    def apply_method_from_selected_source(self):
        """按当前默认来源执行“应用方法”主按钮。"""
        source_mode = self.page_basic.get_apply_source_mode()
        if source_mode == "auto_tune":
            self.apply_method_auto_tuned_default()
            return
        self.apply_method_manual()

    def apply_method_manual(self):
        """按当前手动参数执行方法。"""
        self.page_basic.set_apply_source_hint("将按当前参数执行。")
        self.apply_method()

    def apply_method_auto_tuned_default(self):
        """按自动调参推荐参数执行当前方法。"""
        self._enforce_no_prior_action_guard(
            "preset_recommendation",
            dialog_title="自动推荐参数",
            allow_override=True,
            show_dialog=False,
            advisory_only=True,
        )

        if not self._last_auto_tune_result:
            self.page_basic.set_apply_source_hint(
                "当前无可用推荐结果，正在分析当前参数..."
            )
            self.start_auto_tune_current_method(auto_apply_after_finish=True)
            return

        method_key = self.page_basic.get_current_method_key()
        if method_key != self._last_auto_tune_result.get("method_key"):
            self._reset_auto_tune_state("当前方法已变化，请先重新运行自动选参。")
            self.page_basic.set_apply_source_hint(
                "当前推荐结果已过期，正在重新分析当前参数..."
            )
            self.start_auto_tune_current_method(auto_apply_after_finish=True)
            return

        profile_key = str(
            self._last_auto_tune_result.get("recommended_profile", "balanced")
        )
        self.apply_method_from_profile(profile_key)






    def _on_auto_tune_progress(self, current: int, total: int, message: str):
        """自动选参进度回调（节流刷新 UI）。

        Candidate sweeps can emit many progress signals in a short interval.
        Updating QLabel/QProgressBar on every signal makes the main thread feel
        sticky even though the numerical work is unchanged.  This method stores
        only the latest progress payload and flushes it at a bounded cadence.
        """
        safe_total = max(int(total), 1)
        safe_current = max(0, min(int(current), safe_total))
        self._pending_auto_tune_progress = (safe_current, safe_total, str(message))
        monitor = getattr(self, "_perf_monitor", None)
        if monitor is not None:
            monitor.record("autotune.progress_request", 0.0)
        now = time.perf_counter()
        interval = float(getattr(self, "_auto_tune_progress_min_interval_s", 0.12))
        if (now - float(getattr(self, "_last_auto_tune_progress_flush_ts", 0.0))) >= interval:
            self._flush_auto_tune_progress_update()
            return
        remaining_ms = max(1, int(round((interval - (now - float(self._last_auto_tune_progress_flush_ts))) * 1000.0)))
        if not self._auto_tune_progress_timer.isActive():
            self._auto_tune_progress_timer.start(remaining_ms)

    def _flush_auto_tune_progress_update(self):
        """Apply the latest coalesced AutoTune progress payload to widgets."""
        payload = getattr(self, "_pending_auto_tune_progress", None)
        if payload is None:
            return
        self._pending_auto_tune_progress = None
        safe_current, safe_total, message = payload
        start_ts = time.perf_counter()
        self.status_label.setText(message)
        if self._progress_panel is not None:
            self._progress_panel.setVisible(True)
        if self._progress_bar is not None:
            self._progress_bar.setRange(0, safe_total)
            self._progress_bar.setValue(safe_current)
            self._progress_bar.setFormat(f"候选 {safe_current}/{safe_total}")
        self._last_auto_tune_progress_flush_ts = time.perf_counter()
        monitor = getattr(self, "_perf_monitor", None)
        if monitor is not None:
            monitor.record("autotune.progress_flush_ms", (time.perf_counter() - start_ts) * 1000.0)

    def _on_auto_tune_finished(self, result: dict):
        """自动选参完成。"""
        cancelled = bool(result.get("cancelled"))
        pending_apply = bool(self._pending_apply_after_auto_tune)
        self._set_busy(
            False, text="自动选参完成" if not cancelled else "自动选参已取消"
        )
        if cancelled:
            self._pending_apply_after_auto_tune = False
            self.page_basic.set_apply_source_hint(
                "自动分析已取消，当前未生成自动调参结果。"
            )
            self.page_auto_tune.show_cancelled()
            return

        self._attach_auto_tune_recommendation_label(result)
        self._last_auto_tune_result = result
        self.page_basic.set_auto_tune_result_available(True, result.get("profiles", {}))
        self.page_basic.set_apply_source_hint(
            "已生成自动调参结果，可在“应用方法”右侧切换默认应用来源。"
        )
        self.page_auto_tune.show_result(result)
        self._log(
            f"自动选参完成: {result.get('method_name', result.get('method_key'))} | 推荐参数 {result.get('recommended_params') or result.get('best_params')}"
        )
        self._log_auto_tune_recommendation_label(result)
        if pending_apply:
            self._pending_apply_after_auto_tune = False
            profile_key = str(result.get("recommended_profile", "balanced"))
            self._log(f"自动选参完成后自动应用推荐档：{profile_key}")
            self.apply_method_from_profile(profile_key)

    def _on_auto_tune_error(self, error_msg: str):
        """自动选参失败。"""
        self._set_busy(False, text="自动选参失败")
        self._pending_apply_after_auto_tune = False
        self.page_basic.set_apply_source_hint("自动分析失败，未执行方法。")
        self.page_auto_tune.show_error(error_msg)
        self._log(f"自动选参失败: {error_msg}")
        QMessageBox.warning(self, "自动选参失败", error_msg)

    def _on_auto_stage_finished(self, result: dict):
        """同阶段方法比较完成。"""
        cancelled = bool(result.get("cancelled"))
        self._set_busy(
            False, text="同阶段比较完成" if not cancelled else "同阶段比较已取消"
        )
        if cancelled:
            self.page_auto_tune.show_cancelled()
            return

        self._last_auto_tune_group_result = result
        best_auto = result.get("best_auto_tune_result") or {}
        if best_auto:
            self._attach_auto_tune_recommendation_label(best_auto)
            self._last_auto_tune_result = best_auto
            self.page_auto_tune.set_auto_tune_method_key(
                result.get("best_method_key", best_auto.get("method_key"))
            )
            self.page_basic.set_auto_tune_result_available(
                True, best_auto.get("profiles", {})
            )
            self.page_basic.set_apply_source_hint(
                "已生成同阶段比较推荐，可切换为自动调参推荐执行。"
            )
        self.page_auto_tune.set_stage_compare_result(result)
        self.page_auto_tune.show_result(best_auto)
        self._log_auto_tune_recommendation_label(best_auto)
        self._log(
            f"同阶段比较完成: 推荐 {result.get('best_method_name', result.get('best_method_key'))} | outer score {float(result.get('outer_score', 0.0)):.4f}"
        )

    def _on_auto_stage_error(self, error_msg: str):
        """同阶段方法比较失败。"""
        self._set_busy(False, text="同阶段比较失败")
        self.page_auto_tune.show_error(error_msg)
        self._log(f"同阶段比较失败: {error_msg}")
        QMessageBox.warning(self, "同阶段比较失败", error_msg)

    def _on_auto_comparison_finished(self, result):
        """人工 baseline vs 自动选参对比完成。"""
        cancelled = isinstance(result, dict) and bool(result.get("cancelled"))
        self._set_busy(
            False, text="人工/自动对比完成" if not cancelled else "人工/自动对比已取消"
        )
        if cancelled:
            self.page_auto_tune.show_cancelled()
            return

        self._last_auto_tune_comparison_result = result
        summary = to_summary_dict(result)
        self.page_auto_tune.show_comparison_result(summary)
        self._set_auto_tune_comparison_snapshots(result)
        self._log(
            "人工/自动对比完成: verdict={verdict} | Δscore={delta:.4f}".format(
                verdict=summary.get("verdict"),
                delta=float(
                    (summary.get("metric_delta") or {}).get(
                        "comparison_score", 0.0
                    )
                ),
            )
        )

    def _on_auto_comparison_error(self, error_msg: str):
        """人工/自动对比失败。"""
        self._set_busy(False, text="人工/自动对比失败")
        self.page_auto_tune.show_comparison_error(error_msg)
        self._log(f"人工/自动对比失败: {error_msg}")
        QMessageBox.warning(self, "人工/自动对比失败", error_msg)

    def _set_auto_tune_comparison_snapshots(self, result):
        """把人工/自动结果推入现有 B-scan 对比快照。"""
        manual_header = (result.manual.metadata or {}).get("header_info")
        manual_trace = (result.manual.metadata or {}).get("trace_metadata")
        auto_header = (result.automatic.metadata or {}).get("header_info")
        auto_trace = (result.automatic.metadata or {}).get("trace_metadata")
        self._set_compare_snapshots(
            [
                {
                    "label": "人工 baseline",
                    "data": result.manual.result,
                    "header_info": manual_header,
                    "trace_metadata": manual_trace,
                },
                {
                    "label": "自动选参",
                    "data": result.automatic.result,
                    "header_info": auto_header,
                    "trace_metadata": auto_trace,
                },
            ]
        )
        labels = [snap["label"] for snap in self.compare_snapshots]
        manual_label = next((label for label in labels if label.startswith("人工 baseline")), "")
        auto_label = next((label for label in labels if label.startswith("自动选参")), "")
        if manual_label and auto_label:
            self.page_advanced.compare_var.setChecked(True)
            self.page_advanced.compare_left_combo.setCurrentText(manual_label)
            self.page_advanced.compare_right_combo.setCurrentText(auto_label)
        self.plot_data(self.data)

    def _cleanup_auto_tune_worker(self):
        """清理自动选参线程。"""
        if self._auto_tune_thread:
            self._auto_tune_thread.quit()
            self._auto_tune_thread.wait(5000)
            self._auto_tune_thread = None
        self._auto_tune_worker = None
        self._cancel_in_flight = False
        self.page_basic.btn_cancel.setEnabled(False)
        self._pending_apply_after_auto_tune = False

    def _cleanup_auto_tune_stage_worker(self):
        """清理同阶段比较线程。"""
        if self._auto_tune_stage_thread:
            self._auto_tune_stage_thread.quit()
            self._auto_tune_stage_thread.wait(5000)
            self._auto_tune_stage_thread = None
        self._auto_tune_stage_worker = None
        self._cancel_in_flight = False
        self.page_basic.btn_cancel.setEnabled(False)

    def _cleanup_auto_tune_comparison_worker(self):
        """清理人工/自动对比线程。"""
        if self._auto_tune_comparison_thread:
            self._auto_tune_comparison_thread.quit()
            self._auto_tune_comparison_thread.wait(5000)
            self._auto_tune_comparison_thread = None
        self._auto_tune_comparison_worker = None
        self._cancel_in_flight = False
        self.page_basic.btn_cancel.setEnabled(False)

    def apply_method_from_profile(self, profile_key: str):
        """使用自动选参档位参数并立即执行当前方法。"""
        self._enforce_no_prior_action_guard(
            "preset_recommendation",
            dialog_title="应用推荐档",
            allow_override=True,
            show_dialog=False,
            advisory_only=True,
        )

        if not self._last_auto_tune_result:
            QMessageBox.information(
                self,
                "自动选参结果不存在",
                "请先到“自动选参”页执行自动选参。",
            )
            return

        method_key = self.page_basic.get_current_method_key()
        if method_key != self._last_auto_tune_result.get("method_key"):
            QMessageBox.information(
                self,
                "自动选参结果已过期",
                "当前方法已变化，请重新自动选参。",
            )
            self._reset_auto_tune_state("当前方法已变化，请重新自动选参。")
            return

        profile = (self._last_auto_tune_result.get("profiles", {}) or {}).get(
            profile_key
        )
        if not profile:
            QMessageBox.information(
                self,
                "自动选参档位不可用",
                "当前没有可用的该档位参数，请重新自动选参。",
            )
            return

        apply_params = dict(profile.get("params", {}))
        self.page_basic.apply_method_params(method_key, apply_params)
        self._method_param_overrides[method_key] = dict(apply_params)
        self.page_basic.set_apply_source_hint(
            f"将使用自动调参推荐 - {profile.get('label', profile_key)}"
        )
        self._log(f"使用自动选参{profile.get('label', profile_key)}执行当前方法。")
        self.apply_method()

    def show_auto_tune_details(self):
        """显示自动选参候选评分详情。"""
        if not self._last_auto_tune_result:
            QMessageBox.information(self, "自动选参", "暂无候选评分结果。")
            return
        dialog = AutoTuneResultDialog(self._last_auto_tune_result, self)
        dialog.exec()

    def apply_autotune_recipe_choice(self, recipe_result: dict):
        """Run the currently selected AutoTune workflow recipe.

        The recipe is converted to existing processing methods and executed by
        the standard ProcessingWorker path, so it inherits the same history,
        metadata, warning, and display handling as manual workflows.
        """
        if (
            self._ui_busy
            or self._worker is not None
            or self._auto_tune_worker is not None
            or self._auto_tune_stage_worker is not None
            or self._auto_tune_comparison_worker is not None
        ):
            QMessageBox.information(self, "自动推荐", "当前已有任务在运行，请稍候。")
            return False
        if self.data is None or self.data_path is None:
            QMessageBox.warning(self, "自动推荐", "请先导入数据。")
            return False

        try:
            from core.autotune_recipe_runner import build_recipe_processing_tasks

            tasks, plan = build_recipe_processing_tasks(
                recipe_result,
                out_dir=str(self._default_output_dir()),
            )
        except Exception as exc:
            QMessageBox.warning(self, "自动推荐", f"推荐方案转换失败：{exc}")
            return False

        if not tasks:
            QMessageBox.information(self, "自动推荐", "当前推荐方案没有可执行处理步骤。")
            return False

        self._push_history()
        for task in tasks:
            method_key = task.get("method_key")
            params = dict(task.get("params") or {})
            if method_key:
                self._method_param_overrides[str(method_key)] = dict(params)
                try:
                    self.page_basic.set_method_overrides(str(method_key), params)
                except Exception:
                    pass

        self._last_autotune_recipe_plan = plan.to_dict()
        scoring_record = dict(getattr(plan, "scoring_record", {}) or {})
        step_names = [task.get("method", {}).get("name", task.get("method_key")) for task in tasks]
        self._log(
            "运行自动推荐流程: "
            f"{plan.name} | {len(tasks)} 个可执行步骤 | "
            + " → ".join(str(name) for name in step_names)
        )
        self.page_basic.set_apply_source_hint("正在运行自动推荐流程。")
        self._start_processing_worker(
            tasks,
            run_type="recommended",
            run_label=plan.name,
            profile_key=plan.target_goal,
            execution_mode="sequential",
            run_metadata={
                "autotune_scoring_record": scoring_record,
                "autotune_recipe_plan": plan.to_dict(),
                "autotune_target_goal": plan.target_goal,
                "autotune_roi_mode": plan.roi_mode,
            },
        )
        return True


    def apply_stage_compare_choice(self):
        """将同阶段比较推荐的方法和参数写回日常处理。"""
        self._enforce_no_prior_action_guard(
            "preset_recommendation",
            dialog_title="同阶段推荐写回",
            allow_override=True,
            show_dialog=False,
            advisory_only=True,
        )

        if not self._last_auto_tune_group_result:
            QMessageBox.information(self, "同阶段比较", "暂无可用的阶段比较结果。")
            return

        method_key = self._last_auto_tune_group_result.get("best_method_key")
        best_auto = self._last_auto_tune_group_result.get("best_auto_tune_result") or {}
        params = (
            best_auto.get("recommended_params") or best_auto.get("best_params") or {}
        )
        if not method_key or not params:
            QMessageBox.information(
                self, "同阶段比较", "当前没有可写回的推荐方法参数。"
            )
            return

        self.page_basic.apply_method_params(method_key, dict(params))
        self._method_param_overrides[method_key] = dict(params)
        self.page_basic.set_apply_source_mode("auto_tune")
        self.page_basic.set_apply_source_hint(
            f"已采用同阶段推荐方法：{self._last_auto_tune_group_result.get('best_method_name', method_key)}"
        )
        self._log(
            f"已写回同阶段推荐方法：{self._last_auto_tune_group_result.get('best_method_name', method_key)}"
        )

    def _build_auto_tune_roi_spec(self, roi_mode: str) -> dict:
        return self.autotune_sync_controller._build_auto_tune_roi_spec(roi_mode)

    def _get_manual_roi_bounds(self) -> dict | None:
        return self.autotune_sync_controller._get_manual_roi_bounds()

    def _on_compare_toggled(self, checked: bool):
        """对比模式切换"""
        slider_checked = bool(
            hasattr(self.page_advanced, "slider_compare_var")
            and self.page_advanced.slider_compare_var.isChecked()
        )
        enabled = bool(checked or slider_checked)
        self.page_advanced.compare_left_combo.setEnabled(enabled)
        self.page_advanced.compare_right_combo.setEnabled(enabled)
        if hasattr(self.page_advanced, "compare_controls_row"):
            self.page_advanced.compare_controls_row.setVisible(enabled)
        if hasattr(self.page_advanced, "_refresh_compare_select_visibility"):
            self.page_advanced._refresh_compare_select_visibility()

    def _on_compare_mode_state_changed(self, *args):
        """Keep compare widgets, lineage compare and plot mode synchronized.

        V0.8.8: when compare/slider mode is closed from the display page,
        explicitly tear down lineage slider snapshots and force a single-image
        refresh so the main panel cannot remain visually stuck in slider compare.
        """
        if getattr(self, "_compare_syncing", False):
            return
        try:
            compare_checked = bool(getattr(self.page_advanced, "compare_var", None) and self.page_advanced.compare_var.isChecked())
            diff_checked = bool(getattr(self.page_advanced, "diff_var", None) and self.page_advanced.diff_var.isChecked())
            slider_checked = bool(getattr(self.page_advanced, "slider_compare_var", None) and self.page_advanced.slider_compare_var.isChecked())
            any_compare = bool(compare_checked or diff_checked or slider_checked)
            if not any_compare:
                if hasattr(self.page_advanced, "mode_single") and not self.page_advanced.mode_single.isChecked():
                    self.page_advanced.mode_single.setChecked(True)
                self._main_slider_compare_ratio = 0.5
                controller = getattr(self, "processing_lineage_controller", None)
                if controller is not None and hasattr(controller, "on_compare_mode_disabled"):
                    controller.on_compare_mode_disabled()
                self._last_plot_signature = None
            else:
                controller = getattr(self, "processing_lineage_controller", None)
                if controller is not None and hasattr(controller, "update_step_detail"):
                    controller.update_step_detail()
            self._on_compare_toggled(compare_checked)
            self._update_interaction_mode_status()
        except Exception:
            logger.debug("Compare mode state sync failed", exc_info=True)

    def _request_main_canvas_draw(
        self,
        reason: str = "main",
        *,
        min_interval_s: float | None = None,
        force: bool = False,
    ) -> None:
        """Coalesce high-frequency Matplotlib canvas paints.

        Mouse move, ROI drag and slider-compare gestures can emit many draw
        requests per second.  This helper keeps the UI responsive by merging
        requests into a bounded frame rate.  It changes only rendering cadence;
        numerical arrays and processing results are not touched.
        """
        interval = float(
            self._canvas_draw_min_interval_s
            if min_interval_s is None
            else max(0.0, min_interval_s)
        )
        now = time.perf_counter()
        monitor = getattr(self, "_perf_monitor", None)
        if monitor is not None:
            monitor.record(f"display.canvas_draw_request.{reason}", 0.0)
        if force or (now - float(self._last_canvas_draw_flush_ts)) >= interval:
            self._pending_canvas_draw = False
            try:
                self.canvas.draw_idle()
            finally:
                self._last_canvas_draw_flush_ts = now
                if monitor is not None:
                    monitor.record(f"display.canvas_draw_flush.{reason}", 0.0)
            return

        self._pending_canvas_draw = True
        remaining_ms = max(1, int(round((interval - (now - float(self._last_canvas_draw_flush_ts))) * 1000.0)))
        if not self._canvas_draw_timer.isActive():
            self._canvas_draw_timer.start(remaining_ms)

    def _flush_pending_canvas_draw(self) -> None:
        """Flush one coalesced canvas draw request."""
        if not getattr(self, "_pending_canvas_draw", False):
            return
        self._pending_canvas_draw = False
        self._last_canvas_draw_flush_ts = time.perf_counter()
        try:
            self.canvas.draw_idle()
        finally:
            monitor = getattr(self, "_perf_monitor", None)
            if monitor is not None:
                monitor.record("display.canvas_draw_flush.coalesced", 0.0)

    def _refresh_plot(self):
        """刷新绘图（带防抖）"""
        if self.data is None or self._compare_syncing:
            return
        self._plot_timer.start(30)

    def _do_refresh_plot(self):
        """执行刷新绘图"""
        if self.data is None:
            return
        signature = self._build_plot_signature()
        if signature == self._last_plot_signature:
            self._plot_skip_count += 1
            self._refresh_observability_panel()
            return
        self.plot_data(self.data)

    def _apply_main_plot_theme(self):
        """让主绘图区颜色跟随当前主题。"""
        from core.theme_manager import get_theme_manager

        theme = get_theme_manager().get_current_theme()
        try:
            from ui.theme import is_dark_ui

            effective_dark = is_dark_ui(theme, widget=self)
        except Exception:
            effective_dark = theme == "dark"

        if effective_dark:
            fig_face = "#111820"
            ax_face = "#151A21"
            text_color = "#EAF0F8"
            spine_color = "#3B4654"
        else:
            fig_face = "#ffffff"
            ax_face = "#ffffff"
            text_color = "#333333"
            spine_color = "#bbbbbb"

        self.fig.patch.set_facecolor(fig_face)
        for ax in self.fig.axes:
            ax.set_facecolor(ax_face)
            ax.tick_params(colors=text_color)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color(text_color)
            for spine in ax.spines.values():
                spine.set_color(spine_color)

    def _mark_data_changed(self):
        """标记数据已改变"""
        self._data_revision += 1
        self._view_cache.clear()
        if hasattr(self, "_vmin_vmax_cache"):
            self._vmin_vmax_cache.clear()
        self._clear_display_override()

    def _apply_preset_ui_values(
        self, ui_config: dict | None, preset_key: str | None = None
    ):
        """将预设中的 UI 选项同步到显示页控件。"""
        if not ui_config or self.page_advanced is None:
            return

        checkbox_fields = {
            "normalize": self.page_advanced.normalize_var,
            "demean": self.page_advanced.demean_var,
            "percentile": self.page_advanced.percentile_var,
        }
        for key, widget in checkbox_fields.items():
            if key in ui_config:
                old_block = widget.blockSignals(True)
                try:
                    widget.setChecked(bool(ui_config[key]))
                finally:
                    widget.blockSignals(old_block)

        text_fields = {
            "p_low": self.page_advanced.p_low_edit,
            "p_high": self.page_advanced.p_high_edit,
        }
        for key, widget in text_fields.items():
            if key in ui_config and ui_config[key] is not None:
                old_block = widget.blockSignals(True)
                try:
                    widget.setText(str(ui_config[key]))
                finally:
                    widget.blockSignals(old_block)

    def _apply_preset_method_params(self, method_params: dict | None):
        """将预设中的方法参数同步到基础页和内部覆盖表。"""
        if not method_params:
            return

        for method_key, params in method_params.items():
            resolved = dict(params or {})
            self._method_param_overrides[method_key] = resolved
            if hasattr(self, "page_basic") and self.page_basic is not None:
                self.page_basic.set_method_overrides(method_key, resolved)

        current_method_key = self.page_basic.get_current_method_key()
        if current_method_key in method_params:
            self.page_basic.apply_method_params(
                current_method_key, method_params[current_method_key]
            )

    def _apply_startup_preset_defaults(self):
        """应用启动时预设默认值"""
        self._selected_preset_key = DEFAULT_STARTUP_PRESET_KEY
        preset = GUI_PRESETS_V1.get(DEFAULT_STARTUP_PRESET_KEY)
        if not preset:
            return
        self._apply_preset_ui_values(
            preset.get("ui"), preset_key=DEFAULT_STARTUP_PRESET_KEY
        )
        self._apply_preset_method_params(preset.get("method_params"))

    def _refresh_observability_panel(self):
        """刷新可观测性面板"""
        self.obs_last_plot_label.setText(
            f"最近绘制耗时：{self._last_plot_ms:.2f} ms"
            if self._last_plot_ms
            else "最近绘制耗时：--"
        )
        self.obs_draw_count_label.setText(f"累计绘制次数：{self._plot_draw_count}")
        self.obs_skip_count_label.setText(f"累计跳过重绘：{self._plot_skip_count}")
        self.obs_last_prepare_label.setText(
            f"最近预处理耗时：{self._last_prepare_ms:.2f} ms"
            if self._last_prepare_ms
            else "最近预处理耗时：--"
        )

    def _update_empty_state_and_brief(self):
        """更新空状态面板和数据简介"""
        has_data = self.data is not None
        # 切换空状态卡片和绘图区。未导入数据时只保留导入引导，
        # 隐藏 B-scan 工具栏、链路条和底部读数，避免空状态看起来像半成品绘图页。
        self.plot_stack_host.layout().setCurrentIndex(1 if has_data else 0)
        for widget_name in ("_plot_toolbar_row", "_plot_stepper_bar", "_plot_bottom_status_bar"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setVisible(bool(has_data))
        self._sync_runtime_panels_visibility()
        self.page_basic.data_brief.setText(
            "未加载数据" if not has_data else self._build_data_brief_text()
        )
        # 更新状态栏
        if has_data and self.header_info:
            self.status_label.setText(self._build_status_text())
        elif has_data:
            self.status_label.setText(
                f"{os.path.basename(self.data_path) if self.data_path else 'data'} | shape={self.data.shape}"
            )
        else:
            self.status_label.setText("未加载文件")

        self._update_main_workspace_summary()

        if hasattr(self, "page_quality") and self.page_quality is not None:
            self.page_quality.set_line_summary(self._build_airborne_line_summary_text())
            self.page_quality.set_metadata_summary(
                "\n".join(self._build_airborne_metadata_summary())
            )
            self.page_quality.set_airborne_qc_summary(
                self._build_airborne_qc_summary_text()
            )
            self.page_quality.set_airborne_qc_visualization(
                self._build_airborne_qc_plot_payload()
            )
            self.page_quality.set_airborne_trajectory_visualization(
                self._build_airborne_trajectory_plot_payload()
            )
            if getattr(self, "page_terrain3d", None) is not None:
                self.page_terrain3d.set_airborne_georeference_3d_visualization(
                    self._build_airborne_georeference_3d_plot_payload()
                )
            self.page_quality.set_airborne_anomaly_details(
                self._build_airborne_anomaly_text()
            )

    def _build_airborne_metadata_summary(self) -> list[str]:
        return self.airborne_payload_controller._build_airborne_metadata_summary()

    def _build_status_text(self) -> str:
        """构建顶部状态栏文本。"""
        header = self.header_info or {}
        sample_count = header.get("a_scan_length")
        trace_count = header.get("num_traces")
        if (sample_count is None or trace_count is None) and self.data is not None:
            try:
                sample_count = sample_count if sample_count is not None else int(self.data.shape[0])
                trace_count = trace_count if trace_count is not None else int(self.data.shape[1])
            except Exception:
                pass
        base = (
            f"{os.path.basename(self.data_path) if self.data_path else 'data'} | "
            f"采样:{sample_count or '--'} 道数:{trace_count or '--'}"
        )
        if header.get("has_airborne_metadata"):
            base += f" | 距离:{float(header.get('track_length_m', 0.0)):.1f}m"
        return base

    def _build_data_brief_text(self) -> str:
        """构建基础流程页的数据摘要文本。

        Keep the drawer summary compact.  The detailed values remain available in
        the hover tooltip / log area, while this chip must not run underneath the
        scrollbar or the drawer edge in non-fullscreen windows.
        """
        if self.data is None:
            return "未加载数据"
        summary = [f"{self.data.shape[0]}×{self.data.shape[1]}"]
        if self.header_info and self.header_info.get("has_airborne_metadata"):
            summary.append(f"{float(self.header_info.get('track_length_m', 0.0)):.1f} m")
            summary.append(
                "高 {:.1f}-{:.1f} m".format(
                    float(self.header_info.get("flight_height_min_m", 0.0)),
                    float(self.header_info.get("flight_height_max_m", 0.0)),
                )
            )
        elif self.header_info:
            summary.append(f"{float(self.header_info.get('total_time_ns', 0.0)):.1f} ns")
        return " | ".join(summary)

    def _build_airborne_line_summary_text(self) -> str:
        return self.airborne_payload_controller._build_airborne_line_summary_text()

    def _build_airborne_qc_summary_text(self) -> str:
        return self.airborne_payload_controller._build_airborne_qc_summary_text()

    def _build_airborne_qc_plot_payload(self) -> dict | None:
        return self.airborne_payload_controller._build_airborne_qc_plot_payload()

    def _build_airborne_trajectory_plot_payload(self) -> dict | None:
        return self.airborne_payload_controller._build_airborne_trajectory_plot_payload()

    def _build_airborne_georeference_3d_plot_payload(self) -> dict | None:
        return self.airborne_payload_controller._build_airborne_georeference_3d_plot_payload()

    def _build_airborne_georeference_3d_payload_for(
        self,
        data,
        header_info,
        trace_metadata,
    ) -> dict | None:
        return self.airborne_payload_controller._build_airborne_georeference_3d_payload_for(
            data, header_info, trace_metadata
        )

    def _build_georeference_difference_data(self, raw, current) -> np.ndarray | None:
        return self.airborne_payload_controller._build_georeference_difference_data(raw, current)

    def _build_airborne_anomaly_details(self) -> list[dict]:
        return self.airborne_payload_controller._build_airborne_anomaly_details()

    def _build_airborne_anomaly_text(self) -> str:
        return self.airborne_payload_controller._build_airborne_anomaly_text()

    def _sync_runtime_panels_visibility(self):
        """同步运行时面板可见性"""
        has_data = self.data is not None
        if self._runtime_panel_bar is not None:
            self._runtime_panel_bar.setVisible(has_data)
        if not has_data:
            self._show_runtime_panel(None)
            if self._runtime_panel_container is not None:
                self._runtime_panel_container.setVisible(False)
        elif self._active_runtime_panel is not None:
            self._show_runtime_panel(self._active_runtime_panel)

    # ============ 数据加载 ============

    def load_csv(self):
        """加载数据（兼容旧接口，直接调用导入CSV）"""
        self.import_csv_file()

    def import_csv_file(self):
        """导入 GPR / CSV 数据文件。

        CSV 保持原有 sidecar 同步链路；其它常见 GPR profile 文件走
        ``auto_load_data`` 中央格式路由。
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 GPR 数据文件", "", supported_file_dialog_filter()
        )
        if not path:
            return
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".csv":
            sidecar_kwargs = self._build_sidecar_loader_kwargs(path)
            self._load_with_progress(
                "加载 CSV 文件", self._load_single_csv, path, **sidecar_kwargs
            )
            return
        self._load_with_progress(
            "加载 GPR 数据文件", self._load_common_gpr_file_with_progress, path
        )

    def import_ascans_folder(self):
        """导入 A-scan 文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择 A-scan 文件夹")
        if not folder:
            return
        self._load_with_progress("加载 A-scan 文件夹", self._load_ascans_folder, folder)

    def import_gprmax_out_file(self):
        """导入 gprMax .out 文件"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 gprMax .out 文件",
            "",
            "gprMax 输出文件 (*.out);;所有文件 (*)",
        )
        if not path:
            return
        self._load_with_progress("加载 gprMax .out 文件", self._load_gprmax_out, path)

    def _auto_load_last_data(self):
        """自动加载上次的数据"""
        last_path = _load_last_data_path()
        if not last_path:
            return

        # 检查文件是否存在
        if not os.path.exists(last_path):
            logger.info("Last data path no longer exists: %s", last_path)
            return

        # 显示加载提示并加载数据
        try:
            self._log(f"正在自动加载上次的数据: {os.path.basename(last_path)}")
            if os.path.isdir(last_path):
                self._load_with_progress(
                    "加载 A-scan 文件夹", self._load_ascans_folder, last_path
                )
            else:
                suffix = os.path.splitext(last_path)[1].lower()
                if suffix == ".csv":
                    self._load_with_progress(
                        "加载 CSV 文件", self._load_single_csv, last_path
                    )
                else:
                    self._load_with_progress(
                        "加载 GPR 数据文件", self._load_common_gpr_file_with_progress, last_path
                    )
        except Exception as e:
            logger.warning("Auto load last data failed: %s", e)
            self._log(f"自动加载上次数据失败: {e}")

    def _on_view_style_changed(self):
        """显示形式改变后持久化。"""
        self._save_view_style_to_settings()

    def _save_view_style_to_settings(self):
        """保存当前显示形式到设置文件。"""
        try:
            style = self.page_advanced.get_view_style()
            if style not in {"image", "wiggle"}:
                style = "image"
            settings = _load_app_settings_dict()
            settings["view_style"] = style
            _save_app_settings_dict(settings)
        except Exception as exc:
            logger.warning("Failed to save view style: %s", exc)

    def _restore_view_style_from_settings(self):
        """从设置文件恢复显示形式。"""
        try:
            settings = _load_app_settings_dict()
            style = str(settings.get("view_style", "image")).strip().lower()
            if style not in {"image", "wiggle"}:
                style = "image"
            if hasattr(self, "page_advanced") and self.page_advanced is not None:
                self.page_advanced.set_view_style(style)
        except Exception as exc:
            logger.warning("Failed to restore view style: %s", exc)
            if hasattr(self, "page_advanced") and self.page_advanced is not None:
                self.page_advanced.set_view_style("image")

    def _load_with_progress(self, title, loader_func, *args, **loader_kwargs):
        """使用进度条对话框加载数据"""
        dialog = LoadingProgressDialog(self, title)

        # 创建包装函数来支持进度回调
        def wrapped_loader(*args, progress_callback=None, **kwargs):
            # 修改原始加载函数，支持进度回调
            if loader_func == self._load_single_csv:
                return self._load_single_csv_with_progress(
                    args[0], progress_callback, **loader_kwargs
                )
            elif loader_func == self._load_ascans_folder:
                return self._load_ascans_folder_with_progress(
                    args[0], progress_callback
                )
            else:
                return loader_func(*args, **kwargs)

        dialog.start_loading(wrapped_loader, *args)
        dialog.exec()








    # ============ 绘图方法 ============

    def plot_data(self, data: np.ndarray):
        """绘制数据"""
        start_ts = time.perf_counter()
        self._last_plot_signature = self._build_plot_signature()
        self._apply_main_plot_theme()

        view_data, view_header_info, view_trace_metadata = (
            self._get_active_plot_payload(data)
        )
        if view_data is None:
            return
        display_data, bounds, axis_info = self._prepare_view_data(
            view_data,
            header_info_override=view_header_info,
            trace_metadata_override=view_trace_metadata,
        )
        self._last_display_data = np.asarray(display_data, dtype=np.float32)
        self._last_display_time_axis = np.asarray(
            axis_info.get("time_axis", []), dtype=np.float32
        )
        self._last_display_trace_axis = np.asarray(
            axis_info.get("trace_axis", []), dtype=np.float32
        )
        self._last_display_trace_indices = np.asarray(
            axis_info.get("trace_indices", []), dtype=np.int32
        )
        plot_config = self._resolve_plot_extent_and_labels(
            display_data,
            bounds,
            axis_info,
            header_info_override=view_header_info,
        )
        extent = plot_config["extent"]
        self._last_plot_extent = extent
        cmap = self._get_colormap(view_header_info)

        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception as e:
                logger.debug("Failed to remove main colorbar: %s", e)
            self.cbar = None

        view_style = self.page_advanced.get_view_style()
        try:
            cmap_label = str(self._get_colormap(view_header_info))
        except Exception:
            cmap_label = "默认"
        if getattr(self, "_plot_display_mode_chip", None) is not None:
            style_label = {"image": "图像", "wiggle": "摆动图"}.get(str(view_style), str(view_style))
            self._plot_display_mode_chip.setText(f"显示：{style_label}")
        if getattr(self, "_plot_colormap_chip", None) is not None:
            self._plot_colormap_chip.setText(f"色图：{cmap_label}")
        if getattr(self, "_plot_range_chip", None) is not None:
            range_label = "百分位" if self.page_advanced.percentile_var.isChecked() else "自动"
            if view_header_info and view_header_info.get("display_fixed_unit_range"):
                range_label = "固定 ±1"
            self._plot_range_chip.setText(f"拉伸：{range_label}")
        slider_compare = self._is_main_slider_compare_active()
        if not slider_compare:
            self._slider_compare_render_cache = {}
        if slider_compare and view_style == "wiggle":
            # 绘图刷新可能很频繁；该提示只作为状态展示，不写入全局日志，避免日志被刷新路径污染。
            self._set_runtime_summary("状态：滑动对比优先使用图像分割显示", "info")

        if slider_compare:
            n_panels = 1
        else:
            data_pairs = self._build_compare_data_pairs(
                display_data, header_info_override=view_header_info
            )
            n_panels = len(data_pairs)

        if n_panels != getattr(self, "_last_n_panels", None):
            self.fig.clear()
            self._last_n_panels = n_panels

        axes = self._get_or_create_plot_axes(n_panels)
        self._main_plot_axes = list(axes)
        self._clear_hover_crosshair_artists(draw=False)
        self._clear_axes_artists(axes)

        if slider_compare:
            last_im = self._render_slider_compare_panel(
                axes[0],
                display_data,
                axis_info,
                plot_config,
                cmap,
                header_info_override=view_header_info,
            )
        elif view_style == "wiggle":
            last_im = self._render_wiggle_pairs(
                axes,
                data_pairs,
                axis_info,
                plot_config,
            )
        else:
            last_im = self._render_data_pairs(
                axes,
                data_pairs,
                cmap,
                extent,
                plot_config,
                header_info_override=view_header_info,
            )
        # Main B-scan selection-by-click and hover crosshair overlays have been
        # retired; coordinate readout provides trace/sample inspection without
        # leaving persistent selection markers on the plot.
        self._draw_manual_roi_marker(axes, axis_info)
        if self._main_view_limits and len(axes) == 1 and not slider_compare:
            axes[0].set_xlim(*self._main_view_limits["xlim"])
            axes[0].set_ylim(*self._main_view_limits["ylim"])

        if last_im is not None and view_style != "wiggle":
            self._draw_colorbar_if_needed(
                last_im, axes, header_info_override=view_header_info
            )
        self._polish_main_figure()
        self._request_main_canvas_draw("plot_data", force=True)
        self._update_processing_lineage_display()

        elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
        self._plot_draw_count += 1
        self._last_plot_ms = elapsed_ms
        monitor = getattr(self, "_perf_monitor", None)
        if monitor is not None:
            monitor.record("display.plot_data_ms", elapsed_ms)
        self._refresh_observability_panel()

    # ============ 处理方法 ============

    def apply_method(self):
        """应用当前选中的方法"""
        if self.data is None or self.data_path is None:
            try:
                self.page_basic.set_apply_button_state("error", "未加载数据：请先导入一条 GPR 测线。")
                self._set_runtime_summary("状态：未加载数据", "warning")
            except Exception:
                pass
            QMessageBox.warning(self, "无数据", "请先导入数据。")
            return
        idx = self.page_basic.method_combo.currentIndex()
        method_key = self.page_basic.method_keys[idx]
        method = PROCESSING_METHODS[method_key]
        self._log(f"正在应用: {method['name']}")

        try:
            visible_params = self.page_basic.get_current_params()
        except ValueError as e:
            self.page_basic.set_apply_button_state("error", f"参数错误：{e}")
            self._set_runtime_summary("状态：参数错误", "danger")
            self._log(f"参数错误: {e}")
            QMessageBox.critical(self, "参数错误", str(e))
            return

        # 基础页会隐藏部分高级参数；运行时需要把隐藏参数的默认值/覆盖值一并带上。
        params = self._resolve_method_params(method_key)
        params.update(visible_params)

        method_action = self._classify_method_guard_action(method_key, params)
        if method_action:
            self._enforce_no_prior_action_guard(
                method_action,
                dialog_title="应用方法",
                allow_override=True,
                show_dialog=False,
                advisory_only=True,
            )

        self._push_history()
        self._method_param_overrides[method_key] = dict(params)
        self.page_basic.set_method_overrides(method_key, params)

        out_dir = self._default_output_dir()
        task = {
            "method_key": method_key,
            "method": method,
            "params": params,
            "out_dir": out_dir,
        }
        self._start_processing_worker([task], run_type="single")

    def run_default_pipeline(self):
        """运行默认处理流程"""
        if self.data is None or self.data_path is None:
            QMessageBox.warning(self, "无数据", "请先导入 CSV。")
            return
        if not self._enforce_no_prior_action_guard(
            "workflow_run",
            dialog_title="运行默认流程",
            allow_override=True,
        ):
            return
        try:
            current_method_key = self.page_basic.get_current_method_key()
            visible_params = self.page_basic.get_current_params()
        except ValueError as e:
            QMessageBox.critical(self, "参数错误", str(e))
            return

        if current_method_key:
            merged_params = self._resolve_method_params(current_method_key)
            merged_params.update(visible_params)
            self._method_param_overrides[current_method_key] = dict(merged_params)
            self.page_basic.set_method_overrides(current_method_key, merged_params)

        source_mode = self.page_basic.get_apply_source_mode()
        profile_key = recommended_profile_for_header(self.header_info)
        profile = RECOMMENDED_RUN_PROFILES.get(profile_key, {})
        preset_key = profile.get("preset_key")
        if preset_key:
            self._apply_preset_by_key(preset_key)
        self._apply_preset_method_params(profile.get("method_params"))
        if current_method_key:
            merged_params = self._resolve_method_params(current_method_key)
            merged_params.update(visible_params)
            self._method_param_overrides[current_method_key] = dict(merged_params)
            self.page_basic.set_method_overrides(current_method_key, merged_params)

        self._log(
            f"运行默认高质量流程（{profile.get('label', profile_key)}）："
            + " → ".join(profile.get("order", []))
        )
        order = list(profile.get("order", []))
        current_idx = self.page_basic.method_combo.currentIndex()
        tasks = []
        out_dir = self._default_output_dir()
        for key in order:
            if key in self.page_basic.method_keys:
                tasks.append(
                    self._build_single_task(key, out_dir, param_source_mode=source_mode)
                )
        if not tasks:
            return
        self._push_history()
        self._start_processing_worker(
            tasks, run_type="pipeline", restore_method_idx=current_idx
        )

    def run_recommended_pipeline(self, profile_key: str):
        """运行推荐处理流程"""
        if self.data is None or self.data_path is None:
            QMessageBox.warning(self, "无数据", "请先导入 CSV。")
            return
        if not self._enforce_no_prior_action_guard(
            "preset_recommendation",
            dialog_title="运行推荐流程",
            allow_override=False,
        ):
            return
        profile = RECOMMENDED_RUN_PROFILES.get(profile_key)
        if not profile:
            QMessageBox.warning(self, "配置错误", f"未知推荐配置：{profile_key}")
            return

        preset_key = profile.get("preset_key")
        if preset_key:
            self._apply_preset_by_key(preset_key)
        self._apply_preset_method_params(profile.get("method_params"))

        out_dir = self._default_output_dir()
        current_idx = self.page_basic.method_combo.currentIndex()
        tasks = self._build_tasks_from_order(profile.get("order", []), out_dir)
        if not tasks:
            QMessageBox.warning(self, "无任务", "推荐处理链为空。")
            return

        self._log(f"运行推荐处理链：{profile.get('label', profile_key)}")
        self._push_history()
        self._start_processing_worker(
            tasks,
            run_type="recommended",
            restore_method_idx=current_idx,
            run_label=profile.get("label", profile_key),
            preset_key=preset_key,
            profile_key=profile_key,
        )

    # ============ 报告和导出 ============




















































    # ============ 辅助方法（由于篇幅限制，这里只列出关键方法） ============






















    # ============ 绘图辅助方法 ============







































    # ============ 对比和质量方法 ============

    _MAX_SNAPSHOTS = 8

    def _build_formal_compare_snapshots(self):
        """从共享状态构建正式对比快照。"""
        if self.shared_data is None:
            return []
        return self.shared_data.build_formal_compare_snapshots()

    def _make_unique_compare_label(
        self, base_label: str, existing_labels: set[str]
    ) -> str:
        """为临时对比结果生成不冲突的标签。"""
        base = str(base_label or "结果")
        if base not in existing_labels:
            return base

        candidate = f"{base}（对比）"
        index = 2
        while candidate in existing_labels:
            candidate = f"{base}（对比{index}）"
            index += 1
        return candidate

    def _refresh_compare_snapshots_from_state(self, clear_transient: bool = False):
        """根据共享状态重建正式对比快照，并按需附加临时结果。"""
        if clear_transient:
            self._transient_compare_snapshots = []

        self.compare_snapshots = self._build_formal_compare_snapshots() + [
            {
                "label": snap["label"],
                "data": np.array(snap["data"], copy=False),
                "trace_metadata": snap.get("trace_metadata"),
                "header_info": snap.get("header_info"),
                "source": snap.get("source"),
                "source_index": snap.get("source_index"),
            }
            for snap in self._transient_compare_snapshots
        ]
        self._update_compare_combo_items()

    def _clear_transient_compare_snapshots(self):
        """清除实验性临时对比结果。"""
        self._refresh_compare_snapshots_from_state(clear_transient=True)

    def _update_current_compare_snapshot(self):
        """更新当前对比快照"""
        self._refresh_compare_snapshots_from_state()

    def _set_compare_snapshots(self, snapshots: list):
        """设置临时对比快照。"""
        formal_labels = {
            snap["label"] for snap in self._build_formal_compare_snapshots()
        }
        transient = []
        for snap in snapshots:
            label = self._make_unique_compare_label(
                snap.get("label", "结果"), formal_labels
            )
            formal_labels.add(label)
            transient.append(
                {
                    "label": label,
                    "data": np.array(snap["data"], copy=False),
                    "trace_metadata": snap.get("trace_metadata"),
                    "header_info": snap.get("header_info"),
                    "source": snap.get("source"),
                    "source_index": snap.get("source_index"),
                }
            )

        if len(transient) > self._MAX_SNAPSHOTS:
            transient = transient[-self._MAX_SNAPSHOTS :]

        self._transient_compare_snapshots = transient
        self._refresh_compare_snapshots_from_state()

    def _clone_current_trace_metadata(self):
        meta = self.trace_metadata or {}
        return {k: np.array(v, copy=True) for k, v in meta.items()} if meta else None

    def _update_compare_combo_items(self):
        """更新对比下拉框选项"""
        start_ts = time.perf_counter()
        labels = [s["label"] for s in self.compare_snapshots]
        labels_tuple = tuple(labels)
        combo_widgets = (
            self.page_advanced.single_view_combo,
            self.page_advanced.compare_left_combo,
            self.page_advanced.compare_right_combo,
        )
        try:
            existing = tuple(
                combo_widgets[0].itemText(i) for i in range(combo_widgets[0].count())
            )
            if labels_tuple == getattr(self, "_last_compare_combo_labels", ()) and existing == labels_tuple:
                monitor = getattr(self, "_perf_monitor", None)
                if monitor is not None:
                    monitor.record("display.compare_combo_skip_ms", (time.perf_counter() - start_ts) * 1000.0)
                return
        except Exception:
            pass

        self._compare_syncing = True
        current_single = self.page_advanced.single_view_combo.currentText()
        current_left = self.page_advanced.compare_left_combo.currentText()
        current_right = self.page_advanced.compare_right_combo.currentText()
        self.page_advanced.single_view_combo.clear()
        self.page_advanced.compare_left_combo.clear()
        self.page_advanced.compare_right_combo.clear()
        self.page_advanced.single_view_combo.addItems(labels)
        self.page_advanced.compare_left_combo.addItems(labels)
        self.page_advanced.compare_right_combo.addItems(labels)
        # 保持用户选择（如果仍有效）
        if current_single in labels:
            self.page_advanced.single_view_combo.setCurrentText(current_single)
        elif labels:
            self.page_advanced.single_view_combo.setCurrentIndex(len(labels) - 1)
        if current_left in labels:
            self.page_advanced.compare_left_combo.setCurrentText(current_left)
        if current_right in labels:
            self.page_advanced.compare_right_combo.setCurrentText(current_right)
        # 首次设置或用户未选择时：左=原始，右=最新
        if (
            self.page_advanced.compare_left_combo.currentText() == ""
            and "原始" in labels
        ):
            self.page_advanced.compare_left_combo.setCurrentText("原始")
        if (
            self.page_advanced.compare_right_combo.currentText() == ""
            and len(labels) >= 2
        ):
            self.page_advanced.compare_right_combo.setCurrentIndex(len(labels) - 1)
        self._compare_syncing = False
        self._last_compare_combo_labels = labels_tuple
        monitor = getattr(self, "_perf_monitor", None)
        if monitor is not None:
            monitor.record("display.compare_combo_refresh_ms", (time.perf_counter() - start_ts) * 1000.0)
















    # ============ 工作线程管理 ============







# ... [其他辅助方法将在这里继续]


def apply_theme(app: QApplication):
    """应用主题：qfluentwidgets 主题 + 自定义 QSS 叠加"""
    from core.theme_manager import get_theme_manager

    return get_theme_manager().apply_app_theme(app)


def main():
    log_path = configure_logging()
    app = QApplication(sys.argv)
    theme_name = apply_theme(app)
    qt_font_name = _configure_qt_cjk_font(app)
    version_text = build_version_string("MyGPR")
    logger.info("MyGPR version=%s", version_text)
    win = GPRGuiQt(version_text=version_text)
    logger.info("Runtime log file: %s", log_path)
    win.statusBar().showMessage(
        f"Theme: {theme_name} | QtFont: {qt_font_name} | {version_text}"
    )
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
