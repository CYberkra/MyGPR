#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPR GUI (PyQt6 + themed) - 主入口模块

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
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle

from PyQt6.QtCore import Qt, QObject, QThread, QTimer, pyqtSignal
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
    QTextEdit,
    QScrollArea,
    QSizePolicy,
    QProgressBar,
)

from core.app_paths import get_logs_dir, get_output_dir, get_settings_dir

# 确保本地目录在路径中
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logger = logging.getLogger(__name__)


def configure_logging() -> str:
    """统一配置应用日志。"""
    log_dir = get_logs_dir()
    log_path = os.path.join(log_dir, "gpr_gui.log")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    default_level = logging.DEBUG if os.getenv("GPR_GUI_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"} else logging.INFO
    root_logger.setLevel(default_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
    for noisy_logger in [
        "matplotlib",
        "matplotlib.font_manager",
        "fontTools",
        "PIL",
        "numexpr",
        "PyQt6.uic",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    warnings.filterwarnings(
        "ignore",
        message=r"Glyph .* missing from font\(s\).*",
        category=UserWarning,
    )

    return log_path


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
    compute_quality_metrics,
)
from core.gpr_io import (
    extract_airborne_csv_payload,
    read_ascans_folder,
    subset_trace_metadata,
)
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
from core.shared_data_state import SharedDataState
from PythonModule.kirchhoff_migration import load_cagpr_kir_parameter_file
from qfluentwidgets import FluentIcon

# 导入页面模块
from ui.gui_basic_flow import BasicFlowPage
from ui.gui_auto_tune_page import AutoTunePage
from ui.gui_advanced_settings import AdvancedSettingsPage
from ui.gui_quality_log import QualityLogPage

# 导入新的工作台页面
from ui.gui_workbench import WorkbenchPage
from ui.loading_dialog import LoadingProgressDialog
from ui.auto_tune_result_dialog import AutoTuneResultDialog


def _sanitize_qss(qss: str) -> str:
    """清理 Qt 不支持或会触发布局警告的样式声明。"""
    cleaned_lines = []
    for line in qss.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("cursor:"):
            continue
        if stripped.startswith("max-width: 16777215"):
            continue
        if stripped.startswith("max-height: 16777215"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _get_settings_path() -> str:
    """获取设置文件路径"""
    settings_dir = get_settings_dir()
    return os.path.join(settings_dir, "gpr_gui_settings.json")


def _save_last_data_path(path: str):
    """保存上次加载的数据路径"""
    try:
        # 只保存有效的文件路径
        if not os.path.exists(path):
            return
        settings_path = _get_settings_path()
        settings = {}
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
            except (OSError, json.JSONDecodeError):
                settings = {}
        settings["last_data_path"] = path
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Failed to save last data path: %s", e)


def _load_last_data_path() -> str:
    """加载上次的数据路径"""
    try:
        settings_path = _get_settings_path()
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
                path = settings.get("last_data_path", "")
                # 验证路径是否存在
                if path and os.path.exists(path):
                    return path
    except Exception as e:
        logger.warning("Failed to load last data path: %s", e)
    return ""


class ProcessingWorker(QObject):
    """后台处理工作线程"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    def __init__(
        self,
        base_data: np.ndarray,
        tasks: list,
        base_csv_path: str = None,
        execution_mode: str = "sequential",
        header_info: dict | None = None,
        trace_metadata: dict | None = None,
    ):
        super().__init__()
        self.base_data = np.array(base_data, copy=True)
        self.tasks = tasks
        self.base_csv_path = base_csv_path
        self.header_info = header_info or {}
        self.trace_metadata = trace_metadata or {}
        self._cancel_requested = False
        self.execution_mode = execution_mode

    def request_cancel(self):
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested)

    @staticmethod
    def _compact_result_meta(result_meta: dict | None) -> dict:
        """压缩工作线程中保留的结果元信息，避免重复挂大数组。"""
        compact_meta = dict(result_meta or {})
        compact_meta.pop("display_data", None)
        compact_meta.pop("display_header_info_updates", None)
        compact_meta.pop("display_trace_metadata", None)
        for key in list(compact_meta.keys()):
            if isinstance(compact_meta.get(key), np.ndarray):
                compact_meta.pop(key, None)
        return compact_meta

    def run(self):
        current_data = np.array(self.base_data, copy=True)
        current_header_info = merge_result_header_info(
            self.header_info, None, current_data.shape
        )
        current_trace_metadata = merge_result_trace_metadata(self.trace_metadata, None)
        current_display_data = None
        current_display_header_info = None
        current_display_trace_metadata = None
        outputs = []
        total = len(self.tasks)
        current_method_name = "未知方法"
        try:
            for i, task in enumerate(self.tasks, start=1):
                if self.is_cancel_requested():
                    raise ProcessingCancelled("用户已取消处理")

                step_t0 = time.perf_counter()
                method_key = task["method_key"]
                method = task["method"]
                current_method_name = method.get("name", method_key)
                params = dict(task.get("params", {}))
                param_source_mode = str(task.get("param_source_mode") or "manual")
                out_dir = task["out_dir"]

                self.progress.emit(
                    i - 1, total, f"开始步骤 {i}/{total}: {method['name']}"
                )

                task_input = (
                    self.base_data
                    if self.execution_mode == "independent"
                    else current_data
                )
                if param_source_mode == "auto_tune":
                    tuned_result = auto_tune_method(
                        task_input,
                        method_key,
                        header_info=current_header_info,
                        trace_metadata=current_trace_metadata,
                        base_params=params,
                        cancel_checker=self.is_cancel_requested,
                    )
                    profile_key = str(
                        tuned_result.get("recommended_profile", "balanced")
                    )
                    profile = (tuned_result.get("profiles", {}) or {}).get(
                        profile_key, {}
                    )
                    tuned_params = dict(
                        profile.get("params")
                        or tuned_result.get("recommended_params")
                        or tuned_result.get("best_params")
                        or {}
                    )
                    params.update(tuned_params)
                runtime_params = prepare_runtime_params(
                    method_key,
                    params,
                    current_header_info,
                    current_trace_metadata,
                    task_input.shape,
                )

                newdata, result_meta = run_processing_method(
                    task_input,
                    method_key,
                    runtime_params,
                    cancel_checker=self.is_cancel_requested,
                )
                display_data = result_meta.get("display_data")
                display_header_info = None
                if display_data is not None:
                    display_header_info = merge_result_header_info(
                        current_header_info,
                        {
                            "header_info_updates": result_meta.get(
                                "display_header_info_updates"
                            )
                        },
                        np.asarray(display_data).shape,
                    )
                resolved_header_info = merge_result_header_info(
                    current_header_info, result_meta, newdata.shape
                )
                resolved_trace_metadata = merge_result_trace_metadata(
                    current_trace_metadata, result_meta
                )

                if self.execution_mode != "independent":
                    current_data = newdata
                    current_header_info = resolved_header_info
                    current_trace_metadata = resolved_trace_metadata
                    current_display_data = (
                        np.array(display_data, copy=False)
                        if display_data is not None
                        else None
                    )
                    current_display_header_info = display_header_info
                    current_display_trace_metadata = result_meta.get(
                        "display_trace_metadata"
                    )
                elapsed_ms = (time.perf_counter() - step_t0) * 1000.0
                compact_meta = self._compact_result_meta(result_meta)
                outputs.append(
                    {
                        "method_key": method_key,
                        "method_name": method["name"],
                        "params": dict(runtime_params),
                        "param_source_mode": param_source_mode,
                        "elapsed_ms": elapsed_ms,
                        "data": np.array(newdata, copy=False),
                        "meta": compact_meta,
                        "runtime_warnings": compact_meta.get("runtime_warnings", []),
                        "header_info": resolved_header_info,
                    }
                )
                self.progress.emit(
                    i,
                    total,
                    f"完成步骤 {i}/{total}: {method['name']} ({elapsed_ms:.1f} ms)",
                )

            if self.is_cancel_requested():
                self.finished.emit(
                    {
                        "outputs": outputs,
                        "final_data": current_data,
                        "final_header_info": current_header_info,
                        "final_trace_metadata": current_trace_metadata,
                        "final_display_data": current_display_data,
                        "final_display_header_info": current_display_header_info,
                        "final_display_trace_metadata": current_display_trace_metadata,
                        "cancelled": True,
                        "execution_mode": self.execution_mode,
                    }
                )
            else:
                self.finished.emit(
                    {
                        "outputs": outputs,
                        "final_data": current_data,
                        "final_header_info": current_header_info,
                        "final_trace_metadata": current_trace_metadata,
                        "final_display_data": current_display_data,
                        "final_display_header_info": current_display_header_info,
                        "final_display_trace_metadata": current_display_trace_metadata,
                        "execution_mode": self.execution_mode,
                    }
                )
        except ProcessingCancelled:
            self.finished.emit(
                {
                    "outputs": outputs,
                    "final_data": current_data,
                    "final_header_info": current_header_info,
                    "final_trace_metadata": current_trace_metadata,
                    "final_display_data": current_display_data,
                    "final_display_header_info": current_display_header_info,
                    "final_display_trace_metadata": current_display_trace_metadata,
                    "cancelled": True,
                    "execution_mode": self.execution_mode,
                }
            )
        except Exception as e:
            self.error.emit(build_processing_error_message(e, current_method_name))


class AutoTuneWorker(QObject):
    """后台自动选参工作线程。"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    def __init__(
        self,
        data: np.ndarray,
        method_key: str,
        base_params: dict[str, object],
        header_info: dict | None = None,
        trace_metadata: dict | None = None,
        roi_spec: dict | None = None,
        search_mode: str = "standard",
    ):
        super().__init__()
        self.data = np.array(data, copy=True)
        self.method_key = method_key
        self.base_params = dict(base_params or {})
        self.header_info = header_info or {}
        self.trace_metadata = trace_metadata or {}
        self.roi_spec = dict(roi_spec or {})
        self.search_mode = str(search_mode or "standard")
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested)

    def run(self):
        try:
            result = auto_tune_method(
                self.data,
                self.method_key,
                header_info=self.header_info,
                trace_metadata=self.trace_metadata,
                base_params=self.base_params,
                roi_spec=self.roi_spec,
                search_mode=self.search_mode,
                progress_callback=lambda current, total, message: self.progress.emit(
                    int(current), int(total), str(message)
                ),
                cancel_checker=self.is_cancel_requested,
            )
            result["cancelled"] = self.is_cancel_requested()
            self.finished.emit(result)
        except AutoTuneCancelled:
            self.finished.emit(
                {
                    "method_key": self.method_key,
                    "cancelled": True,
                    "all_trials": [],
                }
            )
        except Exception as e:
            self.error.emit(str(e))


class AutoTuneStageWorker(QObject):
    """后台同阶段方法比较工作线程。"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    def __init__(
        self,
        data: np.ndarray,
        method_keys: list[str],
        base_params_map: dict[str, dict[str, object]],
        header_info: dict | None = None,
        trace_metadata: dict | None = None,
        roi_spec: dict | None = None,
        search_mode: str = "standard",
    ):
        super().__init__()
        self.data = np.array(data, copy=True)
        self.method_keys = list(method_keys)
        self.base_params_map = dict(base_params_map or {})
        self.header_info = header_info or {}
        self.trace_metadata = trace_metadata or {}
        self.roi_spec = dict(roi_spec or {})
        self.search_mode = str(search_mode or "standard")
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested)

    def run(self):
        try:
            result = auto_select_method_group(
                self.data,
                self.method_keys,
                header_info=self.header_info,
                trace_metadata=self.trace_metadata,
                base_params_map=self.base_params_map,
                roi_spec=self.roi_spec,
                search_mode=self.search_mode,
                progress_callback=lambda current, total, message: self.progress.emit(
                    int(current), int(total), str(message)
                ),
                cancel_checker=self.is_cancel_requested,
            )
            result["cancelled"] = self.is_cancel_requested()
            self.finished.emit(result)
        except AutoTuneCancelled:
            self.finished.emit({"cancelled": True, "candidates": []})
        except Exception as e:
            self.error.emit(str(e))


class GPRGuiQt(QMainWindow):
    """GPR GUI主窗口"""

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
        self.version_text = version_text.strip() or "GPR_GUI"
        self.setWindowTitle(self.version_text)
        self.resize(1280, 800)
        self.setMinimumSize(1120, 720)

        self.shared_data = SharedDataState(self)
        self.shared_data.changed.connect(self._on_shared_data_changed)

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
        self._pending_apply_after_auto_tune = False
        self._current_run_context = None
        self._cancel_in_flight = False
        self._last_auto_tune_result = None
        self._last_auto_tune_group_result = None

        # 缓存和状态
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.timeout.connect(self._do_refresh_plot)
        self._ds_cache = {}
        self._view_cache = {}
        self.compare_snapshots = []
        self._transient_compare_snapshots = []
        self._compare_syncing = False
        self._data_revision = 0
        self._last_plot_signature = None
        self._plot_debug_metrics = os.getenv(
            "GPR_GUI_PLOT_DEBUG", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._plot_skip_count = 0
        self._plot_draw_count = 0
        self._last_plot_ms = 0.0
        self._last_prepare_ms = None
        self._last_compare_ms = None
        self._last_n_panels = 1

        # 参数覆盖
        self._method_param_overrides = {}
        self._selected_preset_key = None
        self._last_stolt_adaptive_stats = None
        self._last_stolt_adaptive_reason = ""
        self._last_quality_metrics = None
        self._last_run_summary = None
        self._runtime_warnings = []
        self._sidecar_files = {"rtk": None, "imu": None, "altimeter": None}
        self._trace_timestamps_s = None
        self._quality_thresholds = load_quality_dashboard_thresholds()
        self._ui_busy = False
        self._display_data_override = None
        self._display_header_info_override = None
        self._display_trace_metadata_override = None
        self._selected_trace_index = None
        self._manual_roi_values = None
        self._drag_roi_preview_patch = None
        self._main_view_limits = None
        self._main_press_state = None
        self._main_drag_threshold_px = 8
        self._last_display_trace_axis = np.array([], dtype=np.float32)
        self._last_display_trace_indices = np.array([], dtype=np.int32)
        self._last_display_time_axis = np.array([], dtype=np.float32)
        self._last_display_data = None
        self._last_plot_extent = None
        self._main_plot_axes = []
        self._main_slider_compare_ratio = 0.5

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

    def _setup_ui(self):
        """设置UI"""
        central = QWidget()
        self.setCentralWidget(central)

        self._content_stack = QStackedLayout(central)
        self._content_stack.setContentsMargins(0, 0, 0, 0)
        self._content_stack.setStackingMode(QStackedLayout.StackingMode.StackOne)

        self._main_content_widget = QWidget()
        root_layout = QHBoxLayout(self._main_content_widget)
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("mainSplitter")
        self._main_splitter = splitter
        root_layout.addWidget(splitter)

        # 右侧面板（绘图区）
        right_panel = QWidget()
        self._right_panel = right_panel
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(8)

        # 左侧面板（控制区）
        left_shell = QWidget()
        self._left_shell = left_shell
        left_shell_layout = QVBoxLayout(left_shell)
        left_shell_layout.setContentsMargins(0, 0, 0, 0)
        left_shell_layout.setSpacing(8)

        left_panel = QWidget()
        self._left_panel = left_panel
        left_panel.setMinimumWidth(320)
        left_panel.setMaximumWidth(560)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)

        scroll = QScrollArea()
        self._left_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(left_panel)
        left_shell_layout.addWidget(scroll, 1)

        splitter.addWidget(right_panel)
        splitter.addWidget(left_shell)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([940, 420])

        # 创建多页控制面板
        self.control_tabs = QTabWidget()
        self.control_tabs.setDocumentMode(True)
        self.control_tabs.setUsesScrollButtons(True)
        self.control_tabs.setMovable(False)
        self.control_tabs.tabBar().setElideMode(Qt.TextElideMode.ElideRight)
        left_layout.addWidget(self.control_tabs)

        # 返回工作台按钮
        from qfluentwidgets import PushButton

        self.btn_return_workbench = PushButton("返回工作台")
        self.btn_return_workbench.clicked.connect(self.switch_to_workbench_mode)
        self.btn_return_workbench.setProperty("class", "successBtn")
        left_shell_layout.addWidget(self.btn_return_workbench)

        self._content_stack.addWidget(self._main_content_widget)

        # ===== 原有页面（保留作为日常处理界面）=====
        # 页面1: 日常处理
        self.page_basic = BasicFlowPage(self)
        idx_basic = self.control_tabs.addTab(
            self.page_basic, FluentIcon.HOME.icon(), "日常处理"
        )
        self.control_tabs.setTabToolTip(idx_basic, "日常连续处理操作")

        # 页面2: 调参与实验
        self.page_auto_tune = AutoTunePage(self)
        idx_auto_tune = self.control_tabs.addTab(
            self.page_auto_tune, FluentIcon.SETTING.icon(), "调参与实验"
        )
        self.control_tabs.setTabToolTip(idx_auto_tune, "自动选参、候选评估与方法实验")

        # 页面3: 显示与对比
        self.page_advanced = AdvancedSettingsPage(self)
        idx_advanced = self.control_tabs.addTab(
            self.page_advanced, FluentIcon.VIEW.icon(), "显示与对比"
        )
        self.control_tabs.setTabToolTip(
            idx_advanced, "主图显示、双图对比、裁剪与预览设置"
        )

        # 页面4: 质量
        self.page_quality = QualityLogPage(self)
        self.page_quality.set_trace_selected_callback(
            self._on_trajectory_trace_selected
        )
        idx_quality = self.control_tabs.addTab(
            self.page_quality, FluentIcon.PIE_SINGLE.icon(), "质量与导出"
        )
        self.control_tabs.setTabToolTip(idx_quality, "质量摘要、航迹图、日志与导出入口")

        # ===== 工作台（总控中心 - 独立页面）=====
        self.page_workbench = WorkbenchPage(self, self.shared_data)
        # 添加到 _content_stack，作为最后一个页面
        self._content_stack.addWidget(self.page_workbench)

        # 默认显示主内容区（日常处理界面）
        self._content_stack.setCurrentWidget(self._main_content_widget)
        self.control_tabs.setCurrentWidget(self.page_basic)
        self._relocate_basic_status_brief()
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

        # 绘图区域
        self.fig = Figure(figsize=(9.5, 6.4), dpi=100)
        self._main_ax = self.fig.add_subplot(111)
        self._main_ax.set_title("B-scan")
        self._main_ax.set_xlabel("距离（道索引）")
        self._main_ax.set_ylabel("时间（采样索引）")
        self.canvas = FigureCanvas(self.fig)
        self._main_toolbar = NavigationToolbar(self.canvas, self)
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
        self.canvas.mpl_connect("figure_leave_event", self._on_main_canvas_leave)
        self._last_n_panels = 1

        plot_toolbar_row = QWidget()
        plot_toolbar_layout = QHBoxLayout(plot_toolbar_row)
        plot_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        plot_toolbar_layout.setSpacing(8)
        plot_toolbar_layout.addWidget(self._main_toolbar)
        plot_toolbar_layout.addStretch(1)
        self._plot_coord_label = QLabel("坐标: --")
        self._plot_coord_label.setProperty("class", "hintText")
        plot_toolbar_layout.addWidget(self._plot_coord_label)
        right_layout.addWidget(plot_toolbar_row)

        # 空状态卡片
        self.plot_stack_host = QWidget()
        plot_stack_layout = QStackedLayout(self.plot_stack_host)
        plot_stack_layout.setContentsMargins(0, 0, 0, 0)

        self.empty_state_card = self._create_empty_state_card()
        plot_stack_layout.addWidget(self.empty_state_card)
        plot_stack_layout.addWidget(self.canvas)
        right_layout.addWidget(self.plot_stack_host, 1)

        # 运行信息抽屉：默认收起，避免长期压缩主绘图区。
        self.global_log_box = self._create_global_log_box()
        self.quality_box = self._create_quality_box()
        self.quality_box.setChecked(True)
        self.quality_box.setCheckable(False)
        self._runtime_panel_bar, self._runtime_panel_container = (
            self._create_runtime_panel_drawer()
        )
        right_layout.addWidget(self._runtime_panel_bar)
        right_layout.addWidget(self._runtime_panel_container)

        self._sync_runtime_panels_visibility()

        # 连接信号
        self._connect_signals()

        # 初始化
        self._apply_startup_preset_defaults()
        self._reset_auto_tune_state()
        self._update_manual_roi_status()
        self._refresh_observability_panel()
        self._sync_runtime_panels_visibility()
        self._update_empty_state_and_brief()
        self._log(f"版本: {self.version_text}")
        self._log("欢迎使用。请导入数据开始处理。")

    def _create_empty_state_card(self):
        """创建空状态卡片"""
        card = QFrame()
        card.setObjectName("emptyStateCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(10)

        empty_badge = QLabel("ONBOARDING")
        empty_badge.setProperty("class", "emptyBadge")

        empty_title = QLabel("欢迎来到 GPR 成像工作台")
        empty_title.setProperty("class", "emptyTitle")

        empty_tip = QLabel("从一份 CSV 开始，30 秒看到首张 B-扫图像。")
        empty_tip.setProperty("class", "emptySubtitle")
        empty_tip.setAlignment(Qt.AlignmentFlag.AlignCenter)

        empty_steps = QLabel('① 导入 CSV　　② 选择方法　　③ 点击"应用所选方法"')
        empty_steps.setProperty("class", "emptySteps")
        empty_steps.setAlignment(Qt.AlignmentFlag.AlignCenter)

        empty_hint = QLabel('建议先用"稳健成像"或"一键默认流程"，再按需微调参数。')
        empty_hint.setProperty("class", "emptyHint")
        empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_hint.setWordWrap(True)

        layout.addStretch(1)
        layout.addWidget(empty_badge, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(empty_title, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(empty_tip)
        layout.addWidget(empty_steps)
        layout.addWidget(empty_hint)
        layout.addStretch(1)

        return card

    def _create_progress_panel(self):
        """创建顶部内联进度反馈条。"""
        panel = QFrame()
        panel.setObjectName("progressPanel")
        panel.setVisible(False)
        panel.setFixedWidth(300)

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
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(6)

        title = QLabel("运行信息")
        title.setProperty("class", "topInfoMeta")
        bar_layout.addWidget(title)

        self.btn_toggle_global_log = QPushButton("全局日志")
        self.btn_toggle_global_log.setCheckable(True)
        self.btn_toggle_global_log.clicked.connect(
            lambda checked: self._show_runtime_panel("global_log" if checked else None)
        )
        bar_layout.addWidget(self.btn_toggle_global_log)

        self.btn_toggle_quality = QPushButton("质量摘要")
        self.btn_toggle_quality.setCheckable(True)
        self.btn_toggle_quality.clicked.connect(
            lambda checked: self._show_runtime_panel("quality" if checked else None)
        )
        bar_layout.addWidget(self.btn_toggle_quality)

        btn_collapse = QPushButton("收起")
        btn_collapse.clicked.connect(lambda: self._show_runtime_panel(None))
        bar_layout.addWidget(btn_collapse)
        bar_layout.addStretch(1)

        container = QFrame()
        container.setObjectName("runtimeDrawer")
        container.setVisible(False)
        container.setMaximumHeight(220)
        drawer_layout = QStackedLayout(container)
        drawer_layout.setContentsMargins(0, 0, 0, 0)

        drawer_layout.addWidget(self.global_log_box)
        drawer_layout.addWidget(self.quality_box)

        self._runtime_panel_stack = drawer_layout
        self._runtime_panel_buttons = {
            "global_log": self.btn_toggle_global_log,
            "quality": self.btn_toggle_quality,
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
            index = 0 if panel_key == "global_log" else 1
            self._runtime_panel_stack.setCurrentIndex(index)
            self._runtime_panel_container.setVisible(True)

    def _create_global_log_box(self):
        """创建全局日志面板。"""
        box = QGroupBox("全局日志")
        box.setToolTip("集中查看导入、处理、告警和导出等全局事件")

        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        hint = QLabel(
            "这里汇总当前会话的全局运行日志；绘图性能观测已降级为低频诊断入口。"
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "hintText")
        layout.addWidget(hint)

        self.runtime_log_view = QTextEdit()
        self.runtime_log_view.setReadOnly(True)
        self.runtime_log_view.setPlaceholderText("暂无全局日志")
        self.runtime_log_view.setMinimumHeight(110)
        self.runtime_log_view.setMaximumHeight(150)
        self.runtime_log_view.setToolTip("显示当前会话的全局运行日志")
        self.runtime_log_view.setPlainText(self.page_basic.info.toPlainText().strip())
        layout.addWidget(self.runtime_log_view)

        self.performance_diag_box = self._create_observability_box()
        layout.addWidget(self.performance_diag_box)

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

    def _create_quality_box(self):
        """创建质量面板"""
        box = QGroupBox("质量摘要（可选）")
        box.setCheckable(True)
        box.setChecked(False)
        box.setProperty("class", "lowProfileBox")
        box.setToolTip("显示处理后的数据质量指标")

        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(1)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(1)

        self.quality_focus_label = QLabel("focus_ratio: --")
        self.quality_focus_label.setToolTip("聚焦比：高值表示能量更集中于强反射区域")

        self.quality_hot_label = QLabel("hot_pixels: --")
        self.quality_hot_label.setToolTip("热点像素数：异常高值像素点的数量")

        self.quality_spiky_label = QLabel("spikiness: --")
        self.quality_spiky_label.setToolTip("尖峰度：衡量数据分布的'尖峭'程度")

        self.quality_time_label = QLabel("time_ms: --")
        self.quality_time_label.setToolTip("处理耗时（毫秒）")

        self.quality_alert_label = QLabel("阈值状态: --")
        self.quality_alert_label.setToolTip("质量指标是否超出预设阈值")

        self.quality_track_len_label = QLabel("track_length_m: --")
        self.quality_track_len_label.setToolTip("测线累计长度")

        self.quality_spacing_label = QLabel("trace_spacing_cv: --")
        self.quality_spacing_label.setToolTip("道间距变异系数，越低越稳定")

        self.quality_height_label = QLabel("flight_height_span_m: --")
        self.quality_height_label.setToolTip("飞行高度跨度，越小越稳定")

        self.quality_airborne_alert_label = QLabel("airborne_alerts: --")
        self.quality_airborne_alert_label.setToolTip("航空元数据异常摘要")

        self.btn_quality_export = QPushButton("导出质量快照")
        self.btn_quality_export.setToolTip("将质量指标导出为JSON和CSV文件")
        self.btn_quality_export.clicked.connect(self.export_quality_snapshot)

        for label in [
            self.quality_focus_label,
            self.quality_hot_label,
            self.quality_spiky_label,
            self.quality_time_label,
            self.quality_alert_label,
            self.quality_track_len_label,
            self.quality_spacing_label,
            self.quality_height_label,
            self.quality_airborne_alert_label,
        ]:
            label.setProperty("class", "metricLabel")
            body_layout.addWidget(label)
        body_layout.addWidget(self.btn_quality_export)

        layout.addWidget(body)
        box.toggled.connect(body.setVisible)
        body.setVisible(False)

        return box

    def _relocate_basic_status_brief(self):
        """兼容旧调用：当前版本不再隐藏状态区，仅保留重排逻辑。"""
        return

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
        # ===== 工作台信号连接 =====
        self.page_workbench.data_import_requested.connect(self.load_csv)
        self.page_workbench.save_result_requested.connect(
            self._on_workbench_save_result
        )

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

        # 显示与对比页面
        self.page_advanced.cmap_combo.currentIndexChanged.connect(self._refresh_plot)
        self.page_advanced.view_style_combo.currentIndexChanged.connect(
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
        self.page_advanced.btn_apply_crop.clicked.connect(self._refresh_plot)
        self.page_advanced.btn_reset_crop.clicked.connect(self._reset_crop)
        self.page_advanced.rtk_sidecar_button.clicked.connect(
            lambda: self._pick_sidecar_file("rtk")
        )
        self.page_advanced.rtk_sidecar_clear_button.clicked.connect(
            lambda: self._clear_sidecar_file("rtk")
        )
        self.page_advanced.imu_sidecar_button.clicked.connect(
            lambda: self._pick_sidecar_file("imu")
        )
        self.page_advanced.imu_sidecar_clear_button.clicked.connect(
            lambda: self._clear_sidecar_file("imu")
        )
        self.page_auto_tune.btn_auto_tune.clicked.connect(
            self.start_auto_tune_current_method
        )
        self.page_auto_tune.btn_compare_stage.clicked.connect(
            self.start_auto_select_current_stage
        )
        self.page_auto_tune.btn_view_auto_tune.clicked.connect(
            self.show_auto_tune_details
        )
        self.page_auto_tune.btn_apply_stage_choice.clicked.connect(
            self.apply_stage_compare_choice
        )
        self.page_auto_tune.btn_open_workbench.clicked.connect(
            self.switch_to_workbench_mode
        )
        self.page_advanced.btn_clear_manual_roi.clicked.connect(self._clear_manual_roi)

        # 显示选项
        for cb in [
            self.page_advanced.symmetric_var,
            self.page_advanced.chatgpt_style_var,
            self.page_advanced.compare_var,
            self.page_advanced.cmap_invert_var,
            self.page_advanced.show_cbar_var,
            self.page_advanced.show_grid_var,
            self.page_advanced.percentile_var,
            self.page_advanced.normalize_var,
            self.page_advanced.demean_var,
            self.page_advanced.crop_enable_var,
            self.page_advanced.display_downsample_var,
        ]:
            cb.stateChanged.connect(self._refresh_plot)

        self.page_advanced.compare_var.toggled.connect(self._on_compare_toggled)

        # 质量/日志页面
        self.page_quality.btn_generate_report.clicked.connect(self.generate_report)
        self.page_quality.btn_export_quality_snapshot.clicked.connect(
            self.export_quality_snapshot
        )
        self.page_quality.btn_record_clear.clicked.connect(
            self.page_quality.record.clear
        )
        self.page_quality.btn_record_export.clicked.connect(self.export_record)
        self.page_quality.btn_open_log_dir.clicked.connect(self.open_log_directory)
        self.page_quality.btn_copy_diagnostics.clicked.connect(self.copy_diagnostics)

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
                self.status_label.setText("调参与实验")
            elif tab_key == "advanced" and self.page_advanced is not None:
                self.control_tabs.setCurrentWidget(self.page_advanced)
                self.status_label.setText("显示与对比")
            elif tab_key == "quality" and self.page_quality is not None:
                self.control_tabs.setCurrentWidget(self.page_quality)
                self.status_label.setText("质量与导出")
            else:
                # 默认切换到日常处理页
                if self.page_basic is not None:
                    self.control_tabs.setCurrentWidget(self.page_basic)
                self.status_label.setText("日常处理界面")

            tab_name = {
                "basic": "日常处理",
                "auto_tune": "调参与实验",
                "advanced": "显示与对比",
                "quality": "质量与导出",
            }.get(tab_key, "日常处理")
            self._log(f"切换到: {tab_name}")

    def switch_to_workbench_mode(self):
        """切换到工作台总控页"""
        if self._content_stack is not None and self.page_workbench is not None:
            self._content_stack.setCurrentWidget(self.page_workbench)
            self._sync_workbench_with_main_data()
            self.status_label.setText("工作台总控页")
            self._log("切换到工作台总控页")

    def _on_shared_data_changed(self, payload: dict):
        """共享数据状态变化后，同步相关视图。"""
        reason = (payload or {}).get("reason")
        self._store_trace_timestamps_from_metadata(self.trace_metadata)
        self._normalize_selected_trace_index()
        self._sync_history_action_state()
        self._refresh_compare_snapshots_from_state(
            clear_transient=reason in {"loaded", "current_updated", "undo", "reset"}
        )
        if hasattr(self, "page_workbench") and self.page_workbench is not None:
            self.page_workbench.sync_from_shared_state(payload)
        self._update_empty_state_and_brief()
        if reason == "loaded":
            self._manual_roi_values = None
            self._main_view_limits = None
            self._update_manual_roi_status()
        if reason in {"loaded", "undo", "reset"}:
            self._clear_runtime_warnings()
            self._reset_auto_tune_state("数据已更新，请重新自动选参。")

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
        """清除当前手动框选 ROI。"""
        self._manual_roi_values = None
        self._update_manual_roi_status()
        if self.data is not None:
            self.plot_data(self.data)

    def _update_manual_roi_status(self):
        """同步显示页中的手动 ROI 状态。"""
        if not hasattr(self, "page_advanced") or self.page_advanced is None:
            return
        if self._manual_roi_values is None:
            self.page_advanced.set_manual_roi_status("手动 ROI: 未设置", False)
            return

        vals = self._manual_roi_values
        self.page_advanced.set_manual_roi_status(
            f"手动 ROI: X[{vals['dist_start']:.2f}, {vals['dist_end']:.2f}] | Y[{vals['time_start']:.2f}, {vals['time_end']:.2f}]",
            True,
        )

    def _set_selected_trace_index(self, trace_index: int | None):
        """设置当前选中的道号并刷新主图与航迹图。"""
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
        if hasattr(self, "page_quality") and self.page_quality is not None:
            self.page_quality.set_airborne_trajectory_visualization(
                self._build_airborne_trajectory_plot_payload()
            )
        if self.data is not None:
            self.plot_data(self.data)

    def _on_trajectory_trace_selected(self, trace_index: int):
        """响应航迹图中的 trace 选择。"""
        self._set_selected_trace_index(trace_index)

    def _on_main_canvas_press(self, event):
        """统一记录主图鼠标按下事件。"""
        if self._toolbar_mode_active():
            return
        if (
            event.inaxes not in self._main_plot_axes
            or event.xdata is None
            or event.ydata is None
        ):
            return

        if event.button == 3:
            if self._manual_roi_values is not None:
                self._clear_manual_roi()
            self._capture_main_view_limits_from_axes()
            return

        if event.button != 1:
            return

        if event.dblclick:
            self._reset_main_plot_view_to_default()
            return

        ax = event.inaxes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        self._main_press_state = {
            "x": float(event.x),
            "y": float(event.y),
            "xdata": float(event.xdata),
            "ydata": float(event.ydata),
            "key": str(event.key or ""),
            "axes": ax,
            "dragging": False,
            "slider_mode": self._is_main_slider_compare_active(),
            "xlim_start": (float(xlim[0]), float(xlim[1])),
            "ylim_start": (float(ylim[0]), float(ylim[1])),
        }

    def _on_main_canvas_motion(self, event):
        """鼠标移动时更新坐标读数；支持拖动平移（默认）和 Shift+拖动框选 ROI。"""
        self._update_plot_coord_label(event)
        if self._main_press_state is None or self._toolbar_mode_active():
            return
        if (
            event.x is None
            or event.y is None
            or event.xdata is None
            or event.ydata is None
        ):
            return

        start = self._main_press_state
        dx = float(event.x) - float(start["x"])
        dy = float(event.y) - float(start["y"])
        if np.hypot(dx, dy) < self._main_drag_threshold_px:
            return

        start["dragging"] = True
        key = str(start.get("key", "")).lower()

        if "shift" in key:
            self._draw_drag_roi_preview(start, event)
            return

        if bool(start.get("slider_mode")):
            self._update_main_slider_compare_ratio_from_event(event)
            return

        # Pan like grabbing the image: viewport moves opposite to pointer delta.
        ax = start["axes"]
        dx_data = float(event.xdata) - float(start["xdata"])
        dy_data = float(event.ydata) - float(start["ydata"])
        xlim0 = start["xlim_start"]
        ylim0 = start["ylim_start"]
        ax.set_xlim(xlim0[0] - dx_data, xlim0[1] - dx_data)
        ax.set_ylim(ylim0[0] - dy_data, ylim0[1] - dy_data)
        self.canvas.draw_idle()

    def _on_main_canvas_release(self, event):
        """统一处理主图点击/拖框释放。"""
        if self._main_press_state is None:
            self._capture_main_view_limits_from_axes()
            return

        start = self._main_press_state
        self._main_press_state = None
        self._remove_drag_roi_preview()

        if event.x is None or event.y is None:
            self._capture_main_view_limits_from_axes()
            return

        dx = float(event.x) - float(start["x"])
        dy = float(event.y) - float(start["y"])
        is_drag = (
            bool(start.get("dragging"))
            and np.hypot(dx, dy) >= self._main_drag_threshold_px
        )
        key = str(start.get("key", "")).lower()

        if is_drag and event.xdata is not None and event.ydata is not None:
            if "shift" in key:
                x0, x1 = sorted([float(start["xdata"]), float(event.xdata)])
                y0, y1 = sorted([float(start["ydata"]), float(event.ydata)])
                self._manual_roi_values = {
                    "dist_start": x0,
                    "dist_end": x1,
                    "time_start": y0,
                    "time_end": y1,
                }
                self._update_manual_roi_status()
                if self.data is not None:
                    self.plot_data(self.data)
                return
            if bool(start.get("slider_mode")):
                self._update_main_slider_compare_ratio_from_event(event)
                return
            # Plain drag ends: persist the panned view limits.
            self._capture_main_view_limits_from_axes()
            return

        if "alt" in key and event.xdata is not None:
            self._select_trace_from_x(float(event.xdata))
        self._capture_main_view_limits_from_axes()

    def _is_main_slider_compare_active(self) -> bool:
        """主界面是否处于滑动对比模式。"""
        return bool(
            hasattr(self.page_advanced, "slider_compare_var")
            and self.page_advanced.slider_compare_var.isChecked()
        )

    def _update_main_slider_compare_ratio_from_event(self, event):
        """根据鼠标事件更新主图滑动对比位置。"""
        if (
            event is None
            or event.inaxes not in self._main_plot_axes
            or event.xdata is None
            or self._last_display_trace_axis.size == 0
        ):
            return

        x_min = float(self._last_display_trace_axis[0])
        x_max = float(self._last_display_trace_axis[-1])
        span = x_max - x_min
        if abs(span) < 1.0e-9:
            return

        new_ratio = (float(event.xdata) - x_min) / span
        new_ratio = max(0.0, min(1.0, new_ratio))
        if abs(new_ratio - self._main_slider_compare_ratio) < 1.0e-4:
            return
        self._main_slider_compare_ratio = new_ratio
        self._refresh_plot()

    def _on_main_canvas_scroll(self, event):
        """滚轮缩放主图。"""
        if self._toolbar_mode_active() or self._is_main_slider_compare_active():
            return
        if (
            event.inaxes not in self._main_plot_axes
            or event.xdata is None
            or event.ydata is None
        ):
            return

        ax = event.inaxes
        scale = 0.85 if event.button == "up" else 1.18
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata = float(event.xdata)
        ydata = float(event.ydata)
        ax.set_xlim(
            xdata - (xdata - xlim[0]) * scale,
            xdata + (xlim[1] - xdata) * scale,
        )
        ax.set_ylim(
            ydata - (ydata - ylim[0]) * scale,
            ydata + (ylim[1] - ydata) * scale,
        )
        self._capture_main_view_limits_from_axes()
        self.canvas.draw_idle()

    def _update_plot_coord_label(self, event):
        """更新主图坐标读数。"""
        if self._plot_coord_label is None:
            return
        if (
            event.inaxes not in self._main_plot_axes
            or event.xdata is None
            or event.ydata is None
            or self._last_display_trace_axis.size == 0
            or self._last_display_time_axis.size == 0
            or self._last_display_data is None
        ):
            self._plot_coord_label.setText("坐标: --")
            return

        trace_pos = float(event.xdata)
        time_pos = float(event.ydata)
        trace_idx = int(np.argmin(np.abs(self._last_display_trace_axis - trace_pos)))
        time_idx = int(np.argmin(np.abs(self._last_display_time_axis - time_pos)))
        amplitude = float(self._last_display_data[time_idx, trace_idx])
        trace_value = float(self._last_display_trace_axis[trace_idx])
        time_value = float(self._last_display_time_axis[time_idx])
        raw_trace_idx = (
            int(self._last_display_trace_indices[trace_idx])
            if trace_idx < self._last_display_trace_indices.size
            else trace_idx
        )
        extra_hint = "拖动分隔线调整对比" if self._is_main_slider_compare_active() else "拖框设 ROI"
        self._plot_coord_label.setText(
            f"距离: {trace_value:.2f} | 时间: {time_value:.2f} | 道号: {raw_trace_idx} | 振幅: {amplitude:.4f} | {extra_hint} | 右键清除 ROI | Alt+单击选道"
        )

    def _on_main_canvas_leave(self, event):
        """鼠标离开主图时清空坐标读数。"""
        if self._plot_coord_label is not None:
            self._plot_coord_label.setText("坐标: --")

    def _toolbar_mode_active(self) -> bool:
        return bool(
            self._main_toolbar is not None and getattr(self._main_toolbar, "mode", "")
        )

    def _select_trace_from_x(self, x_value: float):
        """根据 X 坐标选最近 trace。"""
        if (
            self._last_display_trace_axis.size == 0
            or self._last_display_trace_indices.size
            != self._last_display_trace_axis.size
        ):
            return
        nearest = int(np.argmin(np.abs(self._last_display_trace_axis - float(x_value))))
        self._set_selected_trace_index(int(self._last_display_trace_indices[nearest]))

    def _draw_drag_roi_preview(self, start: dict, event):
        """绘制拖框中的临时 ROI。"""
        ax = start.get("axes")
        if ax is None or event.xdata is None or event.ydata is None:
            return
        self._remove_drag_roi_preview()
        x0, x1 = sorted([float(start["xdata"]), float(event.xdata)])
        y0, y1 = sorted([float(start["ydata"]), float(event.ydata)])
        self._drag_roi_preview_patch = Rectangle(
            (x0, y0),
            abs(x1 - x0),
            abs(y1 - y0),
            fill=False,
            edgecolor="#38bdf8",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
            zorder=7,
        )
        ax.add_patch(self._drag_roi_preview_patch)
        self.canvas.draw_idle()

    def _remove_drag_roi_preview(self):
        """移除临时 ROI 预览框。"""
        if self._drag_roi_preview_patch is not None:
            try:
                self._drag_roi_preview_patch.remove()
            except Exception:
                pass
            self._drag_roi_preview_patch = None

    def _capture_main_view_limits_from_axes(self):
        """记录当前主图缩放范围，供重绘后恢复。"""
        if len(self._main_plot_axes) != 1:
            self._main_view_limits = None
            return
        ax = self._main_plot_axes[0]
        self._main_view_limits = {
            "xlim": tuple(float(v) for v in ax.get_xlim()),
            "ylim": tuple(float(v) for v in ax.get_ylim()),
        }

    def _reset_main_plot_view_to_default(self):
        """恢复主图到当前载荷的初始整图视图。"""
        self._main_view_limits = None
        active_data, _, _ = self._get_active_plot_payload(self.data)
        fallback_data = self.data if self.data is not None else active_data
        if fallback_data is None:
            return
        self.plot_data(fallback_data)
        self._capture_main_view_limits_from_axes()

    def _sync_workbench_with_main_data(self):
        """兼容旧调用：实际由共享数据状态统一驱动。"""
        self._on_shared_data_changed({"reason": "manual_sync"})

    def resizeEvent(self, event):
        """窗口尺寸变化时，调整左右区域的相对占比，避免控制区在窄窗口下异常拥挤。
        只在旧界面模式下调整尺寸。"""
        super().resizeEvent(event)

        # 只在旧界面模式下调整
        if (
            self._main_splitter is None
            or self._content_stack is None
            or self._content_stack.currentWidget() != self._main_content_widget
        ):
            return

        total = max(1, self._main_splitter.size().width())
        target_left = max(320, min(520, int(total * 0.30)))
        target_right = max(640, total - target_left)
        self._main_splitter.setSizes([target_right, target_left])

    def _on_workbench_run_method(self, method_id: str, params: dict, source: str):
        """工作台运行单方法"""
        if self.data is None:
            self.page_workbench._log("未加载数据", "ERROR")
            return

        # 获取输入数据
        input_data, source_text = self.page_workbench.resolve_input_data(source)
        input_header_info = self.page_workbench.resolve_input_header_info(source)
        input_trace_metadata = self.page_workbench.resolve_input_trace_metadata(source)

        if input_data is None:
            input_data = self.data
            source_text = "原始数据（fallback）"
            input_header_info = self.header_info
            input_trace_metadata = self.trace_metadata

        # 记录开始时间
        start_time = time.time()

        # 获取方法信息
        method_info = PROCESSING_METHODS.get(method_id)
        if not method_info:
            self.page_workbench._log(f"未知方法: {method_id}", "ERROR")
            return

        method_name = method_info.get("name", method_id)

        # 禁用按钮，显示运行状态
        self.page_workbench.param_editor.set_buttons_for_running()

        # 记录详细信息到日志
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        self.page_workbench._log(f"{'=' * 40}")
        self.page_workbench._log(f"方法: {method_name}")
        self.page_workbench._log(f"输入源: {source_text} ({input_data.shape})")
        self.page_workbench._log(f"参数: {param_str}")
        self.page_workbench._log(f"正在执行...")

        try:
            # 执行方法
            execution = self._apply_single_method(
                input_data,
                method_id,
                params,
                header_info=input_header_info,
                trace_metadata=input_trace_metadata,
            )
            result = execution["result_data"]
            result_header_info = execution["result_header_info"]
            result_trace_metadata = execution["result_trace_metadata"]
            preview_data = execution["preview_data"]
            preview_header_info = execution["preview_header_info"]
            preview_trace_metadata = execution["preview_trace_metadata"]
            runtime_warnings = execution["meta"].get("runtime_warnings", []) or []

            # 计算耗时
            elapsed = (time.time() - start_time) * 1000  # ms

            # 自动切换到结果视图
            self.page_workbench.radio_result.setChecked(True)

            # 更新工作台预览（不直接提交为正式结果）
            self.page_workbench.set_preview_result(
                preview_data,
                f"预览: {method_name}",
                header_info=preview_header_info,
                trace_metadata=preview_trace_metadata,
                commit_data=result,
                commit_header_info=result_header_info,
                commit_trace_metadata=result_trace_metadata,
            )

            for warning in runtime_warnings:
                self.page_workbench._log(
                    f"运行告警: {format_runtime_warning_text(warning)}", "WARN"
                )

            # 记录成功日志
            self.page_workbench._log(f"预览完成！耗时 {elapsed:.1f}ms")
            self.page_workbench._log(
                f"预览: {result.shape} | 范围 [{result.min():.3f}, {result.max():.3f}]"
            )
            self.page_workbench._log(f"{'=' * 40}")

        except Exception as e:
            import traceback

            error_msg = str(e)
            traceback_str = traceback.format_exc()
            self.page_workbench._log(f"执行失败: {error_msg}", "ERROR")
            logger.error("Workbench method failed (%s):\n%s", method_id, traceback_str)

        finally:
            # 恢复按钮状态
            self.page_workbench._update_action_buttons()

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
        """工作台保存结果"""
        preview_data = getattr(self.page_workbench, "preview_data", None)
        preview_header_info = getattr(self.page_workbench, "preview_header_info", None)
        preview_trace_metadata = getattr(
            self.page_workbench, "preview_trace_metadata", None
        )
        preview_commit_data = getattr(self.page_workbench, "preview_commit_data", None)
        preview_commit_header_info = getattr(
            self.page_workbench, "preview_commit_header_info", None
        )
        preview_commit_trace_metadata = getattr(
            self.page_workbench, "preview_commit_trace_metadata", None
        )
        current_result_header_info = getattr(
            self.page_workbench, "current_result_header_info", None
        )
        current_result_trace_metadata = getattr(
            self.page_workbench, "current_result_trace_metadata", None
        )
        committed = (
            np.array(preview_commit_data, copy=True)
            if preview_commit_data is not None
            else np.array(preview_data, copy=True)
            if preview_data is not None
            else (
                np.array(self.page_workbench.current_result, copy=True)
                if self.page_workbench.current_result is not None
                else None
            )
        )

        if committed is not None:
            method_id = self.page_workbench.param_editor.current_method_id
            method_info = PROCESSING_METHODS.get(method_id, {}) if method_id else {}
            method_name = method_info.get("name", method_id or "工作台结果")
            label = method_name
            if preview_data is not None:
                label = f"{method_name}_{datetime.now().strftime('%H:%M:%S')}"

            self.shared_data.apply_current_data(
                committed,
                push_history=True,
                source="workbench_commit",
                label=label,
                header_info=(
                    preview_commit_header_info
                    if preview_commit_data is not None
                    else preview_header_info
                    if preview_data is not None
                    else current_result_header_info
                ),
                trace_metadata=(
                    preview_commit_trace_metadata
                    if preview_commit_data is not None
                    else preview_trace_metadata
                    if preview_data is not None
                    else current_result_trace_metadata
                ),
            )
            self._mark_data_changed()
            self._update_current_compare_snapshot()
            self._update_empty_state_and_brief()
            self.plot_data(self.data)
            try:
                params = self.page_workbench.param_editor.get_current_params()
            except Exception:
                params = {}
            self._set_last_run_summary(
                run_type="workbench_commit",
                label=method_name,
                steps=[
                    {
                        "method_key": method_id or "workbench",
                        "method_name": method_name,
                        "params": params,
                    }
                ],
                notes=["结果由工作台预览提交为正式结果"],
            )
            if preview_data is not None:
                self.page_workbench._log(f"✓ 结果已保存: {label}")
                self.page_workbench._log(f"  形状: {committed.shape}")
                self.page_workbench._log(
                    f"  范围: [{committed.min():.3f}, {committed.max():.3f}]"
                )
            self._log("工作台结果已应用到当前数据")
        else:
            self._log("没有可保存的结果")

    def _apply_style(self):
        """应用样式表 - 使用主题管理器"""
        from core.theme_manager import get_theme_manager

        theme_manager = get_theme_manager()
        stylesheet = theme_manager.get_theme_stylesheet()

        if stylesheet:
            self.setStyleSheet(stylesheet)

    # ============ 日志和帮助方法 ============

    def _log(self, msg: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        self.page_basic.info.append(line)
        self.page_basic.info.ensureCursorVisible()
        if hasattr(self, "runtime_log_view") and self.runtime_log_view is not None:
            self.runtime_log_view.append(line)
            self.runtime_log_view.ensureCursorVisible()
        if hasattr(self, "page_quality") and self.page_quality is not None:
            self.page_quality.append_record(line)

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
            self.page_auto_tune.btn_view_auto_tune,
            self.page_auto_tune.btn_apply_stage_choice,
            self.page_basic.method_combo,
            self.page_quality.btn_generate_report,
            self.page_quality.btn_export_quality_snapshot,
        ]
        for w in controls:
            w.setEnabled(not busy)
        self.page_basic.btn_cancel.setEnabled(busy and (not self._cancel_in_flight))
        if busy:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.status_label.setText(text)
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
        self.shared_data.reset_to_original(push_history=True)
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

    def _on_method_change(self, idx=None):
        """方法选择改变"""
        idx = self.page_basic.method_combo.currentIndex()
        if idx < 0:
            return
        key = self.page_basic.method_keys[idx]
        self.page_basic.render_method_params(key)
        self._reset_auto_tune_state()
        self._update_empty_state_and_brief()

    def _reset_auto_tune_state(self, message: str | None = None):
        """重置自动选参结果摘要。"""
        self._last_auto_tune_result = None
        current_key = self.page_basic.get_current_method_key()
        self.page_basic.set_auto_tune_result_available(False)
        self.page_basic.set_apply_source_hint("当前未生成自动调参结果。")
        self._last_auto_tune_group_result = None
        if hasattr(self, "page_auto_tune") and self.page_auto_tune is not None:
            self.page_auto_tune.reset_for_method(current_key, message=message)

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
        if not self._last_auto_tune_result:
            self.page_basic.set_apply_source_hint(
                "当前无可用推荐结果，正在分析当前参数..."
            )
            self.start_auto_tune_current_method(auto_apply_after_finish=True)
            return

        method_key = self.page_basic.get_current_method_key()
        if method_key != self._last_auto_tune_result.get("method_key"):
            self._reset_auto_tune_state("当前方法已变化，请先重新运行调参与实验。")
            self.page_basic.set_apply_source_hint(
                "当前推荐结果已过期，正在重新分析当前参数..."
            )
            self.start_auto_tune_current_method(auto_apply_after_finish=True)
            return

        profile_key = str(
            self._last_auto_tune_result.get("recommended_profile", "balanced")
        )
        self.apply_method_from_profile(profile_key)

    def start_auto_tune_current_method(self, auto_apply_after_finish: bool = False):
        """对当前方法执行单步自动选参。"""
        if (
            self._ui_busy
            or self._worker is not None
            or self._auto_tune_worker is not None
            or self._auto_tune_stage_worker is not None
        ):
            self._pending_apply_after_auto_tune = False
            QMessageBox.information(self, "自动选参", "当前已有任务在运行，请稍候。")
            return False
        if self.data is None or self.data_path is None:
            self._pending_apply_after_auto_tune = False
            QMessageBox.warning(self, "自动选参", "请先导入数据。")
            return False

        method_key = self.page_basic.get_current_method_key()
        method_info = PROCESSING_METHODS.get(method_key, {}) if method_key else {}
        if not method_key or not method_info.get("auto_tune_enabled"):
            self._pending_apply_after_auto_tune = False
            QMessageBox.information(self, "自动选参", "当前方法暂不支持自动选参。")
            return False

        try:
            current_params = self.page_basic.get_current_params()
        except ValueError as e:
            self._pending_apply_after_auto_tune = False
            QMessageBox.warning(self, "自动选参", str(e))
            return False

        roi_mode = self.page_auto_tune.get_auto_tune_roi_mode()
        search_mode = self.page_auto_tune.get_auto_tune_search_mode()
        roi_spec = self._build_auto_tune_roi_spec(roi_mode)

        self._last_auto_tune_result = None
        self.page_basic.set_auto_tune_result_available(False)
        self.page_auto_tune.show_running(
            roi_spec.get("label", roi_spec.get("source", "全图")),
            search_mode,
        )
        self._set_busy(True, text=f"自动选参: {method_info.get('name', method_key)}")
        self._cancel_in_flight = False
        self._pending_apply_after_auto_tune = bool(auto_apply_after_finish)

        self._auto_tune_thread = QThread(self)
        self._auto_tune_worker = AutoTuneWorker(
            self.data,
            method_key,
            current_params,
            header_info=self.header_info,
            trace_metadata=self.trace_metadata,
            roi_spec=roi_spec,
            search_mode=search_mode,
        )
        self._auto_tune_worker.moveToThread(self._auto_tune_thread)
        self._auto_tune_thread.started.connect(self._auto_tune_worker.run)
        self._auto_tune_worker.progress.connect(self._on_auto_tune_progress)
        self._auto_tune_worker.finished.connect(self._on_auto_tune_finished)
        self._auto_tune_worker.error.connect(self._on_auto_tune_error)
        self._auto_tune_worker.finished.connect(self._auto_tune_thread.quit)
        self._auto_tune_worker.error.connect(self._auto_tune_thread.quit)
        self._auto_tune_thread.finished.connect(self._cleanup_auto_tune_worker)
        self._auto_tune_thread.start()
        return True

    def _get_current_stage_method_keys(self) -> list[str]:
        """获取当前方法所在 stage 的可比较方法列表。"""
        method_key = self.page_basic.get_current_method_key()
        if not method_key:
            return []
        stage = get_auto_tune_stage(method_key)
        if not stage:
            return []
        return [
            key
            for key in get_public_method_keys()
            if PROCESSING_METHODS.get(key, {}).get("auto_tune_enabled")
            and get_auto_tune_stage(key) == stage
        ]

    def start_auto_select_current_stage(self):
        """比较当前 stage 内多个可用方法并推荐最佳方法。"""
        if (
            self._ui_busy
            or self._worker is not None
            or self._auto_tune_worker is not None
            or self._auto_tune_stage_worker is not None
        ):
            QMessageBox.information(self, "同阶段比较", "当前已有任务在运行，请稍候。")
            return False
        if self.data is None or self.data_path is None:
            QMessageBox.warning(self, "同阶段比较", "请先导入数据。")
            return False

        method_keys = self._get_current_stage_method_keys()
        if len(method_keys) < 2:
            QMessageBox.information(
                self,
                "同阶段比较",
                "当前 stage 没有足够多的可比较方法。",
            )
            return False

        base_params_map = {}
        for key in method_keys:
            base_params_map[key] = self._resolve_method_params(key)

        roi_mode = self.page_auto_tune.get_auto_tune_roi_mode()
        search_mode = self.page_auto_tune.get_auto_tune_search_mode()
        roi_spec = self._build_auto_tune_roi_spec(roi_mode)

        self._last_auto_tune_group_result = None
        self.page_auto_tune.set_stage_compare_result(None)
        self.page_auto_tune.show_running(
            roi_spec.get("label", roi_spec.get("source", "全图")),
            f"{search_mode} | 同阶段比较",
        )
        self._set_busy(True, text="同阶段方法比较")
        self._cancel_in_flight = False

        self._auto_tune_stage_thread = QThread(self)
        self._auto_tune_stage_worker = AutoTuneStageWorker(
            self.data,
            method_keys,
            base_params_map,
            header_info=self.header_info,
            trace_metadata=self.trace_metadata,
            roi_spec=roi_spec,
            search_mode=search_mode,
        )
        self._auto_tune_stage_worker.moveToThread(self._auto_tune_stage_thread)
        self._auto_tune_stage_thread.started.connect(self._auto_tune_stage_worker.run)
        self._auto_tune_stage_worker.progress.connect(self._on_auto_tune_progress)
        self._auto_tune_stage_worker.finished.connect(self._on_auto_stage_finished)
        self._auto_tune_stage_worker.error.connect(self._on_auto_stage_error)
        self._auto_tune_stage_worker.finished.connect(self._auto_tune_stage_thread.quit)
        self._auto_tune_stage_worker.error.connect(self._auto_tune_stage_thread.quit)
        self._auto_tune_stage_thread.finished.connect(
            self._cleanup_auto_tune_stage_worker
        )
        self._auto_tune_stage_thread.start()
        return True

    def _on_auto_tune_progress(self, current: int, total: int, message: str):
        """自动选参进度回调。"""
        self.status_label.setText(message)
        if self._progress_panel is not None:
            self._progress_panel.setVisible(True)
        if self._progress_bar is not None:
            safe_total = max(int(total), 1)
            safe_current = max(0, min(int(current), safe_total))
            self._progress_bar.setRange(0, safe_total)
            self._progress_bar.setValue(safe_current)
            self._progress_bar.setFormat(f"候选 {safe_current}/{safe_total}")

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

        self._last_auto_tune_result = result
        self.page_basic.set_auto_tune_result_available(True, result.get("profiles", {}))
        self.page_basic.set_apply_source_hint(
            "已生成自动调参结果，可在“应用方法”右侧切换默认应用来源。"
        )
        self.page_auto_tune.show_result(result)
        self._log(
            f"自动选参完成: {result.get('method_name', result.get('method_key'))} | 推荐参数 {result.get('recommended_params') or result.get('best_params')}"
        )
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
        self._log(
            f"同阶段比较完成: 推荐 {result.get('best_method_name', result.get('best_method_key'))} | outer score {float(result.get('outer_score', 0.0)):.4f}"
        )

    def _on_auto_stage_error(self, error_msg: str):
        """同阶段方法比较失败。"""
        self._set_busy(False, text="同阶段比较失败")
        self.page_auto_tune.show_error(error_msg)
        self._log(f"同阶段比较失败: {error_msg}")
        QMessageBox.warning(self, "同阶段比较失败", error_msg)

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

    def apply_method_from_profile(self, profile_key: str):
        """使用自动选参档位参数并立即执行当前方法。"""
        if not self._last_auto_tune_result:
            QMessageBox.information(
                self,
                "自动选参结果不存在",
                "请先到“调参与实验”页执行自动选参。",
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

    def apply_stage_compare_choice(self):
        """将同阶段比较推荐的方法和参数写回日常处理。"""
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
        """构建自动选参所用 ROI 规格。"""
        mode = str(roi_mode or "prefer_crop")
        if self.data is None:
            return {"mode": "full", "source": "full", "label": "全图"}

        manual_bounds = self._get_manual_roi_bounds()
        if mode != "full" and manual_bounds is not None:
            return {
                "mode": "manual",
                "source": "manual",
                "label": "手动框选 ROI",
                "bounds": manual_bounds,
            }

        crop_bounds = None
        if (
            mode in {"prefer_crop", "crop"}
            and self.page_advanced.crop_enable_var.isChecked()
        ):
            try:
                time_axis = self._build_time_axis(self.data.shape[0])
                trace_axis = self._build_trace_axis(self.data.shape[1])
                crop_bounds = self._get_crop_bounds(self.data, time_axis, trace_axis)
            except Exception:
                crop_bounds = None

        if crop_bounds:
            return {
                "mode": "crop",
                "source": "crop",
                "label": "当前裁剪区",
                "bounds": crop_bounds,
            }

        if mode == "full":
            return {"mode": "full", "source": "full", "label": "全图"}

        return {"mode": "auto", "source": "auto", "label": "自动 ROI"}

    def _get_manual_roi_bounds(self) -> dict | None:
        """将主图手动框选 ROI 转换为当前数据索引边界。"""
        if self.data is None or self._manual_roi_values is None:
            return None

        time_axis = self._build_time_axis(self.data.shape[0])
        trace_axis = self._build_trace_axis(self.data.shape[1])
        vals = self._manual_roi_values
        t0 = min(float(vals["time_start"]), float(vals["time_end"]))
        t1 = max(float(vals["time_start"]), float(vals["time_end"]))
        d0 = min(float(vals["dist_start"]), float(vals["dist_end"]))
        d1 = max(float(vals["dist_start"]), float(vals["dist_end"]))

        return {
            "time_start_idx": self._axis_value_to_index(
                time_axis, t0, self.data.shape[0], "left"
            ),
            "time_end_idx": self._axis_value_to_index(
                time_axis, t1, self.data.shape[0], "right"
            ),
            "dist_start_idx": self._axis_value_to_index(
                trace_axis, d0, self.data.shape[1], "left"
            ),
            "dist_end_idx": self._axis_value_to_index(
                trace_axis, d1, self.data.shape[1], "right"
            ),
        }

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
        if theme == "dark":
            fig_face = "#1f2125"
            ax_face = "#23252a"
            text_color = "#e8e8e8"
            spine_color = "#5a606b"
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
        self._clear_display_override()

    def _apply_preset_ui_values(
        self, ui_config: dict | None, preset_key: str | None = None
    ):
        """将预设中的 UI 选项同步到显示页控件。"""
        if not ui_config or self.page_advanced is None:
            return

        checkbox_fields = {
            "fast_preview": self.page_advanced.fast_preview_var,
            "display_downsample": self.page_advanced.display_downsample_var,
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
            "max_samples": self.page_advanced.max_samples_edit,
            "max_traces": self.page_advanced.max_traces_edit,
            "display_max_samples": self.page_advanced.display_max_samples_edit,
            "display_max_traces": self.page_advanced.display_max_traces_edit,
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
        # 切换空状态卡片和绘图区
        self.plot_stack_host.layout().setCurrentIndex(1 if has_data else 0)
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
            self.page_quality.set_airborne_anomaly_details(
                self._build_airborne_anomaly_text()
            )

    def _build_airborne_metadata_summary(self) -> list[str]:
        """构建航空元数据摘要文本。"""
        info = self.header_info or {}
        if not info.get("has_airborne_metadata"):
            return []

        meta = self.trace_metadata or {}
        lines = []
        track_length = info.get("track_length_m")
        if track_length is not None:
            lines.append(f"测线长度: {float(track_length):.2f} m")
        if meta.get("longitude") is not None and meta.get("latitude") is not None:
            lon = np.asarray(meta.get("longitude"), dtype=np.float64)
            lat = np.asarray(meta.get("latitude"), dtype=np.float64)
            if lon.size > 0 and lat.size > 0:
                lines.append(
                    f"起止经纬度: ({lon[0]:.7f}, {lat[0]:.7f}) -> ({lon[-1]:.7f}, {lat[-1]:.7f})"
                )
        if info.get("ground_elevation_min_m") is not None:
            lines.append(
                "地表高程: {:.2f} ~ {:.2f} m".format(
                    float(info.get("ground_elevation_min_m", 0.0)),
                    float(info.get("ground_elevation_max_m", 0.0)),
                )
            )
        if info.get("flight_height_min_m") is not None:
            lines.append(
                "飞行高度: {:.2f} ~ {:.2f} m".format(
                    float(info.get("flight_height_min_m", 0.0)),
                    float(info.get("flight_height_max_m", 0.0)),
                )
            )
        if info.get("trace_interval_min_m") is not None:
            lines.append(
                "道间距: {:.3f} ~ {:.3f} m (均值 {:.3f} m)".format(
                    float(info.get("trace_interval_min_m", 0.0)),
                    float(info.get("trace_interval_max_m", 0.0)),
                    float(info.get("trace_interval_m", 0.0)),
                )
            )
        return lines

    def _build_status_text(self) -> str:
        """构建顶部状态栏文本。"""
        base = (
            f"{os.path.basename(self.data_path) if self.data_path else 'data'} | "
            f"采样:{self.header_info['a_scan_length']} 道数:{self.header_info['num_traces']}"
        )
        if self.header_info and self.header_info.get("has_airborne_metadata"):
            base += f" | 距离:{float(self.header_info.get('track_length_m', 0.0)):.1f}m"
        return base

    def _build_data_brief_text(self) -> str:
        """构建基础流程页的数据摘要文本。"""
        if self.data is None:
            return "未加载数据"
        summary = [f"数据: {self.data.shape[0]}×{self.data.shape[1]}"]
        if self.header_info and self.header_info.get("has_airborne_metadata"):
            summary.append(
                f"测线长度 {float(self.header_info.get('track_length_m', 0.0)):.1f} m"
            )
            summary.append(
                "飞行高度 {:.1f}-{:.1f} m".format(
                    float(self.header_info.get("flight_height_min_m", 0.0)),
                    float(self.header_info.get("flight_height_max_m", 0.0)),
                )
            )
        elif self.header_info:
            summary.append(
                f"总时窗 {float(self.header_info.get('total_time_ns', 0.0)):.1f} ns"
            )
        return " | ".join(summary)

    def _build_airborne_line_summary_text(self) -> str:
        """构建测线结果卡片文本。"""
        if self.data is None:
            return "暂无测线信息"

        header = self.header_info or {}
        meta = self.trace_metadata or {}
        lines = [
            f"数据文件: {self.data_path or '--'}",
            f"矩阵尺寸: {self.data.shape[0]} × {self.data.shape[1]}",
        ]

        if header:
            lines.append(
                f"采样点数: {header.get('a_scan_length', self.data.shape[0])} | 道数: {header.get('num_traces', self.data.shape[1])}"
            )
        if header.get("has_airborne_metadata"):
            lines.append(f"测线长度: {float(header.get('track_length_m', 0.0)):.2f} m")
            lines.append(
                "道间距: {:.3f} ~ {:.3f} m (均值 {:.3f} m)".format(
                    float(header.get("trace_interval_min_m", 0.0)),
                    float(header.get("trace_interval_max_m", 0.0)),
                    float(header.get("trace_interval_m", 0.0)),
                )
            )
            lines.append(
                "地表高程: {:.2f} ~ {:.2f} m".format(
                    float(header.get("ground_elevation_min_m", 0.0)),
                    float(header.get("ground_elevation_max_m", 0.0)),
                )
            )
            lines.append(
                "飞行高度: {:.2f} ~ {:.2f} m".format(
                    float(header.get("flight_height_min_m", 0.0)),
                    float(header.get("flight_height_max_m", 0.0)),
                )
            )
            if meta.get("longitude") is not None and meta.get("latitude") is not None:
                lon = np.asarray(meta.get("longitude"), dtype=np.float64)
                lat = np.asarray(meta.get("latitude"), dtype=np.float64)
                if lon.size and lat.size:
                    lines.append(f"起点经纬度: {lon[0]:.7f}, {lat[0]:.7f}")
                    lines.append(f"终点经纬度: {lon[-1]:.7f}, {lat[-1]:.7f}")
        else:
            lines.append("航空元数据: 未提供")

        return "\n".join(lines)

    def _build_airborne_qc_summary_text(self) -> str:
        """构建航空质控摘要文本。"""
        qc = self._compute_airborne_qc_metrics()
        if not qc:
            return ""
        lines = [
            f"测线长度: {qc['track_length_m']:.2f} m",
            f"道间距变异系数: {qc['trace_spacing_cv']:.3f}",
            f"飞行高度跨度: {qc['flight_height_span_m']:.2f} m",
            f"道间距离群点: {qc['spacing_outliers']}",
            f"飞行高度离群点: {qc['height_outliers']}",
        ]
        alerts = qc.get("alerts") or []
        lines.append("异常状态: " + (", ".join(alerts) if alerts else "正常"))
        return "\n".join(lines)

    def _build_airborne_qc_plot_payload(self) -> dict | None:
        """构建航空 QC 可视化所需数据。"""
        qc = self._compute_airborne_qc_metrics()
        meta = self.trace_metadata or {}
        if not qc or "trace_distance_m" not in meta or "flight_height_m" not in meta:
            return None

        distance = np.asarray(meta.get("trace_distance_m"), dtype=np.float64)
        flight = np.asarray(meta.get("flight_height_m"), dtype=np.float64)
        if distance.size <= 1:
            return None

        spacing = np.diff(distance)
        spacing_x = 0.5 * (distance[:-1] + distance[1:])
        spacing_mask = np.zeros_like(spacing, dtype=bool)
        for idx in qc.get("spacing_outlier_indices", []):
            if 0 <= idx < spacing_mask.size:
                spacing_mask[idx] = True

        flight_mask = np.zeros_like(flight, dtype=bool)
        for idx in qc.get("height_outlier_indices", []):
            if 0 <= idx < flight_mask.size:
                flight_mask[idx] = True

        return {
            "spacing_x": spacing_x.tolist(),
            "spacing": spacing.tolist(),
            "spacing_mask": spacing_mask.tolist(),
            "distance": distance.tolist(),
            "flight": flight.tolist(),
            "flight_mask": flight_mask.tolist(),
        }

    def _build_airborne_trajectory_plot_payload(self) -> dict | None:
        """构建航空航迹图所需数据。"""
        header = self.header_info or {}
        meta = self.trace_metadata or {}
        if not header.get("has_airborne_metadata"):
            return None

        longitude = np.asarray(meta.get("longitude", []), dtype=np.float64)
        latitude = np.asarray(meta.get("latitude", []), dtype=np.float64)
        n = min(longitude.size, latitude.size)
        if n == 0:
            return None

        longitude = longitude[:n]
        latitude = latitude[:n]
        trace_indices = np.asarray(
            meta.get("trace_index", np.arange(n)), dtype=np.int32
        )
        if trace_indices.size < n:
            trace_indices = np.arange(n, dtype=np.int32)
        else:
            trace_indices = trace_indices[:n]
        anomaly_mask = np.zeros(n, dtype=bool)
        qc = self._compute_airborne_qc_metrics() or {}
        for idx in qc.get("spacing_outlier_indices", []):
            mapped_idx = int(idx) + 1
            if 0 <= mapped_idx < n:
                anomaly_mask[mapped_idx] = True
        for idx in qc.get("height_outlier_indices", []):
            mapped_idx = int(idx)
            if 0 <= mapped_idx < n:
                anomaly_mask[mapped_idx] = True

        finite_mask = np.isfinite(longitude) & np.isfinite(latitude)
        if not np.any(finite_mask):
            return None

        flight_height = np.asarray(meta.get("flight_height_m", []), dtype=np.float64)[:n]

        payload = {
            "longitude": longitude[finite_mask].tolist(),
            "latitude": latitude[finite_mask].tolist(),
            "trace_indices": trace_indices[finite_mask].tolist(),
            "anomaly_mask": anomaly_mask[finite_mask].tolist(),
            "selected_trace_index": self._selected_trace_index,
        }
        if flight_height.size >= n:
            payload["flight_height_m"] = flight_height[finite_mask].tolist()
        return payload

    def _build_airborne_anomaly_details(self) -> list[dict]:
        """构建航空异常明细行。"""
        qc = self._compute_airborne_qc_metrics()
        meta = self.trace_metadata or {}
        if not qc or not meta:
            return []

        distance = np.asarray(meta.get("trace_distance_m", []), dtype=np.float64)
        longitude = np.asarray(meta.get("longitude", []), dtype=np.float64)
        latitude = np.asarray(meta.get("latitude", []), dtype=np.float64)
        flight = np.asarray(meta.get("flight_height_m", []), dtype=np.float64)
        spacing = (
            np.diff(distance) if distance.size > 1 else np.array([], dtype=np.float64)
        )
        spacing_x = (
            0.5 * (distance[:-1] + distance[1:])
            if spacing.size
            else np.array([], dtype=np.float64)
        )

        details: list[dict] = []
        for idx in qc.get("spacing_outlier_indices", []):
            if 0 <= idx < spacing.size:
                lon = (
                    float(longitude[min(idx + 1, longitude.size - 1)])
                    if longitude.size
                    else None
                )
                lat = (
                    float(latitude[min(idx + 1, latitude.size - 1)])
                    if latitude.size
                    else None
                )
                details.append(
                    {
                        "type": "trace_spacing",
                        "index": int(idx),
                        "distance_m": float(spacing_x[idx]),
                        "value": float(spacing[idx]),
                        "longitude": lon,
                        "latitude": lat,
                    }
                )

        for idx in qc.get("height_outlier_indices", []):
            if 0 <= idx < flight.size:
                lon = float(longitude[idx]) if longitude.size > idx else None
                lat = float(latitude[idx]) if latitude.size > idx else None
                details.append(
                    {
                        "type": "flight_height",
                        "index": int(idx),
                        "distance_m": float(distance[idx])
                        if distance.size > idx
                        else None,
                        "value": float(flight[idx]),
                        "longitude": lon,
                        "latitude": lat,
                    }
                )

        return details

    def _build_airborne_anomaly_text(self) -> str:
        """构建航空异常明细文本。"""
        details = self._build_airborne_anomaly_details()
        if not details:
            return "暂无异常明细"

        lines = []
        for item in details:
            if item["type"] == "trace_spacing":
                lines.append(
                    "道间距异常 | idx={} | 距离={:.2f} m | 间距={:.3f} m | lon={:.7f} | lat={:.7f}".format(
                        item["index"],
                        item["distance_m"],
                        item["value"],
                        item["longitude"] if item["longitude"] is not None else 0.0,
                        item["latitude"] if item["latitude"] is not None else 0.0,
                    )
                )
            else:
                lines.append(
                    "飞行高度异常 | idx={} | 距离={:.2f} m | 高度={:.2f} m | lon={:.7f} | lat={:.7f}".format(
                        item["index"],
                        item["distance_m"] if item["distance_m"] is not None else 0.0,
                        item["value"],
                        item["longitude"] if item["longitude"] is not None else 0.0,
                        item["latitude"] if item["latitude"] is not None else 0.0,
                    )
                )
        return "\n".join(lines)

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
        """导入 CSV 文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 CSV 文件", "", "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if not path:
            return
        sidecar_kwargs = self._build_sidecar_loader_kwargs(path)
        self._load_with_progress(
            "加载 CSV 文件", self._load_single_csv, path, **sidecar_kwargs
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
                self._load_with_progress(
                    "加载 CSV 文件", self._load_single_csv, last_path
                )
        except Exception as e:
            logger.warning("Auto load last data failed: %s", e)
            self._log(f"自动加载上次数据失败: {e}")

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

    def _load_single_csv_with_progress(
        self,
        path,
        progress_callback=None,
        *,
        trace_timestamps_s=None,
        rtk_path=None,
        imu_path=None,
    ):
        """带进度回调的CSV加载"""
        try:
            if progress_callback:
                progress_callback(10, "正在检测文件格式...")

            header_info = detect_csv_header(path)
            skip_lines = _detect_skiprows(path)

            if progress_callback:
                progress_callback(20, "正在读取数据...")

            # 使用分块读取
            rows = []
            total_chunks = 0

            for chunk in pd.read_csv(
                path,
                header=None,
                skiprows=skip_lines,
                chunksize=50000,
                na_filter=False,
                low_memory=False,
            ):
                rows.append(chunk)
                total_chunks += 1

                if progress_callback:
                    percent = min(20 + total_chunks * 5, 80)
                    progress_callback(percent, f"已读取 {total_chunks} 个数据块...")

            if progress_callback:
                progress_callback(85, "正在合并数据...")

            df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            raw_data = df.values

            if raw_data.size == 0:
                raise ValueError("CSV 未读取到有效数据")

            if progress_callback:
                progress_callback(90, "正在处理数据...")

            sidecar_kwargs = {}
            if trace_timestamps_s is not None:
                sidecar_kwargs["trace_timestamps_s"] = trace_timestamps_s
            if rtk_path is not None:
                sidecar_kwargs["rtk_path"] = rtk_path
            if imu_path is not None:
                sidecar_kwargs["imu_path"] = imu_path

            payload_extractor = getattr(
                self, "_extract_airborne_payload_with_optional_sidecars", None
            )
            if payload_extractor is None:
                data, trace_metadata, header_info = extract_airborne_csv_payload(
                    raw_data, header_info, **sidecar_kwargs
                )
            else:
                data, trace_metadata, header_info = payload_extractor(
                    raw_data,
                    header_info,
                    sidecar_kwargs,
                )

            data = np.asarray(data, dtype=np.float32)

            if not np.isfinite(data).all():
                finite_mask = np.isfinite(data)
                fill = float(np.mean(data[finite_mask])) if finite_mask.any() else 0.0
                data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)

            if self.page_advanced.fast_preview_var.isChecked():
                data, _, trace_indices = self._downsample_data(
                    data, return_indices=True
                )
                trace_metadata = subset_trace_metadata(trace_metadata, trace_indices)
                if header_info:
                    header_info = dict(header_info)
                    header_info["a_scan_length"] = int(data.shape[0])
                    header_info["num_traces"] = int(data.shape[1])

            if progress_callback:
                progress_callback(100, "加载完成！")

            return {
                "data": data,
                "header_info": header_info,
                "trace_metadata": trace_metadata,
                "path": path,
            }

        except Exception as e:
            raise ValueError(f"CSV 加载失败: {e}")

    def _load_ascans_folder_with_progress(self, folder, progress_callback=None):
        """带进度回调的文件夹加载"""
        try:
            if progress_callback:
                progress_callback(10, "正在扫描文件夹...")

            max_files = 0
            if (
                hasattr(self, "page_advanced")
                and self.page_advanced.fast_preview_var.isChecked()
            ):
                max_files = self._parse_int_edit(
                    self.page_advanced.max_traces_edit, default=200
                )

            def _progress(current, total, msg):
                if progress_callback:
                    percent = int(10 + (current / max(total, 1)) * 80)
                    progress_callback(percent, msg)

            result = read_ascans_folder(
                folder, max_files=max_files, progress_cb=_progress
            )

            if progress_callback:
                progress_callback(95, "正在处理数据...")

            data = result["data"]
            samples = result["samples_per_trace"]
            traces = result["num_traces"]
            time_step_s = result.get("time_step_s")

            total_time_ns = None
            if time_step_s and samples > 0:
                total_time_ns = time_step_s * samples * 1e9

            header_info = {
                "a_scan_length": samples,
                "total_time_ns": total_time_ns if total_time_ns else 0.0,
                "num_traces": traces,
                "trace_interval_m": 0.01,
                "source": "folder",
                "folder_path": folder,
            }

            if not np.isfinite(data).all():
                finite_mask = np.isfinite(data)
                fill = float(np.mean(data[finite_mask])) if finite_mask.any() else 0.0
                data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)

            if data.ndim == 1:
                data = data.reshape(-1, 1)

            if progress_callback:
                progress_callback(100, "加载完成！")

            return {"data": data, "header_info": header_info, "path": folder}

        except Exception as e:
            raise ValueError(f"文件夹加载失败: {e}")

    def _load_gprmax_out(self, path, progress_callback=None):
        """加载 gprMax .out 文件"""
        try:
            from core.gpr_io import read_gprmax_out

            if progress_callback:
                progress_callback(10, "正在读取 .out 文件...")

            # 读取 .out 文件
            result = read_gprmax_out(path)

            if progress_callback:
                progress_callback(50, "正在构建头信息...")

            data = result["data"]
            samples = result["samples_per_trace"]
            traces = result["num_traces"]
            time_step_s = result.get("time_step_s")
            total_time_ns = result.get("total_time_ns")

            # 尝试从同目录的 .in 文件读取 src_steps，得到正确的道间距
            trace_interval_m = 0.1  # 默认值
            try:
                folder = os.path.dirname(path)
                in_files = [f for f in os.listdir(folder) if f.endswith(".in")]
                if in_files:
                    from core.gpr_io import read_gprmax_in

                    in_cfg = read_gprmax_in(os.path.join(folder, in_files[0]))
                    src_steps = in_cfg.get("src_steps")
                    if src_steps and len(src_steps) >= 1 and src_steps[0] > 0:
                        trace_interval_m = src_steps[0]
            except Exception as e:
                logger.debug("Failed to read src_steps from gprMax .in file: %s", e)

            # 构建头信息
            header_info = {
                "a_scan_length": samples,
                "total_time_ns": total_time_ns if total_time_ns else 0.0,
                "num_traces": traces,
                "trace_interval_m": trace_interval_m,
                "source": "gprmax_out",
                "out_path": path,
            }

            if progress_callback:
                progress_callback(100, "加载完成！")

            return {"data": data, "header_info": header_info, "path": path}

        except Exception as e:
            raise ValueError(f"gprMax .out 加载失败: {e}")

    def _on_data_loaded(self, result):
        """数据加载完成回调"""
        if result is None:
            return

        data = result.get("data")
        header_info = result.get("header_info")
        trace_metadata = result.get("trace_metadata")
        path = result.get("path", "")
        import_warnings = list(result.get("runtime_warnings", []) or [])

        if data is None:
            return

        self._clear_runtime_warnings()
        if not np.isfinite(data).all():
            finite_mask = np.isfinite(data)
            fill_value = float(np.mean(data[finite_mask])) if finite_mask.any() else 0.0
            data = np.nan_to_num(
                data, nan=fill_value, posinf=fill_value, neginf=fill_value
            )
            import_warnings.append(
                build_runtime_warning(
                    "data_sanitized",
                    "导入数据包含 NaN/Inf，已使用均值填充。",
                    fill_value=fill_value,
                    path=path,
                )
            )

        # 更新共享数据
        self.shared_data.load_data(
            data,
            path=path,
            header_info=header_info,
            trace_metadata=trace_metadata,
            source="async_load",
        )

        # 更新UI
        self._mark_data_changed()
        self._clear_transient_compare_snapshots()
        self._set_quality_metrics(None)
        self._append_runtime_warnings(import_warnings, source="async_load")

        self._log(f"已加载数据: {data.shape}")
        if header_info:
            self.status_label.setText(self._build_status_text())
            for line in self._build_airborne_metadata_summary():
                self._log(line)
        else:
            self.status_label.setText(os.path.basename(path))

        self._update_empty_state_and_brief()
        self.plot_data(data)

    def _load_ascans_folder(self, folder: str):
        """从文件夹加载 A-scan 数据"""
        self._log(f"正在从文件夹加载 A-scan: {folder}")
        self._set_busy(True, text="读取 A-scan 文件...")
        self._clear_runtime_warnings()
        QApplication.processEvents()
        try:
            import_warnings = []
            max_files = 0
            if self.page_advanced.fast_preview_var.isChecked():
                max_files = self._parse_int_edit(
                    self.page_advanced.max_traces_edit, default=200
                )

            def _progress(current, total, msg):
                self.status_label.setText(f"{msg} ({current}/{total})")
                QApplication.processEvents()

            result = read_ascans_folder(
                folder, max_files=max_files, progress_cb=_progress
            )
            data = result["data"]
            samples = result["samples_per_trace"]
            traces = result["num_traces"]
            time_step_s = result.get("time_step_s")

            # 构造 header_info
            total_time_ns = None
            if time_step_s and samples > 0:
                total_time_ns = time_step_s * samples * 1e9

            header_info = {
                "a_scan_length": samples,
                "total_time_ns": total_time_ns if total_time_ns else 0.0,
                "num_traces": traces,
                "trace_interval_m": 0.01,
                "source": "folder",
                "folder_path": folder,
            }

            if not np.isfinite(data).all():
                finite_mask = np.isfinite(data)
                fill = float(np.mean(data[finite_mask])) if finite_mask.any() else 0.0
                data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)
                import_warnings.append(
                    build_runtime_warning(
                        "data_sanitized",
                        "导入的 A-scan 数据包含 NaN/Inf，已使用均值填充。",
                        fill_value=fill,
                        path=folder,
                    )
                )

            if data.ndim == 1:
                data = data.reshape(-1, 1)

            self.shared_data.load_data(
                data,
                path=folder,
                header_info=header_info,
                trace_metadata=None,
                source="ascans_folder",
            )
            self._mark_data_changed()
            self._clear_transient_compare_snapshots()
            self._set_quality_metrics(None)

            self._log(f"已加载文件夹 A-scan: {traces} 道 x {samples} 采样点")
            if max_files > 0:
                self._log(
                    f"快速预览已启用：仅导入前 {max_files} 道用于预览；如需全量处理请关闭快速预览。"
                )
                import_warnings.append(
                    build_runtime_warning(
                        "preview_downsampled",
                        "当前为快速预览导入，仅载入了部分道数据。",
                        max_traces=max_files,
                        path=folder,
                    )
                )
            self.status_label.setText(
                f"{os.path.basename(folder)} | 采样:{samples} 道数:{traces}"
                + (" | 快速预览" if max_files > 0 else "")
            )

            self._update_empty_state_and_brief()
            self.plot_data(data)
            self._append_runtime_warnings(import_warnings, source="ascans_folder")

        except Exception as e:
            friendly_msg = f"文件夹加载失败:\n{e}"
            QMessageBox.critical(self, "错误", friendly_msg)
            self._log(friendly_msg)
        finally:
            self._set_busy(False, text="就绪")

    def _load_single_csv(
        self,
        path: str,
        *,
        trace_timestamps_s=None,
        rtk_path=None,
        imu_path=None,
    ):
        """加载单个CSV矩阵文件"""

        try:
            self._clear_runtime_warnings()
            import_warnings = []
            header_info = detect_csv_header(path)
            skip_lines = _detect_skiprows(path)

            if self.page_advanced.fast_preview_var.isChecked():
                max_samples = self._parse_int_edit(
                    self.page_advanced.max_samples_edit, default=0
                )
                max_traces = self._parse_int_edit(
                    self.page_advanced.max_traces_edit, default=0
                )

                target_rows = max_samples if max_samples > 0 else 200000
                if header_info:
                    samples = int(header_info["a_scan_length"])
                    traces = int(header_info["num_traces"])
                    required_rows = samples * traces
                    target_rows = max(target_rows, required_rows)
                    if max_samples > 0 and max_traces > 0:
                        target_rows = max(target_rows, max_samples * max_traces)

                rows = []
                count = 0
                for chunk in pd.read_csv(
                    path,
                    header=None,
                    skiprows=skip_lines,
                    chunksize=200000,
                    na_filter=False,
                    low_memory=False,
                ):
                    rows.append(chunk)
                    count += len(chunk)
                    if count >= target_rows:
                        break
                df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
                raw_data = df.values
            else:
                df = pd.read_csv(
                    path,
                    header=None,
                    skiprows=skip_lines,
                    na_filter=False,
                    low_memory=False,
                )
                raw_data = df.values

            if raw_data.size == 0:
                raise ValueError("CSV 未读取到有效数据（空文件或分隔符不匹配）")

            sidecar_kwargs = {}
            if trace_timestamps_s is not None:
                sidecar_kwargs["trace_timestamps_s"] = trace_timestamps_s
            if rtk_path is not None:
                sidecar_kwargs["rtk_path"] = rtk_path
            if imu_path is not None:
                sidecar_kwargs["imu_path"] = imu_path

            payload_extractor = getattr(
                self, "_extract_airborne_payload_with_optional_sidecars", None
            )
            if payload_extractor is None:
                data, trace_metadata, header_info = extract_airborne_csv_payload(
                    raw_data, header_info, **sidecar_kwargs
                )
            else:
                data, trace_metadata, header_info = payload_extractor(
                    raw_data,
                    header_info,
                    sidecar_kwargs,
                )

            try:
                data = np.asarray(data, dtype=np.float32)
            except Exception as conv_err:
                raise ValueError(f"CSV 包含非数值内容，无法转换为数值矩阵: {conv_err}")

            if data.size == 0:
                raise ValueError("CSV 数据矩阵为空")

            if self.page_advanced.fast_preview_var.isChecked():
                data, _, trace_indices = self._downsample_data(
                    data, return_indices=True
                )
                trace_metadata = subset_trace_metadata(trace_metadata, trace_indices)
                if header_info:
                    header_info = dict(header_info)
                    header_info["a_scan_length"] = int(data.shape[0])
                    header_info["num_traces"] = int(data.shape[1])
                self._log("快速预览：数据已降采样，后续处理/导出将基于当前降采样结果。")
                import_warnings.append(
                    build_runtime_warning(
                        "preview_downsampled",
                        "当前为快速预览导入，已对数据降采样。",
                        max_samples=data.shape[0],
                        max_traces=data.shape[1],
                        path=path,
                    )
                )

            if not np.isfinite(data).all():
                finite_mask = np.isfinite(data)
                if finite_mask.any():
                    fill_value = float(np.mean(data[finite_mask]))
                else:
                    fill_value = 0.0
                data = np.nan_to_num(
                    data, nan=fill_value, posinf=fill_value, neginf=fill_value
                )
                self._log(f"检测到 NaN/Inf，已使用 {fill_value:.6g} 填充。")
                import_warnings.append(
                    build_runtime_warning(
                        "data_sanitized",
                        "导入 CSV 数据包含 NaN/Inf，已使用均值填充。",
                        fill_value=fill_value,
                        path=path,
                    )
                )

            if data.ndim == 1:
                data = data.reshape(-1, 1)

            self.shared_data.load_data(
                data,
                path=path,
                header_info=header_info,
                trace_metadata=trace_metadata,
                source="csv_import",
            )
            self._mark_data_changed()
            self._clear_transient_compare_snapshots()
            self._set_quality_metrics(None)

            self._log(f"已加载 CSV： {path}  shape={data.shape}")
            if self.page_advanced.fast_preview_var.isChecked():
                self._log(
                    "提示：当前为快速预览模式。如需全分辨率处理，请关闭快速预览后重新导入。"
                )
            if header_info:
                self.status_label.setText(
                    f"{os.path.basename(path)} | 采样:{header_info['a_scan_length']} 道数:{header_info['num_traces']}"
                    + (
                        " | 快速预览"
                        if self.page_advanced.fast_preview_var.isChecked()
                        else ""
                    )
                )
            else:
                self.status_label.setText(
                    os.path.basename(path)
                    + (
                        " | 快速预览"
                        if self.page_advanced.fast_preview_var.isChecked()
                        else ""
                    )
                )

            if header_info:
                self._log(
                    "检测到头信息： "
                    f"A-scan length={header_info['a_scan_length']} samples; "
                    f"Total time={header_info['total_time_ns']} ns; "
                    f"A-scan count={header_info['num_traces']}; "
                    f"Trace interval={header_info['trace_interval_m']} m"
                )
                for line in self._build_airborne_metadata_summary():
                    self._log(line)
            else:
                self._log("未检测到头信息；使用索引坐标。")

            self._update_empty_state_and_brief()
            self.plot_data(data)
            self._append_runtime_warnings(import_warnings, source="csv_import")

            # 保存数据路径以便下次自动加载
            _save_last_data_path(path)

        except Exception as e:
            friendly_msg = build_csv_load_error_message(e)
            QMessageBox.critical(self, "错误", friendly_msg)
            self._log(f"加载 CSV 失败：\n{friendly_msg}")

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
        slider_compare = self._is_main_slider_compare_active()

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
        self._draw_selected_trace_marker(axes, axis_info)
        self._draw_manual_roi_marker(axes, axis_info)
        if self._main_view_limits and len(axes) == 1 and not slider_compare:
            axes[0].set_xlim(*self._main_view_limits["xlim"])
            axes[0].set_ylim(*self._main_view_limits["ylim"])

        if last_im is not None and view_style != "wiggle":
            self._draw_colorbar_if_needed(
                last_im, axes, header_info_override=view_header_info
            )
        self.canvas.draw_idle()

        elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
        self._plot_draw_count += 1
        self._last_plot_ms = elapsed_ms
        self._refresh_observability_panel()

    # ============ 处理方法 ============

    def apply_method(self):
        """应用当前选中的方法"""
        if self.data is None or self.data_path is None:
            QMessageBox.warning(self, "无数据", "请先导入数据。")
            return
        idx = self.page_basic.method_combo.currentIndex()
        method_key = self.page_basic.method_keys[idx]
        method = PROCESSING_METHODS[method_key]
        self._log(f"正在应用: {method['name']}")

        try:
            visible_params = self.page_basic.get_current_params()
        except ValueError as e:
            QMessageBox.critical(self, "参数错误", str(e))
            return

        # 基础页会隐藏部分高级参数；运行时需要把隐藏参数的默认值/覆盖值一并带上。
        params = self._resolve_method_params(method_key)
        params.update(visible_params)

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
        self._log("运行默认流程v3：zero-time → dewow → 背景抑制 → AGC → 尖锐杂波抑制")
        order = [
            "set_zero_time",
            "dewow",
            "subtracting_average_2D",
            "agcGain",
            "running_average_2D",
        ]
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
        profile = RECOMMENDED_RUN_PROFILES.get(profile_key)
        if not profile:
            QMessageBox.warning(self, "配置错误", f"未知推荐配置：{profile_key}")
            return

        preset_key = profile.get("preset_key")
        if preset_key:
            self._apply_preset_by_key(preset_key)

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

    def generate_report(self):
        """生成报告"""
        if self.data is None or self.data_path is None:
            QMessageBox.warning(self, "无数据", "请先导入数据。")
            return
        out_dir = self._default_output_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(out_dir, f"report_{ts}.md")
        image_path = os.path.join(out_dir, f"report_{ts}.png")

        try:
            self.fig.savefig(image_path, dpi=150)
        except Exception as e:
            self._log(f"报告截图失败: {e}")

        bounds = None
        try:
            report_data = self._apply_preprocess(
                np.asarray(self.data, dtype=np.float32)
            )
            report_time_axis = self._build_time_axis(report_data.shape[0])
            report_trace_axis = self._build_trace_axis(report_data.shape[1])
            bounds = self._get_crop_bounds(
                report_data,
                report_time_axis,
                report_trace_axis,
            )
        except Exception as e:
            self._log(f"报告裁剪信息获取失败: {e}")
            bounds = None

        last_run = self._last_run_summary or {}
        method_key = self.page_basic.method_keys[
            self.page_basic.method_combo.currentIndex()
        ]
        method_name = PROCESSING_METHODS[method_key]["name"]
        try:
            params = self.page_basic.get_current_params()
        except Exception as e:
            self._log(f"参数解析失败: {e}")
            params = {}

        lines = []
        lines.append(f"# GPR GUI Report ({ts})")
        lines.append("")
        lines.append(f"- Data file: {self.data_path}")
        lines.append("")
        lines.append("## Line Summary")
        lines.append("```")
        lines.append(self._build_airborne_line_summary_text())
        lines.append("```")
        if last_run:
            lines.append(f"- Last run: {last_run.get('label', method_name)}")
            lines.append(f"- Run type: {last_run.get('run_type', 'unknown')}")
            lines.append(f"- Run timestamp: {last_run.get('timestamp', '--')}")
            if last_run.get("preset_key"):
                lines.append(f"- Preset: {last_run['preset_key']}")
            if last_run.get("profile_key"):
                lines.append(f"- Profile: {last_run['profile_key']}")
            steps = last_run.get("steps", [])
            if steps:
                lines.append("- Steps:")
                for idx, step in enumerate(steps, start=1):
                    step_name = step.get(
                        "method_name", step.get("method_key", f"step-{idx}")
                    )
                    step_params = step.get("params") or {}
                    step_ms = step.get("elapsed_ms")
                    suffix = f" | {step_ms:.1f} ms" if step_ms is not None else ""
                    lines.append(f"  - [{idx}] {step_name}{suffix}")
                    if step_params:
                        lines.append(f"    params: {step_params}")
            if last_run.get("notes"):
                for note in last_run["notes"]:
                    lines.append(f"- Note: {note}")
            if last_run.get("warnings"):
                lines.append("- Warnings:")
                for warning in last_run.get("warnings", []):
                    lines.append(f"  - {format_runtime_warning_text(warning)}")
        else:
            lines.append(f"- Method: {method_name}")
            if params:
                lines.append(f"- Params: {params}")

        lines.append(f"- Current method selection: {method_name}")
        lines.append(f"- 色图: {self._get_colormap()}")
        lines.append(f"- 显示色标: {self.page_advanced.show_cbar_var.isChecked()}")
        lines.append(f"- 显示网格: {self.page_advanced.show_grid_var.isChecked()}")
        lines.append(
            f"- Symmetric stretch: {self.page_advanced.symmetric_var.isChecked()}"
        )
        if self.page_advanced.percentile_var.isChecked():
            lines.append(
                f"- 百分位拉伸: {self.page_advanced.percentile_var.isChecked()} (low={self.page_advanced.p_low_edit.text()}, high={self.page_advanced.p_high_edit.text()})"
            )
        else:
            lines.append(
                f"- 百分位拉伸: {self.page_advanced.percentile_var.isChecked()}"
            )
        lines.append(f"- Normalize: {self.page_advanced.normalize_var.isChecked()}")
        lines.append(f"- Demean: {self.page_advanced.demean_var.isChecked()}")
        lines.append(
            f"- Display downsample: {self.page_advanced.display_downsample_var.isChecked()} (max_samples={self.page_advanced.display_max_samples_edit.text()}, max_traces={self.page_advanced.display_max_traces_edit.text()})"
        )
        if bounds:
            lines.append(
                f"- Crop: time {bounds['time_start']}~{bounds['time_end']} ; distance {bounds['dist_start']}~{bounds['dist_end']}"
            )
        else:
            lines.append("- Crop: disabled")
        quality = self._last_quality_metrics or {}
        if quality:
            lines.append(
                "- Quality metrics: focus_ratio={focus_ratio:.4f}, hot_pixels={hot_pixels}, spikiness={spikiness:.3f}, time_ms={time_ms:.1f}".format(
                    **quality
                )
            )
        else:
            lines.append("- Quality metrics: --")
        lines.append("")
        lines.append(f"- Screenshot: {image_path}")
        lines.append("")
        lines.append("## Runtime State")
        lines.append(
            f"- Data shape: {self.data.shape if self.data is not None else '--'}"
        )
        lines.append(
            f"- Fast preview import: {self.page_advanced.fast_preview_var.isChecked()}"
        )
        airborne_lines = self._build_airborne_metadata_summary()
        if airborne_lines:
            lines.append("- Airborne metadata:")
            for item in airborne_lines:
                lines.append(f"  - {item}")
        airborne_qc = self._compute_airborne_qc_metrics()
        if airborne_qc:
            lines.append("- Airborne QC:")
            lines.append(f"  - track_length_m: {airborne_qc['track_length_m']:.2f}")
            lines.append(f"  - trace_spacing_cv: {airborne_qc['trace_spacing_cv']:.3f}")
            lines.append(
                f"  - flight_height_span_m: {airborne_qc['flight_height_span_m']:.2f}"
            )
            lines.append(f"  - spacing_outliers: {airborne_qc['spacing_outliers']}")
            lines.append(f"  - height_outliers: {airborne_qc['height_outliers']}")
            lines.append(
                "  - alerts: "
                + (
                    ", ".join(airborne_qc.get("alerts", []))
                    if airborne_qc.get("alerts")
                    else "正常"
                )
            )
        anomaly_lines = self._build_airborne_anomaly_details()
        if anomaly_lines:
            lines.append("- Airborne anomaly details:")
            for item in anomaly_lines:
                lines.append(
                    "  - {type}: idx={index}, distance_m={distance_m}, value={value}, lon={longitude}, lat={latitude}".format(
                        **item
                    )
                )
        if self._runtime_warnings:
            lines.append("- Runtime warnings:")
            for warning in self._runtime_warnings:
                lines.append(f"  - {format_runtime_warning_text(warning)}")
        lines.append("")
        lines.append("## Log")
        log_text = self.page_quality.get_record_text().strip()
        if not log_text:
            log_text = self.page_basic.info.toPlainText().strip()
        lines.append("```")
        lines.append(log_text)
        lines.append("```")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self._log(f"报告已保存: {report_path}")

    def export_record(self):
        """导出记录"""
        text = self.page_quality.get_record_text()
        if not text:
            QMessageBox.information(self, "提示", "记录为空。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存记录", "record.txt", "Text (*.txt);;All files (*)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self._log(f"记录已导出：{path}")

    def open_log_directory(self):
        """打开日志目录。"""
        log_dir = get_logs_dir()
        os.startfile(log_dir)
        self._log(f"已打开日志目录：{log_dir}")

    def copy_diagnostics(self):
        """复制诊断信息到剪贴板。"""
        lines = [
            f"版本: {self.version_text}",
            f"数据路径: {self.data_path}",
            f"数据尺寸: {self.data.shape if self.data is not None else '--'}",
            f"头信息: {self.header_info}",
            f"道元数据键: {sorted((self.trace_metadata or {}).keys())}",
            f"测线摘要: {self._build_airborne_line_summary_text()}",
            f"辅助文件: {self._sidecar_files}",
            f"当前预设: {self._selected_preset_key}",
            f"上次运行: {self._last_run_summary}",
            f"质量指标: {self._last_quality_metrics}",
            f"运行告警: {self._runtime_warnings}",
            f"航空质控: {self._compute_airborne_qc_metrics()}",
            f"日志文件: {os.path.join(get_logs_dir(), 'gpr_gui.log')}",
        ]
        lines.extend(self._build_airborne_metadata_summary())
        details = self._build_airborne_anomaly_details()
        if details:
            lines.append("航空异常明细:")
            lines.extend([str(item) for item in details])
        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        self._log("诊断信息已复制到剪贴板")

    def _pick_sidecar_file(self, kind: str):
        """选择可选 RTK/IMU sidecar 文件；取消选择时保留原状态。"""
        if kind not in {"rtk", "imu"}:
            raise ValueError(f"不支持的 sidecar 类型: {kind}")

        label = "RTK" if kind == "rtk" else "IMU"
        current_path = self._sidecar_files.get(kind)
        initial_dir = os.path.dirname(current_path) if current_path else BASE_DIR
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"选择 {label} sidecar 文件",
            initial_dir,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        self._set_sidecar_file(kind, path)

    def _set_sidecar_file(self, kind: str, path=None) -> None:
        """更新单个 sidecar 路径，并同步高级设置页标签。"""
        if kind not in {"rtk", "imu"}:
            raise ValueError(f"不支持的 sidecar 类型: {kind}")

        normalized = str(path) if path else None
        self._sidecar_files[kind] = normalized
        label_widget = getattr(self.page_advanced, f"{kind}_sidecar_label", None)
        if label_widget is not None:
            label_widget.setText(os.path.basename(normalized) if normalized else "未选择")
            label_widget.setToolTip(normalized or "未选择")
        display = os.path.basename(normalized) if normalized else "未选择"
        self._log(f"{kind.upper()} sidecar：{display}")

    def _clear_sidecar_file(self, kind: str) -> None:
        """仅清除指定 RTK/IMU sidecar 选择。"""
        self._set_sidecar_file(kind, None)

    def _warn_sidecar_ignored(self, kind: str, reason: str) -> None:
        """提示用户已忽略可选 sidecar，同时保持 CSV 正常加载。"""
        message = f"已忽略可选 {kind} 辅助文件，CSV 将按普通数据继续加载。\n原因：{reason}"
        self._log(message.replace("\n", " "))
        if hasattr(self, "status_label"):
            self.status_label.setText(f"已忽略可选 {kind} 辅助文件")
        QMessageBox.warning(self, "可选辅助文件已忽略", message)

    def _store_trace_timestamps_from_metadata(self, trace_metadata) -> None:
        """从既有每道元数据同步 trace 时间戳缓存；不在 GUI 层推导。"""
        timestamps = None
        if isinstance(trace_metadata, dict):
            timestamps = trace_metadata.get("trace_timestamp_s")
        if timestamps is None:
            self._trace_timestamps_s = None
            return
        self._trace_timestamps_s = np.asarray(timestamps, dtype=np.float64).copy()

    def _get_trace_timestamps_for_sidecars(self):
        """仅返回当前会话已存在的 trace 时间戳，不在 GUI 层推导。"""
        self._store_trace_timestamps_from_metadata(self.trace_metadata)
        return getattr(self, "_trace_timestamps_s", None)

    def _read_explicit_trace_timestamps_from_csv(self, path: str):
        """从主 CSV 第 6 列读取显式 trace 时间戳；不根据采样参数推导。"""
        try:
            header_info = detect_csv_header(path)
            if not header_info:
                return None
            samples = int(header_info["a_scan_length"])
            traces = int(header_info["num_traces"])
            required_rows = samples * traces
            skip_lines = _detect_skiprows(path)
            timestamp_col = pd.read_csv(
                path,
                header=None,
                skiprows=skip_lines,
                usecols=[5],
                nrows=required_rows,
                na_filter=False,
                low_memory=False,
            ).iloc[:, 0]
        except (OSError, ValueError, KeyError, IndexError, pd.errors.ParserError):
            return None

        values = pd.to_numeric(timestamp_col, errors="coerce").to_numpy(dtype=np.float64)
        if values.size < required_rows:
            return None
        trace_timestamps_s = values.reshape((traces, samples))[:, 0]
        if not np.isfinite(trace_timestamps_s).all():
            return None
        return trace_timestamps_s.copy()

    def _is_current_data_path(self, path: str) -> bool:
        """Return True only when `path` is the already loaded data source."""
        current_path = getattr(self, "data_path", None)
        if not current_path:
            return False
        try:
            return os.path.abspath(str(current_path)) == os.path.abspath(str(path))
        except (OSError, TypeError, ValueError):
            return False

    def _build_sidecar_loader_kwargs(self, path: str) -> dict:
        """根据当前选择构造 CSV 加载 sidecar kwargs。"""
        rtk_path = self._sidecar_files.get("rtk")
        imu_path = self._sidecar_files.get("imu")
        if not rtk_path and not imu_path:
            return {}

        trace_timestamps_s = self._read_explicit_trace_timestamps_from_csv(path)
        if trace_timestamps_s is None and self._is_current_data_path(path):
            trace_timestamps_s = self._get_trace_timestamps_for_sidecars()
        if trace_timestamps_s is None:
            self._warn_sidecar_ignored(
                "RTK/IMU",
                "缺少可用于对齐的 trace_timestamps_s；本次不会接入辅助文件。",
            )
            return {}

        kwargs = {"trace_timestamps_s": trace_timestamps_s}
        if rtk_path:
            kwargs["rtk_path"] = rtk_path
        if imu_path:
            kwargs["imu_path"] = imu_path
        return kwargs

    def _extract_airborne_payload_with_optional_sidecars(
        self, raw_data: np.ndarray, header_info, sidecar_kwargs: dict
    ):
        """提取 CSV payload；可选 sidecar 出错时警告后回退为普通 CSV。"""
        try:
            return extract_airborne_csv_payload(
                raw_data, header_info, **sidecar_kwargs
            )
        except ValueError as exc:
            if not sidecar_kwargs or not self._is_optional_sidecar_error(exc):
                raise
            self._warn_sidecar_ignored(
                self._describe_selected_sidecars(sidecar_kwargs), str(exc)
            )
            return extract_airborne_csv_payload(raw_data, header_info)

    def _describe_selected_sidecars(self, sidecar_kwargs: dict) -> str:
        """返回用于告警文案的已选 sidecar 类型。"""
        selected = []
        if sidecar_kwargs.get("rtk_path"):
            selected.append("RTK")
        if sidecar_kwargs.get("imu_path"):
            selected.append("IMU")
        return "/".join(selected) or "RTK/IMU"

    def _is_optional_sidecar_error(self, exc: ValueError) -> bool:
        """判断 ValueError 是否来自可选 sidecar 解析/对齐链。"""
        text = str(exc).lower()
        markers = (
            "sidecar",
            "trace_timestamps_s",
            "rtk",
            "imu",
            "timestamp_s",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "parser",
            "parse",
            "optional sidecar integration",
            "unsupported sidecar",
        )
        if any(marker in text for marker in markers):
            return True
        if ("longitude" in text or "latitude" in text) and (
            "sidecar" in text or "rtk" in text or "timestamp" in text
        ):
            return True
        return False

    def export_quality_snapshot(self):
        """导出质量快照"""
        if not self._last_quality_metrics:
            QMessageBox.information(
                self, "无质量指标", "请先运行一次处理流程，再导出质量快照。"
            )
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self._default_output_dir()
        base_name = f"quality_snapshot_{ts}"
        json_path = os.path.join(out_dir, f"{base_name}.json")
        csv_path = os.path.join(out_dir, f"{base_name}.csv")

        selected_method_key = self.page_basic.get_current_method_key()
        selected_method_label = self.page_basic.method_combo.currentText()

        alerts = {
            k: self._is_metric_alert(k, float(self._last_quality_metrics.get(k, 0.0)))
            for k in ["focus_ratio", "hot_pixels", "spikiness", "time_ms"]
        }

        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "data_path": self.data_path,
            "version": self.version_text,
            "preset_key": self._selected_preset_key,
            "line_summary_text": self._build_airborne_line_summary_text(),
            "sidecar_files": dict(self._sidecar_files),
            "airborne_metadata_summary": self._build_airborne_metadata_summary(),
            "airborne_qc": self._compute_airborne_qc_metrics(),
            "airborne_anomaly_details": self._build_airborne_anomaly_details(),
            "selected_method": {
                "key": selected_method_key,
                "label": selected_method_label,
            },
            "metrics": dict(self._last_quality_metrics),
            "thresholds": dict(self._quality_thresholds),
            "alerts": alerts,
            "method_param_overrides": dict(self._method_param_overrides),
            "runtime_warnings": list(self._runtime_warnings),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        rows = []
        for metric in ["focus_ratio", "hot_pixels", "spikiness", "time_ms"]:
            th = self._quality_thresholds.get(metric, {})
            rows.append(
                {
                    "metric": metric,
                    "value": self._last_quality_metrics.get(metric),
                    "threshold_min": th.get("min"),
                    "threshold_max": th.get("max"),
                    "alert": alerts.get(metric, False),
                    "preset_key": self._selected_preset_key,
                    "data_path": self.data_path,
                    "timestamp": payload["timestamp"],
                }
            )
        airborne_qc = payload.get("airborne_qc") or {}
        for metric in [
            "track_length_m",
            "trace_spacing_cv",
            "flight_height_span_m",
            "spacing_outliers",
            "height_outliers",
        ]:
            if metric in airborne_qc:
                rows.append(
                    {
                        "metric": metric,
                        "value": airborne_qc.get(metric),
                        "threshold_min": None,
                        "threshold_max": None,
                        "alert": False,
                        "preset_key": self._selected_preset_key,
                        "data_path": self.data_path,
                        "timestamp": payload["timestamp"],
                    }
                )
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        try:
            json_disp = os.path.relpath(json_path, BASE_DIR)
            csv_disp = os.path.relpath(csv_path, BASE_DIR)
        except ValueError:
            json_disp = json_path
            csv_disp = csv_path
        self._log(f"质量快照已导出: {json_disp}; {csv_disp}")
        self.status_label.setText("质量快照导出完成")
        QMessageBox.information(self, "导出成功", f"已导出:\n{json_path}\n{csv_path}")

    # ============ 辅助方法（由于篇幅限制，这里只列出关键方法） ============

    def _parse_int_edit(self, edit, default: int = 0) -> int:
        """解析整数输入"""
        text = (edit.text() or "").strip()
        if text == "":
            return default
        try:
            return int(float(text))
        except Exception:
            return default

    def _get_colormap(self, header_info_override: dict | None = None):
        """获取当前色图"""
        cmap = (self.page_advanced.cmap_combo.currentText() or "gray").strip()
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        if (
            cmap == "gray"
            and header.get("display_hint") == "signed_migration"
            and not self.page_advanced.cmap_invert_var.isChecked()
        ):
            return "seismic"
        if self.page_advanced.cmap_invert_var.isChecked():
            if cmap.endswith("_r"):
                cmap = cmap[:-2]
            else:
                cmap = cmap + "_r"
        return cmap

    def _set_display_override(
        self,
        data: np.ndarray | None,
        header_info: dict | None = None,
        trace_metadata: dict | None = None,
    ):
        """设置仅用于主图显示的临时数据载荷。"""
        self._display_data_override = (
            np.array(data, copy=False) if data is not None else None
        )
        self._display_header_info_override = dict(header_info or {}) or None
        self._display_trace_metadata_override = trace_metadata

    def _clear_display_override(self):
        """清除主图显示覆盖，回到当前正式数据。"""
        self._display_data_override = None
        self._display_header_info_override = None
        self._display_trace_metadata_override = None

    def _get_active_plot_payload(
        self, fallback_data: np.ndarray | None = None
    ) -> tuple[np.ndarray | None, dict | None, dict | None]:
        """返回当前应绘制的数据载荷。"""
        if self._display_data_override is not None:
            return (
                self._display_data_override,
                self._display_header_info_override or self.header_info,
                self._display_trace_metadata_override,
            )
        return (
            fallback_data if fallback_data is not None else self.data,
            self.header_info,
            self.trace_metadata,
        )

    def _get_single_plot_title(self, header_info_override: dict | None = None) -> str:
        """获取单图模式下的标题。"""
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        title = str(header.get("display_title") or "").strip() if header else ""
        base_title = title or "B-扫"

        current_label = str(getattr(self.shared_data, "current_label", "") or "").strip()
        if current_label and current_label != "原始数据" and current_label not in base_title:
            return f"{base_title} - 当前处理：{current_label}"
        return base_title

    def _resolve_method_params(self, method_key: str):
        """解析方法参数"""
        method = PROCESSING_METHODS[method_key]
        defaults = {p["name"]: p.get("default") for p in method.get("params", [])}
        overrides = self._method_param_overrides.get(method_key, {})
        defaults.update(overrides)
        return defaults

    def _build_tasks_from_order(self, order: list, out_dir: str):
        """从顺序构建任务列表"""
        tasks = []
        for key in order:
            if key not in self.page_basic.method_keys:
                continue
            tasks.append(self._build_single_task(key, out_dir))
        return tasks

    def _build_single_task(
        self, method_key: str, out_dir: str, param_source_mode: str = "manual"
    ) -> dict:
        """构建单个任务字典"""
        method = PROCESSING_METHODS[method_key]
        params = self._resolve_method_params(method_key)
        return {
            "method_key": method_key,
            "method": method,
            "params": params,
            "out_dir": out_dir,
            "param_source_mode": param_source_mode,
        }

    def _apply_preset_by_key(self, preset_key: str):
        """根据预设键应用预设参数"""
        if preset_key not in GUI_PRESETS_V1:
            QMessageBox.warning(self, "预设错误", f"未知预设：{preset_key}")
            return

        preset = GUI_PRESETS_V1[preset_key]
        self._selected_preset_key = preset_key
        self._log(f"应用预设: {preset_key} - {preset['label']}")
        self._apply_preset_ui_values(preset.get("ui"), preset_key=preset_key)
        self._apply_preset_method_params(preset.get("method_params"))
        if self.data is not None:
            self._refresh_plot()

    def backfill_current_method_params(self):
        """回填当前方法的参数"""
        if self.data is None:
            QMessageBox.information(self, "回填参数", "请先加载数据。")
            return

        current_method_key = self.page_basic.get_current_method_key()
        if not current_method_key:
            QMessageBox.warning(self, "回填参数", "请选择一个处理方法。")
            return

        # 从已应用的预设中获取参数，如果没有则从方法默认参数中获取
        method_params = self._method_param_overrides.get(current_method_key, {})
        if not method_params:
            # 如果没有覆盖参数，尝试从方法定义中获取默认参数
            method_definition = PROCESSING_METHODS.get(current_method_key)
            if method_definition and "params" in method_definition:
                method_params = {
                    p["name"]: p.get("default") for p in method_definition["params"]
                }

        if method_params:
            self.page_basic.apply_method_params(current_method_key, method_params)
            self._log(
                f"已回填当前方法({PROCESSING_METHODS[current_method_key]['name']})的参数。"
            )
        else:
            QMessageBox.information(self, "回填参数", "当前方法没有可回填的参数。")

    def import_tzt_as_migration_defaults(self):
        """导入 TZT 文件作为迁移默认值"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Kirchhoff 参数文件",
            "",
            "Kirchhoff 参数 (*.txt *.tzt);;所有文件 (*)",
        )
        if not path:
            return
        try:
            params = load_cagpr_kir_parameter_file(path)
            if not params:
                QMessageBox.information(
                    self, "参数导入", "未识别到可用的 Kirchhoff 参数。"
                )
                return
            self._method_param_overrides["kirchhoff_migration"] = dict(params)
            self.page_basic.apply_method_params("kirchhoff_migration", params)
            current_key = self.page_basic.get_current_method_key()
            if current_key != "kirchhoff_migration":
                idx = self.page_basic.method_keys.index("kirchhoff_migration")
                self.page_basic.method_combo.setCurrentIndex(idx)
            self._log(f"已从参数文件导入 Kirchhoff 默认配置: {path}")
        except Exception as e:
            QMessageBox.warning(
                self, "参数导入失败", f"无法导入 Kirchhoff 参数文件:\n{e}"
            )
            self._log(f"导入 Kirchhoff 参数文件失败: {e}")

    def _reset_crop(self):
        """重置裁剪设置"""
        self.page_advanced.crop_enable_var.setChecked(False)
        self.page_advanced.time_start_edit.setText("")
        self.page_advanced.time_end_edit.setText("")
        self.page_advanced.dist_start_edit.setText("")
        self.page_advanced.dist_end_edit.setText("")
        self._log("裁剪设置已重置。")
        self._refresh_plot()

    def cancel_processing(self):
        """取消处理"""
        if (
            (self._worker is None or self._worker_thread is None)
            and (self._auto_tune_worker is None or self._auto_tune_thread is None)
            and (
                self._auto_tune_stage_worker is None
                or self._auto_tune_stage_thread is None
            )
        ):
            self.status_label.setText("当前无可取消任务")
            return
        if self._cancel_in_flight:
            self.status_label.setText("正在取消...（等待当前步骤安全退出）")
            return
        try:
            self._cancel_in_flight = True
            if self._worker is not None:
                self._worker.request_cancel()
            if self._auto_tune_worker is not None:
                self._auto_tune_worker.request_cancel()
            if self._auto_tune_stage_worker is not None:
                self._auto_tune_stage_worker.request_cancel()
            self.page_basic.btn_cancel.setEnabled(False)
            self.status_label.setText("正在取消...（等待当前步骤安全退出）")
            self._log("收到取消请求：将于当前步骤完成后停止")
        except Exception as e:
            self._cancel_in_flight = False
            self._log(f"取消请求失败: {e}")

    # ============ 绘图辅助方法 ============

    def _build_plot_signature(self):
        """构建绘图签名用于缓存判断"""
        plot_data, plot_header_info, _ = self._get_active_plot_payload(self.data)
        if plot_data is None:
            return None
        header = plot_header_info or {}
        skip_preprocess = bool(header.get("display_skip_preprocess"))
        skip_downsample = bool(header.get("display_skip_downsample"))
        slider_compare = bool(
            hasattr(self.page_advanced, "slider_compare_var")
            and self.page_advanced.slider_compare_var.isChecked()
        )
        sig = {
            "shape": plot_data.shape,
            "revision": self._data_revision,
            "display_override": self._display_data_override is not None,
            "manual_roi": tuple(sorted((self._manual_roi_values or {}).items())),
            "view_limits": tuple(sorted((self._main_view_limits or {}).items())),
            "cmap": self._get_colormap(plot_header_info),
            "view_style": self.page_advanced.get_view_style(),
            "symmetric": self.page_advanced.symmetric_var.isChecked(),
            "chatgpt_style": self.page_advanced.chatgpt_style_var.isChecked(),
            "compare": self.page_advanced.compare_var.isChecked(),
            "slider_compare": slider_compare,
            "cmap_invert": self.page_advanced.cmap_invert_var.isChecked(),
            "show_cbar": self.page_advanced.show_cbar_var.isChecked(),
            "show_grid": self.page_advanced.show_grid_var.isChecked(),
            "percentile": self.page_advanced.percentile_var.isChecked(),
            "normalize": False
            if skip_preprocess
            else self.page_advanced.normalize_var.isChecked(),
            "demean": False
            if skip_preprocess
            else self.page_advanced.demean_var.isChecked(),
            "crop": self.page_advanced.crop_enable_var.isChecked(),
            "downsample": False
            if skip_downsample
            else self.page_advanced.display_downsample_var.isChecked(),
        }
        if self.page_advanced.compare_var.isChecked() or slider_compare:
            sig["left"] = self.page_advanced.compare_left_combo.currentText()
            sig["right"] = self.page_advanced.compare_right_combo.currentText()
            sig["diff"] = self.page_advanced.diff_var.isChecked()
            sig["slider_ratio"] = round(float(self._main_slider_compare_ratio), 4)
        return tuple(sorted(sig.items()))

    def _prepare_view_data(
        self,
        data: np.ndarray,
        header_info_override: dict | None = None,
        trace_metadata_override: dict | None = None,
    ):
        """准备用于显示的数据（裁剪→降采样→预处理，减少内存分配）"""
        start_ts = time.perf_counter()
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        display_data = np.array(data, copy=False)
        time_axis = self._build_time_axis(display_data.shape[0], header_info_override)
        trace_axis = self._build_trace_axis(
            display_data.shape[1], trace_metadata_override, header_info_override
        )
        trace_indices = np.arange(display_data.shape[1], dtype=np.int32)
        bounds = (
            self._get_crop_bounds(display_data, time_axis, trace_axis)
            if self.page_advanced.crop_enable_var.isChecked()
            else None
        )
        if bounds:
            t0, t1 = bounds["time_start_idx"], bounds["time_end_idx"]
            d0, d1 = bounds["dist_start_idx"], bounds["dist_end_idx"]
            display_data = display_data[t0:t1, d0:d1]
            time_axis = time_axis[t0:t1]
            trace_axis = trace_axis[d0:d1]
            trace_indices = trace_indices[d0:d1]
        if self.page_advanced.display_downsample_var.isChecked() and not header.get(
            "display_skip_downsample"
        ):
            max_samples = self._parse_int_edit(
                self.page_advanced.display_max_samples_edit, default=800
            )
            max_traces = self._parse_int_edit(
                self.page_advanced.display_max_traces_edit, default=400
            )
            display_data, sample_idx, trace_idx = self._downsample_for_display(
                display_data, max_samples, max_traces, return_indices=True
            )
            time_axis = time_axis[sample_idx]
            trace_axis = trace_axis[trace_idx]
            trace_indices = trace_indices[trace_idx]
        display_data = self._apply_preprocess(
            display_data, header_info_override=header_info_override
        )
        display_data = self._apply_display_transform(
            display_data, header_info_override=header_info_override
        )
        self._last_prepare_ms = (time.perf_counter() - start_ts) * 1000.0
        return (
            display_data,
            bounds,
            {
                "time_axis": time_axis,
                "trace_axis": trace_axis,
                "trace_indices": trace_indices,
            },
        )

    def _build_time_axis(
        self, n_samples: int, header_info_override: dict | None = None
    ) -> np.ndarray:
        """构建时间轴（ns 或采样索引）。"""
        header = (
            header_info_override
            if header_info_override is not None
            else self.header_info
        )
        if header and header.get("is_elevation"):
            elevation_top = header.get("elevation_top_m")
            depth_step = header.get("depth_step_m")
            if (
                elevation_top is not None
                and depth_step is not None
                and float(depth_step) > 0
            ):
                return float(elevation_top) - np.arange(
                    n_samples, dtype=np.float32
                ) * float(depth_step)
            elevation_bottom = header.get("elevation_bottom_m")
            if elevation_top is not None and elevation_bottom is not None:
                return np.linspace(
                    float(elevation_top),
                    float(elevation_bottom),
                    n_samples,
                    dtype=np.float32,
                )
        if header and header.get("is_depth"):
            depth_step = header.get("depth_step_m")
            if depth_step is not None and float(depth_step) > 0:
                return np.arange(n_samples, dtype=np.float32) * float(depth_step)
            depth_max = header.get("depth_max_m")
            if depth_max is not None:
                return np.linspace(0.0, float(depth_max), n_samples, dtype=np.float32)
        if header and header.get("total_time_ns"):
            total_time_ns = float(header["total_time_ns"])
            return np.linspace(0.0, total_time_ns, n_samples, dtype=np.float32)
        return np.arange(n_samples, dtype=np.float32)

    def _build_trace_axis(
        self,
        n_traces: int,
        trace_metadata_override: dict | None = None,
        header_info_override: dict | None = None,
    ) -> np.ndarray:
        """构建距离轴（真实距离或均匀道距）。"""
        meta = (
            trace_metadata_override
            if trace_metadata_override is not None
            else (self.trace_metadata or {})
        )
        distance = meta.get("trace_distance_m")
        if distance is not None:
            distance = np.asarray(distance, dtype=np.float32)
            if distance.ndim == 1 and distance.size >= n_traces:
                return distance[:n_traces]
        header = (
            header_info_override
            if header_info_override is not None
            else self.header_info
        )
        if header and header.get("trace_interval_m") is not None:
            interval = float(header.get("trace_interval_m", 0.0))
            return np.arange(n_traces, dtype=np.float32) * interval
        return np.arange(n_traces, dtype=np.float32)

    def _apply_preprocess(
        self, data: np.ndarray, header_info_override: dict | None = None
    ) -> np.ndarray:
        """应用预处理（无预处理时跳过拷贝）"""
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        if header.get("display_skip_preprocess"):
            return data
        do_norm = self.page_advanced.normalize_var.isChecked()
        do_demean = self.page_advanced.demean_var.isChecked()
        if not do_norm and not do_demean:
            return data
        result = np.array(data, copy=True)
        if do_norm:
            max_val = np.max(np.abs(result))
            if max_val > 0:
                result /= max_val
        if self.page_advanced.demean_var.isChecked():
            result -= np.mean(result, axis=0, keepdims=True)
        return result

    def _apply_display_transform(
        self, data: np.ndarray, header_info_override: dict | None = None
    ) -> np.ndarray:
        """应用仅用于显示的变换，例如 CaGPR 的对比度拉伸。"""
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        contrast = header.get("display_cagpr_contrast")
        if contrast is None:
            return data
        return self._normalize_cagpr_display(data, float(contrast))

    def _normalize_cagpr_display(self, data: np.ndarray, contrast: float) -> np.ndarray:
        """Match CaGPR's display clipping and normalization path."""
        arr = np.asarray(data, dtype=np.float64)
        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return np.zeros_like(arr, dtype=np.float64)
        data_min = float(np.nanmin(arr))
        data_max = float(np.nanmax(arr))
        center = float(np.nanmean(arr))
        scale = 1.0 / (1.0 + contrast) if contrast > 0 else 1.0 - contrast
        half_range = max((data_max - data_min) * 0.5 * scale, 1.0e-12)
        vmin = center - half_range
        vmax = center + half_range
        clipped = np.array(arr, copy=True)
        clipped[finite_mask] = np.clip(clipped[finite_mask], vmin, vmax)
        normalized = np.full_like(clipped, np.nan, dtype=np.float64)
        normalized[finite_mask] = (
            2.0 * (clipped[finite_mask] - vmin) / max(vmax - vmin, 1.0e-12) - 1.0
        )
        return normalized

    def _get_crop_bounds(
        self, data: np.ndarray, time_axis: np.ndarray, trace_axis: np.ndarray
    ):
        """获取裁剪边界"""
        n_samples, n_traces = data.shape[0], data.shape[1]
        t_start = self._parse_float_edit(
            self.page_advanced.time_start_edit, default=None
        )
        t_end = self._parse_float_edit(self.page_advanced.time_end_edit, default=None)
        d_start = self._parse_float_edit(
            self.page_advanced.dist_start_edit, default=None
        )
        d_end = self._parse_float_edit(self.page_advanced.dist_end_edit, default=None)
        use_physical_time = bool(
            self.header_info
            and (
                self.header_info.get("total_time_ns")
                or self.header_info.get("is_depth")
                or self.header_info.get("is_elevation")
            )
        )
        use_physical_dist = bool(
            self.trace_metadata is not None
            and "trace_distance_m" in self.trace_metadata
        ) or bool(self.header_info and self.header_info.get("trace_interval_m"))

        time_start_idx = (
            self._axis_value_to_index(time_axis, t_start, n_samples, "left")
            if t_start is not None and use_physical_time
            else max(0, int(t_start))
            if t_start is not None
            else 0
        )
        time_end_idx = (
            self._axis_value_to_index(time_axis, t_end, n_samples, "right")
            if t_end is not None and use_physical_time
            else min(n_samples, int(t_end))
            if t_end is not None
            else n_samples
        )
        dist_start_idx = (
            self._axis_value_to_index(trace_axis, d_start, n_traces, "left")
            if d_start is not None and use_physical_dist
            else max(0, int(d_start))
            if d_start is not None
            else 0
        )
        dist_end_idx = (
            self._axis_value_to_index(trace_axis, d_end, n_traces, "right")
            if d_end is not None and use_physical_dist
            else min(n_traces, int(d_end))
            if d_end is not None
            else n_traces
        )

        time_start_idx = max(0, min(time_start_idx, n_samples))
        time_end_idx = max(time_start_idx + 1, min(time_end_idx, n_samples))
        dist_start_idx = max(0, min(dist_start_idx, n_traces))
        dist_end_idx = max(dist_start_idx + 1, min(dist_end_idx, n_traces))

        return {
            "time_start_idx": time_start_idx,
            "time_end_idx": time_end_idx,
            "dist_start_idx": dist_start_idx,
            "dist_end_idx": dist_end_idx,
            "time_start": t_start if t_start is not None else 0,
            "time_end": t_end if t_end is not None else n_samples,
            "dist_start": d_start if d_start is not None else 0,
            "dist_end": d_end if d_end is not None else n_traces,
        }

    def _axis_value_to_index(
        self, axis: np.ndarray, value: float, fallback_size: int, side: str
    ) -> int:
        if axis is None or len(axis) == 0:
            return 0 if side == "left" else fallback_size
        if len(axis) > 1 and axis[0] > axis[-1]:
            reversed_axis = axis[::-1]
            reversed_idx = int(np.searchsorted(reversed_axis, float(value), side=side))
            idx = len(axis) - reversed_idx
            return max(0, min(idx, fallback_size))
        idx = int(np.searchsorted(axis, float(value), side=side))
        return max(0, min(idx, fallback_size))

    def _parse_float_edit(self, edit, default: float = None):
        """解析浮点数输入"""
        text = (edit.text() or "").strip()
        if text == "":
            return default
        try:
            return float(text)
        except Exception:
            return default

    def _downsample_for_display(
        self,
        data: np.ndarray,
        max_samples: int,
        max_traces: int,
        return_indices: bool = False,
    ) -> np.ndarray:
        """为显示降采样数据"""
        n_samples, n_traces = data.shape[0], data.shape[1]
        ds_samples = max(1, n_samples // max_samples) if n_samples > max_samples else 1
        ds_traces = max(1, n_traces // max_traces) if n_traces > max_traces else 1
        sampled = data[::ds_samples, ::ds_traces]
        if return_indices:
            sample_idx = np.arange(0, n_samples, ds_samples, dtype=int)
            trace_idx = np.arange(0, n_traces, ds_traces, dtype=int)
            return sampled, sample_idx, trace_idx
        return sampled

    def _downsample_data(self, data: np.ndarray, return_indices: bool = False):
        """降采样数据（用于快速预览）"""
        max_samples = self._parse_int_edit(
            self.page_advanced.max_samples_edit, default=512
        )
        max_traces = self._parse_int_edit(
            self.page_advanced.max_traces_edit, default=200
        )
        return self._downsample_for_display(
            data, max_samples, max_traces, return_indices=return_indices
        )

    def _resolve_plot_extent_and_labels(
        self,
        display_data: np.ndarray,
        bounds: dict,
        axis_info: dict,
        header_info_override: dict | None = None,
    ):
        """解析绘图范围和标签"""
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        time_axis = np.asarray(axis_info.get("time_axis", []), dtype=np.float32)
        trace_axis = np.asarray(axis_info.get("trace_axis", []), dtype=np.float32)
        if time_axis.size > 1 and trace_axis.size > 1:
            extent = [trace_axis[0], trace_axis[-1], time_axis[-1], time_axis[0]]
            xlabel = "距离 (m)"
            ylabel = (
                "高程 (m)"
                if header.get("is_elevation")
                else "深度 (m)"
                if header.get("is_depth")
                else (
                    "时间 (ns)" if header.get("total_time_ns") else "时间（采样索引）"
                )
            )
        else:
            n_samples, n_traces = display_data.shape[0], display_data.shape[1]
            extent = [0, n_traces, n_samples, 0]
            xlabel, ylabel = "距离（道索引）", "时间（采样索引）"
        xlabel = str(header.get("display_xlabel") or xlabel)
        ylabel = str(header.get("display_ylabel") or ylabel)
        return {"extent": extent, "xlabel": xlabel, "ylabel": ylabel}

    def _build_compare_data_pairs(
        self, display_data: np.ndarray, header_info_override: dict | None = None
    ):
        """构建对比数据对（复用已处理的 display_data，避免重复 _prepare_view_data）"""
        if not self.page_advanced.compare_var.isChecked():
            return [(self._get_single_plot_title(header_info_override), display_data)]
        left_label = self.page_advanced.compare_left_combo.currentText()
        right_label = self.page_advanced.compare_right_combo.currentText()

        def _get_prepared(label):
            if label == "当前":
                return display_data
            snap = next(
                (s for s in self.compare_snapshots if s["label"] == label), None
            )
            if snap and snap["data"] is not None:
                return self._prepare_view_data(
                    snap["data"],
                    header_info_override=snap.get("header_info"),
                    trace_metadata_override=snap.get("trace_metadata"),
                )[0]
            return display_data

        left_data = _get_prepared(left_label)
        right_data = _get_prepared(right_label)

        if self.page_advanced.diff_var.isChecked():
            # 差异视图：对齐尺寸后取绝对差
            min_rows = min(left_data.shape[0], right_data.shape[0])
            min_cols = min(left_data.shape[1], right_data.shape[1])
            diff = np.abs(
                left_data[:min_rows, :min_cols] - right_data[:min_rows, :min_cols]
            )
            return [(f"|{left_label} - {right_label}|", diff)]

        if left_label == right_label:
            return [(left_label, left_data)]

        return [(left_label, left_data), (right_label, right_data)]

    def _build_slider_compare_pair(
        self, display_data: np.ndarray, header_info_override: dict | None = None
    ):
        """构建滑动对比所需的左右数据。"""
        left_label = self.page_advanced.compare_left_combo.currentText() or "原始"
        right_label = self.page_advanced.compare_right_combo.currentText() or "当前"

        def _get_prepared(label):
            if label == "当前":
                return display_data
            snap = next(
                (s for s in self.compare_snapshots if s["label"] == label), None
            )
            if snap and snap["data"] is not None:
                return self._prepare_view_data(
                    snap["data"],
                    header_info_override=snap.get("header_info"),
                    trace_metadata_override=snap.get("trace_metadata"),
                )[0]
            return display_data

        left_data = _get_prepared(left_label)
        right_data = _get_prepared(right_label)
        return left_label, left_data, right_label, right_data

    def _create_plot_axes(self, n_panels: int):
        """创建绘图坐标轴"""
        if n_panels == 1:
            return [self.fig.add_subplot(111)]
        return [self.fig.add_subplot(1, n_panels, i + 1) for i in range(n_panels)]

    def _get_or_create_plot_axes(self, n_panels: int):
        """获取或创建绘图坐标轴（复用已有）"""
        existing = self.fig.axes
        if len(existing) == n_panels:
            return existing
        return self._create_plot_axes(n_panels)

    def _clear_axes_artists(self, axes):
        """清除坐标轴上的艺术家对象"""
        for ax in axes:
            ax.cla()
            ax.set_title("B-扫")

    def _render_data_pairs(
        self,
        axes,
        data_pairs,
        cmap,
        extent,
        plot_config,
        header_info_override: dict | None = None,
    ):
        """渲染数据对（对比模式下统一色标）"""
        is_compare = len(data_pairs) > 1
        if is_compare:
            finite_parts = [
                d[np.isfinite(d)] for _, d in data_pairs if np.isfinite(d).any()
            ]
            if finite_parts:
                all_data = np.concatenate(finite_parts)
                shared_vmin, shared_vmax = self._compute_vmin_vmax(
                    all_data, header_info_override=header_info_override
                )
            else:
                shared_vmin, shared_vmax = -1.0, 1.0
        else:
            shared_vmin, shared_vmax = None, None

        last_im = None
        for ax, (label, data) in zip(axes, data_pairs):
            im = self._render_single_panel(
                ax,
                data,
                cmap,
                extent,
                plot_config,
                label,
                vmin_override=shared_vmin,
                vmax_override=shared_vmax,
                header_info_override=header_info_override,
            )
            if im:
                last_im = im
        return last_im

    def _render_wiggle_pairs(self, axes, data_pairs, axis_info: dict, plot_config: dict):
        """以摆动图形式渲染数据对。"""
        for ax, (label, data) in zip(axes, data_pairs):
            self._render_wiggle_panel(ax, data, label, axis_info, plot_config)
        return None

    def _render_wiggle_panel(
        self, ax, data: np.ndarray, title: str, axis_info: dict, plot_config: dict
    ):
        """渲染单个摆动图面板。"""
        from core.theme_manager import get_theme_manager

        ax.clear()
        ax.set_axis_on()

        if data.ndim != 2 or data.size == 0:
            placeholder = "#b7bcc6" if get_theme_manager().get_current_theme() == "dark" else "#888"
            ax.text(
                0.5,
                0.5,
                "摆动图需要二维数据",
                ha="center",
                va="center",
                fontsize=12,
                color=placeholder,
            )
            return

        x_axis = np.asarray(axis_info.get("trace_axis", []), dtype=np.float32)
        y_axis = np.asarray(axis_info.get("time_axis", []), dtype=np.float32)
        if x_axis.size != data.shape[1]:
            x_axis = np.arange(data.shape[1], dtype=np.float32)
        if y_axis.size != data.shape[0]:
            y_axis = np.arange(data.shape[0], dtype=np.float32)

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

        theme = get_theme_manager().get_current_theme()
        line_color = "#f5f5f5" if theme == "dark" else "#111111"
        fill_color = "#8fb7ff" if theme == "dark" else "#4a4a4a"

        for trace_idx in trace_indices:
            trace = np.asarray(data[:, trace_idx], dtype=float)
            trace = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)
            wiggle = x_axis[trace_idx] + (trace / amp_ref) * wiggle_scale
            ax.plot(wiggle, y_axis, color=line_color, linewidth=0.8)
            ax.fill_betweenx(
                y_axis,
                x_axis[trace_idx],
                wiggle,
                where=wiggle >= x_axis[trace_idx],
                color=fill_color,
                alpha=0.25,
                interpolate=True,
            )

        ax.set_title(f"{title} - 摆动图")
        ax.set_xlabel(plot_config["xlabel"])
        ax.set_ylabel(plot_config["ylabel"])
        if x_axis.size > 0:
            ax.set_xlim(x_axis[0] - spacing * 0.5, x_axis[-1] + spacing * 0.5)
        if y_axis.size > 0:
            ax.set_ylim(y_axis[-1], y_axis[0])
        if self.page_advanced.show_grid_var.isChecked():
            ax.grid(True, linestyle=":", alpha=0.3)

    def _render_slider_compare_panel(
        self,
        ax,
        display_data: np.ndarray,
        axis_info: dict,
        plot_config: dict,
        cmap,
        header_info_override: dict | None = None,
    ):
        """渲染主界面的滑动对比图。"""
        from core.theme_manager import get_theme_manager

        left_label, left_data, right_label, right_data = self._build_slider_compare_pair(
            display_data, header_info_override=header_info_override
        )

        try:
            left_data = np.asarray(left_data, dtype=np.float32)
            right_data = np.asarray(right_data, dtype=np.float32)
        except Exception:
            return None

        if left_data.ndim != 2 or right_data.ndim != 2:
            return None

        min_rows = min(left_data.shape[0], right_data.shape[0])
        min_cols = min(left_data.shape[1], right_data.shape[1])
        left_data = left_data[:min_rows, :min_cols]
        right_data = right_data[:min_rows, :min_cols]
        merged = np.array(right_data, copy=True)

        split_idx = int(round(self._main_slider_compare_ratio * max(min_cols - 1, 1)))
        split_idx = max(0, min(split_idx, min_cols - 1))
        merged[:, : split_idx + 1] = left_data[:, : split_idx + 1]

        finite_parts = [
            part[np.isfinite(part)] for part in [left_data, right_data] if np.isfinite(part).any()
        ]
        if finite_parts:
            all_data = np.concatenate(finite_parts)
            vmin, vmax = self._compute_vmin_vmax(
                all_data, header_info_override=header_info_override
            )
        else:
            vmin, vmax = -1.0, 1.0

        im = ax.imshow(
            merged,
            cmap=cmap,
            aspect="auto",
            extent=plot_config["extent"],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(plot_config["xlabel"])
        ax.set_ylabel(plot_config["ylabel"])
        ax.set_title(f"滑动对比 ({left_label} | {right_label})")

        trace_axis = np.asarray(axis_info.get("trace_axis", []), dtype=np.float32)
        if trace_axis.size == min_cols:
            split_x = float(trace_axis[split_idx])
        else:
            extent = plot_config["extent"]
            split_x = float(extent[0] + (extent[1] - extent[0]) * self._main_slider_compare_ratio)

        theme = get_theme_manager().get_current_theme()
        is_dark = theme == "dark"
        divider_color = "#d9e6ff" if is_dark else "#ffffff"
        label_text_color = "#f5f5f5" if is_dark else "#ffffff"
        label_bg_color = "#111318" if is_dark else "#000000"
        x0, x1, y0, y1 = plot_config["extent"]
        label_y = min(y0, y1) + abs(y1 - y0) * 0.08
        left_x = x0 + abs(x1 - x0) * 0.15
        right_x = x0 + abs(x1 - x0) * 0.85

        ax.axvline(x=split_x, color=divider_color, linewidth=1.6, alpha=0.85)
        ax.text(
            left_x,
            label_y,
            left_label,
            color=label_text_color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=label_bg_color, edgecolor="none", alpha=0.58),
        )
        ax.text(
            right_x,
            label_y,
            right_label,
            color=label_text_color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=label_bg_color, edgecolor="none", alpha=0.58),
        )
        if self.page_advanced.show_grid_var.isChecked():
            ax.grid(True, linestyle=":", alpha=0.3)
        return im

    def _draw_selected_trace_marker(self, axes, axis_info: dict):
        """在主图上绘制当前选中道号的竖线。"""
        if self._selected_trace_index is None:
            return

        trace_axis = np.asarray(axis_info.get("trace_axis", []), dtype=np.float32)
        trace_indices = np.asarray(axis_info.get("trace_indices", []), dtype=np.int32)
        if trace_axis.size == 0 or trace_indices.size != trace_axis.size:
            return

        matches = np.flatnonzero(trace_indices == int(self._selected_trace_index))
        if matches.size == 0:
            return

        selected_x = float(trace_axis[int(matches[0])])
        for ax in axes:
            ax.axvline(
                selected_x,
                color="#64748b",
                linewidth=0.8,
                linestyle=":",
                alpha=0.45,
                zorder=5,
            )

    def _draw_manual_roi_marker(self, axes, axis_info: dict):
        """在主图上绘制当前手动 ROI。"""
        if self._manual_roi_values is None or not axes:
            return

        trace_axis = np.asarray(axis_info.get("trace_axis", []), dtype=np.float32)
        time_axis = np.asarray(axis_info.get("time_axis", []), dtype=np.float32)
        if trace_axis.size == 0 or time_axis.size == 0:
            return

        vals = self._manual_roi_values
        x0 = float(vals["dist_start"])
        x1 = float(vals["dist_end"])
        y0 = float(vals["time_start"])
        y1 = float(vals["time_end"])
        rect_x = min(x0, x1)
        rect_y = min(y0, y1)
        rect_w = abs(x1 - x0)
        rect_h = abs(y1 - y0)
        if rect_w <= 0 or rect_h <= 0:
            return

        for ax in axes[:1]:
            patch = Rectangle(
                (rect_x, rect_y),
                rect_w,
                rect_h,
                fill=False,
                edgecolor="#0ea5e9",
                linewidth=1.4,
                linestyle="--",
                alpha=0.95,
                zorder=6,
            )
            ax.add_patch(patch)

    def _render_single_panel(
        self,
        ax,
        data,
        cmap,
        extent,
        plot_config,
        title,
        vmin_override=None,
        vmax_override=None,
        header_info_override: dict | None = None,
    ):
        """渲染单个面板"""
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        if vmin_override is not None and vmax_override is not None:
            vmin, vmax = vmin_override, vmax_override
        else:
            vmin, vmax = self._compute_vmin_vmax(
                data, header_info_override=header_info_override
            )
        render_data = data
        render_cmap = cmap
        bad_color = header.get("display_bad_color")
        if bad_color:
            render_data = np.ma.masked_invalid(np.asarray(data, dtype=np.float64))
            render_cmap = (
                plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
            )
            render_cmap.set_bad(str(bad_color))
            ax.set_facecolor(str(bad_color))
        im = ax.imshow(
            render_data,
            cmap=render_cmap,
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(plot_config["xlabel"])
        ax.set_ylabel(plot_config["ylabel"])
        ax.set_title(title)
        if self.page_advanced.show_grid_var.isChecked():
            ax.grid(True, linestyle=":", alpha=0.5)
        return im

    def _compute_vmin_vmax(
        self, data: np.ndarray, header_info_override: dict | None = None
    ):
        """计算vmin和vmax"""
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        finite_data = np.asarray(data, dtype=np.float64)
        finite_data = finite_data[np.isfinite(finite_data)]
        if finite_data.size == 0:
            return -1.0, 1.0
        if header.get("display_fixed_unit_range"):
            return -1.0, 1.0
        if self.page_advanced.percentile_var.isChecked():
            p_low = self._parse_float_edit(self.page_advanced.p_low_edit, default=1.0)
            p_high = self._parse_float_edit(
                self.page_advanced.p_high_edit, default=99.0
            )
            vmin, vmax = np.percentile(finite_data, [p_low, p_high])
            return vmin, vmax
        if self.page_advanced.chatgpt_style_var.isChecked():
            vmin, vmax = np.percentile(finite_data, [0.5, 99.5])
            return vmin, vmax
        if header.get("display_center_zero"):
            percentile_high = header.get("display_percentile_abs_high")
            if percentile_high is not None:
                vmax = float(np.percentile(np.abs(finite_data), float(percentile_high)))
            else:
                vmax = float(np.max(np.abs(finite_data))) if finite_data.size else 1.0
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            return -vmax, vmax
        vmax = np.max(np.abs(finite_data))
        vmin = (
            -vmax
            if self.page_advanced.symmetric_var.isChecked()
            else np.min(finite_data)
        )
        return vmin, vmax

    def _draw_colorbar_if_needed(
        self, im, axes, header_info_override: dict | None = None
    ):
        """根据需要绘制色标"""
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        if not self.page_advanced.show_cbar_var.isChecked() and not header.get(
            "display_show_cbar"
        ):
            return
        if len(axes) == 1:
            self.cbar = self.fig.colorbar(im, ax=axes[0])
        else:
            self.cbar = self.fig.colorbar(im, ax=axes)
        label = header.get("display_colorbar_label")
        if label:
            self.cbar.set_label(str(label))

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
        self._compare_syncing = True
        current_left = self.page_advanced.compare_left_combo.currentText()
        current_right = self.page_advanced.compare_right_combo.currentText()
        self.page_advanced.compare_left_combo.clear()
        self.page_advanced.compare_right_combo.clear()
        labels = [s["label"] for s in self.compare_snapshots]
        self.page_advanced.compare_left_combo.addItems(labels)
        self.page_advanced.compare_right_combo.addItems(labels)
        # 保持用户选择（如果仍有效）
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

    def _set_quality_metrics(self, metrics: dict):
        """设置质量指标"""
        self._last_quality_metrics = metrics
        airborne_qc = self._compute_airborne_qc_metrics()
        if metrics is None:
            self.quality_focus_label.setText("focus_ratio: --")
            self.quality_hot_label.setText("hot_pixels: --")
            self.quality_spiky_label.setText("spikiness: --")
            self.quality_time_label.setText("time_ms: --")
            self.quality_alert_label.setText("阈值状态: --")
        else:
            self.quality_focus_label.setText(
                f"focus_ratio: {metrics.get('focus_ratio', 0):.4f}"
            )
            self.quality_hot_label.setText(
                f"hot_pixels: {metrics.get('hot_pixels', 0)}"
            )
            self.quality_spiky_label.setText(
                f"spikiness: {metrics.get('spikiness', 0):.3f}"
            )
            self.quality_time_label.setText(f"time_ms: {metrics.get('time_ms', 0):.1f}")
            alerts = []
            for k in ["focus_ratio", "hot_pixels", "spikiness", "time_ms"]:
                if self._is_metric_alert(k, float(metrics.get(k, 0))):
                    alerts.append(k)
            self.quality_alert_label.setText(
                f"阈值状态: {', '.join(alerts) if alerts else '正常'}"
            )

        if airborne_qc:
            self.quality_track_len_label.setText(
                f"track_length_m: {airborne_qc['track_length_m']:.2f}"
            )
            self.quality_spacing_label.setText(
                f"trace_spacing_cv: {airborne_qc['trace_spacing_cv']:.3f}"
            )
            self.quality_height_label.setText(
                f"flight_height_span_m: {airborne_qc['flight_height_span_m']:.2f}"
            )
            self.quality_airborne_alert_label.setText(
                "airborne_alerts: "
                + (
                    ", ".join(airborne_qc.get("alerts", []))
                    if airborne_qc.get("alerts")
                    else "正常"
                )
            )
            if hasattr(self, "page_quality") and self.page_quality is not None:
                self.page_quality.set_airborne_qc_summary(
                    self._build_airborne_qc_summary_text()
                )
                self.page_quality.set_airborne_qc_visualization(
                    self._build_airborne_qc_plot_payload()
                )
                self.page_quality.set_airborne_trajectory_visualization(
                    self._build_airborne_trajectory_plot_payload()
                )
                self.page_quality.set_airborne_anomaly_details(
                    self._build_airborne_anomaly_text()
                )
        else:
            self.quality_track_len_label.setText("track_length_m: --")
            self.quality_spacing_label.setText("trace_spacing_cv: --")
            self.quality_height_label.setText("flight_height_span_m: --")
            self.quality_airborne_alert_label.setText("airborne_alerts: --")
            if hasattr(self, "page_quality") and self.page_quality is not None:
                self.page_quality.set_airborne_qc_summary("")
                self.page_quality.set_airborne_qc_visualization(None)
                self.page_quality.set_airborne_trajectory_visualization(None)
                self.page_quality.set_airborne_anomaly_details("")

    def _compute_airborne_qc_metrics(self) -> dict | None:
        """基于当前每道元数据计算第一批航空 QC 指标。"""
        header = self.header_info or {}
        meta = self.trace_metadata or {}
        if not header.get("has_airborne_metadata"):
            return None

        distance = np.asarray(meta.get("trace_distance_m", []), dtype=np.float64)
        flight = np.asarray(meta.get("flight_height_m", []), dtype=np.float64)
        spacing = (
            np.diff(distance) if distance.size > 1 else np.array([], dtype=np.float64)
        )
        spacing_mean = float(np.mean(spacing)) if spacing.size else 0.0
        spacing_std = float(np.std(spacing)) if spacing.size else 0.0
        spacing_cv = spacing_std / spacing_mean if spacing_mean > 1e-9 else 0.0
        flight_span = float(np.max(flight) - np.min(flight)) if flight.size else 0.0
        spacing_outliers = (
            int(np.sum(np.abs(spacing - spacing_mean) > max(spacing_std * 2.5, 0.5)))
            if spacing.size
            else 0
        )
        spacing_outlier_indices = (
            np.flatnonzero(
                np.abs(spacing - spacing_mean) > max(spacing_std * 2.5, 0.5)
            ).tolist()
            if spacing.size
            else []
        )
        flight_outliers = (
            int(
                np.sum(
                    np.abs(flight - np.median(flight)) > max(np.std(flight) * 2.5, 0.5)
                )
            )
            if flight.size
            else 0
        )
        flight_outlier_indices = (
            np.flatnonzero(
                np.abs(flight - np.median(flight)) > max(np.std(flight) * 2.5, 0.5)
            ).tolist()
            if flight.size
            else []
        )
        alerts = []
        if spacing_cv > 0.25:
            alerts.append("spacing_cv_high")
        if flight_span > 3.0:
            alerts.append("flight_span_high")
        if spacing_outliers > 0:
            alerts.append(f"spacing_outliers={spacing_outliers}")
        if flight_outliers > 0:
            alerts.append(f"height_outliers={flight_outliers}")
        return {
            "track_length_m": float(header.get("track_length_m", 0.0)),
            "trace_spacing_cv": float(spacing_cv),
            "flight_height_span_m": float(flight_span),
            "spacing_outliers": spacing_outliers,
            "height_outliers": flight_outliers,
            "spacing_outlier_indices": spacing_outlier_indices,
            "height_outlier_indices": flight_outlier_indices,
            "alerts": alerts,
        }

    def _set_last_run_summary(
        self,
        run_type: str,
        label: str,
        steps: list,
        preset_key: str = None,
        profile_key: str = None,
        notes: list | None = None,
        warnings: list | None = None,
    ):
        """记录最近一次真实执行摘要，供报告和诊断使用。"""
        self._last_run_summary = {
            "run_type": run_type,
            "label": label,
            "steps": steps,
            "preset_key": preset_key,
            "profile_key": profile_key,
            "notes": notes or [],
            "warnings": warnings or [],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _is_metric_alert(self, metric: str, value: float) -> bool:
        """检查指标是否超出阈值"""
        thresholds = self._quality_thresholds.get(metric, {})
        min_v = thresholds.get("min")
        max_v = thresholds.get("max")
        if min_v is not None and value < min_v:
            return True
        if max_v is not None and value > max_v:
            return True
        return False

    def _save_pipeline_comparison(self, outputs: list) -> str | None:
        """导出默认/推荐流程对比图（Raw / 中间关键步 / Final）"""
        try:
            if self.original_data is None or self.data is None:
                return None
            out_dir = self._default_output_dir()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"pipeline_compare_{ts}.png")

            raw = np.asarray(self.original_data)
            final = np.asarray(self.data)

            # 选两个中间关键步骤：第1步、倒数第2步（若存在）
            mids = []
            if len(outputs) >= 1:
                mids.append(
                    (
                        outputs[0].get("method_name", "Step1"),
                        np.asarray(outputs[0].get("data")),
                    )
                )
            if len(outputs) >= 3:
                mids.append(
                    (
                        outputs[-2].get("method_name", "StepN-1"),
                        np.asarray(outputs[-2].get("data")),
                    )
                )

            items = [("Raw", raw)] + mids + [("Final", final)]
            n = len(items)
            fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 3.6), dpi=150)
            if n == 1:
                axs = [axs]
            for ax, (title, arr) in zip(axs, items):
                arr = np.asarray(arr)
                vmax = float(np.nanmax(np.abs(arr))) if arr.size else 1.0
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1.0
                ax.imshow(
                    arr,
                    cmap="seismic",
                    aspect="auto",
                    vmin=-vmax,
                    vmax=vmax,
                    origin="upper",
                )
                ax.set_title(title)
                ax.set_xlabel("Trace")
                ax.set_ylabel("Sample")
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)
            return out_path
        except Exception as e:
            self._log(f"对比图导出失败: {e}")
            return None

    # ============ 工作线程管理 ============

    def _start_processing_worker(
        self,
        tasks: list,
        run_type: str = "single",
        restore_method_idx: int = None,
        run_label: str = None,
        preset_key: str = None,
        profile_key: str = None,
        execution_mode: str = "sequential",
    ):
        """启动后台处理工作线程"""
        self._current_run_context = {
            "run_type": run_type,
            "restore_method_idx": restore_method_idx,
            "run_label": run_label,
            "preset_key": preset_key,
            "profile_key": profile_key,
        }
        self._cancel_in_flight = False
        self._set_busy(True, text=f"处理中 ({run_type})...")

        self._worker_thread = QThread(self)
        self._worker = ProcessingWorker(
            self.data,
            tasks,
            self.data_path,
            execution_mode=execution_mode,
            header_info=self.header_info,
            trace_metadata=self.trace_metadata,
        )
        self._worker.moveToThread(self._worker_thread)

        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker_thread.started.connect(self._worker.run)

        self._worker_thread.start()

    def _on_worker_finished(self, result: dict):
        """工作线程完成回调"""
        self._set_busy(False, text="就绪")
        outputs = result.get("outputs", [])
        for item in outputs:
            self._append_runtime_warnings(
                item.get("runtime_warnings", []),
                source=item.get("method_key") or item.get("method_name"),
                log=False,
            )
        final_data = result.get("final_data")
        final_header_info = result.get("final_header_info")
        final_trace_metadata = result.get("final_trace_metadata")
        final_display_data = result.get("final_display_data")
        final_display_header_info = result.get("final_display_header_info")
        final_display_trace_metadata = result.get("final_display_trace_metadata")
        cancelled = result.get("cancelled", False)
        ctx = self._current_run_context or {}
        run_type = ctx.get("run_type", "")

        if cancelled:
            self._log("处理已取消。")
            self.status_label.setText("已取消")
        elif final_data is not None:
            is_kirchhoff = (
                len(outputs) == 1
                and outputs[0].get("method_key") == "kirchhoff_migration"
            )
            if len(outputs) == 1:
                snap_label = outputs[0].get(
                    "method_name", outputs[0].get("method_key", "处理")
                )
            else:
                names = [
                    o.get("method_name", o.get("method_key", "?")) for o in outputs
                ]
                snap_label = (
                    f"{names[0]}+{len(names) - 1}步" if len(names) > 1 else names[0]
                )
            self.shared_data.apply_current_data(
                final_data,
                push_history=False,
                source=run_type or "worker",
                label=snap_label,
                header_info=final_header_info,
                trace_metadata=final_trace_metadata,
            )
            self._mark_data_changed()
            if final_display_data is not None:
                self._set_display_override(
                    final_display_data,
                    header_info=final_display_header_info,
                    trace_metadata=final_display_trace_metadata,
                )
            if is_kirchhoff and self.page_advanced.compare_var.isChecked():
                self.page_advanced.compare_var.setChecked(False)
                self._log("Kirchhoff 迁移结果已切换为单图显示。")
            self._refresh_compare_snapshots_from_state()
            self._update_empty_state_and_brief()
            self.plot_data(self.data)
            self._log(f"处理完成：共 {len(outputs)} 个步骤")

            # Log processing results (for both Kirchhoff and normal cases)
            for k, item in enumerate(outputs, start=1):
                name = item.get("method_name", item.get("method_key", f"step-{k}"))
                ms = item.get("elapsed_ms")
                mapped = (item.get("meta") or {}).get("mapped_params", {})
                backend = mapped.get("execution_backend")
                fallback_reason = mapped.get("fallback_reason")
                if ms is not None:
                    suffix = f" | backend={backend}" if backend else ""
                    self._log(f"  [{k}] {name}: {ms:.1f} ms{suffix}")
                else:
                    self._log(f"  [{k}] {name}")
                if fallback_reason:
                    self._log(f"      fallback: {fallback_reason}")
                    self._append_runtime_warnings(
                        [
                            build_runtime_warning(
                                "method_fallback",
                                "方法执行触发了回退路径。",
                                method=name,
                                reason=fallback_reason,
                            )
                        ],
                        source=item.get("method_key") or name,
                    )
                self._append_runtime_warnings(
                    item.get("runtime_warnings", []),
                    source=item.get("method_key") or name,
                )
            self.status_label.setText(f"完成: {len(outputs)} 步骤")

            # 计算质量指标
            start_ts = time.perf_counter()
            metrics = compute_quality_metrics(self.data)
            metrics["time_ms"] = (time.perf_counter() - start_ts) * 1000.0
            self._set_quality_metrics(metrics)

            # 对于一键/推荐流程，自动导出对比图
            if run_type in {"pipeline", "recommended"}:
                compare_path = self._save_pipeline_comparison(outputs)
                if compare_path:
                    self._log(f"对比图已导出：{compare_path}")

            self._set_last_run_summary(
                run_type=run_type,
                label=ctx.get("run_label")
                or (outputs[-1].get("method_name") if outputs else run_type),
                steps=[
                    {
                        "method_key": item.get("method_key"),
                        "method_name": item.get("method_name"),
                        "params": item.get("params", {}),
                        "elapsed_ms": item.get("elapsed_ms"),
                    }
                    for item in outputs
                ],
                preset_key=ctx.get("preset_key"),
                profile_key=ctx.get("profile_key"),
                warnings=list(self._runtime_warnings),
            )

        self._cleanup_worker()

        # 恢复方法选择
        if ctx.get("restore_method_idx") is not None:
            self.page_basic.method_combo.setCurrentIndex(ctx["restore_method_idx"])

    def _on_worker_error(self, error_msg: str):
        """工作线程错误回调"""
        self._set_busy(False, text="错误")
        hint = self._build_error_hint(error_msg)
        self._log(f"处理错误: {error_msg}")
        self._log(f"处理建议: {hint}")
        QMessageBox.critical(self, "处理错误", f"{error_msg}\n\n{hint}")
        self._cleanup_worker()

    def _on_worker_progress(self, current: int, total: int, message: str):
        """工作线程进度回调"""
        self.status_label.setText(f"{message} ({current}/{total})")
        if self._progress_panel is not None:
            self._progress_panel.setVisible(True)
        if self._progress_bar is not None:
            safe_total = max(int(total), 1)
            safe_current = max(0, min(int(current), safe_total))
            self._progress_bar.setRange(0, safe_total)
            self._progress_bar.setValue(safe_current)
            self._progress_bar.setFormat(f"步骤 {safe_current}/{safe_total}")
        self._log(message)

    def _cleanup_worker(self):
        """清理工作线程"""
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait(5000)
            self._worker_thread = None
        self._worker = None
        self._cancel_in_flight = False
        self.page_basic.btn_cancel.setEnabled(False)
        if self._progress_bar is not None:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(0)
            self._progress_bar.setFormat("等待开始")
        if self._progress_panel is not None:
            self._progress_panel.setVisible(False)


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
    version_text = build_version_string("GPR_GUI")
    logger.info("GPR GUI version=%s", version_text)
    win = GPRGuiQt(version_text=version_text)
    logger.info("Runtime log file: %s", log_path)
    win.statusBar().showMessage(
        f"Theme: {theme_name} | QtFont: {qt_font_name} | {version_text}"
    )
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
