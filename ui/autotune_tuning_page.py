#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoTune 自动推荐页（工程向导版）。

GX-UI-010 light polish pass:
- 专门服务 AutoTune 参数推荐，不引入 Research Lab / 3D viewer。
- 针对 MyGPR 当前右侧工作区宽度重排为“顶部状态 + 标签页”结构。
- 仅做 UI 状态联动，不触发 AutoTune/gprMax/Evidence 执行。
- 保留 legacy AutoTunePage 兼容层，避免 app_qt 既有信号/状态调用断裂。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import time

import numpy as np

from PyQt6.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QGridLayout,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QCheckBox,
    QComboBox,
    QTabWidget,
    QStackedWidget,
    QSpinBox,
    QPushButton,
    QSizePolicy,
    QHeaderView,
    QScrollArea,
    QToolButton,
    QAbstractItemView,
)

from ui.gui_auto_tune_page import AutoTunePage


@dataclass
class AutoTuneRecommendationState:
    """UI-local state only; does not execute AutoTune or alter scoring logic."""

    data_source: str = "未载入"
    data_type: str = "未识别"
    file_path: str | None = None
    source_label: str = "未载入"
    data_shape: tuple[int, int] | None = None
    component: str | None = None
    processing_stage: str = "原始数据"
    workflow_step: str = "背景抑制"
    target_goal: str = "均衡推荐"
    roi_mode: str = "none"
    candidate_methods: set[str] = field(default_factory=lambda: {"baseline", "mean", "median", "svd"})
    svd_rank_min: int = 1
    svd_rank_max: int = 5
    svd_rank_step: int = 1
    roi_trace_start: int = 35
    roi_trace_end: int = 50
    roi_sample_start: int = 350
    roi_sample_end: int = 950
    scoring_metrics: set[str] = field(default_factory=lambda: {"roi_retention", "residual", "cnr", "shape"})
    no_prior_warning: bool = True
    display_only_flag: bool = True
    manual_review_required: bool = True
    claim_boundary_required: bool = True
    synthetic_gt_available: bool = True
    recommendation_status: str = "未运行"
    selected_candidate_name: str = "未生成"
    selected_candidate_params: str = "--"
    selected_candidate_score: float = 0.0
    backend_mode: str = "UI 预览"
    backend_message: str = "未运行"
    backend_results: list[dict] = field(default_factory=list)
    background_results: list[dict] = field(default_factory=list)
    workflow_param_overrides: dict[str, str] = field(default_factory=dict)
    workflow_order: list[str] = field(default_factory=list)
    workflow_customized: bool = False
    workflow_order_override: bool = False
    workflow_param_override: bool = False

    @property
    def roi_is_set(self) -> bool:
        if self.roi_mode == "none":
            return False
        return self.roi_trace_end > self.roi_trace_start and self.roi_sample_end > self.roi_sample_start

    @property
    def candidate_count(self) -> int:
        count = 0
        for method in self.candidate_methods:
            if method == "svd":
                span = max(0, self.svd_rank_max - self.svd_rank_min)
                count += span // max(1, self.svd_rank_step) + 1
            else:
                count += 1
        return count

    @property
    def scoring_count(self) -> int:
        return len(self.scoring_metrics)

    @property
    def risk_level(self) -> str:
        risk = 0
        if self.roi_mode == "manual" and not self.roi_is_set:
            risk += 2
        if self.candidate_count == 0:
            risk += 2
        if self.scoring_count == 0:
            risk += 2
        if self.no_prior_warning:
            risk += 1
        if self.manual_review_required:
            risk += 1
        if risk >= 4:
            return "高"
        if risk >= 2:
            return "中"
        return "低"

    @property
    def recommendation_ready(self) -> bool:
        roi_ok = self.roi_mode != "manual" or self.roi_is_set
        return roi_ok and self.candidate_count > 0 and self.scoring_count > 0


class AutoTuneTuningPage(QWidget):
    """AutoTune 参数推荐页；legacy AutoTunePage 仅作为兼容层。"""

    recipe_run_requested = pyqtSignal(dict)

    _WORKFLOW_STEPS = ["背景抑制", "增益", "Dewow", "频带滤波", "显示增强"]

    _TARGET_GOALS = [
        "均衡推荐",
        "连续界面保留",
        "滑坡基覆界面 / 潜在滑移面",
        "局部异常增强",
        "裂隙/破碎带保留",
        "含水软弱带",
        "深部弱反射增强",
    ]

    _ROI_MODES = [
        ("none", "全图"),
        ("auto", "自动"),
        ("manual", "手动"),
    ]

    _CANDIDATES = [
        ("baseline", "不处理基线", "保留原始输入，用作对照。"),
        ("mean", "均值背景扣除", "对所有 trace 求均值背景并扣除。"),
        ("median", "中位数背景扣除", "对异常值更稳健的中位数背景扣除。"),
        ("svd", "SVD 背景抑制", "按 rank 抑制低秩背景；需警惕目标被削弱。"),
        ("sliding", "滑动窗口背景", "局部窗口背景估计；当前作为实验性候选。"),
    ]

    _SCORING = [
        ("roi_retention", "ROI 能量保持", "目标窗口内响应保留程度。"),
        ("residual", "背景残差降低", "ROI 外背景/杂波残差下降程度。"),
        ("cnr", "CNR/SNR 提升", "目标与背景可分性变化。"),
        ("shape", "目标形态稳定性", "双曲线/目标区域连续性与位移变化。"),
        ("rmse", "RMSE vs target_response", "仅 synthetic paired 数据可用于量化。"),
    ]

    _DELEGATED_NAMES = {
        "btn_auto_tune",
        "btn_compare_stage",
        "btn_compare_manual_auto",
        "btn_export_comparison",
        "btn_view_auto_tune",
        "btn_apply_stage_choice",
        "btn_open_workbench",
        "get_auto_tune_roi_mode",
        "get_auto_tune_search_mode",
        "reset_for_method",
        "show_running",
        "show_result",
        "show_error",
        "show_cancelled",
        "set_stage_compare_result",
        "set_auto_tune_method_key",
        "show_comparison_running",
        "show_comparison_result",
        "show_comparison_error",
        "set_evidence_export_result",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.state = AutoTuneRecommendationState()
        self._candidate_rows_cache_key = None
        self._candidate_rows_cache: list[tuple[str, str, float, str]] = []
        self._current_data = None
        self._target_response_data = None
        self._target_response_label = None
        self._local_style_theme_key: str | None = None
        self._candidate_checks: dict[str, QCheckBox] = {}
        self._scoring_checks: dict[str, QCheckBox] = {}
        self._safety_checks: dict[str, QCheckBox] = {}
        self._legacy_page = AutoTunePage(self)
        self._legacy_page.parent_window = parent
        self._build_ui()
        self._apply_local_style()
        self._sync_controls_from_state()
        self._refresh_from_state()

    def __getattr__(self, name):
        legacy_page = self.__dict__.get("_legacy_page")
        if legacy_page is not None and hasattr(legacy_page, name):
            return getattr(legacy_page, name)
        raise AttributeError(name)

    # ------------------------------------------------------------------
    # Public data-binding API
    # ------------------------------------------------------------------

    def set_loaded_dataset(
        self,
        *,
        file_path: str | None = None,
        data_shape: tuple[int, int] | None = None,
        data_type: str | None = None,
        component: str | None = None,
        processing_stage: str | None = None,
        source_label: str | None = None,
        data_array=None,
        target_response_array=None,
        target_response_label: str | None = None,
    ) -> None:
        """Synchronize the page with the dataset loaded by the main window.

        This method is intentionally metadata-only. It does not execute AutoTune,
        does not mutate production scoring, and does not write Evidence artifacts.
        """
        self.state.data_source = "已载入"
        self.state.file_path = file_path
        self.state.data_shape = data_shape
        self.state.data_type = data_type or self._infer_data_type(file_path)
        self.state.component = component
        self.state.processing_stage = processing_stage or "原始数据"
        self.state.source_label = source_label or self._format_source_label(file_path)
        self._current_data = data_array
        self._target_response_data = target_response_array
        self._target_response_label = target_response_label
        self.state.synthetic_gt_available = target_response_array is not None
        self.state.backend_mode = "真实候选" if data_array is not None else "UI 预览"
        if data_array is not None and target_response_array is not None:
            self.state.backend_message = "已绑定当前 B-scan 数据和 synthetic target_response"
        elif data_array is not None:
            self.state.backend_message = "已绑定当前 B-scan 数据；未绑定 target_response"
        else:
            self.state.backend_message = "仅绑定元数据"
        self.state.backend_results = []
        self.state.background_results = []
        self.state.recommendation_status = "未运行"
        self.state.selected_candidate_name = "未生成"
        self.state.selected_candidate_params = "--"
        self.state.selected_candidate_score = 0.0
        self._candidate_rows_cache_key = None
        self._candidate_rows_cache = []
        self._reset_workflow_customization()
        self._refresh_from_state()

    def clear_loaded_dataset(self) -> None:
        """Return the data status to the initial unloaded state."""
        self.state.data_source = "未载入"
        self.state.data_type = "未识别"
        self.state.file_path = None
        self.state.source_label = "未载入"
        self.state.data_shape = None
        self.state.component = None
        self.state.processing_stage = "原始数据"
        self._current_data = None
        self._target_response_data = None
        self._target_response_label = None
        self.state.synthetic_gt_available = False
        self.state.backend_mode = "UI 预览"
        self.state.backend_message = "未运行"
        self.state.backend_results = []
        self.state.background_results = []
        self._candidate_rows_cache_key = None
        self._candidate_rows_cache = []
        self._reset_workflow_customization()
        self._refresh_from_state()

    def _infer_data_type(self, file_path: str | None) -> str:
        if not file_path:
            return "unknown"
        lower = str(file_path).lower()
        if lower.endswith(".csv"):
            return "CSV"
        if lower.endswith(".out"):
            return "gprMax .out"
        if lower.endswith(".dzt"):
            return "GPR DZT"
        if lower.endswith(".npy"):
            return "NumPy"
        return "unknown"

    def _format_source_label(self, file_path: str | None) -> str:
        if not file_path:
            return "已载入数据"
        try:
            from pathlib import Path

            return Path(file_path).name or str(file_path)
        except Exception:
            return str(file_path)

    def _shape_text(self) -> str:
        if not self.state.data_shape:
            return "尺寸: --"
        if len(self.state.data_shape) >= 2:
            return f"尺寸: {self.state.data_shape[0]} × {self.state.data_shape[1]}（采样点 × 道）"
        return f"尺寸: {self.state.data_shape}"

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(self._build_header())
        root.addWidget(self._build_top_control_bar())
        self._recipe_card = self._build_main_recipe_card()
        self._recipe_card.setVisible(False)
        root.addWidget(self._recipe_card)
        root.addWidget(self._build_recipe_detail_tabs(), 1)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setObjectName("AdvancedFoldToggle")
        self.advanced_toggle.setText("高级设置与审计明细 ▸")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolTip("展开区域坐标、候选空间、评分边界、报告审计和运行日志。")
        self.advanced_toggle.toggled.connect(self._toggle_advanced_panel)
        root.addWidget(self.advanced_toggle)

        self.advanced_panel = self._build_advanced_settings_tab()
        self.advanced_panel.setObjectName("AutoTuneAdvancedPanel")
        self.advanced_panel.setVisible(False)
        root.addWidget(self.advanced_panel)

        # legacy AutoTunePage 仍保留为信号兼容层。主窗口既有 start_auto_tune_* 连接仍指向
        # 这些隐藏按钮；本页新按钮通过 click() 转发，不改变后端算法与评分逻辑。
        self._legacy_page.hide()
        root.addWidget(self._legacy_page)

    def _build_top_control_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("AutoTuneTopControlBar")
        layout = QGridLayout(bar)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)

        target_label = QLabel("目标")
        target_label.setObjectName("FieldLabel")
        layout.addWidget(target_label, 0, 0)
        self.target_goal_combo = QComboBox()
        self.target_goal_combo.addItems(self._TARGET_GOALS)
        self.target_goal_combo.currentTextChanged.connect(self._on_target_goal_changed)
        self.target_goal_combo.setMinimumWidth(0)
        self.target_goal_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.target_goal_combo, 0, 1, 1, 3)

        region_label = QLabel("ROI")
        region_label.setObjectName("FieldLabel")
        layout.addWidget(region_label, 1, 0)
        self.region_mode_combo = QComboBox()
        for key, label in self._ROI_MODES:
            self.region_mode_combo.addItem(label, key)
        self.region_mode_combo.currentIndexChanged.connect(self._on_region_mode_changed)
        self.region_mode_combo.setMinimumWidth(0)
        self.region_mode_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.region_mode_combo, 1, 1)

        self.btn_auto_roi = QPushButton("自动")
        self.btn_auto_roi.setToolTip("基于当前 B-scan 响应生成 ROI 建议；该区域不是目标真值标签。")
        self.btn_auto_roi.setEnabled(False)
        self.btn_auto_roi.clicked.connect(self._on_auto_roi_clicked)
        layout.addWidget(self.btn_auto_roi, 1, 2)

        self.btn_pick_roi = QPushButton("框选")
        self.btn_pick_roi.setCheckable(True)
        self.btn_pick_roi.setChecked(False)
        self.btn_pick_roi.setToolTip("选择“手动”后，点击此按钮并在右侧 B-scan 上按住左键拖拽。")
        self.btn_pick_roi.toggled.connect(self._on_pick_roi_toggled)
        layout.addWidget(self.btn_pick_roi, 1, 3)

        self.btn_generate_recommendation = QPushButton("生成")
        self.btn_generate_recommendation.setObjectName("PrimaryButton")
        self.btn_generate_recommendation.setToolTip("按当前目标、ROI 和候选空间生成推荐流程与参数。")
        self.btn_generate_recommendation.clicked.connect(self._on_run_recommendation_preview)
        layout.addWidget(self.btn_generate_recommendation, 2, 0, 1, 1)

        self.btn_run_recommendation_compact = QPushButton("运行")
        self.btn_run_recommendation_compact.setToolTip("应用并运行当前推荐流程。")
        self.btn_run_recommendation_compact.clicked.connect(self._on_recipe_run_requested)
        layout.addWidget(self.btn_run_recommendation_compact, 2, 1, 1, 3)

        # 顶部报告按钮已移除；保留隐藏按钮作为旧刷新/测试代码的兼容对象。
        self.btn_export_step_report = QPushButton("报告")
        self.btn_export_step_report.clicked.connect(lambda: self._legacy_page.btn_export_comparison.click())
        self.btn_export_step_report.hide()

        for col in range(4):
            layout.setColumnStretch(col, 1 if col else 0)
        return bar

    def _build_main_recipe_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("AutoTuneRecipeCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title_row = QHBoxLayout()
        title = QLabel("推荐方案")
        title.setObjectName("CardTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)
        self.recipe_score_chip = self._chip("未生成")
        title_row.addWidget(self.recipe_score_chip)
        layout.addLayout(title_row)

        self.recipe_flow_label = QLabel("Raw → Dewow → 频带滤波 → 背景抑制 → 增益")
        self.recipe_flow_label.setObjectName("AutoTuneRecipeFlow")
        self.recipe_flow_label.setWordWrap(True)
        layout.addWidget(self.recipe_flow_label)

        self.recipe_param_label = QLabel("推荐参数会在生成后显示。")
        self.recipe_param_label.setObjectName("Hint")
        self.recipe_param_label.setWordWrap(True)
        layout.addWidget(self.recipe_param_label)

        action_row = QHBoxLayout()
        self.btn_stage_compare_wizard = QPushButton("查看对比")
        self.btn_stage_compare_wizard.clicked.connect(lambda: self._legacy_page.btn_compare_stage.click())
        self.btn_view_details_wizard = QPushButton("高级明细")
        self.btn_view_details_wizard.clicked.connect(lambda: self.advanced_toggle.setChecked(True))
        self.btn_apply_step_recommendation = QPushButton("应用并运行推荐流程")
        self.btn_apply_step_recommendation.clicked.connect(self._on_recipe_run_requested)
        self.btn_view_step_details = QPushButton("查看参数")
        self.btn_view_step_details.clicked.connect(lambda: self.detail_tabs.setCurrentIndex(0))
        for btn in [self.btn_apply_step_recommendation, self.btn_stage_compare_wizard, self.btn_view_step_details, self.btn_view_details_wizard]:
            action_row.addWidget(btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)
        return card

    def _build_recipe_detail_tabs(self) -> QTabWidget:
        self.detail_tabs = QTabWidget()
        self.detail_tabs.setObjectName("AutoTuneRecipeDetailTabs")
        self.detail_tabs.tabBar().setObjectName("AutoTuneRecipeDetailTabBar")
        self.detail_tabs.tabBar().setStyleSheet(
            "QTabBar#AutoTuneRecipeDetailTabBar::tab {"
            "min-width: 34px; max-width: 44px; padding: 6px 4px; margin-right: 1px; font-size: 12px;"
            "}"
            "QTabBar#AutoTuneRecipeDetailTabBar::tab:selected { font-weight: 800; }"
        )
        self.detail_tabs.setUsesScrollButtons(False)
        self.detail_tabs.addTab(self._build_workflow_detail_tab(), "流程")
        self.detail_tabs.addTab(self._build_candidate_compare_tab(), "候选")
        self.detail_tabs.addTab(self._build_result_note_tab(), "说明")
        # Hidden compatibility buffers: older report/test code still updates these widgets.
        self.parameter_table = QTableWidget(0, 4)
        self.recommended_text = self._mini_text(40)
        self.score_text = self._mini_text(40)
        self.apply_report_text = self._mini_text(40)
        self.parameter_table.hide()
        self.recommended_text.hide()
        self.score_text.hide()
        self.apply_report_text.hide()
        return self.detail_tabs

    def _build_workflow_detail_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.workflow_table = QTableWidget(0, 4)
        self.workflow_table.setHorizontalHeaderLabels(["步骤", "参数", "处理方式", "说明"])
        self.workflow_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.workflow_table.horizontalHeader().setStretchLastSection(True)
        self.workflow_table.setMinimumHeight(180)
        self.workflow_table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.EditKeyPressed
        )
        self.workflow_table.setDragEnabled(True)
        self.workflow_table.setAcceptDrops(True)
        self.workflow_table.setDropIndicatorShown(True)
        self.workflow_table.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.workflow_table.setDragDropOverwriteMode(False)
        self.workflow_table.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.workflow_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.workflow_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.workflow_table.verticalHeader().setVisible(False)
        self.workflow_table.setAlternatingRowColors(True)
        self.workflow_table.itemChanged.connect(self._on_workflow_table_item_changed)
        try:
            self.workflow_table.model().rowsMoved.connect(self._on_workflow_rows_moved)
        except Exception:
            pass
        hint = QLabel("双击参数可修改；拖动可调整部分处理步骤顺序。")
        hint.setObjectName("Hint")
        layout.addWidget(hint)
        layout.addWidget(self.workflow_table, 1)
        return page

    def _build_parameter_detail_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.parameter_table = QTableWidget(0, 4)
        self.parameter_table.setHorizontalHeaderLabels(["步骤", "参数项", "推荐值", "说明"])
        self.parameter_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.parameter_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_table.setMinimumHeight(180)
        self.parameter_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.parameter_table.verticalHeader().setVisible(False)
        self.parameter_table.setAlternatingRowColors(True)
        layout.addWidget(self.parameter_table, 1)

        # Hidden compatibility buffers: older tests and report code read these text widgets.
        self.recommended_text = self._mini_text(40)
        self.score_text = self._mini_text(40)
        self.apply_report_text = self._mini_text(40)
        self.recommended_text.hide()
        self.score_text.hide()
        self.apply_report_text.hide()
        return page

    def _build_candidate_compare_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.ranking_table = QTableWidget(0, 4)
        self.ranking_table.setHorizontalHeaderLabels(["排名", "候选流程", "分数", "说明"])
        self.ranking_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.ranking_table.horizontalHeader().setStretchLastSection(True)
        self.ranking_table.setMinimumHeight(92)
        self.ranking_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ranking_table.verticalHeader().setVisible(False)
        self.ranking_table.setAlternatingRowColors(True)
        layout.addWidget(self.ranking_table, 1)

        self.candidate_step_table = QTableWidget(0, 4)
        self.candidate_step_table.setHorizontalHeaderLabels(["步骤", "候选数量", "状态", "说明"])
        self.candidate_step_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.candidate_step_table.horizontalHeader().setStretchLastSection(True)
        self.candidate_step_table.setMinimumHeight(92)
        self.candidate_step_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.candidate_step_table.verticalHeader().setVisible(False)
        self.candidate_step_table.setAlternatingRowColors(True)
        layout.addWidget(self.candidate_step_table, 1)
        return page

    def _build_roi_detail_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.roi_mode_hint = QLabel("ROI 坐标用于记录关注范围；普通推荐可保持全图。")
        self.roi_mode_hint.setObjectName("Hint")
        self.roi_mode_hint.setWordWrap(True)
        layout.addWidget(self.roi_mode_hint)

        coord_box = QGroupBox("坐标")
        coord_layout = QGridLayout(coord_box)
        coord_layout.setContentsMargins(10, 12, 10, 10)
        coord_layout.setHorizontalSpacing(8)
        coord_layout.setVerticalSpacing(8)
        self.roi_trace_start = self._spin(0, 100000, self._on_roi_changed)
        self.roi_trace_end = self._spin(0, 100000, self._on_roi_changed)
        self.roi_sample_start = self._spin(0, 100000, self._on_roi_changed)
        self.roi_sample_end = self._spin(0, 100000, self._on_roi_changed)
        for row, (label, widget) in enumerate(
            [
                ("Trace 起点", self.roi_trace_start),
                ("Trace 终点", self.roi_trace_end),
                ("Sample 起点", self.roi_sample_start),
                ("Sample 终点", self.roi_sample_end),
            ]
        ):
            coord_layout.addWidget(QLabel(label), row, 0)
            coord_layout.addWidget(widget, row, 1)
        layout.addWidget(coord_box)
        self.roi_picker_status_label = QLabel("ROI：全图")
        self.roi_picker_status_label.setProperty("class", "hintText")
        layout.addWidget(self.roi_picker_status_label)
        layout.addStretch(1)
        return page

    def _build_result_note_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.result_notes_text = self._mini_text(150)
        self.risk_text = self.result_notes_text  # compatibility with existing refresh code
        self.boundary_text = self._mini_text(150)
        layout.addWidget(self.result_notes_text)
        layout.addWidget(self.boundary_text)
        return page

    def _build_report_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.apply_report_text = self._mini_text(180)
        layout.addWidget(self.apply_report_text, 1)
        return page

    def _build_wizard_stepper(self) -> QFrame:
        rail = QFrame()
        rail.setObjectName("WizardStepperRail")
        rail.setMinimumWidth(88)
        rail.setMaximumWidth(104)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self._wizard_buttons = []
        steps = [
            ("1 目标", "选择自动推荐的目标倾向"),
            ("2 ROI", "选择是否使用 ROI"),
            ("3 推荐", "生成推荐并查看推荐理由"),
            ("4 应用", "采用推荐、对比或导出报告"),
        ]
        for index, (text, tip) in enumerate(steps):
            btn = QPushButton(text)
            btn.setObjectName("WizardStepButton")
            btn.setCheckable(True)
            btn.setMinimumHeight(32)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda checked=False, i=index: self._switch_wizard_step(i))
            layout.addWidget(btn)
            self._wizard_buttons.append(btn)
        layout.addStretch(1)
        return rail

    def _toggle_advanced_panel(self, checked: bool) -> None:
        panel = getattr(self, "advanced_panel", None)
        toggle = getattr(self, "advanced_toggle", None)
        if panel is not None:
            panel.setVisible(bool(checked))
        if toggle is not None:
            toggle.setText("高级设置与审计明细 ▾" if checked else "高级设置与审计明细 ▸")
        if checked:
            self._refresh_trial_table()
            self._refresh_audit_text()

    def _switch_wizard_step(self, index: int) -> None:
        if not hasattr(self, "main_tabs"):
            return
        index = max(0, min(index, self.main_tabs.count() - 1))
        self.main_tabs.setCurrentIndex(index)
        for i, btn in enumerate(getattr(self, "_wizard_buttons", [])):
            btn.setChecked(i == index)

    def _scroll(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setObjectName("AutoTuneDrawerScroll")
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setViewportMargins(0, 0, 8, 0)
        area.setWidget(widget)
        return area

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("AutoTuneHeader")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 9, 14, 9)
        layout.setSpacing(6)

        title = QLabel("自动推荐")
        title.setObjectName("AutoTuneTitle")
        subtitle = QLabel("选择目标和 ROI 后，软件生成推荐处理流程与参数。")
        subtitle.setObjectName("AutoTuneSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.btn_load_data = self._action_button("数据状态", primary=False)
        self.btn_run_autotune_mvp = self._action_button("生成推荐", primary=True)
        self.btn_export_evidence_mvp = self._action_button("导出报告", primary=False)
        self.btn_load_data.setToolTip("请在左侧主工作区导入数据；本页会自动同步当前数据状态。")
        self.btn_run_autotune_mvp.setToolTip("在当前数据、ROI 和候选空间下生成推荐流程与参数。")
        self.btn_export_evidence_mvp.setToolTip("导出自动推荐报告；详细候选记录在高级设置与审计明细中查看。")
        self.btn_load_data.setEnabled(False)
        self.btn_run_autotune_mvp.setEnabled(False)
        self.btn_export_evidence_mvp.setEnabled(False)
        self.btn_run_autotune_mvp.clicked.connect(self._on_run_recommendation_preview)
        # Header action buttons are kept for compatibility but are not shown;
        # the compact top control bar below owns the visible workflow actions.
        self.btn_load_data.hide()
        self.btn_run_autotune_mvp.hide()
        self.btn_export_evidence_mvp.hide()

        self.next_step_hint = QLabel("导入数据后点击“生成推荐”。")
        self.next_step_hint.setObjectName("NextStepHint")
        self.next_step_hint.setWordWrap(True)
        layout.addWidget(self.next_step_hint)

        chips = QGridLayout()
        chips.setHorizontalSpacing(5)
        chips.setVerticalSpacing(5)
        self.chip_data = self._chip("")
        self.chip_step = self._chip("")
        self.chip_roi = self._chip("")
        self.chip_candidates = self._chip("")
        self.chip_status = self._chip("")
        for i, chip in enumerate([self.chip_data, self.chip_step, self.chip_roi, self.chip_candidates, self.chip_status]):
            chips.addWidget(chip, i // 3, i % 3)
        layout.addLayout(chips)
        return frame

    def _build_data_step_tab(self) -> QWidget:
        """Wizard step 1: data and processing step selection."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        layout.addWidget(self._build_goal_box())
        layout.addWidget(self._build_data_status_box())
        layout.addWidget(self._build_step_box())

        quick_box = QGroupBox("推荐流程")
        quick_layout = QVBoxLayout(quick_box)
        quick_layout.setContentsMargins(10, 12, 10, 10)
        quick_layout.setSpacing(6)
        flow = QLabel("选择目标倾向 → 检查数据 → 生成推荐 → 应用 / 对比 / 导出")
        flow.setObjectName("Hint")
        flow.setWordWrap(True)
        quick_layout.addWidget(flow)
        note = QLabel("默认使用均衡推荐；需要突出局部异常、连续界面、裂隙/破碎带或深部弱反射时，可主动切换目标倾向。")
        note.setObjectName("Hint")
        note.setWordWrap(True)
        quick_layout.addWidget(note)
        layout.addWidget(quick_box)
        layout.addStretch(1)
        return page

    def _build_goal_box(self) -> QGroupBox:
        box = QGroupBox("1. 选择目标倾向")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(8)
        self.target_goal_combo = QComboBox()
        self.target_goal_combo.addItems(self._TARGET_GOALS)
        self.target_goal_combo.currentTextChanged.connect(self._on_target_goal_changed)
        layout.addWidget(self.target_goal_combo)
        self.target_goal_desc = QLabel("")
        self.target_goal_desc.setObjectName("Hint")
        self.target_goal_desc.setWordWrap(True)
        layout.addWidget(self.target_goal_desc)
        return box

    def _build_data_status_box(self) -> QGroupBox:
        box = QGroupBox("当前数据")
        layout = QGridLayout(box)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)
        self.data_status_label = QLabel("未载入")
        self.data_status_label.setObjectName("InlineTitle")
        self.data_source_label = QLabel("请先在主工作区导入 GPR / UAV-GPR 数据。")
        self.data_source_label.setWordWrap(True)
        self.data_shape_label = QLabel("尺寸：--")
        self.data_stage_label = QLabel("当前阶段：原始数据")
        for label in [self.data_source_label, self.data_shape_label, self.data_stage_label]:
            label.setObjectName("Hint")
            label.setWordWrap(True)
        layout.addWidget(QLabel("状态"), 0, 0)
        layout.addWidget(self.data_status_label, 0, 1)
        layout.addWidget(QLabel("文件"), 1, 0)
        layout.addWidget(self.data_source_label, 1, 1)
        layout.addWidget(QLabel("尺寸"), 2, 0)
        layout.addWidget(self.data_shape_label, 2, 1)
        layout.addWidget(QLabel("阶段"), 3, 0)
        layout.addWidget(self.data_stage_label, 3, 1)
        return box

    def _build_target_region_tab(self) -> QWidget:
        """Wizard step 2: target/ROI and review boundaries."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        intro = QGroupBox("ROI")
        intro_layout = QVBoxLayout(intro)
        intro_layout.setContentsMargins(10, 12, 10, 10)
        intro_layout.setSpacing(8)
        hint = QLabel("默认使用全图 ROI；也可以选择自动或手动。")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        intro_layout.addWidget(hint)
        layout.addWidget(intro)

        layout.addWidget(self._build_roi_box())

        review_box = QGroupBox("结果说明")
        review_layout = QVBoxLayout(review_box)
        review_layout.setContentsMargins(10, 12, 10, 10)
        review_layout.setSpacing(8)
        review_hint = QLabel("真实数据、低置信度推荐、过平滑问题、候选冲突或手动重点区域参与时，都应提示人工确认。")
        review_hint.setObjectName("Hint")
        review_hint.setWordWrap(True)
        review_layout.addWidget(review_hint)
        layout.addWidget(review_box)
        layout.addStretch(1)
        return page

    def _build_recommendation_workflow_tab(self) -> QWidget:
        """Wizard step 3/4: run recommendation, review result and apply/export."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        action_box = QGroupBox("生成推荐")
        action_layout = QVBoxLayout(action_box)
        action_layout.setContentsMargins(10, 12, 10, 10)
        action_layout.setSpacing(8)
        hint = QLabel("点击生成推荐后，优先查看推荐参数、推荐理由和结果说明；候选空间与评分细节在高级设置中查看。")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        action_layout.addWidget(hint)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_generate_recommendation = QPushButton("生成推荐")
        self.btn_generate_recommendation.setObjectName("PrimaryButton")
        self.btn_generate_recommendation.setToolTip("按当前目标倾向、评分指标、候选空间和重点区域生成推荐。")
        self.btn_generate_recommendation.clicked.connect(self._on_run_recommendation_preview)
        self.btn_stage_compare_wizard = QPushButton("查看对比")
        self.btn_stage_compare_wizard.setToolTip("比较同一处理阶段内的候选方法。")
        self.btn_stage_compare_wizard.clicked.connect(lambda: self._legacy_page.btn_compare_stage.click())
        self.btn_view_details_wizard = QPushButton("高级明细")
        self.btn_view_details_wizard.clicked.connect(lambda: self._legacy_page.btn_view_auto_tune.click())
        self.btn_apply_wizard = QPushButton("应用推荐")
        self.btn_apply_wizard.clicked.connect(self._on_recipe_run_requested)
        row.addWidget(self.btn_generate_recommendation)
        row.addWidget(self.btn_stage_compare_wizard)
        row.addWidget(self.btn_view_details_wizard)
        row.addWidget(self.btn_apply_wizard)
        row.addStretch(1)
        action_layout.addLayout(row)
        layout.addWidget(action_box)

        layout.addWidget(self._build_top3_box())
        layout.addWidget(self._build_recommend_summary_box())
        layout.addStretch(1)
        return page

    def _build_top3_box(self) -> QGroupBox:
        ranking_box = QGroupBox("候选 Top-3")
        ranking_layout = QVBoxLayout(ranking_box)
        ranking_layout.setContentsMargins(10, 12, 10, 10)
        self.ranking_table = QTableWidget(0, 4)
        self.ranking_table.setHorizontalHeaderLabels(["排名", "候选", "分数", "结果说明"])
        self.ranking_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ranking_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ranking_table.verticalHeader().setVisible(False)
        self.ranking_table.setAlternatingRowColors(True)
        ranking_layout.addWidget(self.ranking_table)
        return ranking_box

    def _build_apply_report_tab(self) -> QWidget:
        """Wizard step 4: apply the recommendation or export a concise report."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        apply_box = QGroupBox("4. 应用与报告")
        apply_layout = QVBoxLayout(apply_box)
        apply_layout.setContentsMargins(10, 14, 10, 10)
        apply_layout.setSpacing(8)
        info = QLabel("确认推荐结果后，可一键应用、查看前后对比，或导出自动推荐报告。应用后仍可撤销和查看参数。")
        info.setObjectName("Hint")
        info.setWordWrap(True)
        apply_layout.addWidget(info)

        action_row = QWidget()
        row = QHBoxLayout(action_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.btn_apply_step_recommendation = QPushButton("应用推荐")
        self.btn_apply_step_recommendation.setToolTip("将当前推荐参数应用到对应处理步骤。")
        self.btn_apply_step_recommendation.clicked.connect(self._on_recipe_run_requested)
        self.btn_export_step_report = QPushButton("导出推荐报告")
        self.btn_export_step_report.setToolTip("导出推荐参数、结果说明和结论范围。")
        self.btn_export_step_report.clicked.connect(lambda: self._legacy_page.btn_export_comparison.click())
        self.btn_view_step_details = QPushButton("查看高级明细")
        self.btn_view_step_details.clicked.connect(lambda: self.advanced_toggle.setChecked(True))
        row.addWidget(self.btn_apply_step_recommendation)
        row.addWidget(self.btn_export_step_report)
        row.addWidget(self.btn_view_step_details)
        apply_layout.addWidget(action_row)
        layout.addWidget(apply_box)

        status_box = QGroupBox("当前推荐状态")
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(10, 14, 10, 10)
        self.apply_report_text = self._mini_text(150)
        status_layout.addWidget(self.apply_report_text)
        layout.addWidget(status_box, 1)
        layout.addStretch(1)
        return page

    def _build_recommend_summary_box(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.recommended_text = self._mini_text(92)
        self.score_text = self._mini_text(84)
        self.risk_text = self._mini_text(84)
        self.boundary_text = self._mini_text(84)
        for title, widget in [
            ("总推荐", self.recommended_text),
            ("评分概览", self.score_text),
            ("结果说明", self.risk_text),
            ("结论说明", self.boundary_text),
        ]:
            group = QGroupBox(title)
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(10, 12, 10, 10)
            group_layout.addWidget(widget)
            layout.addWidget(group)
        return page

    def _build_advanced_settings_tab(self) -> QTabWidget:
        """Advanced settings and audit details; hidden behind the last tab by default."""
        tabs = QTabWidget()
        tabs.addTab(self._scroll(self._build_roi_detail_tab()), "区域设置")

        candidate_page = QWidget()
        candidate_layout = QVBoxLayout(candidate_page)
        candidate_layout.setContentsMargins(8, 8, 8, 8)
        candidate_layout.setSpacing(10)
        candidate_layout.addWidget(self._build_step_box())
        candidate_layout.addWidget(self._build_candidate_box())
        candidate_layout.addStretch(1)
        tabs.addTab(self._scroll(candidate_page), "候选空间")

        score_page = QWidget()
        score_layout = QVBoxLayout(score_page)
        score_layout.setContentsMargins(8, 8, 8, 8)
        score_layout.setSpacing(10)
        score_layout.addWidget(self._build_scoring_box())
        score_layout.addWidget(self._build_safety_box())
        score_layout.addStretch(1)
        tabs.addTab(self._scroll(score_page), "评分与边界")

        compare_page = QWidget()
        compare_layout = QVBoxLayout(compare_page)
        compare_layout.setContentsMargins(6, 6, 6, 6)
        compare_layout.setSpacing(10)
        self.raw_card = self._preview_card("输入数据", "输入数据预览", "输入")
        self.candidate_card = self._preview_card("候选输出", "候选结果预览", "候选")
        self.recommended_card = self._preview_card("推荐输出", "推荐结果预览", "推荐")
        compare_layout.addWidget(self.raw_card)
        compare_layout.addWidget(self.candidate_card)
        compare_layout.addWidget(self.recommended_card)
        compare_layout.addStretch(1)
        tabs.addTab(self._scroll(compare_page), "预览")

        tabs.addTab(self._build_audit_tab(), "报告 / 审计")
        return tabs

    def _build_config_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        layout.addWidget(self._build_step_box())
        layout.addWidget(self._build_candidate_box())
        layout.addWidget(self._build_roi_box())
        layout.addWidget(self._build_scoring_box())
        layout.addWidget(self._build_safety_box())
        layout.addStretch(1)
        return page

    def _build_compare_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        self.raw_card = self._preview_card("原始数据", "原始数据预览", "输入")
        self.candidate_card = self._preview_card("当前候选", "候选结果预览", "候选")
        self.recommended_card = self._preview_card("推荐结果", "推荐结果预览", "推荐")
        layout.addWidget(self.raw_card)
        layout.addWidget(self.candidate_card)
        layout.addWidget(self.recommended_card)

        ranking_box = QGroupBox("候选排名 Top 3")
        ranking_layout = QVBoxLayout(ranking_box)
        self.ranking_table = QTableWidget(0, 4)
        self.ranking_table.setHorizontalHeaderLabels(["排名", "候选", "分数", "状态"])
        self.ranking_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ranking_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ranking_table.verticalHeader().setVisible(False)
        self.ranking_table.setAlternatingRowColors(True)
        ranking_layout.addWidget(self.ranking_table)
        layout.addWidget(ranking_box, 1)
        return page

    def _build_recommend_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        self.recommended_text = self._mini_text(84)
        self.score_text = self._mini_text(96)
        self.risk_text = self._mini_text(96)
        self.boundary_text = self._mini_text(104)
        for title, widget in [
            ("推荐参数", self.recommended_text),
            ("评分解释", self.score_text),
            ("结果说明", self.risk_text),
            ("结论边界", self.boundary_text),
        ]:
            group = QGroupBox(title)
            group_layout = QVBoxLayout(group)
            group_layout.addWidget(widget)
            layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _build_audit_tab(self) -> QTabWidget:
        tabs = QTabWidget()
        self.trial_table = QTableWidget(0, 8)
        self.trial_table.setHorizontalHeaderLabels(["候选流程", "关键参数", "总分", "流程评分", "背景抑制", "响应保留", "连续/深部", "说明"])
        self.trial_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.trial_table.horizontalHeader().setStretchLastSection(True)
        self.trial_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.trial_table.verticalHeader().setVisible(False)
        self.trial_table.setAlternatingRowColors(True)

        self.metrics_text = self._read_only_text()
        self.logs_text = self._read_only_text()
        self.warnings_text = self._read_only_text()
        self.claim_text = self._read_only_text()

        tabs.addTab(self.trial_table, "候选记录")
        tabs.addTab(self.metrics_text, "指标")
        tabs.addTab(self.logs_text, "日志")
        tabs.addTab(self.warnings_text, "结果说明")
        tabs.addTab(self.claim_text, "结论说明")
        return tabs

    def _build_step_box(self) -> QGroupBox:
        box = QGroupBox("1. 处理步骤")
        layout = QVBoxLayout(box)
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItems(self._WORKFLOW_STEPS)
        self.workflow_combo.currentTextChanged.connect(self._on_workflow_changed)
        layout.addWidget(self.workflow_combo)
        hint = QLabel("高级入口用于选择候选处理步骤；普通用户通常无需修改。")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return box

    def _build_candidate_box(self) -> QGroupBox:
        box = QGroupBox("高级候选设置")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        for key, label, desc in self._CANDIDATES:
            layout.addWidget(self._candidate_row(key, label, desc))

        rank_panel = QFrame()
        rank_panel.setObjectName("RankPanel")
        rank_layout = QGridLayout(rank_panel)
        rank_layout.setContentsMargins(10, 8, 10, 8)
        rank_layout.setHorizontalSpacing(8)
        rank_layout.setVerticalSpacing(6)

        title = QLabel("SVD rank sweep")
        title.setObjectName("InlineTitle")
        title.setToolTip("仅在启用 SVD 背景抑制时生效。")
        rank_layout.addWidget(title, 0, 0, 1, 3)

        self.svd_rank_min_spin = self._spin(1, 64, self._on_svd_rank_changed)
        self.svd_rank_max_spin = self._spin(1, 64, self._on_svd_rank_changed)
        self.svd_rank_step_spin = self._spin(1, 16, self._on_svd_rank_changed)

        for col, (label, widget) in enumerate(
            [
                ("最小", self.svd_rank_min_spin),
                ("最大", self.svd_rank_max_spin),
                ("步长", self.svd_rank_step_spin),
            ]
        ):
            lab = QLabel(label)
            lab.setObjectName("FieldLabel")
            rank_layout.addWidget(lab, 1, col)
            rank_layout.addWidget(widget, 2, col)

        layout.addWidget(rank_panel)
        return box

    def _candidate_row(self, key: str, label: str, desc: str) -> QFrame:
        """Create a compact product-style candidate checklist row."""
        row = QFrame()
        row.setObjectName("CandidateRow")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(10, 8, 10, 8)
        row_layout.setSpacing(8)

        cb = QCheckBox()
        cb.setToolTip(desc)
        cb.toggled.connect(lambda checked, k=key: self._on_candidate_toggled(k, checked))
        self._candidate_checks[key] = cb
        row_layout.addWidget(cb, 0, Qt.AlignmentFlag.AlignTop)

        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(2)
        title = QLabel(label)
        title.setObjectName("CandidateTitle")
        title.setWordWrap(True)
        subtitle = QLabel(desc)
        subtitle.setObjectName("CandidateSubtitle")
        subtitle.setWordWrap(True)
        text_box.addWidget(title)
        text_box.addWidget(subtitle)
        row_layout.addLayout(text_box, 1)

        tag_map = {
            "baseline": "基线",
            "mean": "fast",
            "median": "robust",
            "svd": "秩扫描",
            "sliding": "experimental",
        }
        tag = QLabel(tag_map.get(key, "candidate"))
        tag.setObjectName("MethodTag")
        tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row_layout.addWidget(tag, 0, Qt.AlignmentFlag.AlignTop)
        return row

    def _build_roi_box(self) -> QGroupBox:
        box = QGroupBox("ROI 设置")
        layout = QGridLayout(box)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.region_mode_combo = QComboBox()
        for key, label in self._ROI_MODES:
            self.region_mode_combo.addItem(label, key)
        self.region_mode_combo.currentIndexChanged.connect(self._on_region_mode_changed)
        layout.addWidget(QLabel("模式"), 0, 0)
        layout.addWidget(self.region_mode_combo, 0, 1)

        self.roi_mode_hint = QLabel("默认使用全图 ROI；适合快速推荐。")
        self.roi_mode_hint.setObjectName("Hint")
        self.roi_mode_hint.setWordWrap(True)
        layout.addWidget(self.roi_mode_hint, 1, 0, 1, 2)

        self.roi_trace_start = self._spin(0, 100000, self._on_roi_changed)
        self.roi_trace_end = self._spin(0, 100000, self._on_roi_changed)
        self.roi_sample_start = self._spin(0, 100000, self._on_roi_changed)
        self.roi_sample_end = self._spin(0, 100000, self._on_roi_changed)
        for row, (label, widget) in enumerate(
            [
                ("Trace 起点", self.roi_trace_start),
                ("Trace 终点", self.roi_trace_end),
                ("Sample 起点", self.roi_sample_start),
                ("Sample 终点", self.roi_sample_end),
            ],
            start=2,
        ):
            layout.addWidget(QLabel(label), row, 0)
            layout.addWidget(widget, row, 1)

        self.btn_pick_roi = QPushButton("框选")
        self.btn_pick_roi.setCheckable(True)
        self.btn_pick_roi.setChecked(False)
        self.btn_pick_roi.setToolTip("选择“手动”后，在右侧 B-scan 上拖拽框选 ROI。")
        self.btn_pick_roi.toggled.connect(self._on_pick_roi_toggled)
        self.btn_auto_roi = QPushButton("自动")
        self.btn_auto_roi.setEnabled(False)
        self.btn_auto_roi.clicked.connect(self._on_auto_roi_clicked)
        self.roi_picker_status_label = QLabel("ROI：全图")
        self.roi_picker_status_label.setProperty("class", "hintText")
        layout.addWidget(self.btn_pick_roi, 6, 0)
        layout.addWidget(self.btn_auto_roi, 6, 1)
        layout.addWidget(self.roi_picker_status_label, 7, 0, 1, 2)
        self._update_roi_mode_controls()
        return box

    def set_plot_roi_picker_status(self, enabled: bool) -> None:
        """由主窗口同步图上 ROI 框选开关状态。"""
        text = "ROI：框选中" if enabled else self._roi_mode_label()
        if hasattr(self, "roi_picker_status_label"):
            self.roi_picker_status_label.setText(text)
        if hasattr(self, "btn_pick_roi"):
            self.btn_pick_roi.setText("框选中" if enabled else "框选")

    def _build_scoring_box(self) -> QGroupBox:
        box = QGroupBox("高级评分指标")
        layout = QVBoxLayout(box)
        for key, label, desc in self._SCORING:
            cb = QCheckBox(label)
            cb.setToolTip(desc)
            cb.toggled.connect(lambda checked, k=key: self._on_scoring_toggled(k, checked))
            self._scoring_checks[key] = cb
            layout.addWidget(cb)
        return box

    def _build_safety_box(self) -> QGroupBox:
        box = QGroupBox("结论范围与结论说明")
        layout = QVBoxLayout(box)
        for key, label in [
            ("no_prior_warning", "真实 no-prior 数据结果说明"),
            ("display_only_flag", "仅显示增强标记"),
            ("manual_review_required", "人工确认说明"),
            ("claim_boundary_required", "自动生成结论范围说明"),
        ]:
            cb = QCheckBox(label)
            cb.toggled.connect(lambda checked, k=key: self._on_safety_toggled(k, checked))
            self._safety_checks[key] = cb
            layout.addWidget(cb)
        return box

    def _preview_card(self, title: str, main_text: str, tag: str) -> QFrame:
        card = QFrame()
        card.setObjectName("PreviewCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("CardTitle")
        tag_label = QLabel(tag)
        tag_label.setObjectName("MiniTag")
        tag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(title_label)
        header.addStretch(1)
        header.addWidget(tag_label)
        layout.addLayout(header)

        canvas = QLabel(main_text + "\n\n等待主工作区数据同步")
        canvas.setObjectName("PreviewCanvas")
        canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas.setMinimumHeight(96)
        canvas.setWordWrap(True)
        layout.addWidget(canvas, 1)
        if tag in {"Input", "输入"}:
            self.raw_preview_canvas = canvas
        elif tag in {"Candidate", "候选"}:
            self.candidate_preview_canvas = canvas
        elif tag in {"Recommended", "推荐"}:
            self.recommended_preview_canvas = canvas
        return card

    # ------------------------------------------------------------------
    # Small widget helpers
    # ------------------------------------------------------------------

    def _section_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("PanelTitle")
        return label

    def _chip(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("StatusChip")
        label.setProperty("tone", "neutral")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumWidth(112)
        return label

    def _set_chip(self, label: QLabel, text: str, tone: str = "neutral") -> None:
        if label.text() == text and label.property("tone") == tone:
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

    def _action_button(self, text: str, *, primary: bool) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("PrimaryButton" if primary else "SecondaryButton")
        btn.setProperty("actionRole", "primary" if primary else "secondary")
        btn.setMinimumWidth(58)
        btn.setMinimumHeight(32)
        return btn

    def _spin(self, low: int, high: int, callback) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.valueChanged.connect(callback)
        spin.setMinimumWidth(82)
        return spin

    def _read_only_text(self) -> QTextEdit:
        text = QTextEdit()
        text.setReadOnly(True)
        text.setMinimumHeight(180)
        return text

    def _mini_text(self, height: int) -> QTextEdit:
        text = QTextEdit()
        text.setReadOnly(True)
        text.setMinimumHeight(height)
        text.setMaximumHeight(max(height + 90, 180))
        return text

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _sync_controls_from_state(self) -> None:
        widgets = [
            self.workflow_combo,
            getattr(self, "target_goal_combo", None),
            getattr(self, "region_mode_combo", None),
            *self._candidate_checks.values(),
            self.svd_rank_min_spin,
            self.svd_rank_max_spin,
            self.svd_rank_step_spin,
            self.roi_trace_start,
            self.roi_trace_end,
            self.roi_sample_start,
            self.roi_sample_end,
            *self._scoring_checks.values(),
            *self._safety_checks.values(),
        ]
        widgets = [w for w in widgets if w is not None]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.workflow_combo.setCurrentText(self.state.workflow_step)
            if getattr(self, "target_goal_combo", None) is not None:
                self.target_goal_combo.setCurrentText(self.state.target_goal)
            if getattr(self, "region_mode_combo", None) is not None:
                for i in range(self.region_mode_combo.count()):
                    if self.region_mode_combo.itemData(i) == self.state.roi_mode:
                        self.region_mode_combo.setCurrentIndex(i)
                        break
            for key, cb in self._candidate_checks.items():
                cb.setChecked(key in self.state.candidate_methods)
            self.svd_rank_min_spin.setValue(self.state.svd_rank_min)
            self.svd_rank_max_spin.setValue(self.state.svd_rank_max)
            self.svd_rank_step_spin.setValue(self.state.svd_rank_step)
            self.roi_trace_start.setValue(self.state.roi_trace_start)
            self.roi_trace_end.setValue(self.state.roi_trace_end)
            self.roi_sample_start.setValue(self.state.roi_sample_start)
            self.roi_sample_end.setValue(self.state.roi_sample_end)
            for key, cb in self._scoring_checks.items():
                cb.setChecked(key in self.state.scoring_metrics)
            self._safety_checks["no_prior_warning"].setChecked(self.state.no_prior_warning)
            self._safety_checks["display_only_flag"].setChecked(self.state.display_only_flag)
            self._safety_checks["manual_review_required"].setChecked(self.state.manual_review_required)
            self._safety_checks["claim_boundary_required"].setChecked(self.state.claim_boundary_required)
        finally:
            for widget in widgets:
                widget.blockSignals(False)


    def _reset_workflow_customization(self) -> None:
        """Clear user workflow edits when the recommendation context changes."""
        self.state.workflow_param_overrides = {}
        self.state.workflow_order = []
        self.state.workflow_customized = False
        self.state.workflow_order_override = False
        self.state.workflow_param_override = False

    def _current_workflow_keys_from_table(self) -> list[str]:
        table = getattr(self, "workflow_table", None)
        if table is None:
            return []
        keys: list[str] = []
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            key = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if key:
                keys.append(str(key))
        return keys

    def _workflow_order_is_valid(self, keys: list[str]) -> tuple[bool, str]:
        if not keys:
            return True, ""
        index = {key: pos for pos, key in enumerate(keys)}
        if "zero_time" in index and index["zero_time"] != 0:
            return False, "零时校正必须保持在流程最前。"
        if "dewow" in index and "zero_time" in index and index["dewow"] < index["zero_time"]:
            return False, "Dewow 不能移动到零时校正之前。"
        if "background" in index and "bandpass" in index and index["background"] < index["bandpass"]:
            return False, "背景抑制不建议移动到频带滤波之前。"
        if "gain" in index:
            fixed_after_gain = [key for key in ("zero_time", "dewow", "bandpass", "background") if key in index and index[key] > index["gain"]]
            if fixed_after_gain:
                return False, "增益不能移动到基础校正或背景抑制之前。"
        return True, ""

    def _on_workflow_rows_moved(self, *args) -> None:
        keys = self._current_workflow_keys_from_table()
        ok, reason = self._workflow_order_is_valid(keys)
        if not ok:
            self.state.backend_message = reason
            self._refresh_from_state()
            return
        self.state.workflow_order = keys
        self.state.workflow_order_override = True
        self.state.workflow_customized = True
        self.state.manual_review_required = True
        self.state.backend_message = "已调整流程顺序；运行时将记录为自定义流程。"
        self._refresh_from_state()

    def _on_workflow_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item is None or item.column() != 1:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        if not key:
            return
        text = str(item.text()).strip() or "--"
        self.state.workflow_param_overrides[str(key)] = text
        self.state.workflow_param_override = True
        self.state.workflow_customized = True
        self.state.manual_review_required = True
        self.state.backend_message = "已修改流程参数；运行时将记录为自定义参数。"
        self._refresh_from_state()

    def _invalidate_candidate_cache(self) -> None:
        self._candidate_rows_cache_key = None
        self._candidate_rows_cache = []

    def _clear_backend_recommendation(self, status: str = "需重新生成") -> None:
        had_result = bool(self.state.backend_results) or self.state.recommendation_status == "已生成"
        self.state.backend_results = []
        self.state.background_results = []
        self.state.selected_candidate_name = "未生成"
        self.state.selected_candidate_params = "--"
        self.state.selected_candidate_score = 0.0
        if had_result or self.state.recommendation_status == "已生成":
            self.state.recommendation_status = status

    def _mark_recommendation_dirty(self) -> None:
        self._invalidate_candidate_cache()
        self._clear_backend_recommendation("需重新生成")
        self._reset_workflow_customization()

    def _on_target_goal_changed(self, value: str) -> None:
        self.state.target_goal = value or "均衡推荐"
        self._mark_recommendation_dirty()
        self._refresh_from_state()

    def _on_region_mode_changed(self) -> None:
        combo = getattr(self, "region_mode_combo", None)
        if combo is None:
            return
        self.state.roi_mode = str(combo.currentData() or "none")
        if self.state.roi_mode != "manual":
            self._set_host_roi_picker(False)
        self._update_roi_mode_controls()
        self._mark_recommendation_dirty()
        self._refresh_from_state()

    def _set_host_roi_picker(self, enabled: bool) -> None:
        parent = getattr(self, "parent_window", None)
        setter = getattr(parent, "_set_manual_roi_pick_enabled", None)
        if callable(setter):
            try:
                setter(bool(enabled))
            except Exception:
                pass

    def _on_pick_roi_toggled(self, checked: bool) -> None:
        if checked and self.state.roi_mode != "manual":
            self.state.roi_mode = "manual"
            combo = getattr(self, "region_mode_combo", None)
            if combo is not None:
                old = combo.blockSignals(True)
                for i in range(combo.count()):
                    if combo.itemData(i) == "manual":
                        combo.setCurrentIndex(i)
                        break
                combo.blockSignals(old)
        self._set_host_roi_picker(bool(checked and self.state.roi_mode == "manual"))
        self.set_plot_roi_picker_status(bool(checked and self.state.roi_mode == "manual"))

    def set_manual_roi_from_bounds(self, bounds: dict | None, *, activate: bool = True) -> None:
        """Receive ROI index bounds from the main B-scan picker.

        The main plot works in display-axis units.  ``AutoTuneSyncController``
        converts those units to sample/trace indices and calls this method so
        the AutoTune runner uses exactly the region the user dragged.
        """
        if not bounds:
            return
        try:
            trace_start = int(bounds.get("dist_start_idx", bounds.get("trace_start", 0)))
            trace_end = int(bounds.get("dist_end_idx", bounds.get("trace_end", trace_start + 1)))
            sample_start = int(bounds.get("time_start_idx", bounds.get("sample_start", 0)))
            sample_end = int(bounds.get("time_end_idx", bounds.get("sample_end", sample_start + 1)))
        except Exception:
            return
        if trace_end <= trace_start or sample_end <= sample_start:
            return
        self.state.roi_trace_start = trace_start
        self.state.roi_trace_end = trace_end
        self.state.roi_sample_start = sample_start
        self.state.roi_sample_end = sample_end
        if activate:
            self.state.roi_mode = "manual"
        self._sync_controls_from_state()
        self._mark_recommendation_dirty()
        self._refresh_from_state()

    def _on_auto_roi_clicked(self) -> None:
        roi = self._calculate_auto_roi()
        if roi is None:
            self.state.backend_message = "当前数据不足，无法自动建议重点区域"
            self._refresh_from_state()
            return
        self.state.roi_mode = "auto"
        self.state.roi_trace_start = roi["trace_start"]
        self.state.roi_trace_end = roi["trace_end"]
        self.state.roi_sample_start = roi["sample_start"]
        self.state.roi_sample_end = roi["sample_end"]
        self.state.backend_message = "已基于当前 B-scan 响应强度生成自动建议区域；该区域不是目标真值标签"
        self._sync_controls_from_state()
        self._mark_recommendation_dirty()
        self._refresh_from_state()

    def _calculate_auto_roi(self) -> dict | None:
        if self._current_data is None:
            return None
        arr = np.asarray(self._current_data, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return None
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        fill = float(np.nanmedian(finite))
        arr = np.where(np.isfinite(arr), arr, fill)
        amp = np.abs(arr)
        try:
            threshold = float(np.percentile(amp, 92.0))
        except Exception:
            threshold = float(np.max(amp))
        mask = amp >= threshold
        if not np.any(mask):
            s0, s1 = arr.shape[0] // 4, max(arr.shape[0] // 4 + 1, arr.shape[0] * 3 // 4)
            t0, t1 = arr.shape[1] // 4, max(arr.shape[1] // 4 + 1, arr.shape[1] * 3 // 4)
        else:
            sample_idx, trace_idx = np.where(mask)
            s0, s1 = int(sample_idx.min()), int(sample_idx.max()) + 1
            t0, t1 = int(trace_idx.min()), int(trace_idx.max()) + 1
            s_pad = max(3, int(arr.shape[0] * 0.04))
            t_pad = max(2, int(arr.shape[1] * 0.04))
            s0, s1 = s0 - s_pad, s1 + s_pad
            t0, t1 = t0 - t_pad, t1 + t_pad
        s0 = max(0, min(arr.shape[0] - 1, s0))
        s1 = max(s0 + 1, min(arr.shape[0], s1))
        t0 = max(0, min(arr.shape[1] - 1, t0))
        t1 = max(t0 + 1, min(arr.shape[1], t1))
        return {"sample_start": s0, "sample_end": s1, "trace_start": t0, "trace_end": t1}

    def _update_roi_mode_controls(self) -> None:
        mode = getattr(self.state, "roi_mode", "none")
        manual = mode == "manual"
        auto = mode == "auto"
        enabled = manual or auto
        for name in ["roi_trace_start", "roi_trace_end", "roi_sample_start", "roi_sample_end"]:
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(enabled)
        if getattr(self, "btn_pick_roi", None) is not None:
            self.btn_pick_roi.setEnabled(manual)
            checked = bool(manual and self.btn_pick_roi.isChecked())
            old = self.btn_pick_roi.blockSignals(True)
            self.btn_pick_roi.setChecked(checked)
            self.btn_pick_roi.blockSignals(old)
        if getattr(self, "btn_auto_roi", None) is not None:
            self.btn_auto_roi.setEnabled(auto)
        if getattr(self, "roi_mode_hint", None) is not None:
            if mode == "manual":
                text = "手动 ROI：在 B-scan 上框选重点区域。"
            elif mode == "auto":
                text = "自动 ROI：按当前响应估计关注范围，不代表目标真值。"
            else:
                text = "ROI：全图。"
            self.roi_mode_hint.setText(text)
        if getattr(self, "roi_picker_status_label", None) is not None:
            status = {
                "none": "ROI：全图",
                "auto": "ROI：自动",
                "manual": "ROI：手动",
            }.get(mode, "ROI：全图")
            self.roi_picker_status_label.setText(status)

    def _on_workflow_changed(self, value: str) -> None:
        self.state.workflow_step = value
        self._mark_recommendation_dirty()
        self._refresh_from_state()

    def _on_candidate_toggled(self, key: str, checked: bool) -> None:
        if checked:
            self.state.candidate_methods.add(key)
        else:
            self.state.candidate_methods.discard(key)
        self._mark_recommendation_dirty()
        self._refresh_from_state()

    def _on_svd_rank_changed(self) -> None:
        self.state.svd_rank_min = self.svd_rank_min_spin.value()
        self.state.svd_rank_max = max(self.svd_rank_max_spin.value(), self.state.svd_rank_min)
        self.state.svd_rank_step = max(1, self.svd_rank_step_spin.value())
        if self.svd_rank_max_spin.value() != self.state.svd_rank_max:
            self.svd_rank_max_spin.blockSignals(True)
            self.svd_rank_max_spin.setValue(self.state.svd_rank_max)
            self.svd_rank_max_spin.blockSignals(False)
        self._mark_recommendation_dirty()
        self._refresh_from_state()

    def _on_roi_changed(self) -> None:
        self.state.roi_trace_start = self.roi_trace_start.value()
        self.state.roi_trace_end = self.roi_trace_end.value()
        self.state.roi_sample_start = self.roi_sample_start.value()
        self.state.roi_sample_end = self.roi_sample_end.value()
        self._mark_recommendation_dirty()
        self._refresh_from_state()

    def _on_scoring_toggled(self, key: str, checked: bool) -> None:
        if checked:
            self.state.scoring_metrics.add(key)
        else:
            self.state.scoring_metrics.discard(key)
        self._mark_recommendation_dirty()
        self._refresh_from_state()

    def _on_safety_toggled(self, key: str, checked: bool) -> None:
        setattr(self.state, key, checked)
        self._mark_recommendation_dirty()
        self._refresh_from_state()

    def _on_run_recommendation_preview(self) -> None:
        """Run the minimal real backend when data is available.

        For the 背景抑制 step, this calls ``core.autotune_background_runner``
        on the currently loaded B-scan array. If only metadata is available,
        it falls back to the deterministic UI preview rows.
        """
        run_start_ts = time.perf_counter()
        if self.sender() is getattr(self, "btn_generate_recommendation", None):
            self._reset_workflow_customization()
        if self.state.data_source != "已载入":
            self.state.recommendation_status = "需载入数据"
            self.state.backend_message = "未载入数据"
        elif not self.state.recommendation_ready:
            self.state.recommendation_status = "配置未完整"
            self.state.backend_message = "目标区域、候选空间或评分指标未完整"
        else:
            results = []
            background_results = []
            if self._current_data is not None:
                try:
                    from core.autotune_background_runner import run_background_candidates
                    from core.autotune_workflow_planner import plan_workflow_recipes

                    ranks = list(range(
                        int(self.state.svd_rank_min),
                        int(self.state.svd_rank_max) + 1,
                        max(1, int(self.state.svd_rank_step)),
                    ))
                    roi = None if self.state.roi_mode == "none" else {
                        "trace_start": self.state.roi_trace_start,
                        "trace_end": self.state.roi_trace_end,
                        "sample_start": self.state.roi_sample_start,
                        "sample_end": self.state.roi_sample_end,
                    }
                    background_results = run_background_candidates(
                        self._current_data,
                        candidate_methods=sorted(self.state.candidate_methods),
                        svd_ranks=ranks,
                        roi=roi,
                        target_goal=self.state.target_goal,
                        scoring_metrics=sorted(self.state.scoring_metrics),
                        target_response=self._target_response_data,
                        use_v1_candidate_space=True,
                    )
                    results = plan_workflow_recipes(
                        self._current_data,
                        target_goal=self.state.target_goal,
                        roi_mode=self.state.roi_mode,
                        scoring_metrics=sorted(self.state.scoring_metrics),
                        target_response=self._target_response_data,
                        background_results=background_results,
                        max_candidates=12,
                    )
                    self.state.background_results = background_results
                    self.state.backend_results = results
                    self.state.backend_mode = "流程推荐"
                    hash_label = ""
                    if background_results and background_results[0].get("candidate_space_hash"):
                        hash_label = f"，候选空间 {str(background_results[0].get('candidate_space_hash'))[:8]}"
                    self.state.backend_message = f"已生成 {len(results)} 个候选流程，背景参数候选 {len(background_results)} 个{hash_label}"
                except Exception as exc:
                    self.state.background_results = []
                    self.state.backend_results = []
                    self.state.backend_mode = "UI 预览"
                    self.state.backend_message = f"流程推荐运行失败，已回退 UI 预览：{exc}"
            else:
                self.state.background_results = []
                self.state.backend_results = []
                self.state.backend_mode = "UI 预览"
                self.state.backend_message = "当前仅绑定元数据，使用 UI 预览"

            rows = self._candidate_rows()
            if rows:
                best = rows[0]
                self.state.selected_candidate_name = best[0]
                self.state.selected_candidate_params = best[1]
                self.state.selected_candidate_score = best[2]
                self.state.recommendation_status = "已生成"
            else:
                self.state.recommendation_status = "无候选"
        self._refresh_from_state()
        self._record_ui_perf("autotune.generate_preview_ms", (time.perf_counter() - run_start_ts) * 1000.0)


    def _selected_recipe_payload(self) -> dict | None:
        """Return the currently selected recipe row for execution/reporting."""
        if self.state.recommendation_status != "已生成":
            return None
        recipe = self._current_recipe() if hasattr(self, "_current_recipe") else None
        if recipe is None:
            return None
        if self.state.backend_results:
            payload = dict(self.state.backend_results[0])
        else:
            payload = {
                "name": "AutoTune 推荐流程",
                "target_goal": recipe.target_goal,
                "roi_mode": recipe.roi_mode,
                "score": recipe.score,
            }
        payload["recipe_steps"] = self._recipe_steps_payload(recipe.steps)
        payload.update(self._custom_workflow_metadata())
        if payload.get("manual_override"):
            payload["warning"] = str(payload.get("warning") or "用户修改了推荐流程或参数，需人工复核。")
        return payload

    def _on_recipe_run_requested(self) -> None:
        """Emit the selected workflow recipe to the main window executor."""
        if self.state.recommendation_status != "已生成":
            self._on_run_recommendation_preview()
        payload = self._selected_recipe_payload()
        if not payload:
            self.state.backend_message = "请先生成推荐"
            self._refresh_from_state()
            return
        self.recipe_run_requested.emit(payload)

    # ------------------------------------------------------------------
    # Lightweight refresh helpers
    # ------------------------------------------------------------------

    def _record_ui_perf(self, name: str, elapsed_ms: float) -> None:
        parent = getattr(self, "parent_window", None)
        monitor = getattr(parent, "_perf_monitor", None)
        if monitor is not None:
            try:
                monitor.record(name, float(elapsed_ms))
            except Exception:
                pass

    def _table_signature(self, rows) -> tuple:
        return tuple(tuple(str(value) for value in row) for row in rows)

    def _apply_table_rows(
        self,
        table: QTableWidget,
        rows,
        *,
        role_col: int | None = None,
        role_values=None,
        editable_cols: set[int] | None = None,
        row_height: int = 28,
        perf_name: str = "autotune.table_refresh",
    ) -> None:
        """Batch-replace table rows with signature-based skip."""
        rows = [tuple(str(value) for value in row) for row in rows]
        sig = self._table_signature(rows)
        attr = f"_mygpr_last_sig_{id(table)}"
        if getattr(self, attr, None) == sig:
            self._record_ui_perf(f"{perf_name}.skip", 0.0)
            return
        setattr(self, attr, sig)
        start_ts = time.perf_counter()
        editable_cols = set(editable_cols or set())
        table.setUpdatesEnabled(False)
        with QSignalBlocker(table):
            try:
                table.setRowCount(len(rows))
                for row_idx, values in enumerate(rows):
                    role_value = None
                    if role_values is not None and row_idx < len(role_values):
                        role_value = role_values[row_idx]
                    for col, value in enumerate(values):
                        item = QTableWidgetItem(value)
                        item.setToolTip(value)
                        if role_col is not None and col == role_col and role_value is not None:
                            item.setData(Qt.ItemDataRole.UserRole, role_value)
                        flags = item.flags()
                        if col in editable_cols:
                            flags |= Qt.ItemFlag.ItemIsEditable
                        else:
                            flags &= ~Qt.ItemFlag.ItemIsEditable
                        item.setFlags(flags)
                        table.setItem(row_idx, col, item)
                    table.setRowHeight(row_idx, row_height)
            finally:
                table.setUpdatesEnabled(True)
        self._record_ui_perf(perf_name, (time.perf_counter() - start_ts) * 1000.0)

    # ------------------------------------------------------------------
    # State -> UI refresh
    # ------------------------------------------------------------------

    def _refresh_from_state(self) -> None:
        refresh_start_ts = time.perf_counter()
        data_label = self.state.source_label if self.state.data_source == "已载入" else "未载入"
        if len(data_label) > 18:
            data_label = data_label[:15] + "..."
        self._set_chip(self.chip_data, f"数据：{data_label}", "good" if self.state.data_source == "已载入" else "neutral")
        self._set_chip(self.chip_step, f"目标：{self.state.target_goal}", "neutral")
        roi_text = {"none": "ROI：全图", "auto": "ROI：自动", "manual": "ROI：手动"}.get(self.state.roi_mode, "ROI：全图")
        roi_tone = "neutral" if self.state.roi_mode == "none" else ("good" if self.state.roi_is_set or self.state.roi_mode == "auto" else "warning")
        self._set_chip(self.chip_roi, roi_text, roi_tone)
        self._set_chip(self.chip_candidates, f"候选：{self.state.candidate_count}", "good" if self.state.candidate_count else "warning")
        if self.state.data_source != "已载入":
            status_text = "未就绪"
            status_tone = "neutral"
        elif self.state.recommendation_status == "已生成":
            status_text = "已生成"
            status_tone = "good"
        elif self.state.recommendation_ready:
            status_text = "待推荐"
            status_tone = "warning"
        else:
            status_text = "需配置"
            status_tone = "warning"
        self._set_chip(self.chip_status, f"状态：{status_text}", status_tone)
        if getattr(self, "chip_risk", None) is not None:
            self._set_chip(self.chip_risk, f"提示：{self.state.risk_level}", "neutral")
        ready_for_recommendation = self.state.data_source == "已载入" and self.state.recommendation_ready
        self.btn_run_autotune_mvp.setEnabled(ready_for_recommendation)
        for btn_name in [
            "btn_generate_recommendation",
            "btn_stage_compare_wizard",
            "btn_view_details_wizard",
            "btn_apply_wizard",
            "btn_apply_step_recommendation",
            "btn_export_step_report",
            "btn_view_step_details",
        ]:
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(self.state.data_source == "已载入")
        if getattr(self, "data_status_label", None) is not None:
            self.data_status_label.setText("已载入" if self.state.data_source == "已载入" else "未载入")
        if getattr(self, "data_source_label", None) is not None:
            self.data_source_label.setText(self.state.source_label if self.state.data_source == "已载入" else "请先在主工作区导入 GPR / UAV-GPR 数据。")
        if getattr(self, "data_shape_label", None) is not None:
            self.data_shape_label.setText(self._shape_text())
        if getattr(self, "data_stage_label", None) is not None:
            self.data_stage_label.setText(f"当前阶段：{self.state.processing_stage}")
        if getattr(self, "target_goal_desc", None) is not None:
            self.target_goal_desc.setText(self._target_goal_description(self.state.target_goal))
        self._update_roi_mode_controls()
        if getattr(self, "next_step_hint", None) is not None:
            if self.state.data_source != "已载入":
                hint = "下一步：在左侧主工作区导入数据，AutoTune 页会自动同步。"
                tone = "neutral"
            elif self.state.roi_mode == "manual" and not self.state.roi_is_set:
                hint = "下一步：请完成手动 ROI 或切回全图。"
                tone = "warning"
            elif self.state.candidate_count == 0:
                hint = "下一步：至少启用一个候选方法。"
                tone = "warning"
            elif self.state.recommendation_status == "已生成":
                if self.state.workflow_customized:
                    hint = "已自定义流程或参数，可运行；结果将记录为人工修改。"
                    tone = "warning"
                else:
                    hint = "推荐已生成，可直接运行，也可查看候选对比。"
                    tone = "good"
            else:
                hint = "点击“生成”，查看流程和候选对比。"
                tone = "good"
            self.next_step_hint.setText(hint)
            try:
                from ui.theme import set_dynamic_property, repolish

                if set_dynamic_property(self.next_step_hint, "tone", tone):
                    repolish(self.next_step_hint)
            except Exception:
                self.next_step_hint.setProperty("tone", tone)


        self._refresh_preview_cards()
        self._refresh_recommendation_text()
        self._refresh_apply_report_text()
        self._refresh_ranking()
        if getattr(self, "advanced_panel", None) is not None and self.advanced_panel.isVisible():
            self._refresh_trial_table()
            self._refresh_audit_text()
        self._record_ui_perf("autotune.page_refresh_ms", (time.perf_counter() - refresh_start_ts) * 1000.0)

    def _candidate_rows(self) -> list[tuple[str, str, float, str]]:
        if self.state.backend_results:
            rows = [
                (
                    str(item.get("name", "候选")),
                    str(item.get("params", "--")),
                    float(item.get("score", 0.0)),
                    str(item.get("status", "备选")),
                )
                for item in self.state.backend_results
            ]
            rows.sort(key=lambda x: x[2], reverse=True)
            return rows

        key = (
            tuple(sorted(self.state.candidate_methods)),
            self.state.svd_rank_min,
            self.state.svd_rank_max,
            self.state.svd_rank_step,
        )
        if key == self._candidate_rows_cache_key:
            return list(self._candidate_rows_cache)

        rows: list[tuple[str, str, float, str]] = []
        if "svd" in self.state.candidate_methods:
            rank = self.state.svd_rank_min
            while rank <= self.state.svd_rank_max:
                score = 0.70 + min(rank, 5) * 0.028
                rows.append((f"SVD 背景抑制 rank={rank}", f"rank={rank}", min(score, 0.88), "备选" if rank > 3 else "推荐"))
                rank += max(1, self.state.svd_rank_step)
        if "median" in self.state.candidate_methods:
            rows.append(("中位数背景扣除", "method=median", 0.77, "稳健候选"))
        if "mean" in self.state.candidate_methods:
            rows.append(("均值背景扣除", "method=mean", 0.71, "基线+"))
        if "sliding" in self.state.candidate_methods:
            rows.append(("滑动窗口背景", "window=preview", 0.66, "实验性"))
        if "baseline" in self.state.candidate_methods:
            rows.append(("不处理基线", "method=none", 0.52, "对照"))
        rows.sort(key=lambda x: x[2], reverse=True)
        self._candidate_rows_cache_key = key
        self._candidate_rows_cache = list(rows)
        return list(rows)

    def _current_backend_item(self) -> dict:
        if self.state.backend_results and self.state.recommendation_status == "已生成":
            try:
                return dict(self.state.backend_results[0])
            except Exception:
                return {}
        return {}

    def _current_scoring_record(self) -> dict:
        item = self._current_backend_item()
        record = item.get("autotune_scoring_record") if isinstance(item, dict) else None
        if isinstance(record, dict) and record:
            return record
        if item:
            try:
                from core.autotune_scoring_record import build_scoring_v2_record

                return build_scoring_v2_record(
                    item,
                    target_goal=self.state.target_goal,
                    roi_mode=self.state.roi_mode,
                    target_response_available=self._target_response_data is not None,
                )
            except Exception:
                return {}
        return {}

    def _format_scoring_record_for_ui(self, record: dict | None = None) -> str:
        record = record if record is not None else self._current_scoring_record()
        if not record:
            return "scoring v2：等待生成推荐后写入候选记录。"
        try:
            from core.autotune_scoring_record import summarize_record

            return summarize_record(record)
        except Exception:
            return f"scoring v2：{float(record.get('final_score', 0.0) or 0.0):.2f}"

    def _score_term(self, item: dict, key: str, *, section: str = "workflow") -> float:
        record = item.get("autotune_scoring_record") if isinstance(item.get("autotune_scoring_record"), dict) else {}
        source = {}
        if section == "background":
            background = record.get("background_score") if isinstance(record.get("background_score"), dict) else {}
            source = background.get("terms") if isinstance(background.get("terms"), dict) else {}
        else:
            workflow = record.get("workflow_score") if isinstance(record.get("workflow_score"), dict) else {}
            source = workflow.get("terms") if isinstance(workflow.get("terms"), dict) else {}
        if key in source:
            try:
                return float(source[key])
            except Exception:
                return 0.0
        terms = item.get("scoring_terms") if isinstance(item.get("scoring_terms"), dict) else {}
        for candidate_key in (key, f"v2_{key}"):
            if candidate_key in terms:
                try:
                    return float(terms[candidate_key])
                except Exception:
                    return 0.0
        return 0.0


    def _refresh_preview_cards(self) -> None:
        loaded = self.state.data_source == "已载入"
        if loaded:
            lines = [
                f"已载入：{self.state.source_label}",
                self._shape_text(),
                f"类型：{self.state.data_type}",
                f"阶段：{self.state.processing_stage}",
            ]
            if self.state.component:
                lines.append(f"分量：{self.state.component}")
            lines.append(f"后端：{self.state.backend_mode} · {self.state.backend_message}")
            raw_text = "\n".join(lines)
        else:
            raw_text = "未载入数据\n\n请先在左侧主工作区导入 CSV / GPR 数据。"

        if hasattr(self, "raw_preview_canvas"):
            self.raw_preview_canvas.setText(raw_text)
        if hasattr(self, "candidate_preview_canvas"):
            self.candidate_preview_canvas.setText(
                f"当前候选输出\n\n候选数：{self.state.candidate_count}\nROI：Trace {self.state.roi_trace_start}-{self.state.roi_trace_end} / Sample {self.state.roi_sample_start}-{self.state.roi_sample_end}" if loaded else "当前候选输出\n\n等待数据载入"
            )
        if hasattr(self, "recommended_preview_canvas"):
            self.recommended_preview_canvas.setText(
                f"推荐候选输出\n\n状态：{self.state.recommendation_status}\n推荐：{self.state.selected_candidate_name}\n后端：{self.state.backend_mode}" if loaded else "推荐候选输出\n\n等待数据载入"
            )


    def _apply_workflow_ui_overrides(self, recipe):
        """Apply local table order/parameter edits to a recipe object."""
        if recipe is None:
            return None
        try:
            from core.autotune_recipe import AutoTuneRecipe, AutoTuneRecipeStep
        except Exception:
            return recipe
        steps = list(recipe.steps)
        if self.state.workflow_param_overrides:
            rewritten = []
            for step in steps:
                params = self.state.workflow_param_overrides.get(step.key, step.params)
                source = "user" if step.key in self.state.workflow_param_overrides else step.source
                rewritten.append(
                    AutoTuneRecipeStep(
                        key=step.key,
                        label=step.label,
                        method=step.method,
                        params=params,
                        enabled=step.enabled,
                        source=source,
                    )
                )
            steps = rewritten
        if self.state.workflow_order:
            by_key = {step.key: step for step in steps}
            ordered = [by_key[key] for key in self.state.workflow_order if key in by_key]
            ordered.extend(step for step in steps if step.key not in self.state.workflow_order)
            steps = ordered
        notes = tuple(recipe.notes)
        if self.state.workflow_customized:
            notes = notes + ("用户已修改推荐流程或参数，运行和导出时应按自定义流程记录。",)
        return AutoTuneRecipe(
            target_goal=recipe.target_goal,
            roi_mode=recipe.roi_mode,
            steps=tuple(steps),
            score=recipe.score,
            data_mode=recipe.data_mode,
            notes=notes,
        )

    def _recipe_steps_payload(self, steps) -> list[dict]:
        payload = []
        for order, step in enumerate(steps):
            payload.append(
                {
                    "key": step.key,
                    "label": step.label,
                    "method": step.method,
                    "params": step.params,
                    "enabled": step.enabled,
                    "source": step.source,
                    "ui_order": order,
                    "manual_override": step.source == "user",
                }
            )
        return payload

    def _custom_workflow_metadata(self) -> dict:
        return {
            "manual_override": bool(self.state.workflow_customized),
            "workflow_override": bool(self.state.workflow_order_override),
            "parameter_override": bool(self.state.workflow_param_override),
            "manual_review_required": bool(self.state.manual_review_required or self.state.workflow_customized),
            "workflow_order": list(self.state.workflow_order),
            "parameter_overrides": dict(self.state.workflow_param_overrides),
        }

    def _current_recipe(self):
        rows = self._candidate_rows()
        best = rows[0] if rows else ("--", "--", 0.0, "")
        recommended_name = self.state.selected_candidate_name if self.state.recommendation_status == "已生成" else best[0]
        recommended_params = self.state.selected_candidate_params if self.state.recommendation_status == "已生成" else best[1]
        recommended_score = self.state.selected_candidate_score if self.state.recommendation_status == "已生成" else best[2]
        try:
            from core.autotune_recipe import build_workflow_recipe

            recipe_steps = None
            if self.state.backend_results and self.state.recommendation_status == "已生成":
                recipe_steps = self.state.backend_results[0].get("recipe_steps")
            recipe = build_workflow_recipe(
                target_goal=self.state.target_goal,
                roi_mode=self.state.roi_mode,
                best_candidate_name=recommended_name,
                best_candidate_params=recommended_params,
                best_score=recommended_score,
                target_response_available=self._target_response_data is not None,
                backend_mode=self.state.backend_mode,
                recipe_steps=recipe_steps,
            )
            return self._apply_workflow_ui_overrides(recipe)
        except Exception:
            return None

    def _step_ui_note(self, step) -> str:
        key = str(getattr(step, "key", ""))
        if key == "zero_time":
            return "保持当前数据校正"
        if key == "background":
            return "作为推荐流程的核心处理项"
        if key == "gain":
            return "用于增强可读性，结果需结合原图复核"
        if key == "denoise":
            return "轻度处理，避免过平滑"
        return "按目标倾向与数据特征生成"

    def _refresh_parameter_table(self, recipe) -> None:
        table = getattr(self, "parameter_table", None)
        if table is None:
            return
        rows = []
        if recipe is not None:
            for step in recipe.steps:
                if not step.enabled:
                    continue
                rows.append((step.label, step.method, step.params, self._step_ui_note(step)))
        if not rows:
            rows = [
                ("Dewow", "window", "auto", "按当前数据生成推荐值"),
                ("频带滤波", "range", "auto", "按采样与频谱信息生成推荐值"),
                ("背景抑制", "method / params", self.state.selected_candidate_params, "根据当前候选空间推荐"),
                ("增益", "mode / window", "auto", "按目标倾向生成推荐值"),
            ]
        self._apply_table_rows(table, rows, row_height=28, perf_name="autotune.parameter_table_refresh_ms")


    def _refresh_candidate_step_table(self) -> None:
        table = getattr(self, "candidate_step_table", None)
        if table is None:
            return
        background_count = max(0, self.state.candidate_count)
        rows = [
            ("Dewow", "auto", "已生成", "随推荐流程给出窗口建议"),
            ("频带滤波", "auto", "已生成", "随推荐流程给出频带建议"),
            ("背景抑制", str(background_count), "已参与推荐", "使用当前候选空间比较"),
            ("增益", "auto", "已生成", "随目标倾向给出增强建议"),
            ("去噪", "可选", "按流程启用", "仅在对应目标倾向下启用"),
            ("迁移", "后续", "未默认启用", "保留为高级模块"),
        ]
        self._apply_table_rows(table, rows, row_height=26, perf_name="autotune.candidate_step_table_refresh_ms")


    def _refresh_recommendation_text(self) -> None:
        rows = self._candidate_rows()
        best = rows[0] if rows else ("无候选", "--", 0.0, "未就绪")
        recommended_name = self.state.selected_candidate_name if self.state.recommendation_status == "已生成" else best[0]
        recommended_params = self.state.selected_candidate_params if self.state.recommendation_status == "已生成" else best[1]
        recommended_score = self.state.selected_candidate_score if self.state.recommendation_status == "已生成" else best[2]
        recipe = self._current_recipe()

        if recipe is not None:
            if getattr(self, "recipe_flow_label", None) is not None:
                self.recipe_flow_label.setText(recipe.flow_text)
            if getattr(self, "recipe_param_label", None) is not None:
                param_lines = recipe.parameter_text.splitlines()
                self.recipe_param_label.setText("  ·  ".join(param_lines[:4]) if param_lines else "推荐参数会在生成后显示。")
            if getattr(self, "recipe_score_chip", None) is not None:
                chip_text = "未生成" if self.state.recommendation_status != "已生成" else f"分数 {recipe.score:.2f}"
                self._set_chip(self.recipe_score_chip, chip_text, "good" if self.state.recommendation_status == "已生成" else "neutral")
            if getattr(self, "workflow_table", None) is not None:
                with QSignalBlocker(self.workflow_table):
                    self.workflow_table.setRowCount(len(recipe.steps))
                    for row_idx, step in enumerate(recipe.steps):
                        values = [step.label, step.params, step.method, self._step_ui_note(step)]
                        for col, value in enumerate(values):
                            cell_text = str(value)
                            cell_item = QTableWidgetItem(cell_text)
                            cell_item.setToolTip(cell_text)
                            cell_item.setData(Qt.ItemDataRole.UserRole, step.key)
                            flags = cell_item.flags()
                            if col == 1:
                                flags |= Qt.ItemFlag.ItemIsEditable
                            else:
                                flags &= ~Qt.ItemFlag.ItemIsEditable
                            cell_item.setFlags(flags)
                            self.workflow_table.setItem(row_idx, col, cell_item)
                        self.workflow_table.setRowHeight(row_idx, 28)
                self._refresh_parameter_table(recipe)

        self.recommended_text.setText(
            "\n".join(
                [
                    f"目标：{self.state.target_goal}",
                    f"状态：{self.state.recommendation_status}",
                    f"当前推荐：{recommended_name}",
                    f"参数：{recommended_params}",
                    f"分数：{recommended_score:.2f}",
                    f"ROI：{self._roi_mode_label()}",
                    f"流程：{recipe.flow_text if recipe is not None else self._pipeline_summary()}",
                ]
            )
        )
        metric_names = [self._metric_label(k) for k in sorted(self.state.scoring_metrics)]
        scoring_record = self._current_scoring_record()
        score_lines = [
            f"目标倾向：{self.state.target_goal}",
            self._target_goal_description(self.state.target_goal),
            "",
            "推荐依据：",
            *[f"- {name}" for name in metric_names],
            "",
            "综合状态：",
            "可生成推荐" if self.state.recommendation_ready else "配置未完整",
            "",
            "scoring v2 记录：",
            self._format_scoring_record_for_ui(scoring_record),
            "目标权重：" + self._active_weight_summary(),
        ]
        self.score_text.setText("\n".join(score_lines))

        notes = []
        if self._target_response_data is not None:
            notes.append("检测到参考响应，可使用有参考指标参与排序。")
        else:
            notes.append("未检测到参考响应，使用当前数据的启发式指标排序。")
        if self.state.display_only_flag:
            notes.append("显示增强仅影响视觉呈现，不作为幅值定量依据。")
        if self.state.roi_mode == "manual" and not self.state.roi_is_set:
            notes.append("手动 ROI 尚未完成。")
        elif self.state.roi_mode == "none":
            notes.append("当前使用全图推荐。")
        if recipe is not None:
            notes.extend(recipe.notes)
        if self.state.backend_results:
            first = self.state.backend_results[0]
            message = str(first.get("note") or first.get("warning") or "").strip()
            if first.get("background_low_benefit") and not message:
                message = "背景抑制收益较弱，已采用温和背景抑制方法。"
            if message and message not in notes:
                notes.append(message)
        self.risk_text.setText("\n".join(f"• {item}" for item in notes))
        self.boundary_text.setText(self._claim_boundary_text())

    def _refresh_apply_report_text(self) -> None:
        text = getattr(self, "apply_report_text", None)
        if text is None:
            return
        rows = self._candidate_rows()[:3]
        if self.state.data_source != "已载入":
            text.setText("尚未载入数据。请先在主工作区导入测线，然后返回自动选参。")
            return
        scoring_record = self._current_scoring_record()
        lines = [
            f"推荐状态：{self.state.recommendation_status}",
            f"目标倾向：{self.state.target_goal}",
            f"处理步骤：{self.state.workflow_step}",
            f"ROI：{self._roi_mode_label()}",
            f"数据：{self.state.source_label}",
            "",
            "scoring v2：",
            self._format_scoring_record_for_ui(scoring_record),
            "",
            "Top 候选：",
        ]
        for idx, (name, params, score, status) in enumerate(rows, start=1):
            lines.append(f"{idx}. {name} | {params} | {score:.2f} | {status}")
        lines.extend([
            "",
            "说明：真实数据结果应结合原始图像、现场资料和处理记录共同判断。",
        ])
        text.setText("\n".join(lines))

    def _refresh_ranking(self) -> None:
        role_values = []
        if self.state.backend_results and self.state.recommendation_status == "已生成":
            items = list(self.state.backend_results)[:3]
            rows = []
            for row_idx, item_data in enumerate(items):
                rank_text = "推荐" if row_idx == 0 else str(row_idx + 1)
                name = str(item_data.get("name", "候选"))
                score = f"{float(item_data.get('score', 0.0)):.2f}"
                status_text = (
                    f"流程 {self._score_term(item_data, 'workflow_fit'):.2f} / "
                    f"背景 {self._score_term(item_data, 'background_suppression', section='background'):.2f} / "
                    f"保留 {self._score_term(item_data, 'response_preservation', section='background'):.2f}"
                )
                rows.append((rank_text, name, score, status_text))
                role_values.append("recommended" if row_idx == 0 else None)
            self._apply_table_rows(
                self.ranking_table,
                rows,
                role_col=0,
                role_values=role_values,
                row_height=30,
                perf_name="autotune.ranking_table_refresh_ms",
            )
            self._refresh_candidate_step_table()
            return

        rows = []
        for row_idx, (name, params, score, status) in enumerate(self._candidate_rows()[:3]):
            rank_text = "推荐" if row_idx == 0 else str(row_idx + 1)
            status_text = "推荐" if row_idx == 0 else status
            rows.append((rank_text, name, f"{score:.2f}", status_text))
            role_values.append("recommended" if row_idx == 0 else None)
        self._apply_table_rows(
            self.ranking_table,
            rows,
            role_col=0,
            role_values=role_values,
            row_height=30,
            perf_name="autotune.ranking_table_refresh_ms",
        )
        self._refresh_candidate_step_table()


    def _refresh_trial_table(self) -> None:
        if self.state.backend_results and self.state.recommendation_status == "已生成":
            rows = []
            role_values = []
            for row_idx, item in enumerate(list(self.state.backend_results)):
                rows.append((
                    str(item.get("name", "候选")),
                    str(item.get("params", "--")),
                    f"{float(item.get('score', 0.0)):.2f}",
                    f"{self._score_term(item, 'workflow_fit'):.2f}",
                    f"{self._score_term(item, 'background_suppression', section='background'):.2f}",
                    f"{self._score_term(item, 'response_preservation', section='background'):.2f}",
                    f"{max(self._score_term(item, 'continuity', section='background'), self._score_term(item, 'deep_balance', section='background')):.2f}",
                    str(item.get("warning") or item.get("status") or "备选"),
                ))
                role_values.append("recommended" if row_idx == 0 else None)
            self._apply_table_rows(
                self.trial_table,
                rows,
                role_col=0,
                role_values=role_values,
                row_height=28,
                perf_name="autotune.trial_table_refresh_ms",
            )
            return

        rows = []
        role_values = []
        for row_idx, (name, params, score, status) in enumerate(self._candidate_rows()):
            rows.append((
                name,
                params,
                f"{score:.2f}",
                "预览",
                "预览",
                "预览",
                "预览",
                "推荐" if row_idx == 0 else status,
            ))
            role_values.append("recommended" if row_idx == 0 else None)
        self._apply_table_rows(
            self.trial_table,
            rows,
            role_col=0,
            role_values=role_values,
            row_height=28,
            perf_name="autotune.trial_table_refresh_ms",
        )


    def _refresh_audit_text(self) -> None:
        self.metrics_text.setText(
            "\n".join(
                [
                    f"数据状态 = {self.state.data_source}",
                    f"数据名称 = {self.state.source_label}",
                    f"数据类型 = {self.state.data_type}",
                    f"数据尺寸 = {self.state.data_shape}",
                    f"处理阶段 = {self.state.processing_stage}",
                    f"目标倾向 = {self.state.target_goal}",
                    f"ROI 模式 = {self._roi_mode_label()}",
                    f"候选数量 = {self.state.candidate_count}",
                    f"指标数量 = {self.state.scoring_count}",
                    f"ROI 已设置 = {self.state.roi_is_set}",
                    f"结果提示等级 = {self.state.risk_level}",
                    f"推荐状态 = {self.state.recommendation_status}",
                    "",
                    f"后端模式 = {self.state.backend_mode}",
                    f"后端信息 = {self.state.backend_message}",
                    f"目标权重 = {self._active_weight_summary()}",
                    "",
                    self._format_scoring_record_for_ui(),
                    "说明：当前使用有边界的流程模板和背景参数候选生成推荐 recipe；不是全局自由搜索。",
                ]
            )
        )
        self.logs_text.setText(
            "\n".join(
                [
                    "自动推荐向导已加载。",
                    "数据状态会从主工作区导入流程自动同步。",
                    f"当前数据：{self.state.source_label if self.state.data_source == '已载入' else '未载入'}",
                    "开始推荐按钮会生成候选处理流程，并为背景抑制、增益等步骤填入推荐参数。",
                    "修改目标、ROI 或候选空间会刷新候选流程和结果说明。",
                ]
            )
        )
        self.warnings_text.setText(
            "\n".join(
                [
                    "推荐方案来自当前候选空间，不代表全局最优 workflow。",
                    "真实数据没有参考响应时，排序依据为启发式指标。",
                    "报告中会记录目标倾向、ROI 模式、候选空间和参数来源。",
                ]
            )
        )
        self.claim_text.setText(self._claim_boundary_text())

    def _claim_boundary_text(self) -> str:
        return (
            f"该推荐表示在当前处理目标“{self.state.target_goal}”、候选空间、ROI 模式和评分指标设置下，"
            "系统生成的一套处理流程与参数建议。"
            "\n\n它不代表全局最优处理流程，也不等同于真实场地验证。"
            "真实数据结果应结合原始图像、现场资料和处理记录共同判断。"
        )

    def _active_weight_summary(self) -> str:
        if self.state.backend_results:
            weights = self.state.backend_results[0].get("scoring_weights") or {}
            if isinstance(weights, dict) and weights:
                return ", ".join(f"{self._metric_label(k)}={float(v):.2f}" for k, v in weights.items())
        try:
            from core.autotune_scoring_weights import resolve_scoring_weights

            _, _, weights = resolve_scoring_weights(
                target_goal=self.state.target_goal,
                scoring_metrics=sorted(self.state.scoring_metrics),
                target_response_available=self._target_response_data is not None,
            )
            return ", ".join(f"{self._metric_label(k)}={float(v):.2f}" for k, v in weights.items())
        except Exception:
            return "--"

    def _target_goal_description(self, goal: str) -> str:
        mapping = {
            "均衡推荐": "默认模式：兼顾杂波削弱、结构保留和显示可读性，适合新用户快速获得稳健建议。",
            "局部异常增强": "偏向突出管线、空洞、孤立强散射等局部异常；可能削弱部分连续层状反射。",
            "连续界面保留": "偏向保留地层分界、连续或缓倾斜反射。",
            "滑坡基覆界面 / 潜在滑移面": "偏向保留较连续、较弱或深部的界面响应，适合滑坡基覆界面相关处理方案。",
            "裂隙/破碎带保留": "偏向保留断续反射和局部破碎响应；尖峰抑制和平滑不宜过强。",
            "含水软弱带": "偏向保留衰减、弱反射和带状连续响应，增益与背景抑制保持温和。",
            "深部弱反射增强": "偏向提升深部弱反射可见性；AGC / 增益后不适合直接做幅值定量解释。",
        }
        return mapping.get(goal, mapping["均衡推荐"])

    def _roi_mode_label(self) -> str:
        if self.state.roi_mode == "manual":
            if self.state.roi_is_set:
                return (
                    f"手动 ROI T{self.state.roi_trace_start}-{self.state.roi_trace_end} / "
                    f"S{self.state.roi_sample_start}-{self.state.roi_sample_end}"
                )
            return "手动 ROI 未完成"
        if self.state.roi_mode == "auto":
            return "自动 ROI"
        return "全图"

    def _pipeline_summary(self) -> str:
        recipe = self._current_recipe() if hasattr(self, "_current_recipe") else None
        if recipe is not None:
            return recipe.flow_text
        return "零时校正 → Dewow → 频带滤波 → 背景抑制 → 增益"

    def _metric_label(self, key: str) -> str:
        for metric_key, label, _ in self._SCORING:
            if metric_key == key:
                return label
        return key

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    def refresh_theme(self, theme: str | None = None) -> None:
        """Force-refresh local AutoTune styling after an app theme switch."""
        self._local_style_theme_key = None
        self._apply_local_style(theme_override=theme)

    def _apply_local_style(self, theme_override: str | None = None) -> None:
        """Refresh AutoTune page theme properties.

        All AutoTune-specific QSS now lives in ``ui.theme``. The page only
        exposes objectName / dynamic properties, which prevents local QSS from
        drifting out of sync with the global light/dark theme.
        """
        try:
            from core.theme_manager import get_theme_manager
            from ui.theme import get_effective_theme_key, repolish

            explicit_theme = theme_override if theme_override is not None else get_theme_manager().get_current_theme()
            theme_key = get_effective_theme_key(explicit_theme, widget=self)
        except Exception:
            try:
                from ui.theme import get_effective_theme_key, repolish

                theme_key = get_effective_theme_key(theme_override, widget=self)
            except Exception:
                theme_key = "light"
                repolish = None

        if self._local_style_theme_key == theme_key:
            return
        self._local_style_theme_key = theme_key
        self.setProperty("effectiveTheme", theme_key)
        if repolish is not None:
            repolish(self)
        else:
            try:
                self.style().unpolish(self)
                self.style().polish(self)
                self.update()
            except Exception:
                pass
