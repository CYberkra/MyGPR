#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoTune 参数推荐控制台（高级视觉精修版）。

GX-UI-010 light polish pass:
- 专门服务 AutoTune 参数推荐，不引入 Research Lab / 3D viewer。
- 针对 MyGPR 当前右侧工作区宽度重排为“顶部状态 + 标签页”结构。
- 仅做 UI 状态联动，不触发 AutoTune/gprMax/Evidence 执行。
- 保留 legacy AutoTunePage 兼容层，避免 app_qt 既有信号/状态调用断裂。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtCore import Qt
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
    QSpinBox,
    QPushButton,
    QSizePolicy,
    QHeaderView,
    QScrollArea,
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

    @property
    def roi_is_set(self) -> bool:
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
        if not self.roi_is_set:
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
        return self.roi_is_set and self.candidate_count > 0 and self.scoring_count > 0


class AutoTuneTuningPage(QWidget):
    """AutoTune 参数推荐页；legacy AutoTunePage 仅作为兼容层。"""

    _WORKFLOW_STEPS = ["背景抑制", "增益", "Dewow", "频带滤波", "显示增强"]

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
        ("shape", "目标形态稳定性", "双曲线/目标区域连续性与位移风险。"),
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
        self.state.backend_mode = "真实候选" if data_array is not None else "UI 预览"
        self.state.backend_message = "已绑定当前 B-scan 数据" if data_array is not None else "仅绑定元数据"
        self.state.backend_results = []
        self.state.recommendation_status = "未运行"
        self.state.selected_candidate_name = "未生成"
        self.state.selected_candidate_params = "--"
        self.state.selected_candidate_score = 0.0
        self._candidate_rows_cache_key = None
        self._candidate_rows_cache = []
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
        self.state.backend_mode = "UI 预览"
        self.state.backend_message = "未运行"
        self.state.backend_results = []
        self._candidate_rows_cache_key = None
        self._candidate_rows_cache = []
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

        self.main_tabs = QTabWidget()
        self.main_tabs.setObjectName("MainTabs")
        self.main_tabs.addTab(self._scroll(self._build_config_tab()), "配置")
        self.main_tabs.addTab(self._scroll(self._build_compare_tab()), "对比")
        self.main_tabs.addTab(self._scroll(self._build_recommend_tab()), "推荐")
        self.main_tabs.addTab(self._build_audit_tab(), "审计")
        root.addWidget(self.main_tabs, 1)

        self._legacy_page.hide()
        root.addWidget(self._legacy_page)

    def _scroll(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(widget)
        return area

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("AutoTuneHeader")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 9, 14, 9)
        layout.setSpacing(6)

        title = QLabel("AutoTune 参数推荐")
        title.setObjectName("AutoTuneTitle")
        subtitle = QLabel("ROI 感知候选比较 · 固定流程参数推荐 · 风险提示")
        subtitle.setObjectName("AutoTuneSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.btn_load_data = self._action_button("数据", primary=False)
        self.btn_run_autotune_mvp = self._action_button("推荐", primary=True)
        self.btn_export_evidence_mvp = self._action_button("报告", primary=False)
        self.btn_load_data.setToolTip("请在左侧主工作区导入数据；本页会自动同步当前数据状态。")
        self.btn_run_autotune_mvp.setToolTip("生成安全的 UI 预览推荐，不调用生产 AutoTune。")
        self.btn_export_evidence_mvp.setToolTip("Evidence 导出将在后续阶段接入。")
        self.btn_load_data.setEnabled(False)
        self.btn_run_autotune_mvp.setEnabled(False)
        self.btn_export_evidence_mvp.setEnabled(False)
        self.btn_run_autotune_mvp.clicked.connect(self._on_run_recommendation_preview)
        actions.addWidget(self.btn_load_data)
        actions.addWidget(self.btn_run_autotune_mvp)
        actions.addWidget(self.btn_export_evidence_mvp)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.next_step_hint = QLabel("下一步：载入数据并确认 ROI 后生成推荐。")
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
        self.chip_risk = self._chip("")
        for i, chip in enumerate(
            [self.chip_data, self.chip_step, self.chip_roi, self.chip_candidates, self.chip_status, self.chip_risk]
        ):
            chips.addWidget(chip, i // 2, i % 2)
        layout.addLayout(chips)
        return frame

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
            ("风险提示", self.risk_text),
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
        self.trial_table = QTableWidget(0, 7)
        self.trial_table.setHorizontalHeaderLabels(["候选", "参数", "分数", "ROI", "残差", "CNR/SNR", "风险"])
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
        tabs.addTab(self.warnings_text, "风险")
        tabs.addTab(self.claim_text, "边界")
        return tabs

    def _build_step_box(self) -> QGroupBox:
        box = QGroupBox("1. 处理步骤")
        layout = QVBoxLayout(box)
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItems(self._WORKFLOW_STEPS)
        self.workflow_combo.currentTextChanged.connect(self._on_workflow_changed)
        layout.addWidget(self.workflow_combo)
        hint = QLabel("当前 MVP 优先支持“背景抑制”参数推荐；其他步骤保留为后续入口。")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return box

    def _build_candidate_box(self) -> QGroupBox:
        box = QGroupBox("2. 候选空间")
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
        box = QGroupBox("3. ROI 设置")
        layout = QGridLayout(box)
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
            layout.addWidget(QLabel(label), row, 0)
            layout.addWidget(widget, row, 1)
        self.btn_pick_roi = QCheckBox("启用图上框选 ROI")
        self.btn_pick_roi.setChecked(False)
        self.btn_pick_roi.setToolTip("默认关闭。开启后在主 B-scan 图中左键拖拽即可写入手动 ROI；关闭时左键只用于选道。")
        self.btn_auto_roi = QPushButton("自动建议 ROI")
        self.btn_auto_roi.setEnabled(False)
        self.roi_picker_status_label = QLabel("图上框选：关闭（默认）")
        self.roi_picker_status_label.setProperty("class", "hintText")
        layout.addWidget(self.btn_pick_roi, 4, 0)
        layout.addWidget(self.btn_auto_roi, 4, 1)
        layout.addWidget(self.roi_picker_status_label, 5, 0, 1, 2)
        return box

    def set_plot_roi_picker_status(self, enabled: bool) -> None:
        """由主窗口同步图上 ROI 框选开关状态。"""
        text = "图上框选：开启（左键拖拽主 B-scan 写入 ROI）" if enabled else "图上框选：关闭（默认）"
        if hasattr(self, "roi_picker_status_label"):
            self.roi_picker_status_label.setText(text)

    def _build_scoring_box(self) -> QGroupBox:
        box = QGroupBox("4. 评分指标")
        layout = QVBoxLayout(box)
        for key, label, desc in self._SCORING:
            cb = QCheckBox(label)
            cb.setToolTip(desc)
            cb.toggled.connect(lambda checked, k=key: self._on_scoring_toggled(k, checked))
            self._scoring_checks[key] = cb
            layout.addWidget(cb)
        return box

    def _build_safety_box(self) -> QGroupBox:
        box = QGroupBox("5. 安全边界")
        layout = QVBoxLayout(box)
        for key, label in [
            ("no_prior_warning", "no-prior 风险提示"),
            ("display_only_flag", "display-only 增强标记"),
            ("manual_review_required", "人工复核要求"),
            ("claim_boundary_required", "自动生成 claim boundary"),
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
        btn.setMinimumHeight(34)
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
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.workflow_combo.setCurrentText(self.state.workflow_step)
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

    def _invalidate_candidate_cache(self) -> None:
        self._candidate_rows_cache_key = None

    def _mark_recommendation_dirty(self) -> None:
        self._invalidate_candidate_cache()
        if self.state.recommendation_status == "已生成":
            self.state.recommendation_status = "需重新生成"

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
        if self.state.data_source != "已载入":
            self.state.recommendation_status = "需载入数据"
            self.state.backend_message = "未载入数据"
        elif not self.state.recommendation_ready:
            self.state.recommendation_status = "配置未完整"
            self.state.backend_message = "ROI、候选空间或评分指标未完整"
        else:
            results = []
            if self._current_data is not None and self.state.workflow_step == "背景抑制":
                try:
                    from core.autotune_background_runner import run_background_candidates

                    ranks = list(range(
                        int(self.state.svd_rank_min),
                        int(self.state.svd_rank_max) + 1,
                        max(1, int(self.state.svd_rank_step)),
                    ))
                    roi = {
                        "trace_start": self.state.roi_trace_start,
                        "trace_end": self.state.roi_trace_end,
                        "sample_start": self.state.roi_sample_start,
                        "sample_end": self.state.roi_sample_end,
                    }
                    results = run_background_candidates(
                        self._current_data,
                        candidate_methods=sorted(self.state.candidate_methods),
                        svd_ranks=ranks,
                        roi=roi,
                    )
                    self.state.backend_results = results
                    self.state.backend_mode = "真实候选"
                    self.state.backend_message = f"已完成真实背景抑制候选比较，共 {len(results)} 个 trial"
                except Exception as exc:
                    self.state.backend_results = []
                    self.state.backend_mode = "UI 预览"
                    self.state.backend_message = f"真实候选运行失败，已回退 UI 预览：{exc}"
            else:
                self.state.backend_results = []
                self.state.backend_mode = "UI 预览"
                self.state.backend_message = "当前步骤尚未接入真实 runner，使用 UI 预览"

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

    # ------------------------------------------------------------------
    # State -> UI refresh
    # ------------------------------------------------------------------

    def _refresh_from_state(self) -> None:
        data_label = self.state.source_label if self.state.data_source == "已载入" else "未载入"
        if len(data_label) > 18:
            data_label = data_label[:15] + "..."
        self._set_chip(self.chip_data, f"数据：{data_label}", "good" if self.state.data_source == "已载入" else "neutral")
        self._set_chip(self.chip_step, f"步骤：{self.state.workflow_step}", "neutral")
        self._set_chip(self.chip_roi, "ROI：已设置" if self.state.roi_is_set else "ROI：未设置", "good" if self.state.roi_is_set else "warning")
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
        risk_tone = "danger" if self.state.risk_level == "高" else "warning" if self.state.risk_level == "中" else "good"
        self._set_chip(self.chip_risk, f"风险：{self.state.risk_level}", risk_tone)
        self.btn_run_autotune_mvp.setEnabled(self.state.data_source == "已载入" and self.state.recommendation_ready)
        if getattr(self, "next_step_hint", None) is not None:
            if self.state.data_source != "已载入":
                hint = "下一步：在左侧主工作区导入数据，AutoTune 页会自动同步。"
                tone = "neutral"
            elif not self.state.roi_is_set:
                hint = "下一步：设置 ROI，避免推荐只基于全图统计。"
                tone = "warning"
            elif self.state.candidate_count == 0:
                hint = "下一步：至少启用一个候选方法。"
                tone = "warning"
            elif self.state.recommendation_status == "已生成":
                hint = "下一步：审查推荐理由、风险提示和候选记录。"
                tone = "good"
            else:
                hint = "下一步：点击“推荐”生成候选排名和结论边界。"
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
        self._refresh_ranking()
        self._refresh_trial_table()
        self._refresh_audit_text()

    def _candidate_rows(self) -> list[tuple[str, str, float, str]]:
        if self.state.backend_results:
            rows = [
                (
                    str(item.get("name", "候选")),
                    str(item.get("params", "--")),
                    float(item.get("score", 0.0)),
                    str(item.get("status", "需复核")),
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
                rows.append((f"SVD 背景抑制 rank={rank}", f"rank={rank}", min(score, 0.88), "需复核" if rank > 3 else "推荐候选"))
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

    def _refresh_recommendation_text(self) -> None:
        rows = self._candidate_rows()
        best = rows[0] if rows else ("无候选", "--", 0.0, "未就绪")
        recommended_name = self.state.selected_candidate_name if self.state.recommendation_status == "已生成" else best[0]
        recommended_params = self.state.selected_candidate_params if self.state.recommendation_status == "已生成" else best[1]
        recommended_score = self.state.selected_candidate_score if self.state.recommendation_status == "已生成" else best[2]
        self.recommended_text.setText(
            "\n".join(
                [
                    f"推荐状态：{self.state.recommendation_status}",
                    f"推荐候选：{recommended_name}",
                    f"参数：{recommended_params}",
                    f"预览分数：{recommended_score:.2f}",
                    f"数据：{self.state.source_label if self.state.data_source == '已载入' else '未载入'}",
                    f"{self._shape_text()}",
                    f"候选数：{self.state.candidate_count}",
                    f"ROI：Trace {self.state.roi_trace_start}-{self.state.roi_trace_end}, "
                    f"Sample {self.state.roi_sample_start}-{self.state.roi_sample_end}",
                    f"后端模式：{self.state.backend_mode}",
                    f"后端信息：{self.state.backend_message}",
                    "说明：背景抑制步骤已接入真实候选 runner；其余步骤仍为 UI 预览。",
                ]
            )
        )
        metric_names = [self._metric_label(k) for k in sorted(self.state.scoring_metrics)]
        self.score_text.setText(
            "\n".join(
                [
                    "启用指标：",
                    *[f"- {name}" for name in metric_names],
                    "",
                    f"综合状态：{'可进行推荐' if self.state.recommendation_ready else '配置未完整'}",
                ]
            )
        )
        warnings = []
        if self.state.no_prior_warning:
            warnings.append("无先验风险：真实数据不能作地下结构真实性声明。")
        if self.state.display_only_flag:
            warnings.append("仅显示增强：不能用于幅值定量结论。")
        if self.state.manual_review_required:
            warnings.append("人工复核：推荐结果需人工确认。")
        if not self.state.roi_is_set:
            warnings.append("ROI 未有效设置。")
        if not warnings:
            warnings.append("当前未启用额外风险提示。")
        self.risk_text.setText("\n".join(f"⚠ {w}" for w in warnings))
        self.boundary_text.setText(self._claim_boundary_text())

    def _refresh_ranking(self) -> None:
        rows = self._candidate_rows()[:3]
        self.ranking_table.setRowCount(len(rows))
        for row_idx, (name, params, score, status) in enumerate(rows):
            rank_text = "推荐" if row_idx == 0 else str(row_idx + 1)
            status_text = "推荐候选" if row_idx == 0 else status
            for col, value in enumerate([rank_text, name, f"{score:.2f}", status_text]):
                item = QTableWidgetItem(value)
                if row_idx == 0:
                    item.setData(Qt.ItemDataRole.UserRole, "recommended")
                self.ranking_table.setItem(row_idx, col, item)
            self.ranking_table.setRowHeight(row_idx, 30)

    def _refresh_trial_table(self) -> None:
        rows = self._candidate_rows()
        self.trial_table.setRowCount(len(rows))
        for row_idx, (name, params, score, status) in enumerate(rows):
            values = [
                name,
                params,
                f"{score:.2f}",
                "已设置" if self.state.roi_is_set else "缺失",
                "预览",
                "预览",
                "推荐候选" if row_idx == 0 else status,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if row_idx == 0:
                    item.setData(Qt.ItemDataRole.UserRole, "recommended")
                self.trial_table.setItem(row_idx, col, item)
            self.trial_table.setRowHeight(row_idx, 28)

    def _refresh_audit_text(self) -> None:
        self.metrics_text.setText(
            "\n".join(
                [
                    f"数据状态 = {self.state.data_source}",
                    f"数据名称 = {self.state.source_label}",
                    f"数据类型 = {self.state.data_type}",
                    f"数据尺寸 = {self.state.data_shape}",
                    f"处理阶段 = {self.state.processing_stage}",
                    f"候选数量 = {self.state.candidate_count}",
                    f"指标数量 = {self.state.scoring_count}",
                    f"ROI 已设置 = {self.state.roi_is_set}",
                    f"风险等级 = {self.state.risk_level}",
                    f"推荐状态 = {self.state.recommendation_status}",
                    "",
                    f"后端模式 = {self.state.backend_mode}",
                    f"后端信息 = {self.state.backend_message}",
                    "MVP 说明：背景抑制使用真实候选 runner；仍不代表生产 AutoTune 全流程。",
                ]
            )
        )
        self.logs_text.setText(
            "\n".join(
                [
                    "AutoTune 参数推荐控制台已加载。",
                    "数据状态会从主工作区导入流程自动同步。",
                    f"当前数据：{self.state.source_label if self.state.data_source == '已载入' else '未载入'}",
                    "开始推荐按钮会在背景抑制步骤运行真实候选比较；其他步骤保留 UI 预览。",
                    "修改左侧配置会刷新候选排名、Trial Table 和风险文本。",
                ]
            )
        )
        self.warnings_text.setText(
            "\n".join(
                [
                    "不代表全局最优 workflow。",
                    "不代表 field validation。",
                    "不证明 AutoTune 优于专家。",
                    "真实 no-prior 数据必须保留人工复核边界。",
                ]
            )
        )
        self.claim_text.setText(self._claim_boundary_text())

    def _claim_boundary_text(self) -> str:
        return (
            "该推荐仅表示在当前固定 workflow、候选空间、ROI 和评分指标设置下，"
            "候选参数在 UI 预览评分语义中表现较好。"
            "\n\n它不代表全局最优处理流程，不代表真实场地验证，"
            "也不能证明 AutoTune 在所有数据上优于专家。"
        )

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
