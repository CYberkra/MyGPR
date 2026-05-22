#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 调参与实验页面。"""

import json
import os
import platform
import subprocess

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QGroupBox,
    QTextEdit,
    QScrollArea,
    QFrame,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
from qfluentwidgets import PushButton, FluentIcon, SegmentedWidget

from core.methods_registry import PROCESSING_METHODS, get_method_display_name


class AutoTunePage(QWidget):
    """调参与实验页面。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self._method_key = None
        self._supports_auto_tune = False
        self._last_result = None
        self._last_stage_compare_result = None
        self._last_comparison_result = None
        self._last_evidence_bundle = None
        self._truth_side_by_side_preview_cache_key = None
        self._truth_side_by_side_preview_cache_pixmap = QPixmap()
        self.setup_ui()

    def setup_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer_layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        title = QLabel("调参与实验")
        title.setProperty("class", "sectionTitle")
        layout.addWidget(title)

        page_hint = QLabel(
            "本页集中处理候选参数分析、同阶段方法比较与实验验证。分析完成后，可回到“日常处理”页把“应用方法”的默认来源切换为“使用自动调参参数”。"
        )
        page_hint.setWordWrap(True)
        page_hint.setProperty("class", "hintText")
        layout.addWidget(page_hint)

        # ========== 流程概览 ==========
        flow_box = QGroupBox("实验流程")
        flow_box.setProperty("class", "calloutBox")
        flow_layout = QVBoxLayout(flow_box)
        flow_layout.setContentsMargins(10, 14, 10, 10)
        flow_layout.setSpacing(8)

        flow_hint = QLabel(
            "推荐顺序：先配置 ROI 与搜索强度，再做自动选参或同阶段比较，最后查看结果并决定是否采用推荐方案。"
        )
        flow_hint.setWordWrap(True)
        flow_hint.setProperty("class", "hintText")
        flow_layout.addWidget(flow_hint)

        flow_row = QWidget()
        flow_row_layout = QHBoxLayout(flow_row)
        flow_row_layout.setContentsMargins(0, 0, 0, 0)
        flow_row_layout.setSpacing(8)
        for text in ["① 参数配置", "② 实验执行", "③ 结果查看"]:
            chip = QLabel(text)
            chip.setProperty("class", "statusChip")
            flow_row_layout.addWidget(chip)
        flow_row_layout.addStretch(1)
        flow_layout.addWidget(flow_row)
        layout.addWidget(flow_box)

        # ========== 顶部标签 ==========
        self.segmented = SegmentedWidget(self)
        self.segmented.addItem("config", "参数配置")
        self.segmented.addItem("actions", "实验执行")
        self.segmented.addItem("results", "结果查看")
        self.segmented.addItem("truth", "真值验证")
        layout.addWidget(self.segmented)

        self.stack = QStackedWidget(self)
        layout.addWidget(self.stack)

        self.page_config = self._build_config_page()
        self.page_actions = self._build_actions_page()
        self.page_results = self._build_results_page()
        self.page_truth = self._build_truth_validation_page()

        self.stack.addWidget(self.page_config)
        self.stack.addWidget(self.page_actions)
        self.stack.addWidget(self.page_results)
        self.stack.addWidget(self.page_truth)

        self.segmented.setCurrentItem("config")
        self.stack.setCurrentIndex(0)
        self.segmented.currentItemChanged.connect(self._on_segment_changed)

        layout.addStretch(1)
        self.reset_for_method(None)

    def _build_config_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        method_box = QGroupBox("当前方法与实验边界")
        method_box.setToolTip("先确认当前方法是否支持自动选参，再决定是在本页快速实验还是进入工作台")
        method_layout = QVBoxLayout(method_box)
        method_layout.setContentsMargins(10, 14, 10, 10)
        method_layout.setSpacing(8)

        self.auto_tune_method_label = QLabel("当前方法：未选择")
        self.auto_tune_method_label.setProperty("class", "titleSmall")
        method_layout.addWidget(self.auto_tune_method_label)

        method_hint = QLabel(
            "本页适合做单方法自动选参和同阶段快速比较。需要跨方法串联、手工反复试验或长期保留实验链路时，再进入工作台。"
        )
        method_hint.setWordWrap(True)
        method_hint.setProperty("class", "hintText")
        method_layout.addWidget(method_hint)
        layout.addWidget(method_box)

        config_box = QGroupBox("实验配置")
        config_box.setToolTip("控制评分区域和搜索深度")
        config_layout = QVBoxLayout(config_box)
        config_layout.setContentsMargins(10, 14, 10, 10)
        config_layout.setSpacing(10)

        config_hint = QLabel(
            "ROI 决定评分聚焦区域，搜索决定候选数量与细化强度。一般先用“当前裁剪区优先 + 标准”。"
        )
        config_hint.setWordWrap(True)
        config_hint.setProperty("class", "hintText")
        config_layout.addWidget(config_hint)

        form_row = QWidget()
        form_layout = QGridLayout(form_row)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(10)

        form_layout.addWidget(QLabel("ROI 来源"), 0, 0)
        self.auto_tune_roi_combo = QComboBox()
        self.auto_tune_roi_combo.addItem("当前裁剪区优先", "prefer_crop")
        self.auto_tune_roi_combo.addItem("自动 ROI", "auto")
        self.auto_tune_roi_combo.addItem("全图", "full")
        self.auto_tune_roi_combo.setToolTip("自动选参评分时优先使用哪一块区域")
        form_layout.addWidget(self.auto_tune_roi_combo, 0, 1)

        form_layout.addWidget(QLabel("搜索强度"), 1, 0)
        self.auto_tune_search_combo = QComboBox()
        self.auto_tune_search_combo.addItem("快速", "fast")
        self.auto_tune_search_combo.addItem("标准", "standard")
        self.auto_tune_search_combo.addItem("深入", "thorough")
        self.auto_tune_search_combo.setCurrentIndex(1)
        self.auto_tune_search_combo.setToolTip("控制粗筛/细化的候选数量与搜索深度")
        form_layout.addWidget(self.auto_tune_search_combo, 1, 1)

        config_layout.addWidget(form_row)
        layout.addWidget(config_box)
        layout.addStretch(1)
        return page

    def _build_actions_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        action_box = QGroupBox("快速实验动作")
        action_box.setToolTip("对当前方法进行自动量化评估、候选搜索、同阶段比较与实验结果查看")
        action_layout = QVBoxLayout(action_box)
        action_layout.setContentsMargins(10, 14, 10, 10)
        action_layout.setSpacing(10)

        auto_hint = QLabel(
            "建议先完成候选分析或同阶段比较，再决定是否把“应用方法”的默认来源切换为自动调参参数。"
        )
        auto_hint.setWordWrap(True)
        auto_hint.setProperty("class", "hintText")
        action_layout.addWidget(auto_hint)

        primary_row = QWidget()
        primary_layout = QHBoxLayout(primary_row)
        primary_layout.setContentsMargins(0, 0, 0, 0)
        primary_layout.setSpacing(8)

        self.btn_auto_tune = PushButton(FluentIcon.SETTING, "自动选参")
        self.btn_auto_tune.setToolTip("对当前方法的候选参数自动评分并生成推荐参数")
        self.btn_compare_stage = PushButton(FluentIcon.FILTER, "同阶段实验比较")
        self.btn_compare_stage.setToolTip("比较当前 stage 内多个可用方法，推荐更合适的方法")
        self.btn_compare_stage.setEnabled(False)
        self.btn_compare_manual_auto = PushButton(FluentIcon.VIEW, "人工/自动对比")
        self.btn_compare_manual_auto.setToolTip("用经验/当前参数 baseline 与自动选参结果生成科研对比")
        self.btn_view_auto_tune = PushButton(FluentIcon.VIEW, "查看实验结果")
        self.btn_view_auto_tune.setEnabled(False)
        self.btn_view_auto_tune.setToolTip("查看候选参数、阶段比较与推荐理由")

        primary_layout.addWidget(self.btn_auto_tune)
        primary_layout.addWidget(self.btn_compare_stage)
        primary_layout.addWidget(self.btn_compare_manual_auto)
        primary_layout.addWidget(self.btn_view_auto_tune)
        primary_layout.addStretch(1)
        action_layout.addWidget(primary_row)

        adopt_box = QGroupBox("结果采用")
        adopt_box.setProperty("class", "lowProfileBox")
        adopt_layout = QVBoxLayout(adopt_box)
        adopt_layout.setContentsMargins(8, 12, 8, 8)
        adopt_layout.setSpacing(8)

        adopt_hint = QLabel("如果同阶段比较已经给出明确推荐，可以直接把推荐方法和参数写回日常处理。")
        adopt_hint.setWordWrap(True)
        adopt_hint.setProperty("class", "hintText")
        adopt_layout.addWidget(adopt_hint)

        self.btn_apply_stage_choice = PushButton(FluentIcon.ACCEPT, "采用推荐方案")
        self.btn_apply_stage_choice.setEnabled(False)
        self.btn_apply_stage_choice.setToolTip("将同阶段实验比较推荐的方法和参数写回日常处理")
        adopt_layout.addWidget(self.btn_apply_stage_choice)
        action_layout.addWidget(adopt_box)
        layout.addWidget(action_box)

        bridge_box = QGroupBox("深度实验入口")
        bridge_box.setToolTip("帮助区分本页快速实验与 Workbench 深度实验的使用边界")
        bridge_layout = QVBoxLayout(bridge_box)
        bridge_layout.setContentsMargins(10, 14, 10, 10)
        bridge_layout.setSpacing(8)

        bridge_hint = QLabel(
            "当你需要手工串联多步方法、反复试错或长期保留实验链路时，跳到工作台继续。"
        )
        bridge_hint.setWordWrap(True)
        bridge_hint.setProperty("class", "hintText")
        bridge_layout.addWidget(bridge_hint)

        bridge_row = QWidget()
        bridge_row_layout = QHBoxLayout(bridge_row)
        bridge_row_layout.setContentsMargins(0, 0, 0, 0)
        bridge_row_layout.setSpacing(8)
        self.btn_open_workbench = PushButton(FluentIcon.APPLICATION, "进入工作台深度实验")
        self.btn_open_workbench.setToolTip("需要跨方法组合、手工调参与长链路实验时，跳转到工作台继续")
        bridge_row_layout.addWidget(self.btn_open_workbench)
        bridge_row_layout.addStretch(1)
        bridge_layout.addWidget(bridge_row)
        layout.addWidget(bridge_box)

        layout.addStretch(1)
        return page

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        overview_box = QGroupBox("结果概览与推荐")
        overview_layout = QVBoxLayout(overview_box)
        overview_layout.setContentsMargins(10, 14, 10, 10)
        overview_layout.setSpacing(10)

        result_hint = QLabel("先看状态、稳定性和阶段比较，再决定是否把推荐方案写回日常处理。")
        result_hint.setWordWrap(True)
        result_hint.setProperty("class", "hintText")
        overview_layout.addWidget(result_hint)

        overview_grid = QWidget()
        grid = QGridLayout(overview_grid)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel("状态"), 0, 0)
        self.result_state_label = QLabel("未分析")
        self.result_state_label.setProperty("class", "titleSmall")
        grid.addWidget(self.result_state_label, 0, 1)

        grid.addWidget(QLabel("推荐档"), 0, 2)
        self.recommended_profile_label = QLabel("--")
        grid.addWidget(self.recommended_profile_label, 0, 3)

        grid.addWidget(QLabel("稳定性"), 1, 0)
        self.selection_confidence_label = QLabel("--")
        grid.addWidget(self.selection_confidence_label, 1, 1)

        grid.addWidget(QLabel("候选统计"), 1, 2)
        self.execution_stats_label = QLabel("--")
        self.execution_stats_label.setWordWrap(True)
        grid.addWidget(self.execution_stats_label, 1, 3)

        grid.addWidget(QLabel("风险提示"), 2, 0)
        self.risk_hint_label = QLabel("--")
        self.risk_hint_label.setWordWrap(True)
        self.risk_hint_label.setProperty("class", "hintText")
        grid.addWidget(self.risk_hint_label, 2, 1, 1, 3)

        grid.addWidget(QLabel("阶段比较"), 3, 0)
        self.stage_compare_label = QLabel("--")
        self.stage_compare_label.setWordWrap(True)
        grid.addWidget(self.stage_compare_label, 3, 1, 1, 3)

        overview_layout.addWidget(overview_grid)
        layout.addWidget(overview_box)

        detail_segmented_box = QGroupBox("结果详情查看")
        detail_layout = QVBoxLayout(detail_segmented_box)
        detail_layout.setContentsMargins(10, 14, 10, 10)
        detail_layout.setSpacing(10)

        detail_hint = QLabel("把自动选参与同阶段比较拆开查看，减少文本挤压并提升判读速度。")
        detail_hint.setWordWrap(True)
        detail_hint.setProperty("class", "hintText")
        detail_layout.addWidget(detail_hint)

        self.result_segmented = SegmentedWidget(self)
        self.result_segmented.addItem("auto", "自动选参结果")
        self.result_segmented.addItem("stage", "阶段比较结果")
        self.result_segmented.addItem("comparison", "人工/自动对比")
        detail_layout.addWidget(self.result_segmented)

        self.result_stack = QStackedWidget(self)
        detail_layout.addWidget(self.result_stack)

        auto_panel = QWidget()
        auto_layout = QVBoxLayout(auto_panel)
        auto_layout.setContentsMargins(0, 0, 0, 0)
        auto_layout.setSpacing(8)
        auto_panel_hint = QLabel("显示推荐参数、候选评分摘要与三档候选。")
        auto_panel_hint.setWordWrap(True)
        auto_panel_hint.setProperty("class", "hintText")
        auto_layout.addWidget(auto_panel_hint)
        self.auto_tune_summary = QTextEdit()
        self.auto_tune_summary.setReadOnly(True)
        self.auto_tune_summary.setMaximumHeight(260)
        self.auto_tune_summary.setPlaceholderText("自动选参结果将在这里显示：推荐参数、候选评分摘要与三档候选。")
        auto_layout.addWidget(self.auto_tune_summary)
        self.result_stack.addWidget(auto_panel)

        stage_panel = QWidget()
        stage_layout = QVBoxLayout(stage_panel)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(8)
        stage_panel_hint = QLabel("显示推荐方法、outer score、比较方法列表和推荐理由。")
        stage_panel_hint.setWordWrap(True)
        stage_panel_hint.setProperty("class", "hintText")
        stage_layout.addWidget(stage_panel_hint)
        self.stage_compare_summary = QTextEdit()
        self.stage_compare_summary.setReadOnly(True)
        self.stage_compare_summary.setMaximumHeight(220)
        self.stage_compare_summary.setPlaceholderText("同阶段比较结果会显示在这里：推荐方法、outer score、比较方法列表和推荐理由。")
        stage_layout.addWidget(self.stage_compare_summary)
        self.result_stack.addWidget(stage_panel)

        comparison_panel = QWidget()
        comparison_layout = QVBoxLayout(comparison_panel)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        comparison_layout.setSpacing(8)
        comparison_panel_hint = QLabel("显示人工 baseline 与自动选参的参数、评分差异和图像对比入口。")
        comparison_panel_hint.setWordWrap(True)
        comparison_panel_hint.setProperty("class", "hintText")
        comparison_layout.addWidget(comparison_panel_hint)
        self.comparison_summary = QTextEdit()
        self.comparison_summary.setReadOnly(True)
        self.comparison_summary.setMaximumHeight(260)
        self.comparison_summary.setPlaceholderText("人工/自动对比结果会显示在这里：pipeline、参数差异、评分差异和结论。")
        comparison_layout.addWidget(self.comparison_summary)

        comparison_action_row = QWidget()
        comparison_action_layout = QHBoxLayout(comparison_action_row)
        comparison_action_layout.setContentsMargins(0, 0, 0, 0)
        comparison_action_layout.setSpacing(8)
        self.btn_export_comparison = PushButton(FluentIcon.SAVE, "导出对比证据")
        self.btn_export_comparison.setEnabled(False)
        self.btn_export_comparison.setToolTip(
            "导出 summary JSON、manual/auto/side-by-side PNG、params/metrics CSV"
        )
        comparison_action_layout.addWidget(self.btn_export_comparison)
        comparison_action_layout.addStretch(1)
        comparison_layout.addWidget(comparison_action_row)
        self.result_stack.addWidget(comparison_panel)

        self.result_segmented.setCurrentItem("auto")
        self.result_stack.setCurrentIndex(0)
        self.result_segmented.currentItemChanged.connect(self._on_result_segment_changed)

        layout.addWidget(detail_segmented_box)
        layout.addStretch(1)
        return page

    def _build_truth_validation_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        status_box = QGroupBox("数据集真值状态")
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(10, 14, 10, 10)
        status_layout.setSpacing(10)

        self.truth_status_label = QLabel("当前数据未检测到 gprMax ground_truth.yaml，仍可做普通自动选参，但不能做真值验证。")
        self.truth_status_label.setWordWrap(True)
        self.truth_status_label.setProperty("class", "hintText")
        status_layout.addWidget(self.truth_status_label)

        status_grid = QWidget()
        grid = QGridLayout(status_grid)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        rows = [
            ("当前输入文件", "truth_input_file_label"),
            ("manifest", "truth_manifest_label"),
            ("ground_truth", "truth_loaded_label"),
            ("scenario_id", "truth_scenario_label"),
            ("target", "truth_target_label"),
            ("target ROI", "truth_target_roi_label"),
            ("background ROI", "truth_background_roi_label"),
            ("warning/error", "truth_warning_label"),
        ]
        for row, (name, attr) in enumerate(rows):
            grid.addWidget(QLabel(name), row, 0)
            label = QLabel("--")
            label.setWordWrap(True)
            if attr in {"truth_scenario_label", "truth_target_label"}:
                label.setProperty("class", "titleSmall")
            setattr(self, attr, label)
            grid.addWidget(label, row, 1)
        status_layout.addWidget(status_grid)
        layout.addWidget(status_box)

        bscan_box = QGroupBox("B-scan 对比入口")
        bscan_layout = QVBoxLayout(bscan_box)
        bscan_layout.setContentsMargins(10, 14, 10, 10)
        bscan_layout.setSpacing(8)
        self.truth_bscan_status_label = QLabel("尚未运行人工/自动对比。运行后可查看 Raw / Manual / AutoTune 状态，并导出 side-by-side Evidence。")
        self.truth_bscan_status_label.setWordWrap(True)
        self.truth_bscan_status_label.setProperty("class", "hintText")
        bscan_layout.addWidget(self.truth_bscan_status_label)
        self.truth_side_by_side_preview = QLabel("暂无 side-by-side 预览。导出 Evidence 后会显示缩略图。")
        self.truth_side_by_side_preview.setMinimumHeight(180)
        self.truth_side_by_side_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.truth_side_by_side_preview.setWordWrap(True)
        self.truth_side_by_side_preview.setStyleSheet(
            "border: 1px solid #d0d7de; border-radius: 6px; background: #f6f8fa; color: #57606a;"
        )
        bscan_layout.addWidget(self.truth_side_by_side_preview)
        self.btn_truth_open_side_by_side = PushButton(FluentIcon.VIEW, "打开 side-by-side 图片")
        self.btn_truth_open_side_by_side.setEnabled(False)
        self.btn_truth_open_side_by_side.clicked.connect(self._open_truth_side_by_side)
        bscan_layout.addWidget(self.btn_truth_open_side_by_side)
        layout.addWidget(bscan_box)

        metrics_box = QGroupBox("Truth metrics")
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setContentsMargins(10, 14, 10, 10)
        metrics_layout.setSpacing(8)
        metrics_grid = QWidget()
        metrics_grid_layout = QGridLayout(metrics_grid)
        metrics_grid_layout.setContentsMargins(0, 0, 0, 0)
        metrics_grid_layout.setHorizontalSpacing(16)
        metrics_grid_layout.setVerticalSpacing(8)
        self.truth_metric_labels = {}
        for row, metric_key in enumerate(
            [
                "truth_score",
                "truth_target_energy_preservation",
                "truth_target_saliency_gain",
                "truth_background_energy_reduction",
                "truth_false_positive_ratio",
                "truth_target_count",
            ]
        ):
            metrics_grid_layout.addWidget(QLabel(metric_key), row, 0)
            label = QLabel("--")
            label.setWordWrap(True)
            self.truth_metric_labels[metric_key] = label
            metrics_grid_layout.addWidget(label, row, 1)
        metrics_layout.addWidget(metrics_grid)
        layout.addWidget(metrics_box)

        params_box = QGroupBox("参数对比表")
        params_layout = QVBoxLayout(params_box)
        params_layout.setContentsMargins(10, 14, 10, 10)
        params_layout.setSpacing(8)
        self.truth_params_table = QTableWidget(0, 3)
        self.truth_params_table.setHorizontalHeaderLabels(["method", "manual 参数", "AutoTune 参数"])
        self.truth_params_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.truth_params_table.setMinimumHeight(160)
        self.truth_params_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        params_layout.addWidget(self.truth_params_table)
        layout.addWidget(params_box)

        evidence_box = QGroupBox("Evidence 操作")
        evidence_layout = QVBoxLayout(evidence_box)
        evidence_layout.setContentsMargins(10, 14, 10, 10)
        evidence_layout.setSpacing(8)
        evidence_row = QWidget()
        evidence_row_layout = QHBoxLayout(evidence_row)
        evidence_row_layout.setContentsMargins(0, 0, 0, 0)
        evidence_row_layout.setSpacing(8)
        self.btn_truth_run_comparison = PushButton(FluentIcon.VIEW, "运行人工/自动对比")
        self.btn_truth_export_evidence = PushButton(FluentIcon.SAVE, "导出 Evidence")
        self.btn_truth_open_output = PushButton(FluentIcon.FOLDER, "打开输出目录")
        self.btn_truth_open_report = PushButton(FluentIcon.DOCUMENT, "查看 Markdown 报告")
        self.btn_truth_export_evidence.setEnabled(False)
        self.btn_truth_open_output.setEnabled(False)
        self.btn_truth_open_report.setEnabled(False)
        self.btn_truth_run_comparison.clicked.connect(self.btn_compare_manual_auto.click)
        self.btn_truth_export_evidence.clicked.connect(self.btn_export_comparison.click)
        self.btn_truth_open_output.clicked.connect(self._open_truth_output_dir)
        self.btn_truth_open_report.clicked.connect(self._open_truth_report)
        evidence_row_layout.addWidget(self.btn_truth_run_comparison)
        evidence_row_layout.addWidget(self.btn_truth_export_evidence)
        evidence_row_layout.addWidget(self.btn_truth_open_output)
        evidence_row_layout.addWidget(self.btn_truth_open_report)
        evidence_row_layout.addStretch(1)
        evidence_layout.addWidget(evidence_row)
        self.truth_evidence_label = QLabel("Evidence 尚未导出。")
        self.truth_evidence_label.setWordWrap(True)
        self.truth_evidence_label.setProperty("class", "hintText")
        evidence_layout.addWidget(self.truth_evidence_label)
        layout.addWidget(evidence_box)

        layout.addStretch(1)
        self.refresh_truth_validation()
        return page

    def _on_segment_changed(self, route_key: str):
        if route_key == "truth":
            self.refresh_truth_validation()
        mapping = {"config": 0, "actions": 1, "results": 2, "truth": 3}
        self.stack.setCurrentIndex(mapping.get(route_key, 0))

    def _on_result_segment_changed(self, route_key: str):
        mapping = {"auto": 0, "stage": 1, "comparison": 2}
        self.result_stack.setCurrentIndex(mapping.get(route_key, 0))

    def refresh_truth_validation(self):
        """刷新 gprMax 真值验证面板。"""
        ground_truth = self._current_ground_truth()
        incomplete_ground_truth_info = self._incomplete_ground_truth_info()
        header_info = self._current_header_info()
        comparison = self._last_comparison_result or {}
        input_file = self._current_input_file()

        self.truth_input_file_label.setText(input_file or "--")
        manifest_path = (
            header_info.get("ground_truth_manifest_path")
            or ((ground_truth or {}).get("source_paths") or {}).get("manifest_file")
        )
        self.truth_manifest_label.setText("已找到" if manifest_path else "未找到")
        if manifest_path:
            self.truth_manifest_label.setToolTip(str(manifest_path))

        if ground_truth:
            self.truth_status_label.setText("真值验证已启用。")
            self.truth_status_label.setStyleSheet("color: #137333;")
            self.truth_loaded_label.setText("已加载")
            self.truth_scenario_label.setText(str(ground_truth.get("scenario_id") or "--"))
            self.truth_target_label.setText(self._format_truth_target(ground_truth))
            self.truth_target_roi_label.setText(self._format_truth_target_rois(ground_truth))
            self.truth_background_roi_label.setText(self._format_truth_background_rois(ground_truth))
        elif incomplete_ground_truth_info:
            self.truth_status_label.setText(
                "真值验证结果存在，但缺少完整 target/background ROI；请加载原始 gprMax manifest + ground_truth.yaml。"
            )
            self.truth_status_label.setStyleSheet("color: #9a6700;")
            self.truth_loaded_label.setText("仅有结果摘要")
            self.truth_scenario_label.setText(str(incomplete_ground_truth_info.get("scenario_id") or "--"))
            self.truth_target_label.setText("--")
            self.truth_target_roi_label.setText("--")
            self.truth_background_roi_label.setText("--")
        else:
            self.truth_status_label.setText(
                "当前数据未检测到 gprMax ground_truth.yaml，仍可做普通自动选参，但不能做真值验证。"
            )
            self.truth_status_label.setStyleSheet("color: #9a6700;")
            self.truth_loaded_label.setText("未加载")
            self.truth_scenario_label.setText("--")
            self.truth_target_label.setText("--")
            self.truth_target_roi_label.setText("--")
            self.truth_background_roi_label.setText("--")

        warning = self._truth_warning_text(header_info, ground_truth)
        self.truth_warning_label.setText(warning or "--")
        self._refresh_truth_bscan_status(comparison)
        self._refresh_truth_metrics(comparison)
        self._refresh_truth_params_table(comparison)
        self._refresh_truth_evidence_controls()

    def _current_header_info(self) -> dict:
        header = getattr(self.parent_window, "header_info", None)
        return dict(header or {}) if isinstance(header, dict) else {}

    def _current_ground_truth(self) -> dict | None:
        header = self._current_header_info()
        ground_truth = header.get("ground_truth")
        if isinstance(ground_truth, dict):
            return ground_truth
        comparison_info = (self._last_comparison_result or {}).get("ground_truth_info")
        if (
            isinstance(comparison_info, dict)
            and comparison_info.get("enabled")
            and self._has_complete_truth_rois(comparison_info)
        ):
            return comparison_info
        return None

    def _incomplete_ground_truth_info(self) -> dict | None:
        header = self._current_header_info()
        if isinstance(header.get("ground_truth"), dict):
            return None
        comparison_info = (self._last_comparison_result or {}).get("ground_truth_info")
        if (
            isinstance(comparison_info, dict)
            and comparison_info.get("enabled")
            and not self._has_complete_truth_rois(comparison_info)
        ):
            return comparison_info
        return None

    def _has_complete_truth_rois(self, ground_truth: dict) -> bool:
        return isinstance(ground_truth.get("targets"), list) and isinstance(
            ground_truth.get("background_rois"),
            list,
        )

    def _current_input_file(self) -> str:
        data_path = getattr(self.parent_window, "data_path", None)
        if data_path:
            return str(data_path)
        header = self._current_header_info()
        return str(header.get("out_path") or header.get("source_path") or "--")

    def _truth_warning_text(self, header_info: dict, ground_truth: dict | None) -> str:
        parts = []
        if header_info.get("ground_truth_load_error"):
            parts.append(str(header_info.get("ground_truth_load_error")))
        if ground_truth:
            parts.extend(str(item) for item in ground_truth.get("conversion_warnings", []) or [])
        return "；".join(parts)

    def _format_truth_target(self, ground_truth: dict) -> str:
        targets = list(ground_truth.get("targets") or [])
        if not targets:
            return "无 target（no-target 场景）"
        target = targets[0] if isinstance(targets[0], dict) else {}
        parts = [
            f"type={target.get('type', '--')}",
            f"material={target.get('material', '--')}",
            f"depth={self._format_optional_number(target.get('depth_m'), suffix=' m')}",
        ]
        center_x = target.get("center_x_m")
        center_y = target.get("center_y_m")
        radius = target.get("radius_m")
        if center_x is not None or center_y is not None:
            parts.append(
                f"center=({self._format_optional_number(center_x)}, {self._format_optional_number(center_y)}) m"
            )
        if radius is not None:
            parts.append(f"radius={self._format_optional_number(radius, suffix=' m')}")
        if len(targets) > 1:
            parts.append(f"target_count={len(targets)}")
        return " | ".join(parts)

    def _format_truth_target_rois(self, ground_truth: dict) -> str:
        rois = []
        for target in ground_truth.get("targets", []) or []:
            if isinstance(target, dict) and isinstance(target.get("roi"), dict):
                target_id = target.get("id") or target.get("target_id") or "target"
                rois.append(f"{target_id}: {self._format_roi(target['roi'])}")
        return "；".join(rois) if rois else "--"

    def _format_truth_background_rois(self, ground_truth: dict) -> str:
        rois = [
            self._format_roi(roi)
            for roi in ground_truth.get("background_rois", []) or []
            if isinstance(roi, dict)
        ]
        return "；".join(rois) if rois else "未提供，metrics 将 fallback 到 analysis_roi - target_roi"

    def _format_roi(self, roi: dict) -> str:
        t0 = int(roi.get("time_start_idx", 0))
        t1 = int(roi.get("time_end_idx", t0 + 1))
        d0 = int(roi.get("dist_start_idx", 0))
        d1 = int(roi.get("dist_end_idx", d0 + 1))
        return (
            f"MyGPR half-open time=[{t0},{t1}), trace=[{d0},{d1})；"
            f"显示闭区间 time={t0}-{max(t0, t1 - 1)}, trace={d0}-{max(d0, d1 - 1)}"
        )

    def _format_optional_number(self, value, *, suffix: str = "") -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):.4g}{suffix}"
        except (TypeError, ValueError):
            return f"{value}{suffix}"

    def _refresh_truth_bscan_status(self, comparison: dict):
        artifacts = (self._last_evidence_bundle or {}).get("artifacts") or {}
        side_by_side = artifacts.get("side_by_side_png")
        self._refresh_truth_side_by_side_preview(side_by_side)
        if not comparison:
            if side_by_side:
                self.truth_bscan_status_label.setText(
                    f"已记录 side-by-side Evidence 预览路径: {side_by_side}"
                )
            else:
                self.truth_bscan_status_label.setText(
                    "尚未运行人工/自动对比。运行后可查看 Raw / Manual / AutoTune 状态，并导出 side-by-side Evidence。"
                )
            return
        if side_by_side:
            self.truth_bscan_status_label.setText(
                f"Raw / Manual / AutoTune 对比已生成；side-by-side Evidence: {side_by_side}"
            )
        else:
            self.truth_bscan_status_label.setText(
                "已生成 Raw / Manual / AutoTune side-by-side 对比状态，可导出 Evidence 查看图像、参数表和报告。"
            )

    def _refresh_truth_side_by_side_preview(self, side_by_side_path):
        self._truth_side_by_side_path = str(side_by_side_path or "")
        self.btn_truth_open_side_by_side.setEnabled(False)
        self.truth_side_by_side_preview.clear()
        self.truth_side_by_side_preview.setPixmap(QPixmap())
        self.truth_side_by_side_preview.setToolTip("")
        if not side_by_side_path:
            self._truth_side_by_side_preview_cache_key = None
            self._truth_side_by_side_preview_cache_pixmap = QPixmap()
            self.truth_side_by_side_preview.setText("暂无 side-by-side 预览。导出 Evidence 后会显示缩略图。")
            return

        path_text = str(side_by_side_path)
        self.truth_side_by_side_preview.setToolTip(path_text)
        exists = os.path.exists(path_text)
        self.btn_truth_open_side_by_side.setEnabled(exists)
        if not exists:
            self.truth_side_by_side_preview.setText(f"side-by-side PNG 路径已记录，但文件不存在：\n{path_text}")
            return

        stat = os.stat(path_text)
        cache_key = (
            path_text,
            int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
            int(stat.st_size),
        )
        if (
            cache_key == self._truth_side_by_side_preview_cache_key
            and not self._truth_side_by_side_preview_cache_pixmap.isNull()
        ):
            self.truth_side_by_side_preview.setPixmap(
                self._truth_side_by_side_preview_cache_pixmap
            )
            return

        pixmap = QPixmap(path_text)
        if pixmap.isNull():
            self._truth_side_by_side_preview_cache_key = None
            self._truth_side_by_side_preview_cache_pixmap = QPixmap()
            self.truth_side_by_side_preview.setText(f"side-by-side PNG 路径已记录，但无法载入缩略图：\n{path_text}")
            return

        scaled = pixmap.scaled(
            720,
            260,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._truth_side_by_side_preview_cache_key = cache_key
        self._truth_side_by_side_preview_cache_pixmap = scaled
        self.truth_side_by_side_preview.setPixmap(scaled)

    def _refresh_truth_metrics(self, comparison: dict):
        manual_metrics = ((comparison.get("manual") or {}).get("metrics") or {})
        auto_metrics = ((comparison.get("automatic") or {}).get("metrics") or {})
        delta_metrics = comparison.get("metric_delta") or {}
        for metric_key, label in self.truth_metric_labels.items():
            label.setText(
                self._format_truth_metric_value(
                    metric_key,
                    manual_metrics,
                    auto_metrics,
                    delta_metrics,
                )
            )

    def _format_truth_metric_value(
        self,
        metric_key: str,
        manual_metrics: dict,
        auto_metrics: dict,
        delta_metrics: dict,
    ) -> str:
        manual_value = manual_metrics.get(metric_key)
        auto_value = auto_metrics.get(metric_key)
        delta_value = delta_metrics.get(metric_key)
        if manual_value is None and auto_value is None:
            return "--"
        parts = []
        if manual_value is not None:
            parts.append(f"manual={self._metric_text(manual_value)}")
        if auto_value is not None:
            parts.append(f"auto={self._metric_text(auto_value)}")
        if delta_value is not None:
            parts.append(f"Δ={self._metric_text(delta_value)}")
        return " | ".join(parts)

    def _metric_text(self, value) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    def _refresh_truth_params_table(self, comparison: dict):
        manual = comparison.get("manual") or {}
        automatic = comparison.get("automatic") or {}
        pipeline = list(manual.get("pipeline") or automatic.get("pipeline") or [])
        manual_params = manual.get("params_by_method") or {}
        auto_params = automatic.get("params_by_method") or {}
        if not pipeline:
            pipeline = sorted(set(manual_params) | set(auto_params))
        self.truth_params_table.setRowCount(len(pipeline))
        for row, method_key in enumerate(pipeline):
            self.truth_params_table.setItem(row, 0, QTableWidgetItem(str(method_key)))
            self.truth_params_table.setItem(
                row,
                1,
                QTableWidgetItem(json.dumps(manual_params.get(method_key, {}), ensure_ascii=False)),
            )
            self.truth_params_table.setItem(
                row,
                2,
                QTableWidgetItem(json.dumps(auto_params.get(method_key, {}), ensure_ascii=False)),
            )

    def _refresh_truth_evidence_controls(self):
        can_export = bool(self._last_comparison_result)
        self.btn_truth_export_evidence.setEnabled(can_export)
        artifacts = (self._last_evidence_bundle or {}).get("artifacts") or {}
        output_dir = (self._last_evidence_bundle or {}).get("output_dir")
        report_path = artifacts.get("report_md")
        self.btn_truth_open_output.setEnabled(bool(output_dir))
        self.btn_truth_open_report.setEnabled(bool(report_path))
        if output_dir or report_path:
            self.truth_evidence_label.setText(
                "Evidence 已导出: "
                + "；".join(
                    str(item)
                    for item in [output_dir, report_path]
                    if item
                )
            )
        else:
            self.truth_evidence_label.setText("Evidence 尚未导出。")

    def set_evidence_export_result(self, bundle: dict | None):
        """记录最近一次 Evidence 导出结果并刷新真值验证入口。"""
        self._last_evidence_bundle = dict(bundle or {}) if bundle else None
        self.refresh_truth_validation()

    def _open_truth_output_dir(self):
        output_dir = (self._last_evidence_bundle or {}).get("output_dir")
        if output_dir:
            self._open_path(str(output_dir))

    def _open_truth_report(self):
        report_path = ((self._last_evidence_bundle or {}).get("artifacts") or {}).get("report_md")
        if report_path:
            self._open_path(str(report_path))

    def _open_truth_side_by_side(self):
        side_by_side_path = getattr(self, "_truth_side_by_side_path", "")
        if side_by_side_path:
            self._open_path(side_by_side_path)

    def _open_path(self, path: str) -> bool:
        path_text = str(path or "")
        if not path_text or not os.path.exists(path_text):
            self._set_truth_open_warning(f"打开路径失败：路径不存在 {path_text or '--'}")
            return False
        try:
            if os.name == "nt":
                os.startfile(path_text)
            elif platform.system() == "Darwin":
                result = subprocess.run(["open", path_text], check=False)
                if result.returncode != 0:
                    self._set_truth_open_warning(f"打开路径失败：open 返回 {result.returncode}")
                    return False
            else:
                result = subprocess.run(["xdg-open", path_text], check=False)
                if result.returncode != 0:
                    self._set_truth_open_warning(f"打开路径失败：xdg-open 返回 {result.returncode}")
                    return False
        except Exception as exc:
            self._set_truth_open_warning(f"打开路径失败：{exc}")
            return False
        return True

    def _set_truth_open_warning(self, message: str):
        if hasattr(self, "truth_warning_label"):
            self.truth_warning_label.setText(message)
        if hasattr(self, "truth_evidence_label"):
            self.truth_evidence_label.setText(message)

    def get_auto_tune_roi_mode(self) -> str:
        """获取自动选参 ROI 来源模式。"""
        return str(self.auto_tune_roi_combo.currentData() or "prefer_crop")

    def get_auto_tune_search_mode(self) -> str:
        """获取自动选参搜索模式。"""
        return str(self.auto_tune_search_combo.currentData() or "standard")

    def set_auto_tune_summary(self, text: str):
        """设置自动选参摘要。"""
        self.auto_tune_summary.setPlainText(text)

    def _set_result_overview(
        self,
        *,
        state: str,
        recommended: str = "--",
        confidence: str = "--",
        stats: str = "--",
        risk: str = "--",
    ):
        """设置结果概况区。"""
        self.result_state_label.setText(state)
        self.recommended_profile_label.setText(recommended)
        self.selection_confidence_label.setText(confidence)
        self.execution_stats_label.setText(stats)
        self.risk_hint_label.setText(risk)

    def set_stage_compare_result(self, result: dict | None):
        """设置同阶段方法比较结果摘要。"""
        self._last_stage_compare_result = dict(result or {}) if result else None
        if not result:
            self.stage_compare_label.setText("--")
            self.stage_compare_summary.clear()
            self.btn_apply_stage_choice.setEnabled(False)
            return

        best_name = result.get(
            "best_method_name", result.get("best_method_key", "未知方法")
        )
        outer_score = float(result.get("outer_score", 0.0))
        candidate_count = len(result.get("candidates", []))
        stage = result.get("stage", "--")
        self.stage_compare_label.setText(
            f"Stage：{stage} | 推荐方法：{best_name} | outer score={outer_score:.4f} | 比较方法数={candidate_count}"
        )
        self.stage_compare_summary.setPlainText(
            self._format_stage_compare_summary(result)
        )
        self.btn_apply_stage_choice.setEnabled(True)

    def show_comparison_running(self, roi_label: str, search_mode: str):
        """显示人工/自动对比运行中状态。"""
        self._last_comparison_result = None
        self._last_evidence_bundle = None
        self.btn_export_comparison.setEnabled(False)
        self.refresh_truth_validation()
        self._set_result_overview(
            state="对比中",
            stats=f"ROI={roi_label} | 搜索={search_mode}",
            risk="正在以同一输入、同一 ROI 运行人工 baseline 与自动选参分支。",
        )
        self.result_segmented.setCurrentItem("comparison")
        self.comparison_summary.setPlainText(
            f"正在生成人工/自动对比...\nROI 来源: {roi_label}\n搜索模式: {search_mode}"
        )

    def show_comparison_result(self, summary: dict):
        """显示人工 baseline vs 自动选参对比结果摘要。"""
        self._last_comparison_result = dict(summary or {})
        self._last_evidence_bundle = None
        self.btn_export_comparison.setEnabled(True)
        verdict = str(summary.get("verdict") or "tie")
        verdict_label = {
            "auto_better": "自动选参更优",
            "manual_better": "人工 baseline 更优",
            "tie": "差异不明显",
        }.get(verdict, verdict)
        delta_score = float(
            (summary.get("metric_delta") or {}).get("comparison_score", 0.0)
        )
        manual_score = float(
            ((summary.get("manual") or {}).get("metrics") or {}).get(
                "comparison_score", 0.0
            )
        )
        auto_score = float(
            ((summary.get("automatic") or {}).get("metrics") or {}).get(
                "comparison_score", 0.0
            )
        )
        warning_count = len((summary.get("manual") or {}).get("warnings", []) or []) + len(
            (summary.get("automatic") or {}).get("warnings", []) or []
        )
        self._set_result_overview(
            state="对比完成",
            recommended=verdict_label,
            confidence=f"Δscore={delta_score:.4f}",
            stats=f"人工 {manual_score:.4f} | 自动 {auto_score:.4f}",
            risk=(
                f"存在 {warning_count} 条运行提示，导出前建议核查。"
                if warning_count
                else "已生成同尺度图像快照，可在主图对比区查看。"
            ),
        )
        self.result_segmented.setCurrentItem("comparison")
        self.comparison_summary.setPlainText(self._format_comparison_summary(summary))
        self.refresh_truth_validation()

    def show_comparison_error(self, error_msg: str):
        """显示人工/自动对比失败状态。"""
        self._last_comparison_result = None
        self._last_evidence_bundle = None
        self.btn_export_comparison.setEnabled(False)
        self.refresh_truth_validation()
        self._set_result_overview(state="对比失败", risk="当前没有可用对比结果。")
        self.result_segmented.setCurrentItem("comparison")
        self.comparison_summary.setPlainText(f"人工/自动对比失败:\n{error_msg}")

    def _format_comparison_summary(self, summary: dict) -> str:
        """格式化人工/自动对比摘要。"""
        lines = []
        verdict = str(summary.get("verdict") or "tie")
        verdict_label = {
            "auto_better": "自动选参更优",
            "manual_better": "人工 baseline 更优",
            "tie": "差异不明显",
        }.get(verdict, verdict)
        lines.append(f"结论: {verdict_label}")
        roi_info = summary.get("roi_info", {}) or {}
        lines.append(f"ROI: {roi_info.get('label', roi_info.get('source', '--'))}")
        lines.append(f"Baseline profile: {summary.get('baseline_profile_key', '--')}")

        manual = summary.get("manual", {}) or {}
        automatic = summary.get("automatic", {}) or {}
        lines.append("")
        lines.append(
            "评分: 人工 {manual:.4f} | 自动 {auto:.4f} | Δ {delta:.4f}".format(
                manual=float((manual.get("metrics") or {}).get("comparison_score", 0.0)),
                auto=float(
                    (automatic.get("metrics") or {}).get("comparison_score", 0.0)
                ),
                delta=float(
                    (summary.get("metric_delta") or {}).get("comparison_score", 0.0)
                ),
            )
        )
        pipeline = manual.get("pipeline") or automatic.get("pipeline") or []
        if pipeline:
            lines.append("Pipeline: " + " -> ".join(str(item) for item in pipeline))

        lines.append("")
        lines.append("人工 baseline 参数:")
        manual_params = manual.get("params_by_method", {}) or {}
        for method_key in pipeline:
            lines.append(
                f"- {method_key}: {json.dumps(manual_params.get(method_key, {}), ensure_ascii=False)}"
            )

        lines.append("")
        lines.append("自动选参参数:")
        auto_params = automatic.get("params_by_method", {}) or {}
        for method_key in pipeline:
            lines.append(
                f"- {method_key}: {json.dumps(auto_params.get(method_key, {}), ensure_ascii=False)}"
            )

        auto_tune_results = automatic.get("auto_tune_results", {}) or {}
        if auto_tune_results:
            lines.append("")
            lines.append("自动推荐理由:")
            for method_key in pipeline:
                item = auto_tune_results.get(method_key)
                if not item:
                    continue
                reason = item.get("best_reason") or "--"
                lines.append(f"- {method_key}: {reason}")
                domain = item.get("parameter_domain") or {}
                notes = list(domain.get("notes") or [])
                if notes:
                    lines.append("  参数域: " + "；".join(notes[:3]))

        warnings = (manual.get("warnings") or []) + (automatic.get("warnings") or [])
        if warnings:
            lines.append("")
            lines.append("运行提示:")
            for warning in warnings[:8]:
                lines.append(f"- {warning}")
        return "\n".join(lines)

    def _format_stage_compare_summary(self, result: dict) -> str:
        """格式化同阶段方法比较结果。"""
        lines = []
        lines.append(f"Stage: {result.get('stage', '--')}")
        lines.append(
            f"推荐方法: {result.get('best_method_name', result.get('best_method_key', '未知方法'))}"
        )
        lines.append(f"推荐 outer score: {float(result.get('outer_score', 0.0)):.4f}")
        reason = result.get("outer_reason")
        if reason:
            lines.append(f"推荐理由: {reason}")

        candidates = sorted(
            list(result.get("candidates", []) or []),
            key=lambda item: float(item.get("outer_score", 0.0)),
            reverse=True,
        )
        if candidates:
            lines.append("")
            lines.append("比较结果:")
            for item in candidates:
                lines.append(
                    f"- {item.get('method_name', item.get('method_key'))} | outer score={float(item.get('outer_score', 0.0)):.4f} | champion={item.get('champion_profile', '--')}"
                )
                item_reason = item.get("outer_reason")
                if item_reason:
                    lines.append(f"  说明: {item_reason}")
        return "\n".join(lines)

    def set_auto_tune_result_available(self, available: bool):
        """设置候选结果入口状态。"""
        self.btn_view_auto_tune.setEnabled(bool(available))
        if not available:
            self._last_result = None

    def set_auto_tune_method_key(self, method_key: str | None):
        """根据当前方法刷新调参与实验区基础状态。"""
        self._method_key = method_key
        if not method_key:
            self._supports_auto_tune = False
            self.auto_tune_method_label.setText("当前方法：未选择")
            self.btn_auto_tune.setEnabled(False)
            return

        method_info = PROCESSING_METHODS.get(method_key, {})
        method_name = get_method_display_name(method_key)
        self.auto_tune_method_label.setText(f"当前方法：{method_name}")
        enabled = bool(method_info.get("auto_tune_enabled"))
        self._supports_auto_tune = enabled
        self.btn_auto_tune.setEnabled(enabled)

    def reset_for_method(self, method_key: str | None, message: str | None = None):
        """切换方法后，重置当前 auto-tune 页面状态。"""
        self.set_auto_tune_result_available(False)
        self.set_auto_tune_method_key(method_key)
        self.set_stage_compare_result(None)
        self._last_comparison_result = None
        self._last_evidence_bundle = None
        self.comparison_summary.clear()
        self.btn_export_comparison.setEnabled(False)
        self.refresh_truth_validation()
        if not method_key:
            self._set_result_overview(state="未分析")
            self.set_auto_tune_summary(
                "请先在“日常处理”页选择一个方法，再进入调参与实验。"
            )
            return
        if not self._supports_auto_tune:
            self._set_result_overview(state="当前方法不支持")
            self.set_auto_tune_summary("当前方法暂未接入自动选参，暂不支持实验比较。")
            return
        self._set_result_overview(state="等待分析")
        self.btn_compare_stage.setEnabled(True)
        self.set_auto_tune_summary(
            message
            or "支持自动选参：先完成参数实验，再回到“日常处理”页把“应用方法”的默认来源切换为自动调参参数。"
        )

    def show_running(self, roi_label: str, search_mode: str):
        """显示正在分析的状态。"""
        self.set_auto_tune_result_available(False)
        self._set_result_overview(
            state="分析中",
            stats=f"ROI={roi_label} | 搜索={search_mode}",
            risk="正在生成候选评分，请等待结果稳定后再决定是否应用。",
        )
        self.set_stage_compare_result(None)
        self.set_auto_tune_summary(
            f"正在分析候选参数，请稍候...\nROI 来源: {roi_label}\n搜索模式: {search_mode}"
        )

    def show_cancelled(self):
        """显示分析取消状态。"""
        self.set_auto_tune_result_available(False)
        self._set_result_overview(state="已取消")
        self.set_auto_tune_summary("自动选参已取消。")

    def show_error(self, error_msg: str):
        """显示分析失败状态。"""
        self.set_auto_tune_result_available(False)
        self._set_result_overview(state="失败", risk="当前没有可用推荐结果。")
        self.set_auto_tune_summary(f"自动选参失败:\n{error_msg}")

    def show_result(self, result: dict):
        """显示分析完成结果。"""
        self._last_result = dict(result or {})
        self.set_auto_tune_result_available(True)
        stats = self._format_execution_stats(result)
        recommended = self._format_recommended_profile(result)
        confidence = self._format_selection_confidence(result)
        risk = self._build_risk_hint(result)
        self._set_result_overview(
            state="结果可用",
            recommended=recommended,
            confidence=confidence,
            stats=stats,
            risk=risk,
        )
        self.set_auto_tune_summary(self._format_result_summary(result))

    def _format_recommended_profile(self, result: dict) -> str:
        recommended_key = result.get("recommended_profile", "balanced")
        return (result.get("profiles", {}) or {}).get(recommended_key, {}).get(
            "label"
        ) or str(recommended_key)

    def _format_selection_confidence(self, result: dict) -> str:
        confidence = float(result.get("selection_confidence", 0.0))
        margin = float(result.get("selection_margin", 0.0))
        if confidence >= 0.75:
            level = "高"
        elif confidence >= 0.45:
            level = "中"
        else:
            level = "低"
        return f"{level} ({confidence:.2f}, margin={margin:.3f})"

    def _format_execution_stats(self, result: dict) -> str:
        stats = result.get("execution_stats", {}) or {}
        total = int(stats.get("total_trial_count", len(result.get("all_trials", []))))
        valid = int(stats.get("valid_trial_count", total))
        failed = int(
            stats.get("failed_trial_count", len(result.get("failed_trials", [])))
        )
        cache_hits = int(stats.get("cache_hit_count", 0))
        adjusted = int(stats.get("constraint_adjustment_count", 0))
        parts = [
            f"总候选 {total}",
            f"有效 {valid}",
            f"失败 {failed}",
            f"缓存命中 {cache_hits}",
        ]
        if adjusted:
            parts.append(f"参数约束 {adjusted}")
        return " | ".join(parts)

    def _build_risk_hint(self, result: dict) -> str:
        label_info = result.get("recommendation_label_info") or {}
        if label_info:
            label = str(label_info.get("recommendation_label") or "--")
            severity = str(label_info.get("severity") or "--")
            flags = ", ".join(label_info.get("risk_flags") or []) or "--"
            if label != "normal":
                return f"推荐标签: {label} ({severity}) | 风险标记: {flags}"

        recommended_key = str(result.get("recommended_profile", "balanced"))
        confidence = float(result.get("selection_confidence", 0.0))
        failed = len(result.get("failed_trials", []))
        adjusted = int(
            (result.get("execution_stats", {}) or {}).get(
                "constraint_adjustment_count", 0
            )
        )
        risk_flags = list(result.get("risk_flags") or [])
        risk_reason = str(result.get("risk_reason") or "")
        recommendation = str(result.get("selection_recommendation") or "")
        parameter_domain = result.get("parameter_domain", {}) or {}
        domain_notes = list(parameter_domain.get("notes") or [])
        if risk_flags:
            detail = "；".join(risk_flags[:4])
            if risk_reason:
                return f"自动选参存在风险标记: {detail}。{risk_reason}"
            return f"自动选参存在风险标记: {detail}。建议结合参数域与候选明细复核。"
        if failed > 0:
            return "存在失败候选，建议查看候选评分明细，确认推荐结果是否稳定。"
        if adjusted > 0:
            return "部分候选参数已按当前数据尺度限制，建议核查 requested/effective 参数差异。"
        if domain_notes:
            return "；".join(domain_notes[:2])
        if recommendation == "adopt_auto" and confidence >= 0.75:
            return "当前推荐已通过稳定性与域约束检查，可优先采用。"
        if recommended_key == "aggressive":
            return "当前推荐偏增强，建议重点核查过处理、过曝或结构损伤风险。"
        if confidence < 0.45:
            return "当前推荐稳定性偏低，建议优先对比平衡档和保守档。"
        return "当前推荐结果较稳，可先从平衡档开始验证。"

    def _format_result_summary(self, result: dict) -> str:
        """格式化自动选参摘要。"""
        lines = []
        lines.append(
            f"方法: {result.get('method_name', result.get('method_key', '未知方法'))}"
        )
        roi_info = result.get("roi_info", {}) or {}
        roi_label = roi_info.get("label") or roi_info.get("source") or "全图"
        lines.append(f"ROI 来源: {roi_label}")
        lines.append(
            f"粗筛/细化: {len(result.get('coarse_trials', []))} / {len(result.get('fine_trials', []))}"
        )
        lines.append(f"候选数量: {len(result.get('all_trials', []))}")
        recommended_label = self._format_recommended_profile(result)
        lines.append(f"推荐调试档: {recommended_label}")
        lines.append(f"稳定性: {self._format_selection_confidence(result)}")
        risk_flags = list(result.get("risk_flags") or [])
        if risk_flags:
            lines.append("风险标记: " + ", ".join(risk_flags))
        selection_recommendation = result.get("selection_recommendation")
        if selection_recommendation:
            lines.append(f"建议动作: {selection_recommendation}")
        lines.append(f"候选统计: {self._format_execution_stats(result)}")
        lines.append(
            f"总分最高: {float(result.get('best_score', 0.0)):.4f} | 参数 {json.dumps(result.get('best_params', {}), ensure_ascii=False)}"
        )
        label_info = result.get("recommendation_label_info") or {}
        if label_info:
            lines.append(
                "推荐标签: {label} | severity={severity} | manual_review_recommended={manual}".format(
                    label=label_info.get("recommendation_label", "--"),
                    severity=label_info.get("severity", "--"),
                    manual=bool(label_info.get("manual_review_recommended", False)),
                )
            )
            risk_flags = ", ".join(label_info.get("risk_flags") or [])
            if risk_flags:
                lines.append(f"标签风险依据: {risk_flags}")
            for msg in list(label_info.get("user_log_messages") or [])[:2]:
                lines.append(f"提示: {msg}")
        best_constraint_warnings = result.get("best_constraint_warnings", []) or []
        if best_constraint_warnings:
            warning = best_constraint_warnings[0]
            details = warning.get("details", {}) or {}
            parameter = details.get("parameter", "--")
            requested = details.get("requested", "--")
            effective = details.get("effective", "--")
            lines.append(
                f"参数约束: 推荐候选 {parameter} requested={requested} -> effective={effective}"
            )
        parameter_domain = result.get("parameter_domain", {}) or {}
        domain_notes = list(parameter_domain.get("notes") or [])
        if domain_notes:
            lines.append("参数域提示:")
            for note in domain_notes[:3]:
                lines.append(f"  - {note}")
        if parameter_domain.get("risk_reason"):
            lines.append(f"风险说明: {parameter_domain.get('risk_reason')}")
        profiles = result.get("profiles", {}) or {}
        for key in ["conservative", "balanced", "aggressive"]:
            profile = profiles.get(key)
            if not profile:
                continue
            params_text = json.dumps(profile.get("params", {}), ensure_ascii=False)
            lines.append(
                f"{profile.get('label', key)}: score={float(profile.get('score', 0.0)):.4f} | {params_text}"
            )
            metrics = profile.get("metrics", {}) or {}
            compact = []
            for metric_key, value in list(metrics.items())[:3]:
                if isinstance(value, (int, float)):
                    compact.append(f"{metric_key}={value:.4f}")
                else:
                    compact.append(f"{metric_key}={value}")
            if compact:
                lines.append("  指标: " + ", ".join(compact))
            reason = profile.get("reason")
            if reason:
                lines.append("  说明: " + str(reason))
        return "\n".join(lines)
