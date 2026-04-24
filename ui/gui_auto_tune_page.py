#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 调参与实验页面。"""

import json

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
        layout.addWidget(self.segmented)

        self.stack = QStackedWidget(self)
        layout.addWidget(self.stack)

        self.page_config = self._build_config_page()
        self.page_actions = self._build_actions_page()
        self.page_results = self._build_results_page()

        self.stack.addWidget(self.page_config)
        self.stack.addWidget(self.page_actions)
        self.stack.addWidget(self.page_results)

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
        self.btn_view_auto_tune = PushButton(FluentIcon.VIEW, "查看实验结果")
        self.btn_view_auto_tune.setEnabled(False)
        self.btn_view_auto_tune.setToolTip("查看候选参数、阶段比较与推荐理由")

        primary_layout.addWidget(self.btn_auto_tune)
        primary_layout.addWidget(self.btn_compare_stage)
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

        self.result_segmented.setCurrentItem("auto")
        self.result_stack.setCurrentIndex(0)
        self.result_segmented.currentItemChanged.connect(self._on_result_segment_changed)

        layout.addWidget(detail_segmented_box)
        layout.addStretch(1)
        return page

    def _on_segment_changed(self, route_key: str):
        mapping = {"config": 0, "actions": 1, "results": 2}
        self.stack.setCurrentIndex(mapping.get(route_key, 0))

    def _on_result_segment_changed(self, route_key: str):
        mapping = {"auto": 0, "stage": 1}
        self.result_stack.setCurrentIndex(mapping.get(route_key, 0))

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
        return f"总候选 {total} | 有效 {valid} | 失败 {failed} | 缓存命中 {cache_hits}"

    def _build_risk_hint(self, result: dict) -> str:
        recommended_key = str(result.get("recommended_profile", "balanced"))
        confidence = float(result.get("selection_confidence", 0.0))
        failed = len(result.get("failed_trials", []))
        if failed > 0:
            return "存在失败候选，建议查看候选评分明细，确认推荐结果是否稳定。"
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
        lines.append(f"候选统计: {self._format_execution_stats(result)}")
        lines.append(
            f"总分最高: {float(result.get('best_score', 0.0)):.4f} | 参数 {json.dumps(result.get('best_params', {}), ensure_ascii=False)}"
        )
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
