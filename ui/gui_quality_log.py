#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 质量与导出页面 - 包含处理记录、质量指标显示等功能。"""

import os

import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QTextEdit,
    QGroupBox,
    QScrollArea,
    QFrame,
    QStackedWidget,
    QDialog,
    QToolButton,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer
from qfluentwidgets import PushButton, FluentIcon, SegmentedWidget

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import colormaps, colors

from core.theme_manager import get_theme_manager


class QualityLogPage(QWidget):
    """质量与导出页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self._trace_selected_callback = None
        self._trajectory_longitude = np.array([], dtype=np.float64)
        self._trajectory_latitude = np.array([], dtype=np.float64)
        self._trajectory_trace_indices = np.array([], dtype=np.int32)
        self._selected_trace_index = None
        self._georef3d_bundle = {"raw": None, "current": None, "diff": None}
        self._georef3d_view_state = None
        self._georef3d_force_default_view = True
        self._georef3d_redraw_timer = QTimer(self)
        self._georef3d_redraw_timer.setSingleShot(True)
        self._georef3d_redraw_timer.timeout.connect(self._redraw_airborne_georeference_3d)
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

        # ========== 页面说明 ==========
        hero = QFrame()
        hero.setObjectName("QualityHeroCard")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(16, 14, 16, 14)
        hero_layout.setSpacing(12)

        hero_mark = QLabel("QC")
        hero_mark.setObjectName("QualityHeroMark")
        hero_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hero_mark.setFixedSize(48, 48)

        hero_text = QWidget()
        hero_text_layout = QVBoxLayout(hero_text)
        hero_text_layout.setContentsMargins(0, 0, 0, 0)
        hero_text_layout.setSpacing(2)
        title = QLabel("质量与报告")
        title.setProperty("class", "sectionTitle")
        hint = QLabel(
            "查看数据质量、处理记录、运行摘要，并导出项目报告或处理记录包。"
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "hintText")
        hero_text_layout.addWidget(title)
        hero_text_layout.addWidget(hint)

        hero_badge = QLabel("QC · 处理记录 · 报告")
        hero_badge.setObjectName("QualityHeroBadge")
        hero_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hero_layout.addWidget(hero_mark)
        hero_layout.addWidget(hero_text, 1)
        hero_layout.addWidget(hero_badge)
        layout.addWidget(hero)

        # 工程化 QC 状态卡：先给用户一行结论，再进入详细摘要/图表。
        self.quality_status_row = QFrame()
        self.quality_status_row.setObjectName("QualityStatusCardRow")
        status_layout = QHBoxLayout(self.quality_status_row)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)
        self.quality_status_cards = {}
        for key, title, value in [
            ("data", "数据状态", "待导入"),
            ("metadata", "元数据", "未接入"),
            ("chart", "图表", "待计算"),
            ("anomaly", "异常数量", "--"),
            ("report", "报告状态", "待生成"),
        ]:
            card = QFrame()
            card.setObjectName("QualityStatusCard")
            card_l = QVBoxLayout(card)
            card_l.setContentsMargins(10, 8, 10, 8)
            card_l.setSpacing(2)
            title_label = QLabel(title)
            title_label.setObjectName("QualityStatusTitle")
            value_label = QLabel(value)
            value_label.setObjectName("QualityStatusValue")
            value_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            card_l.addWidget(title_label)
            card_l.addWidget(value_label)
            status_layout.addWidget(card, 1)
            self.quality_status_cards[key] = value_label
        layout.addWidget(self.quality_status_row)

        # V0.8.43: quality page uses a compact dashboard switcher.
        # Only one responsibility group is shown at a time: QC, records, report, or advanced diagnostics.
        self.quality_section_segmented = SegmentedWidget(self)
        self.quality_section_segmented.addItem("qc", "数据质量")
        self.quality_section_segmented.addItem("record", "处理记录")
        self.quality_section_segmented.addItem("report", "报告导出")
        self.quality_section_segmented.addItem("advanced", "高级")
        layout.addWidget(self.quality_section_segmented)

        self.quality_section_stack = QStackedWidget(self)
        layout.addWidget(self.quality_section_stack, 1)

        qc_dashboard = QWidget()
        qc_dashboard_layout = QVBoxLayout(qc_dashboard)
        qc_dashboard_layout.setContentsMargins(0, 0, 0, 0)
        qc_dashboard_layout.setSpacing(10)

        record_dashboard = QWidget()
        record_dashboard_layout = QVBoxLayout(record_dashboard)
        record_dashboard_layout.setContentsMargins(0, 0, 0, 0)
        record_dashboard_layout.setSpacing(10)

        report_dashboard = QWidget()
        report_dashboard_layout = QVBoxLayout(report_dashboard)
        report_dashboard_layout.setContentsMargins(0, 0, 0, 0)
        report_dashboard_layout.setSpacing(10)

        advanced_dashboard = QWidget()
        advanced_dashboard_layout = QVBoxLayout(advanced_dashboard)
        advanced_dashboard_layout.setContentsMargins(0, 0, 0, 0)
        advanced_dashboard_layout.setSpacing(10)

        # ========== 质量概览区 ==========
        summary_box = QGroupBox("质量概览")
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.setContentsMargins(10, 14, 10, 10)
        summary_layout.setSpacing(10)

        summary_hint = QLabel(
            "查看测线摘要、元数据完整性、质量检查结论和异常明细。"
        )
        summary_hint.setWordWrap(True)
        summary_hint.setProperty("class", "hintText")
        summary_layout.addWidget(summary_hint)

        self.summary_segmented = SegmentedWidget(self)
        self.summary_segmented.addItem("line", "测线结果")
        self.summary_segmented.addItem("meta", "航空元数据")
        self.summary_segmented.addItem("qc", "航空质控")
        self.summary_segmented.addItem("anomaly", "异常明细")
        summary_layout.addWidget(self.summary_segmented)

        self.summary_stack = QStackedWidget(self)
        summary_layout.addWidget(self.summary_stack)

        self.line_summary = QTextEdit()
        self.line_summary.setReadOnly(True)
        self.line_summary.setPlaceholderText("暂无测线结果信息")
        self.line_summary.setMinimumHeight(150)
        self.line_summary.setMaximumHeight(220)
        self.summary_stack.addWidget(self._wrap_text_panel(
            "测线结果卡片",
            "查看当前测线的核心结果、长度、阶段摘要与结论。",
            self.line_summary,
        ))

        self.metadata_summary = QTextEdit()
        self.metadata_summary.setReadOnly(True)
        self.metadata_summary.setPlaceholderText("暂无航空元数据")
        self.metadata_summary.setMinimumHeight(150)
        self.metadata_summary.setMaximumHeight(220)
        self.summary_stack.addWidget(self._wrap_text_panel(
            "航空元数据摘要",
            "用于确认经纬度、轨迹、时间、高程和飞行高度等元数据是否完整；完整空间图请进入“空间”。",
            self.metadata_summary,
        ))

        self.airborne_qc_summary = QTextEdit()
        self.airborne_qc_summary.setReadOnly(True)
        self.airborne_qc_summary.setPlaceholderText("暂无航空质控摘要")
        self.airborne_qc_summary.setMinimumHeight(150)
        self.airborne_qc_summary.setMaximumHeight(220)
        self.summary_stack.addWidget(self._wrap_text_panel(
            "航空质控摘要",
            "优先查看异常级别、稳定性和是否存在明显风险。",
            self.airborne_qc_summary,
        ))

        self.airborne_anomaly_details = QTextEdit()
        self.airborne_anomaly_details.setReadOnly(True)
        self.airborne_anomaly_details.setPlaceholderText("暂无航空异常明细")
        self.airborne_anomaly_details.setMinimumHeight(180)
        self.airborne_anomaly_details.setMaximumHeight(260)
        self.summary_stack.addWidget(self._wrap_text_panel(
            "航空异常明细",
            "用于追查具体异常点、异常段和可疑区间。",
            self.airborne_anomaly_details,
        ))

        self.summary_segmented.setCurrentItem("line")
        self.summary_stack.setCurrentIndex(0)
        qc_dashboard_layout.addWidget(summary_box)

        # ========== 质量图表 ==========
        visual_box = QGroupBox("质量图表")
        visual_layout = QVBoxLayout(visual_box)
        visual_layout.setContentsMargins(10, 14, 10, 10)
        visual_layout.setSpacing(10)

        visual_hint = QLabel(
            "用于判断道间距、飞行高度和异常点分布是否稳定；完整航迹位置请在“空间 > 测线轨迹”中查看。"
        )
        visual_hint.setWordWrap(True)
        visual_hint.setProperty("class", "hintText")
        visual_layout.addWidget(visual_hint)

        self.visual_stack = QStackedWidget(self)
        visual_layout.addWidget(self.visual_stack)

        qc_panel = QWidget()
        qc_panel_layout = QVBoxLayout(qc_panel)
        qc_panel_layout.setContentsMargins(0, 0, 0, 0)
        qc_panel_layout.setSpacing(8)
        qc_hint = QLabel("质量页只判断是否存在异常；航迹位置、地形剖面和空间联动放在空间成果页。")
        qc_hint.setWordWrap(True)
        qc_hint.setProperty("class", "hintText")
        qc_panel_layout.addWidget(qc_hint)
        self.qc_fig = Figure(figsize=(6, 3.2), dpi=100)
        self.qc_canvas = FigureCanvas(self.qc_fig)
        self.qc_ax_spacing = self.qc_fig.add_subplot(211)
        self.qc_ax_height = self.qc_fig.add_subplot(212)
        qc_panel_layout.addWidget(self.qc_canvas)
        self.visual_stack.addWidget(qc_panel)

        trajectory_summary_panel = QFrame()
        trajectory_summary_panel.setObjectName("ReportChecklistCard")
        trajectory_summary_layout = QHBoxLayout(trajectory_summary_panel)
        trajectory_summary_layout.setContentsMargins(10, 8, 10, 8)
        trajectory_summary_layout.setSpacing(8)
        self.trajectory_status_label = QLabel("航迹：未加载｜查看详情请进入 空间 > 测线轨迹")
        self.trajectory_status_label.setProperty("class", "hintText")
        self.trajectory_status_label.setWordWrap(True)
        self.btn_open_trajectory_space = PushButton(FluentIcon.VIEW, "查看测线轨迹")
        self.btn_open_trajectory_space.setToolTip("切换到空间成果页的测线轨迹视图")
        trajectory_summary_layout.addWidget(self.trajectory_status_label, 1)
        trajectory_summary_layout.addWidget(self.btn_open_trajectory_space)
        visual_layout.addWidget(trajectory_summary_panel)

        self.visual_stack.setCurrentIndex(0)
        qc_dashboard_layout.addWidget(visual_box)

        # ========== 报告与导出 ==========
        action_box = QGroupBox("报告与导出")
        action_layout = QVBoxLayout(action_box)
        action_layout.setContentsMargins(10, 14, 10, 10)
        action_layout.setSpacing(8)

        action_hint = QLabel(
            "面向工程使用导出项目报告、质量检查表和处理记录包；科研审计细节保留在高级导出中。"
        )
        action_hint.setWordWrap(True)
        action_hint.setProperty("class", "hintText")
        action_layout.addWidget(action_hint)

        # 报告导出前检查：保留底层 Evidence 能力，但前端不暴露科研术语。
        checklist_panel = QFrame()
        checklist_panel.setObjectName("ReportChecklistCard")
        checklist_layout = QGridLayout(checklist_panel)
        checklist_layout.setContentsMargins(0, 0, 0, 0)
        checklist_layout.setHorizontalSpacing(8)
        checklist_layout.setVerticalSpacing(6)
        checklist_items = [
            ("数据文件", "待检查"),
            ("处理步骤", "待检查"),
            ("参数记录", "待检查"),
            ("图像输出", "可生成"),
            ("质量指标", "待计算"),
            ("检查提示", "待检查"),
            ("结论说明", "待填写"),
        ]
        self.evidence_checklist_labels = []
        for i, (name, status) in enumerate(checklist_items):
            label = QLabel(f"{name}：{status}")
            label.setObjectName("ReportCheckChip")
            label.setProperty("tone", "warning" if "复核" in status or "待" in status else "neutral")
            checklist_layout.addWidget(label, i // 2, i % 2)
            self.evidence_checklist_labels.append(label)
        action_layout.addWidget(checklist_panel)

        self.btn_generate_report = PushButton(FluentIcon.DOCUMENT, "生成项目报告")
        self.btn_generate_report.setToolTip(
            "导出当前图像、运行摘要和日志到 Markdown + HTML 报告包"
        )
        self.btn_export_quality_snapshot = PushButton(
            FluentIcon.DOWNLOAD, "导出质量检查表"
        )
        self.btn_export_quality_snapshot.setToolTip(
            "导出当前质量指标、阈值与航空质控摘要"
        )
        self.btn_export_replay_evidence = PushButton(
            FluentIcon.SAVE, "导出处理记录包"
        )
        self.btn_export_replay_evidence.setToolTip(
            "导出当前处理历史、参数记录和高级审计文件"
        )
        self.btn_open_log_dir = PushButton(FluentIcon.FOLDER, "打开输出目录")
        self.btn_open_log_dir.setToolTip("打开日志和输出目录")

        action_row_top = QWidget()
        action_row_top_layout = QHBoxLayout(action_row_top)
        action_row_top_layout.setContentsMargins(0, 0, 0, 0)
        action_row_top_layout.setSpacing(8)
        action_row_top_layout.addWidget(self.btn_generate_report)
        action_row_top_layout.addWidget(self.btn_export_quality_snapshot)
        action_row_top_layout.addStretch(1)
        action_layout.addWidget(action_row_top)

        action_row_bottom = QWidget()
        action_row_bottom_layout = QHBoxLayout(action_row_bottom)
        action_row_bottom_layout.setContentsMargins(0, 0, 0, 0)
        action_row_bottom_layout.setSpacing(8)
        action_row_bottom_layout.addWidget(self.btn_export_replay_evidence)
        action_row_bottom_layout.addWidget(self.btn_open_log_dir)
        action_row_bottom_layout.addStretch(1)
        action_layout.addWidget(action_row_bottom)
        report_dashboard_layout.addWidget(action_box)

        # ========== 运行记录区 ==========
        record_box = QGroupBox("处理记录")
        record_layout = QVBoxLayout(record_box)
        record_layout.setContentsMargins(10, 14, 10, 10)
        record_layout.setSpacing(8)

        record_hint = QLabel("保留处理历史、导出前核对记录和可追溯操作信息。")
        record_hint.setWordWrap(True)
        record_hint.setProperty("class", "hintText")
        record_layout.addWidget(record_hint)

        record_tools_row = QWidget()
        record_tools_layout = QHBoxLayout(record_tools_row)
        record_tools_layout.setContentsMargins(0, 0, 0, 0)
        record_tools_layout.setSpacing(8)
        self.btn_record_clear = PushButton(FluentIcon.DELETE, "清空记录")
        self.btn_record_clear.setToolTip("清空当前页面中的处理记录")
        self.btn_record_export = PushButton(FluentIcon.SAVE_AS, "导出记录")
        self.btn_record_export.setToolTip("导出处理记录到文本文件")
        record_tools_layout.addWidget(self.btn_record_clear)
        record_tools_layout.addWidget(self.btn_record_export)
        record_tools_layout.addStretch(1)
        record_layout.addWidget(record_tools_row)

        self.record = QTextEdit()
        self.record.setReadOnly(True)
        self.record.setPlaceholderText("暂无记录")
        self.record.setMinimumHeight(180)
        self.record.setMaximumHeight(280)
        self.record.setToolTip("处理操作历史，包含时间戳和方法信息")
        record_layout.addWidget(self.record)
        record_dashboard_layout.addWidget(record_box)

        # ========== 高级诊断 ==========
        diagnostic_box = QGroupBox("高级诊断")
        diagnostic_layout = QVBoxLayout(diagnostic_box)
        diagnostic_layout.setContentsMargins(10, 14, 10, 10)
        diagnostic_layout.setSpacing(8)
        diagnostic_hint = QLabel(
            "面向开发调试和科研审计，普通工程处理通常无需使用。"
        )
        diagnostic_hint.setWordWrap(True)
        diagnostic_hint.setProperty("class", "hintText")
        diagnostic_layout.addWidget(diagnostic_hint)
        diagnostic_row = QWidget()
        diagnostic_row_layout = QHBoxLayout(diagnostic_row)
        diagnostic_row_layout.setContentsMargins(0, 0, 0, 0)
        diagnostic_row_layout.setSpacing(8)
        self.btn_copy_diagnostics = PushButton(FluentIcon.COPY, "复制诊断信息")
        self.btn_copy_diagnostics.setToolTip("复制当前环境、数据和日志摘要")
        diagnostic_row_layout.addWidget(self.btn_copy_diagnostics)
        diagnostic_row_layout.addStretch(1)
        diagnostic_layout.addWidget(diagnostic_row)
        advanced_dashboard_layout.addWidget(diagnostic_box)

        qc_dashboard_layout.addStretch(1)
        record_dashboard_layout.addStretch(1)
        report_dashboard_layout.addStretch(1)
        advanced_dashboard_layout.addStretch(1)

        self.quality_section_stack.addWidget(qc_dashboard)
        self.quality_section_stack.addWidget(record_dashboard)
        self.quality_section_stack.addWidget(report_dashboard)
        self.quality_section_stack.addWidget(advanced_dashboard)
        self.quality_section_segmented.setCurrentItem("qc")
        self.quality_section_stack.setCurrentIndex(0)

        self.summary_segmented.currentItemChanged.connect(self._on_summary_segment_changed)
        self.quality_section_segmented.currentItemChanged.connect(self._on_quality_section_changed)

        self.set_airborne_qc_visualization(None)
        self.set_airborne_trajectory_visualization(None)
        self.btn_open_trajectory_space.clicked.connect(self._open_space_track_view)
    def resizeEvent(self, event):
        """页面尺寸变化时更新三维预览图内控制位置。"""
        super().resizeEvent(event)
        self._position_georef3d_overlay_controls()

    def _create_georef3d_overlay_button(
        self,
        text: str,
        tooltip: str,
        *,
        checked: bool = False,
        checkable: bool = True,
    ) -> QToolButton:
        """创建画布内轻量控制按钮。"""
        parent = getattr(self, "_georef3d_overlay_parent", self.georef3d_canvas)
        button = QToolButton(parent)
        button.setText(text)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        button.setCheckable(checkable)
        if checkable:
            button.setChecked(checked)
        button.setMinimumHeight(24)
        button.setObjectName("QualityToolButton")
        button.adjustSize()
        button.show()
        button.raise_()
        return button

    def _position_georef3d_overlay_controls(self) -> None:
        """把三维预览控制按钮放在画布内角落，避免占用主布局空间。"""
        if not hasattr(self, "georef3d_canvas"):
            return
        canvas = self.georef3d_canvas
        parent = getattr(self, "_georef3d_overlay_parent", canvas)
        origin = canvas.mapTo(parent, canvas.rect().topLeft())
        margin = 8
        gap = 5
        y = origin.y() + margin
        x = origin.x() + margin
        for button in [
            getattr(self, "btn_georef3d_raw", None),
            getattr(self, "btn_georef3d_current", None),
        ]:
            if button is None:
                continue
            button.adjustSize()
            button.move(x, y)
            button.raise_()
            x += button.width() + gap

        right_buttons = [
            getattr(self, "btn_georef3d_bscan", None),
            getattr(self, "btn_georef3d_diff", None),
            getattr(self, "btn_georef3d_reset_view", None),
            getattr(self, "btn_georef3d_expand", None),
        ]
        active = [button for button in right_buttons if button is not None]
        for button in active:
            button.adjustSize()
        total_width = sum(button.width() for button in active) + gap * max(len(active) - 1, 0)
        x = max(origin.x() + margin, origin.x() + canvas.width() - total_width - margin)
        for button in active:
            button.move(x, y)
            button.raise_()
            x += button.width() + gap

    def _wrap_text_panel(self, title: str, hint: str, text_edit: QTextEdit) -> QWidget:
        """包装摘要文本面板。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setProperty("class", "titleSmall")
        layout.addWidget(title_label)

        hint_label = QLabel(hint)
        hint_label.setWordWrap(True)
        hint_label.setProperty("class", "hintText")
        layout.addWidget(hint_label)

        layout.addWidget(text_edit)
        return panel

    def _on_summary_segment_changed(self, route_key: str):
        mapping = {
            "line": 0,
            "meta": 1,
            "qc": 2,
            "anomaly": 3,
        }
        self.summary_stack.setCurrentIndex(mapping.get(route_key, 0))

    def _on_quality_section_changed(self, route_key: str):
        mapping = {
            "qc": 0,
            "record": 1,
            "report": 2,
            "advanced": 3,
        }
        self.quality_section_stack.setCurrentIndex(mapping.get(route_key, 0))

    def _open_space_track_view(self):
        """切换到空间成果页的测线轨迹视图。"""
        parent = getattr(self, "parent_window", None)
        if parent is None:
            return
        terrain_page = getattr(parent, "page_terrain3d", None)
        try:
            if hasattr(parent, "_switch_side_workspace"):
                parent._switch_side_workspace(4)
            if terrain_page is not None and hasattr(terrain_page, "focus_track_view"):
                terrain_page.focus_track_view()
        except Exception:
            return

    def _axis_label_for_user(self, payload: dict | None, axis: str) -> str:
        """Return user-facing 3D axis labels; never expose internal metadata field names."""
        payload = payload or {}
        if axis == "x":
            return "沿测线距离 / 局部 X (m)" if payload.get("has_longitude_latitude") else "沿测线距离 (m)"
        if axis == "y":
            return "横向偏移 / 局部 Y (m)" if payload.get("has_longitude_latitude") else "横向偏移 (m)"
        return "高程 / 等效深度 (m)"

    def _format_georef_quality_flags(self, flags) -> str:
        """Map internal provenance flags to short engineering notes."""
        if not flags:
            return ""
        mapping = {
            "derived_local_xy_from_lon_lat": "局部坐标由经纬度换算",
            "derived_distance_from_trace_index": "距离轴由道号估算",
            "missing_ground_elevation": "缺少地表高程",
            "missing_height_agl": "缺少飞行高度",
        }
        out = []
        for flag in list(flags)[:4]:
            out.append(mapping.get(str(flag), "空间元数据提示"))
        return " | ".join(out)

    def _style_3d_axes(self, ax):
        """统一三维图表主题。"""
        palette = self._get_plot_palette()
        ax.set_facecolor(palette["ax_face"])
        ax.tick_params(colors=palette["text"])
        ax.xaxis.label.set_color(palette["text"])
        ax.yaxis.label.set_color(palette["text"])
        ax.zaxis.label.set_color(palette["text"])
        ax.title.set_color(palette["text"])
        try:
            ax.xaxis.pane.set_facecolor(palette["ax_face"])
            ax.yaxis.pane.set_facecolor(palette["ax_face"])
            ax.zaxis.pane.set_facecolor(palette["ax_face"])
            ax.xaxis.pane.set_edgecolor(palette["spine"])
            ax.yaxis.pane.set_edgecolor(palette["spine"])
            ax.zaxis.pane.set_edgecolor(palette["spine"])
        except Exception:
            pass
        try:
            ax.xaxis._axinfo["grid"]["color"] = palette["grid"]
            ax.yaxis._axinfo["grid"]["color"] = palette["grid"]
            ax.zaxis._axinfo["grid"]["color"] = palette["grid"]
        except Exception:
            pass
        return palette

    def _get_plot_palette(self) -> dict:
        """获取当前主题下的图表配色。"""
        theme = get_theme_manager().get_current_theme()
        if theme == "dark":
            return {
                "theme": "dark",
                "fig_face": "#1f2125",
                "ax_face": "#23252a",
                "text": "#e8e8e8",
                "hint": "#b7bcc6",
                "spine": "#5a606b",
                "grid": "#4b515c",
                "legend_face": "#2a2d33",
                "legend_edge": "#434852",
                "line_primary": "#7ab8ff",
                "line_success": "#6dd7a3",
                "line_warning": "#f4bf4f",
                "line_error": "#ff8f8f",
                "line_emphasis": "#c084fc",
            }
        return {
            "theme": "light",
            "fig_face": "#ffffff",
            "ax_face": "#f8f8f8",
            "text": "#333333",
            "hint": "#666666",
            "spine": "#bbbbbb",
            "grid": "#d9dee7",
            "legend_face": "#ffffff",
            "legend_edge": "#d1d5db",
            "line_primary": "#3b82f6",
            "line_success": "#10b981",
            "line_warning": "#f59e0b",
            "line_error": "#ef4444",
            "line_emphasis": "#a855f7",
        }

    def _apply_axes_theme(self, ax, palette: dict, *, grid: bool = True):
        """应用坐标轴主题。"""
        ax.set_facecolor(palette["ax_face"])
        ax.tick_params(colors=palette["text"])
        ax.xaxis.label.set_color(palette["text"])
        ax.yaxis.label.set_color(palette["text"])
        ax.title.set_color(palette["text"])
        for spine in ax.spines.values():
            spine.set_color(palette["spine"])
        ax.grid(grid, linestyle=":", alpha=0.35, color=palette["grid"])

    def _style_figure(self, fig, axes: list):
        """统一图表主题。"""
        palette = self._get_plot_palette()
        fig.patch.set_facecolor(palette["fig_face"])
        for ax in axes:
            self._apply_axes_theme(ax, palette)
            legend = ax.get_legend()
            if legend is not None:
                frame = legend.get_frame()
                frame.set_facecolor(palette["legend_face"])
                frame.set_edgecolor(palette["legend_edge"])
                frame.set_alpha(0.9)
                for text in legend.get_texts():
                    text.set_color(palette["text"])
        return palette

    def _draw_canvas_safely(self, canvas) -> None:
        """低内存环境下避免占位图渲染拖垮窗口创建。"""
        try:
            canvas.draw_idle()
        except MemoryError:
            return

    def _finalize_figure(self, fig, canvas) -> None:
        """统一收尾 2D 图表布局和刷新。"""
        try:
            fig.tight_layout()
        except MemoryError:
            return
        self._draw_canvas_safely(canvas)

    def release_plot_resources(self) -> None:
        """窗口关闭时显式释放 Matplotlib 图表资源。"""
        try:
            self._georef3d_redraw_timer.stop()
        except Exception:
            pass
        for fig_name, canvas_name in [
            ("qc_fig", "qc_canvas"),
            ("georef3d_fig", "georef3d_canvas"),
        ]:
            fig = getattr(self, fig_name, None)
            canvas = getattr(self, canvas_name, None)
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

    def append_record(self, text: str):
        """追加记录"""
        self.record.append(text)

    def clear_record(self):
        """清空记录"""
        self.record.clear()

    def get_record_text(self) -> str:
        """获取记录文本"""
        return self.record.toPlainText()

    def _set_quality_status(self, key: str, value: str) -> None:
        label = getattr(self, "quality_status_cards", {}).get(key)
        if label is not None:
            label.setText(value or "--")

    def set_metadata_summary(self, text: str):
        """设置航空元数据摘要文本。"""
        self.metadata_summary.setPlainText(text or "")
        self._set_quality_status("metadata", "可用" if text else "未接入")

    def set_line_summary(self, text: str):
        """设置测线结果卡片文本。"""
        self.line_summary.setPlainText(text or "")
        self._set_quality_status("data", "已加载" if text else "待导入")

    def set_airborne_qc_summary(self, text: str):
        """设置航空质控摘要文本。"""
        self.airborne_qc_summary.setPlainText(text or "")
        self._set_quality_status("report", "可导出" if text else "待生成")

    def set_trace_selected_callback(self, callback):
        """设置航迹点击后的回调。"""
        self._trace_selected_callback = callback

    def set_airborne_qc_visualization(self, payload: dict | None):
        """绘制航空异常可视化。"""
        self.qc_fig.clear()
        self.qc_ax_spacing = self.qc_fig.add_subplot(211)
        self.qc_ax_height = self.qc_fig.add_subplot(212)
        palette = self._get_plot_palette()

        if not payload:
            self._set_quality_status("chart", "待计算")
            for ax, title in [
                (self.qc_ax_spacing, "道间距"),
                (self.qc_ax_height, "飞行高度"),
            ]:
                ax.set_title(title)
                ax.text(
                    0.5,
                    0.5,
                    "暂无航空数据",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=palette["hint"],
                )
                ax.set_xticks([])
                ax.set_yticks([])
            self._style_figure(self.qc_fig, [self.qc_ax_spacing, self.qc_ax_height])
            self._finalize_figure(self.qc_fig, self.qc_canvas)
            return

        self._set_quality_status("chart", "已计算" if payload.get("flight") else "待计算")

        spacing_x = payload.get("spacing_x", [])
        spacing = payload.get("spacing", [])
        spacing_mask = payload.get("spacing_mask", [])
        distance = payload.get("distance", [])
        flight = payload.get("flight", [])
        flight_mask = payload.get("flight_mask", [])

        ax = self.qc_ax_spacing
        ax.set_title("道间距稳定性")
        if len(spacing_x) and len(spacing):
            ax.plot(spacing_x, spacing, color=palette["line_primary"], linewidth=1.2)
            if len(spacing_mask):
                mask = [bool(x) for x in spacing_mask]
                x_bad = [x for x, m in zip(spacing_x, mask) if m]
                y_bad = [y for y, m in zip(spacing, mask) if m]
                if x_bad:
                    ax.scatter(x_bad, y_bad, color=palette["line_error"], s=20, zorder=3)
        ax.set_ylabel("m")

        ax = self.qc_ax_height
        ax.set_title("飞行高度稳定性")
        if len(distance) and len(flight):
            ax.plot(distance, flight, color=palette["line_success"], linewidth=1.2)
            if len(flight_mask):
                mask = [bool(x) for x in flight_mask]
                x_bad = [x for x, m in zip(distance, mask) if m]
                y_bad = [y for y, m in zip(flight, mask) if m]
                if x_bad:
                    ax.scatter(x_bad, y_bad, color=palette["line_warning"], s=20, zorder=3)
        ax.set_xlabel("距离 (m)")
        ax.set_ylabel("m")

        self._style_figure(self.qc_fig, [self.qc_ax_spacing, self.qc_ax_height])
        self._finalize_figure(self.qc_fig, self.qc_canvas)

    def set_airborne_trajectory_visualization(self, payload: dict | None):
        """更新航迹 QC 摘要；完整航迹图由“空间 > 测线轨迹”负责显示。"""
        if not payload:
            self._trajectory_longitude = np.array([], dtype=np.float64)
            self._trajectory_latitude = np.array([], dtype=np.float64)
            self._trajectory_trace_indices = np.array([], dtype=np.int32)
            self._selected_trace_index = None
            if hasattr(self, "trajectory_status_label"):
                self.trajectory_status_label.setText("航迹：未加载｜查看详情请进入 空间 > 测线轨迹")
            return

        longitude = np.asarray(payload.get("longitude", []), dtype=np.float64)
        latitude = np.asarray(payload.get("latitude", []), dtype=np.float64)
        trace_indices = np.asarray(payload.get("trace_indices", []), dtype=np.int32)
        anomaly_mask = np.asarray(payload.get("anomaly_mask", []), dtype=bool)
        flight_height = np.asarray(payload.get("flight_height_m", []), dtype=np.float64)
        selected_trace_index = payload.get("selected_trace_index")

        if (
            longitude.size == 0
            or latitude.size == 0
            or longitude.size != latitude.size
            or trace_indices.size != longitude.size
        ):
            self.set_airborne_trajectory_visualization(None)
            return

        self._trajectory_longitude = longitude
        self._trajectory_latitude = latitude
        self._trajectory_trace_indices = trace_indices
        self._selected_trace_index = (
            int(selected_trace_index) if selected_trace_index is not None else None
        )

        anomaly_count = int(np.count_nonzero(anomaly_mask)) if anomaly_mask.size == longitude.size else 0
        height_text = "飞行高度可用" if flight_height.size == longitude.size else "飞行高度缺失"
        if hasattr(self, "trajectory_status_label"):
            self.trajectory_status_label.setText(
                f"航迹：{longitude.size} 个位置点｜异常点 {anomaly_count}｜{height_text}｜详情在 空间 > 测线轨迹"
            )

    def set_airborne_georeference_3d_visualization(self, payload: dict | None):
        """设置三维地理参考预览数据并延迟刷新。"""
        had_payload = bool(self._available_georef3d_entries()) if hasattr(self, "_available_georef3d_entries") else False
        if payload and any(key in payload for key in ("raw", "current", "diff")):
            self._georef3d_bundle = {
                "raw": payload.get("raw"),
                "current": payload.get("current"),
                "diff": payload.get("diff"),
            }
        else:
            self._georef3d_bundle = {"raw": None, "current": payload, "diff": None}
        has_payload = bool(self._available_georef3d_entries()) if hasattr(self, "_available_georef3d_entries") else bool(payload)
        # When real spatial data arrives after the empty state, discard the old
        # 0--1 placeholder view. Otherwise Matplotlib restores the empty axes and
        # the 3D scene appears blank even though the payload is valid.
        if has_payload and not had_payload:
            self._georef3d_view_state = None
            self._georef3d_force_default_view = True
        else:
            self._georef3d_force_default_view = self._georef3d_view_state is None
        self._schedule_georef3d_redraw()

    def _schedule_georef3d_redraw(self, *_args):
        """Debounce 3D redraws caused by layer changes."""
        if hasattr(self, "_georef3d_redraw_timer"):
            self._georef3d_redraw_timer.start(120)

    def _on_georef3d_interaction_start(self, _event):
        """记录交互前视角，不在鼠标按下时触发重绘。"""
        self._georef3d_view_state = self._capture_georef3d_view_state()

    def _on_georef3d_interaction_end(self, _event):
        """记录交互后视角，不自动恢复默认视角。"""
        self._georef3d_view_state = self._capture_georef3d_view_state()

    def _capture_georef3d_view_state(self) -> dict | None:
        """Capture current 3D view so redraws do not reset user navigation."""
        ax = getattr(self, "georef3d_ax", None)
        if ax is None or not hasattr(ax, "get_xlim3d"):
            return None
        try:
            return {
                "elev": float(ax.elev),
                "azim": float(ax.azim),
                "xlim": tuple(float(v) for v in ax.get_xlim3d()),
                "ylim": tuple(float(v) for v in ax.get_ylim3d()),
                "zlim": tuple(float(v) for v in ax.get_zlim3d()),
            }
        except Exception:
            return None

    def _is_placeholder_georef3d_view(self, state: dict | None) -> bool:
        """Return True for the 0--1 Matplotlib placeholder view from the empty state."""
        if not state:
            return False
        try:
            ranges = []
            for key in ("xlim", "ylim", "zlim"):
                lo, hi = state.get(key, (None, None))
                ranges.append(abs(float(hi) - float(lo)))
            return all(0.8 <= r <= 1.4 for r in ranges)
        except Exception:
            return False

    def _restore_georef3d_view_state(self, ax, state: dict | None) -> None:
        """Restore a captured 3D view, or apply the default view once."""
        if state:
            try:
                ax.view_init(elev=float(state["elev"]), azim=float(state["azim"]))
                ax.set_xlim3d(*state["xlim"])
                ax.set_ylim3d(*state["ylim"])
                ax.set_zlim3d(*state["zlim"])
                return
            except Exception:
                pass
        if self._georef3d_force_default_view:
            ax.view_init(elev=24, azim=-58)
            self._georef3d_force_default_view = False

    def _reset_georef3d_view(self) -> None:
        """用户显式请求时才恢复默认三维视角。"""
        self._georef3d_view_state = None
        self._georef3d_force_default_view = True
        self._schedule_georef3d_redraw()

    def _select_georef3d_payload(self, entry):
        """Select the cached preview payload, with compatibility for older LOD maps."""
        if not entry:
            return None
        if isinstance(entry, dict) and "payloads_by_lod" in entry:
            payloads = entry.get("payloads_by_lod") or {}
            return (
                payloads.get("auto")
                or payloads.get("medium")
                or payloads.get("high")
                or payloads.get("low")
                or next(iter(payloads.values()), None)
            )
        return entry

    def _plot_georef3d_entry(
        self,
        ax,
        payload: dict,
        *,
        label: str,
        kind: str,
        palette: dict,
        show_bscan: bool | None = None,
    ):
        preview = payload.get("preview") or {}
        curtain_x = np.asarray(preview.get("curtain_x_m", []), dtype=np.float64)
        curtain_y = np.asarray(preview.get("curtain_y_m", []), dtype=np.float64)
        curtain_z = np.asarray(preview.get("curtain_z_m", []), dtype=np.float64)
        amplitude = np.asarray(preview.get("amplitude", []), dtype=np.float64)
        if curtain_x.size == 0 or curtain_x.shape != curtain_y.shape or curtain_x.shape != curtain_z.shape:
            return

        if show_bscan is None:
            show_bscan = self.btn_georef3d_bscan.isChecked()
        if show_bscan and amplitude.size and amplitude.shape == curtain_x.shape:
            finite_amp = amplitude[np.isfinite(amplitude)]
            amp_min = float(preview.get("amplitude_min", float(np.min(finite_amp)) if finite_amp.size else 0.0))
            amp_max = float(preview.get("amplitude_max", float(np.max(finite_amp)) if finite_amp.size else 1.0))
            if not np.isfinite(amp_min) or not np.isfinite(amp_max) or amp_min == amp_max:
                amp_min, amp_max = 0.0, 1.0
            cmap_name = "seismic" if kind == "diff" else "gray"
            cmap = colormaps.get_cmap(cmap_name)
            if kind == "diff":
                vmax = max(abs(amp_min), abs(amp_max), 1.0e-12)
                norm = colors.Normalize(vmin=-vmax, vmax=vmax)
            else:
                norm = colors.Normalize(vmin=amp_min, vmax=amp_max)
            facecolors = cmap(norm(amplitude))
            alpha = 0.55 if kind == "raw" else 0.88
            ax.plot_surface(
                curtain_x,
                curtain_y,
                curtain_z,
                facecolors=facecolors,
                alpha=alpha,
                linewidth=0,
                antialiased=False,
                shade=False,
            )

        x_m = np.asarray(payload.get("local_x_m", []), dtype=np.float64)
        y_m = np.asarray(payload.get("local_y_m", []), dtype=np.float64)
        airborne_z_m = np.asarray(payload.get("airborne_z_m", []), dtype=np.float64)
        if x_m.size and y_m.size and airborne_z_m.size:
            color = {
                "raw": palette["line_warning"],
                "current": palette["line_success"],
                "diff": palette["line_emphasis"],
            }.get(kind, palette["line_primary"])
            linestyle = "--" if kind == "raw" else "-"
            ax.plot(
                x_m,
                y_m,
                airborne_z_m,
                color=color,
                linewidth=1.4,
                linestyle=linestyle,
                label=label,
            )
            if kind == "current":
                ax.scatter([x_m[0]], [y_m[0]], [airborne_z_m[0]], color=palette["line_success"], s=30)
                ax.scatter([x_m[-1]], [y_m[-1]], [airborne_z_m[-1]], color=palette["line_error"], s=30)

    def _visible_georef3d_entries(self) -> list[tuple[str, str, dict]]:
        """Return visible 3D layer payloads without rebuilding preview data."""
        bundle = self._georef3d_bundle or {}
        entries: list[tuple[str, str, dict]] = []
        if self.btn_georef3d_raw.isChecked():
            payload = self._select_georef3d_payload(bundle.get("raw"))
            if payload:
                entries.append(("原始3D", "raw", payload))
        if self.btn_georef3d_current.isChecked():
            payload = self._select_georef3d_payload(bundle.get("current"))
            if payload:
                entries.append(("当前3D", "current", payload))
        if self.btn_georef3d_diff.isChecked():
            payload = self._select_georef3d_payload(bundle.get("diff"))
            if payload:
                entries.append(("差异", "diff", payload))
        return entries

    def _available_georef3d_entries(self) -> dict[str, tuple[str, dict]]:
        """Return all available expanded-preview payloads independent of panel visibility."""
        bundle = self._georef3d_bundle or {}
        available: dict[str, tuple[str, dict]] = {}
        for key, label in (("raw", "原始"), ("current", "当前"), ("diff", "差异")):
            payload = self._select_georef3d_payload(bundle.get(key))
            if payload:
                available[key] = (label, payload)
        return available

    def _open_georef3d_dialog(self) -> None:
        """打开独立大窗口预览；统一使用 Matplotlib 兼容视图。"""
        entries = self._available_georef3d_entries()
        if not entries:
            return
        self._open_georef3d_matplotlib_dialog(
            entries,
            fallback_message="MyGPR 当前不内置 PyVista 预览，已使用 Matplotlib 兼容视图。",
        )

    def _open_georef3d_pyvista_dialog(self, entries: dict[str, tuple[str, dict]]) -> None:
        """Use PyVista/VTK for the expanded 3D motion preview."""
        if str(os.environ.get("QT_QPA_PLATFORM", "")).lower() == "offscreen":
            raise RuntimeError("当前 Qt 平台为 offscreen，跳过 PyVista/VTK 交互预览")

        from pyvistaqt import QtInteractor

        dialog = QDialog(self)
        dialog.setWindowTitle("UAV-GPR 三维运动补偿预览")
        dialog.resize(1180, 820)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title = QLabel("UAV-GPR 三维运动补偿预览")
        title.setProperty("class", "sectionTitle")
        title.setMaximumHeight(28)
        title_hint = QLabel("鼠标旋转/缩放/平移；坐标单位 m；PyVista 彩色坐标轴=X/Y/Z。")
        title_hint.setProperty("class", "hintText")
        title_row.addWidget(title)
        title_row.addWidget(title_hint)
        title_row.addStretch(1)
        layer_buttons = {
            "raw": self._create_dialog_layer_button(
                "👁 原始", checked=self.btn_georef3d_raw.isChecked(), parent=dialog
            ),
            "current": self._create_dialog_layer_button(
                "👁 当前", checked=self.btn_georef3d_current.isChecked(), parent=dialog
            ),
            "diff": self._create_dialog_layer_button(
                "👁 差异", checked=self.btn_georef3d_diff.isChecked(), parent=dialog
            ),
        }
        bscan_button = self._create_dialog_layer_button(
            "👁 B-scan", checked=self.btn_georef3d_bscan.isChecked(), parent=dialog
        )
        export_button = QToolButton(dialog)
        export_button.setText("导出当前视图 PNG")
        export_button.setToolTip("导出当前 PyVista 大窗口视图为 PNG")
        export_button.setAutoRaise(True)
        layer_label = QLabel("图层")
        layer_label.setProperty("class", "hintText")
        title_row.addWidget(layer_label)
        for key, button in layer_buttons.items():
            button.setEnabled(key in entries)
            title_row.addWidget(button)
        title_row.addWidget(bscan_button)
        title_row.addWidget(export_button)
        layout.addLayout(title_row)
        legend = QLabel("原始=橙色虚线，当前=青绿色实线，差异=紫色；绿色点为起点，红色点为终点。")
        legend.setProperty("class", "hintText")
        legend.setMaximumHeight(22)
        layout.addWidget(legend)
        plotter = QtInteractor(dialog)
        plotter.set_background("white")
        plotter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plotter.setMinimumHeight(640)
        layout.addWidget(plotter.interactor, 1)

        def redraw_view(reset_camera: bool = False):
            plotter.clear()
            selected = [
                (key, label, payload)
                for key, (label, payload) in entries.items()
                if layer_buttons[key].isChecked()
            ]
            all_points = self._collect_georef3d_points(
                [(key, payload) for key, _label, payload in selected],
                include_bscan=bscan_button.isChecked(),
            )
            self._add_georef3d_reference_frame(plotter, all_points)
            for key, label, payload in selected:
                self._add_georef3d_payload_to_pyvista(
                    plotter,
                    payload,
                    label=label,
                    kind=key,
                    show_bscan=bscan_button.isChecked(),
                )
            plotter.add_legend(bcolor="white", border=True, face=None)
            plotter.reset_camera_clipping_range()
            if reset_camera and all_points:
                points = np.vstack(all_points)
                finite = points[np.isfinite(points).all(axis=1)]
                if finite.size:
                    center = np.nanmean(finite, axis=0)
                    span = np.maximum(np.nanmax(finite, axis=0) - np.nanmin(finite, axis=0), 1.0)
                    distance = float(max(np.nanmax(span) * 2.2, 8.0))
                    plotter.camera_position = [
                        (
                            float(center[0] + distance),
                            float(center[1] - distance),
                            float(center[2] + distance * 0.55),
                        ),
                        (float(center[0]), float(center[1]), float(center[2])),
                        (0.0, 0.0, 1.0),
                    ]

        for button in list(layer_buttons.values()) + [bscan_button]:
            button.toggled.connect(lambda _checked: redraw_view(False))
        export_button.clicked.connect(lambda: self._export_georef3d_view_png(plotter, dialog))
        redraw_view(True)

        dialog.exec()

    def _collect_georef3d_points(
        self,
        entries: list[tuple[str, dict]],
        *,
        include_bscan: bool,
    ) -> list[np.ndarray]:
        """Collect trajectory and curtain points for camera and reference-frame bounds."""
        all_points: list[np.ndarray] = []
        for _kind, payload in entries:
            x_m = np.asarray(payload.get("local_x_m", []), dtype=np.float64)
            y_m = np.asarray(payload.get("local_y_m", []), dtype=np.float64)
            z_m = np.asarray(payload.get("airborne_z_m", []), dtype=np.float64)
            if x_m.size and y_m.size and z_m.size:
                all_points.append(np.column_stack([x_m, y_m, z_m]))
            if not include_bscan:
                continue
            preview = payload.get("preview") or {}
            curtain_x = np.asarray(preview.get("curtain_x_m", []), dtype=np.float64)
            curtain_y = np.asarray(preview.get("curtain_y_m", []), dtype=np.float64)
            curtain_z = np.asarray(preview.get("curtain_z_m", []), dtype=np.float64)
            if curtain_x.size and curtain_x.shape == curtain_y.shape == curtain_z.shape:
                all_points.append(
                    np.column_stack(
                        [
                            curtain_x.reshape(-1),
                            curtain_y.reshape(-1),
                            curtain_z.reshape(-1),
                        ]
                    )
                )
        return all_points

    def _add_georef3d_reference_frame(self, plotter, all_points: list[np.ndarray]) -> None:
        """Add PyVista axes and an XY reference grid around the expanded preview."""
        import pyvista as pv

        frame = self._georef3d_reference_geometry(all_points)
        origin = np.asarray(frame["origin"], dtype=np.float64)
        size = np.asarray(frame["size"], dtype=np.float64)

        grid = pv.Plane(
            center=(float(origin[0] + size[0] / 2.0), float(origin[1] + size[1] / 2.0), float(origin[2])),
            direction=(0.0, 0.0, 1.0),
            i_size=float(size[0]),
            j_size=float(size[1]),
            i_resolution=10,
            j_resolution=10,
        )
        plotter.add_mesh(grid, color="#bfc7d5", opacity=0.16, show_edges=True, edge_color="#c8d0dd", label="XY参考面")
        plotter.show_bounds(
            grid="front",
            location="outer",
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
            color="#334155",
            font_size=10,
        )
        plotter.add_axes(line_width=3, labels_off=False)
        self._add_georef3d_colored_axis(plotter, "x", origin, size)
        self._add_georef3d_colored_axis(plotter, "y", origin, size)
        self._add_georef3d_colored_axis(plotter, "z", origin, size)

    def _georef3d_reference_geometry(self, all_points: list[np.ndarray]) -> dict[str, np.ndarray | float]:
        """Compute a stable right-handed 3D reference frame around preview data."""
        if all_points:
            points = np.vstack(all_points)
            finite_mask = np.isfinite(points).all(axis=1)
            points = points[finite_mask]
        else:
            points = np.empty((0, 3), dtype=np.float64)
        if points.size:
            min_xyz = np.nanmin(points, axis=0)
            max_xyz = np.nanmax(points, axis=0)
        else:
            min_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            max_xyz = np.array([10.0, 10.0, 5.0], dtype=np.float64)
        span = np.maximum(max_xyz - min_xyz, np.array([1.0, 1.0, 1.0]))
        pad = np.maximum(span * 0.08, np.array([0.5, 0.5, 0.2]))
        origin = min_xyz - pad
        size = span + 2.0 * pad
        spacing = float(max(np.nanmax(size[:2]) / 10.0, 0.5))
        return {"origin": origin, "size": size, "grid_spacing": spacing}

    def _add_georef3d_colored_axis(self, plotter, axis: str, origin: np.ndarray, size: np.ndarray) -> None:
        """Add one colored X/Y/Z axis line with a small text label."""
        import pyvista as pv

        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        colors_by_axis = {
            "x": "#d7191c",
            "y": "#1a9641",
            "z": "#2c7bb6",
        }
        labels_by_axis = {"x": "X", "y": "Y", "z": "Z"}
        end = np.asarray(origin, dtype=np.float64).copy()
        end[axis_index] += float(size[axis_index])
        plotter.add_mesh(
            pv.Line(origin, end),
            color=colors_by_axis[axis],
            line_width=4,
            label=f"{labels_by_axis[axis]}轴",
        )
        try:
            label_pos = end.copy()
            label_pos[axis_index] += max(float(size[axis_index]) * 0.035, 0.12)
            plotter.add_point_labels(
                np.asarray([label_pos], dtype=np.float64),
                [labels_by_axis[axis]],
                text_color=colors_by_axis[axis],
                font_size=18,
                point_size=0,
                shape=None,
                always_visible=True,
            )
        except Exception:
            return

    def _create_dialog_layer_button(
        self,
        text: str,
        *,
        checked: bool,
        parent: QWidget | None = None,
    ) -> QToolButton:
        button = QToolButton(parent or self)
        button.setText(text)
        button.setCheckable(True)
        button.setChecked(checked)
        button.setAutoRaise(True)
        return button

    def _export_georef3d_view_png(self, view, dialog: QDialog) -> None:
        path, _ = QFileDialog.getSaveFileName(
            dialog,
            "导出当前三维视图",
            "uav_gpr_3d_view.png",
            "PNG Images (*.png)",
        )
        if not path:
            return
        try:
            if hasattr(view, "screenshot"):
                view.screenshot(path)
            else:
                pixmap = view.grab()
                if pixmap.isNull() or not pixmap.save(path, "PNG"):
                    raise RuntimeError("当前视图截图为空或保存失败")
        except Exception as exc:
            QMessageBox.warning(dialog, "导出失败", f"无法导出当前视图 PNG：\n{exc}")

    def _add_georef3d_payload_to_pyvista(
        self,
        plotter,
        payload: dict,
        *,
        label: str,
        kind: str,
        show_bscan: bool = True,
    ) -> None:
        """Add one trajectory/curtain payload to a PyVista plotter."""
        import pyvista as pv

        x_m = np.asarray(payload.get("local_x_m", []), dtype=np.float64)
        y_m = np.asarray(payload.get("local_y_m", []), dtype=np.float64)
        z_m = np.asarray(payload.get("airborne_z_m", []), dtype=np.float64)
        if x_m.size and y_m.size and z_m.size:
            pos = np.column_stack([x_m, y_m, z_m]).astype(np.float64)
            color = {
                "raw": "#f97316",
                "current": "#0891b2",
                "diff": "#7c3aed",
            }.get(kind, "#111827")
            if pos.shape[0] >= 2:
                plotter.add_mesh(
                    pv.lines_from_points(pos, close=False),
                    color=color,
                    line_width=4 if kind == "current" else 3,
                    label=label,
                )
            endpoints = np.vstack([pos[0], pos[-1]])
            plotter.add_points(
                endpoints[:1],
                color="#16a34a",
                point_size=12,
                render_points_as_spheres=True,
                label=f"{label}起点",
            )
            plotter.add_points(
                endpoints[1:],
                color="#dc2626",
                point_size=12,
                render_points_as_spheres=True,
                label=f"{label}终点",
            )

        if not show_bscan:
            return
        preview = payload.get("preview") or {}
        curtain_x = np.asarray(preview.get("curtain_x_m", []), dtype=np.float64)
        curtain_y = np.asarray(preview.get("curtain_y_m", []), dtype=np.float64)
        curtain_z = np.asarray(preview.get("curtain_z_m", []), dtype=np.float64)
        amplitude = np.asarray(preview.get("amplitude", []), dtype=np.float64)
        if (
            curtain_x.size == 0
            or curtain_x.shape != curtain_y.shape
            or curtain_x.shape != curtain_z.shape
            or amplitude.shape != curtain_x.shape
        ):
            return
        grid = self._build_georef3d_pyvista_grid(
            curtain_x,
            curtain_y,
            curtain_z,
            amplitude,
        )
        finite_amp = amplitude[np.isfinite(amplitude)]
        if finite_amp.size:
            amp_min = float(np.min(finite_amp))
            amp_max = float(np.max(finite_amp))
        else:
            amp_min, amp_max = 0.0, 1.0
        if not np.isfinite(amp_min) or not np.isfinite(amp_max) or amp_min == amp_max:
            amp_min, amp_max = 0.0, 1.0
        cmap_name = "seismic" if kind == "diff" else "gray"
        clim = (-max(abs(amp_min), abs(amp_max), 1.0e-12), max(abs(amp_min), abs(amp_max), 1.0e-12)) if kind == "diff" else (amp_min, amp_max)
        plotter.add_mesh(
            grid,
            scalars="amplitude",
            cmap=cmap_name,
            clim=clim,
            opacity=0.50 if kind == "raw" else 0.82,
            show_edges=False,
            show_scalar_bar=False,
            label=f"{label} B-scan",
        )

    def _build_georef3d_pyvista_grid(
        self,
        curtain_x: np.ndarray,
        curtain_y: np.ndarray,
        curtain_z: np.ndarray,
        amplitude: np.ndarray,
    ):
        """Convert curtain arrays to a PyVista StructuredGrid."""
        import pyvista as pv

        rows, cols = curtain_x.shape
        grid = pv.StructuredGrid(
            np.asarray(curtain_x, dtype=np.float64),
            np.asarray(curtain_y, dtype=np.float64),
            np.asarray(curtain_z, dtype=np.float64),
        )
        grid.dimensions = (cols, rows, 1)
        grid.point_data["amplitude"] = np.asarray(amplitude, dtype=np.float64).reshape(-1, order="F")
        return grid

    def _open_georef3d_matplotlib_dialog(
        self,
        entries: dict[str, tuple[str, dict]],
        *,
        fallback_message: str | None = None,
    ) -> None:
        """Fallback expanded preview that keeps static export independent from OpenGL."""
        dialog = QDialog(self)
        dialog.setWindowTitle("UAV-GPR 三维运动补偿预览")
        dialog.resize(1180, 820)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        if fallback_message:
            warning_label = QLabel(fallback_message)
            warning_label.setWordWrap(True)
            warning_label.setProperty("class", "hintText")
            layout.addWidget(warning_label)
        controls = QHBoxLayout()
        controls.setSpacing(8)
        title = QLabel("UAV-GPR 三维运动补偿预览")
        title.setProperty("class", "sectionTitle")
        title.setMaximumHeight(28)
        controls.addWidget(title)
        controls.addStretch(1)
        layer_buttons = {
            "raw": self._create_dialog_layer_button(
                "👁 原始", checked=self.btn_georef3d_raw.isChecked(), parent=dialog
            ),
            "current": self._create_dialog_layer_button(
                "👁 当前", checked=self.btn_georef3d_current.isChecked(), parent=dialog
            ),
            "diff": self._create_dialog_layer_button(
                "👁 差异", checked=self.btn_georef3d_diff.isChecked(), parent=dialog
            ),
        }
        bscan_button = self._create_dialog_layer_button(
            "👁 B-scan", checked=self.btn_georef3d_bscan.isChecked(), parent=dialog
        )
        export_button = QToolButton(dialog)
        export_button.setText("导出当前视图 PNG")
        export_button.setToolTip("导出当前 Matplotlib 大窗口视图为 PNG")
        export_button.setAutoRaise(True)
        layer_label = QLabel("图层")
        layer_label.setProperty("class", "hintText")
        controls.addWidget(layer_label)
        for key, button in layer_buttons.items():
            button.setEnabled(key in entries)
            controls.addWidget(button)
        controls.addWidget(bscan_button)
        controls.addWidget(export_button)
        layout.addLayout(controls)
        legend = QLabel("原始=橙色虚线，当前=青绿色实线，差异=紫色；坐标单位 m。")
        legend.setProperty("class", "hintText")
        legend.setMaximumHeight(22)
        layout.addWidget(legend)
        fig = Figure(figsize=(10.8, 7.4), dpi=100)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas.setMinimumHeight(640)
        layout.addWidget(canvas, 1)

        def redraw_dialog():
            fig.clear()
            ax = fig.add_subplot(111, projection="3d")
            palette = self._get_plot_palette()
            fig.patch.set_facecolor(palette["fig_face"])
            selected = [
                (label, key, payload)
                for key, (label, payload) in entries.items()
                if layer_buttons[key].isChecked()
            ]
            if not selected:
                ax.text2D(0.5, 0.5, "未选择三维图层", transform=ax.transAxes, ha="center")
                canvas.draw_idle()
                return
            for label, kind, payload in selected:
                self._plot_georef3d_entry(
                    ax,
                    payload,
                    label=label,
                    kind=kind,
                    palette=palette,
                    show_bscan=bscan_button.isChecked(),
                )
            payload = selected[-1][2]
            ax.set_title("UAV-GPR 三维运动补偿预览")
            ax.set_xlabel(self._axis_label_for_user(payload, "x"))
            ax.set_ylabel(self._axis_label_for_user(payload, "y"))
            ax.set_zlabel(self._axis_label_for_user(payload, "z"))
            ax.view_init(elev=24, azim=-58)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper left")
            self._style_3d_axes(ax)
            canvas.draw_idle()

        for button in list(layer_buttons.values()) + [bscan_button]:
            button.toggled.connect(redraw_dialog)
        export_button.clicked.connect(
            lambda: self._export_georef3d_matplotlib_png(fig, dialog)
        )
        redraw_dialog()
        dialog.exec()

    def _export_georef3d_matplotlib_png(self, fig: Figure, dialog: QDialog) -> None:
        path, _ = QFileDialog.getSaveFileName(
            dialog,
            "导出当前三维视图",
            "uav_gpr_3d_view.png",
            "PNG Images (*.png)",
        )
        if not path:
            return
        try:
            fig.savefig(path, dpi=160, bbox_inches="tight")
        except Exception as exc:
            QMessageBox.warning(dialog, "导出失败", f"无法导出当前视图 PNG：\n{exc}")

    def _redraw_airborne_georeference_3d(self):
        """绘制三维地理参考预览。"""
        previous_view = self._capture_georef3d_view_state()
        self.georef3d_fig.clear()
        self.georef3d_ax = self.georef3d_fig.add_subplot(111, projection="3d")
        ax = self.georef3d_ax
        palette = self._get_plot_palette()
        self.georef3d_fig.patch.set_facecolor(palette["fig_face"])
        entries = self._visible_georef3d_entries()
        if entries and self._is_placeholder_georef3d_view(previous_view):
            previous_view = None

        if not entries:
            ax.set_title("三维地理参考预览")
            ax.text2D(
                0.5,
                0.5,
                "空间数据未接入\n导入轨迹/高程/飞行高度后启用三维预览",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=palette["hint"],
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            self._restore_georef3d_view_state(ax, previous_view or self._georef3d_view_state)
            self._style_3d_axes(ax)
            self.georef3d_fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.92)
            self._georef3d_view_state = self._capture_georef3d_view_state()
            self._position_georef3d_overlay_controls()
            self._draw_canvas_safely(self.georef3d_canvas)
            return

        for label, kind, payload in entries:
            self._plot_georef3d_entry(ax, payload, label=label, kind=kind, palette=palette)

        payload = entries[-1][2]
        preview = payload.get("preview") or {}
        preview_trace_indices = np.asarray(preview.get("trace_indices", []), dtype=np.int32)
        preview_sample_indices = np.asarray(preview.get("sample_indices", []), dtype=np.int32)

        has_georef = bool(payload.get("has_longitude_latitude") and (payload.get("has_ground_elevation") or payload.get("has_height_agl")))
        ax.set_title("三维地理参考预览" if has_georef else "三维剖面预览（未地理参考）")
        ax.set_xlabel(self._axis_label_for_user(payload, "x"))
        ax.set_ylabel(self._axis_label_for_user(payload, "y"))
        ax.set_zlabel(self._axis_label_for_user(payload, "z"))

        bounds = self._compute_georef3d_bounds(entries, include_bscan=self.btn_georef3d_bscan.isChecked())
        if bounds is not None:
            (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

        self._restore_georef3d_view_state(ax, previous_view or self._georef3d_view_state)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left")
        if preview_sample_indices.size and preview_trace_indices.size:
            ax.text2D(
                0.02,
                0.02,
                f"预览网格: {preview_trace_indices.size}x{preview_sample_indices.size}",
                transform=ax.transAxes,
                color=palette["hint"],
            )
        # Do not print raw provenance / quality flags on the engineering view.
        # Detailed spatial metadata status is shown in the right-side property panel
        # and export metadata, keeping the 3D scene visually clean.

        self._style_3d_axes(ax)
        self.georef3d_fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.92)
        self._georef3d_view_state = self._capture_georef3d_view_state()
        self._position_georef3d_overlay_controls()
        self._draw_canvas_safely(self.georef3d_canvas)

    def _compute_georef3d_bounds(
        self,
        entries: list[tuple[str, str, dict]],
        *,
        include_bscan: bool,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
        """Compute robust 3D bounds from all visible payload objects."""
        point_entries = [(kind, payload) for _label, kind, payload in entries]
        points = self._collect_georef3d_points(point_entries, include_bscan=include_bscan)
        if not points:
            return None
        stacked = np.vstack(points)
        finite = stacked[np.isfinite(stacked).all(axis=1)]
        if finite.size == 0:
            return None
        mins = np.min(finite, axis=0)
        maxs = np.max(finite, axis=0)
        span = np.maximum(maxs - mins, 1.0e-6)
        padding = np.maximum(span * 0.08, 1.0e-6)
        return (
            (float(mins[0] - padding[0]), float(maxs[0] + padding[0])),
            (float(mins[1] - padding[1]), float(maxs[1] + padding[1])),
            (float(mins[2] - padding[2]), float(maxs[2] + padding[2])),
        )

    def _on_trajectory_click(self, event):
        """根据点击位置选中最近的航迹点。"""
        if (
            event.inaxes not in (self.trajectory_ax, self.trajectory_height_ax)
            or event.xdata is None
            or event.ydata is None
            or self._trajectory_trace_indices.size == 0
        ):
            return

        if event.inaxes == self.trajectory_height_ax:
            # 在高度剖面点击时，按 x 轴（道号索引）找最近点
            x_axis = np.arange(self._trajectory_trace_indices.size)
            delta = (x_axis - float(event.xdata)) ** 2
        else:
            delta = (self._trajectory_longitude - float(event.xdata)) ** 2 + (
                self._trajectory_latitude - float(event.ydata)
            ) ** 2
        nearest_idx = int(np.argmin(delta))
        callback = self._trace_selected_callback
        if callback is not None:
            callback(int(self._trajectory_trace_indices[nearest_idx]))

    def set_airborne_anomaly_details(self, text: str):
        """设置航空异常明细文本。"""
        self.airborne_anomaly_details.setPlainText(text or "")
        clean = (text or "").strip()
        if not clean:
            value = "--"
        elif "暂无" in clean or "未发现" in clean or "无明显" in clean:
            value = "0"
        else:
            value = str(max(1, len([line for line in clean.splitlines() if line.strip()])))
        self._set_quality_status("anomaly", value)
