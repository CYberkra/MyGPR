#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 质量与导出页面 - 包含处理记录、质量指标显示等功能。"""

import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QGroupBox,
    QScrollArea,
    QFrame,
    QStackedWidget,
)
from qfluentwidgets import PushButton, FluentIcon, SegmentedWidget

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

        title = QLabel("质量与导出")
        title.setProperty("class", "sectionTitle")
        layout.addWidget(title)

        hint = QLabel(
            "集中查看运行记录、质量摘要与航空质控，并从这里完成报告、快照和诊断导出。"
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "hintText")
        layout.addWidget(hint)

        flow_box = QGroupBox("查看顺序")
        flow_box.setProperty("class", "calloutBox")
        flow_layout = QVBoxLayout(flow_box)
        flow_layout.setContentsMargins(10, 14, 10, 10)
        flow_layout.setSpacing(8)

        flow_hint = QLabel("推荐顺序：先看质量摘要，再看图表定位问题，最后导出报告、快照或诊断信息。")
        flow_hint.setWordWrap(True)
        flow_hint.setProperty("class", "hintText")
        flow_layout.addWidget(flow_hint)

        flow_row = QWidget()
        flow_row_layout = QHBoxLayout(flow_row)
        flow_row_layout.setContentsMargins(0, 0, 0, 0)
        flow_row_layout.setSpacing(8)
        for text in ["① 质量摘要", "② 图表查看", "③ 导出与记录"]:
            chip = QLabel(text)
            chip.setProperty("class", "statusChip")
            flow_row_layout.addWidget(chip)
        flow_row_layout.addStretch(1)
        flow_layout.addWidget(flow_row)
        layout.addWidget(flow_box)

        # ========== 顶部动作区 ==========
        action_box = QGroupBox("导出与诊断")
        action_layout = QVBoxLayout(action_box)
        action_layout.setContentsMargins(10, 14, 10, 10)
        action_layout.setSpacing(8)

        action_hint = QLabel(
            "先看质量摘要，再决定是否生成报告、导出快照或复制诊断信息。"
        )
        action_hint.setWordWrap(True)
        action_hint.setProperty("class", "hintText")
        action_layout.addWidget(action_hint)

        self.btn_generate_report = PushButton(FluentIcon.DOCUMENT, "生成报告")
        self.btn_generate_report.setToolTip(
            "导出当前图像、运行摘要和日志到 Markdown 报告"
        )
        self.btn_export_quality_snapshot = PushButton(
            FluentIcon.DOWNLOAD, "导出质量快照"
        )
        self.btn_export_quality_snapshot.setToolTip(
            "导出当前质量指标、阈值与航空质控摘要"
        )
        self.btn_open_log_dir = PushButton(FluentIcon.FOLDER, "打开日志目录")
        self.btn_open_log_dir.setToolTip("打开日志和输出目录")
        self.btn_copy_diagnostics = PushButton(FluentIcon.COPY, "复制诊断信息")
        self.btn_copy_diagnostics.setToolTip("复制当前环境、数据和日志摘要")

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
        action_row_bottom_layout.addWidget(self.btn_open_log_dir)
        action_row_bottom_layout.addWidget(self.btn_copy_diagnostics)
        action_row_bottom_layout.addStretch(1)
        action_layout.addWidget(action_row_bottom)
        layout.addWidget(action_box)

        # ========== 质量摘要区 ==========
        summary_box = QGroupBox("质量摘要")
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.setContentsMargins(10, 14, 10, 10)
        summary_layout.setSpacing(10)

        summary_hint = QLabel(
            "把测线结果、航空元数据、质控结论和异常明细分开看，避免同屏堆叠。"
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
            "用于快速确认经纬度、轨迹、时间和高度等元数据是否完整。",
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
        layout.addWidget(summary_box)

        # ========== 图表区 ==========
        visual_box = QGroupBox("图表查看")
        visual_layout = QVBoxLayout(visual_box)
        visual_layout.setContentsMargins(10, 14, 10, 10)
        visual_layout.setSpacing(10)

        visual_hint = QLabel(
            "先看质量图表判断稳定性，再切到航迹图定位异常位置和选中 trace。"
        )
        visual_hint.setWordWrap(True)
        visual_hint.setProperty("class", "hintText")
        visual_layout.addWidget(visual_hint)

        self.visual_segmented = SegmentedWidget(self)
        self.visual_segmented.addItem("qc_chart", "质量图表")
        self.visual_segmented.addItem("trajectory", "航迹图")
        visual_layout.addWidget(self.visual_segmented)

        self.visual_stack = QStackedWidget(self)
        visual_layout.addWidget(self.visual_stack)

        qc_panel = QWidget()
        qc_panel_layout = QVBoxLayout(qc_panel)
        qc_panel_layout.setContentsMargins(0, 0, 0, 0)
        qc_panel_layout.setSpacing(8)
        qc_hint = QLabel("用于查看道间距稳定性、飞行高度稳定性和异常点分布。")
        qc_hint.setWordWrap(True)
        qc_hint.setProperty("class", "hintText")
        qc_panel_layout.addWidget(qc_hint)
        self.qc_fig = Figure(figsize=(6, 3.2), dpi=100)
        self.qc_canvas = FigureCanvas(self.qc_fig)
        self.qc_ax_spacing = self.qc_fig.add_subplot(211)
        self.qc_ax_height = self.qc_fig.add_subplot(212)
        qc_panel_layout.addWidget(self.qc_canvas)
        self.visual_stack.addWidget(qc_panel)

        trajectory_panel = QWidget()
        trajectory_panel_layout = QVBoxLayout(trajectory_panel)
        trajectory_panel_layout.setContentsMargins(0, 0, 0, 0)
        trajectory_panel_layout.setSpacing(8)
        trajectory_hint = QLabel("点击航迹图中的点可联动主图，定位对应 trace。")
        trajectory_hint.setWordWrap(True)
        trajectory_hint.setProperty("class", "hintText")
        trajectory_panel_layout.addWidget(trajectory_hint)
        self.trajectory_fig = Figure(figsize=(6, 4.8), dpi=100)
        self.trajectory_canvas = FigureCanvas(self.trajectory_fig)
        self.trajectory_ax = self.trajectory_fig.add_subplot(2, 1, 1)
        self.trajectory_height_ax = self.trajectory_fig.add_subplot(2, 1, 2)
        self.trajectory_canvas.mpl_connect(
            "button_press_event", self._on_trajectory_click
        )
        trajectory_panel_layout.addWidget(self.trajectory_canvas)
        self.visual_stack.addWidget(trajectory_panel)

        self.visual_segmented.setCurrentItem("qc_chart")
        self.visual_stack.setCurrentIndex(0)
        layout.addWidget(visual_box)

        # ========== 运行记录区 ==========
        record_box = QGroupBox("运行记录")
        record_layout = QVBoxLayout(record_box)
        record_layout.setContentsMargins(10, 14, 10, 10)
        record_layout.setSpacing(8)

        record_hint = QLabel("保留处理历史、诊断信息和导出前的最终核对记录。")
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
        layout.addWidget(record_box)

        layout.addStretch(1)

        self.summary_segmented.currentItemChanged.connect(self._on_summary_segment_changed)
        self.visual_segmented.currentItemChanged.connect(self._on_visual_segment_changed)

        self.set_airborne_qc_visualization(None)
        self.set_airborne_trajectory_visualization(None)

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

    def _on_visual_segment_changed(self, route_key: str):
        mapping = {
            "qc_chart": 0,
            "trajectory": 1,
        }
        self.visual_stack.setCurrentIndex(mapping.get(route_key, 0))

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

    def append_record(self, text: str):
        """追加记录"""
        self.record.append(text)

    def clear_record(self):
        """清空记录"""
        self.record.clear()

    def get_record_text(self) -> str:
        """获取记录文本"""
        return self.record.toPlainText()

    def set_metadata_summary(self, text: str):
        """设置航空元数据摘要文本。"""
        self.metadata_summary.setPlainText(text or "")

    def set_line_summary(self, text: str):
        """设置测线结果卡片文本。"""
        self.line_summary.setPlainText(text or "")

    def set_airborne_qc_summary(self, text: str):
        """设置航空质控摘要文本。"""
        self.airborne_qc_summary.setPlainText(text or "")

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
            self.qc_fig.tight_layout()
            self.qc_canvas.draw_idle()
            return

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
        self.qc_fig.tight_layout()
        self.qc_canvas.draw_idle()

    def set_airborne_trajectory_visualization(self, payload: dict | None):
        """绘制航空航迹图（含飞行高度剖面）。"""
        self.trajectory_fig.clear()
        self.trajectory_ax = self.trajectory_fig.add_subplot(2, 1, 1)
        self.trajectory_height_ax = self.trajectory_fig.add_subplot(2, 1, 2)
        ax = self.trajectory_ax
        ax_h = self.trajectory_height_ax
        palette = self._get_plot_palette()

        if not payload:
            self._trajectory_longitude = np.array([], dtype=np.float64)
            self._trajectory_latitude = np.array([], dtype=np.float64)
            self._trajectory_trace_indices = np.array([], dtype=np.int32)
            self._selected_trace_index = None
            ax.set_title("航迹图")
            ax.text(
                0.5,
                0.5,
                "暂无航迹数据",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=palette["hint"],
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax_h.set_visible(False)
            self._style_figure(self.trajectory_fig, [ax])
            self.trajectory_fig.tight_layout()
            self.trajectory_canvas.draw_idle()
            return

        longitude = np.asarray(payload.get("longitude", []), dtype=np.float64)
        latitude = np.asarray(payload.get("latitude", []), dtype=np.float64)
        anomaly_mask = np.asarray(payload.get("anomaly_mask", []), dtype=bool)
        trace_indices = np.asarray(payload.get("trace_indices", []), dtype=np.int32)
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

        # ===== 上子图：航迹图 =====
        ax.set_title("航迹图（经纬度）")
        ax.plot(longitude, latitude, color=palette["line_primary"], linewidth=1.4, zorder=1)
        ax.scatter(
            [longitude[0]],
            [latitude[0]],
            color=palette["line_success"],
            s=42,
            zorder=3,
            label="起点",
        )
        ax.scatter(
            [longitude[-1]],
            [latitude[-1]],
            color=palette["line_error"],
            s=42,
            zorder=3,
            label="终点",
        )

        if anomaly_mask.size == longitude.size and np.any(anomaly_mask):
            ax.scatter(
                longitude[anomaly_mask],
                latitude[anomaly_mask],
                color=palette["line_warning"],
                s=26,
                zorder=4,
                label="异常点",
            )

        if self._selected_trace_index is not None:
            selected_mask = trace_indices == self._selected_trace_index
            if np.any(selected_mask):
                ax.scatter(
                    longitude[selected_mask],
                    latitude[selected_mask],
                    color=palette["line_emphasis"],
                    s=64,
                    marker="x",
                    linewidths=1.8,
                    zorder=5,
                    label="当前选中",
                )

        finite_mask = np.isfinite(longitude) & np.isfinite(latitude)
        if np.count_nonzero(finite_mask) > 0:
            lon_valid = longitude[finite_mask]
            lat_valid = latitude[finite_mask]
            lon_span = (
                float(np.max(lon_valid) - np.min(lon_valid)) if lon_valid.size else 0.0
            )
            lat_span = (
                float(np.max(lat_valid) - np.min(lat_valid)) if lat_valid.size else 0.0
            )
            pad_lon = max(lon_span * 0.05, 1e-6)
            pad_lat = max(lat_span * 0.05, 1e-6)
            ax.set_xlim(
                float(np.min(lon_valid) - pad_lon), float(np.max(lon_valid) + pad_lon)
            )
            ax.set_ylim(
                float(np.min(lat_valid) - pad_lat), float(np.max(lat_valid) + pad_lat)
            )

        ax.set_xlabel("经度")
        ax.set_ylabel("纬度")
        ax.legend(loc="best")

        # ===== 下子图：飞行高度剖面 =====
        if flight_height.size == longitude.size:
            ax_h.set_visible(True)
            ax_h.set_title("飞行高度剖面")
            x_axis = np.arange(longitude.size)
            ax_h.fill_between(
                x_axis,
                flight_height,
                alpha=0.25,
                color=palette["line_primary"],
            )
            ax_h.plot(
                x_axis,
                flight_height,
                color=palette["line_primary"],
                linewidth=1.2,
                label="飞行高度",
            )
            h_mean = float(np.mean(flight_height[np.isfinite(flight_height)])) if np.isfinite(flight_height).any() else None
            if h_mean is not None:
                ax_h.axhline(
                    y=h_mean,
                    color=palette["line_emphasis"],
                    linestyle="--",
                    linewidth=1.2,
                    label=f"平均高度: {h_mean:.2f} m",
                )
            if self._selected_trace_index is not None:
                selected_mask = trace_indices == self._selected_trace_index
                if np.any(selected_mask):
                    ax_h.scatter(
                        x_axis[selected_mask],
                        flight_height[selected_mask],
                        color=palette["line_error"],
                        s=48,
                        zorder=5,
                        label="当前选中",
                    )
            ax_h.set_xlabel("道号索引")
            ax_h.set_ylabel("飞行高度 (m)")
            ax_h.legend(loc="best")
        else:
            ax_h.set_visible(False)

        self._style_figure(self.trajectory_fig, [ax, ax_h])
        self.trajectory_fig.tight_layout()
        self.trajectory_canvas.draw_idle()

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
