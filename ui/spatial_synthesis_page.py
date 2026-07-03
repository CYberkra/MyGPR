#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-level map, multi-line, and terrain synthesis page."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.project_service import ProjectService
from core.spatial_synthesis_service import SpatialSynthesisService


class SpatialSynthesisPage(QWidget):
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: ProjectService | None = None
        self.synthesis: dict | None = None
        layout = QVBoxLayout(self)
        top = QFrame()
        top_layout = QHBoxLayout(top)
        title = QLabel("空间成果")
        self.status_label = QLabel("未加载工程")
        self.refresh_button = QPushButton("刷新空间成果")
        top_layout.addWidget(title)
        top_layout.addWidget(self.status_label)
        top_layout.addStretch(1)
        top_layout.addWidget(self.refresh_button)
        layout.addWidget(top)
        self.tabs = QTabWidget()
        self.map_figure = Figure(facecolor="#101820")
        self.map_canvas = FigureCanvas(self.map_figure)
        self.terrain_figure = Figure(facecolor="#101820")
        self.terrain_canvas = FigureCanvas(self.terrain_figure)
        self.summary_table = QTableWidget(0, 3)
        self.summary_table.setHorizontalHeaderLabels(["测线", "状态", "道数"])
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.map_canvas, "测线轨迹与目标关联")
        self.tabs.addTab(self.terrain_canvas, "地形与高度")
        self.tabs.addTab(self.summary_table, "定位状态")
        layout.addWidget(self.tabs)
        self.refresh_button.clicked.connect(self.refresh)

    def open_project(self, project: ProjectService) -> None:
        self.project = project
        self.refresh()

    def refresh(self) -> None:
        if self.project is None:
            return
        self.synthesis = SpatialSynthesisService(self.project).build()
        summary = self.synthesis["summary"]
        if summary["located_line_count"]:
            self.status_label.setText(
                f"已定位 {summary['located_line_count']} 条测线 · "
                f"{summary['track_point_count']} 个轨迹点"
            )
        else:
            self.status_label.setText(
                f"无空间定位 · {summary['unlocated_line_count']} 条测线待补充 RTK/逐道坐标"
            )
        self._draw_map()
        self._draw_terrain()
        self._populate_summary()
        self.status_changed.emit(self.status_label.text())

    def _draw_map(self) -> None:
        self.map_figure.clear()
        axis = self.map_figure.add_subplot(111)
        self._style_axis(axis, "测线轨迹与目标标注", "经度", "纬度")
        for track in (self.synthesis or {}).get("tracks", []):
            axis.plot(track["longitude"], track["latitude"], linewidth=1.7, label=track["name"])
        features = (self.synthesis or {}).get("interpretation_features", [])
        if features:
            axis.scatter(
                [item["longitude"] for item in features],
                [item["latitude"] for item in features],
                c=[item["properties"]["confidence"] for item in features],
                cmap="autumn",
                edgecolors="#FFFFFF",
                s=35,
            )
        if (self.synthesis or {}).get("tracks"):
            axis.legend(loc="best")
        else:
            axis.text(0.5, 0.5, "当前项目还没有可用的空间定位数据", transform=axis.transAxes, ha="center", color="#EAF4F4")
        self.map_figure.tight_layout()
        self.map_canvas.draw_idle()

    def _draw_terrain(self) -> None:
        self.terrain_figure.clear()
        axis = self.terrain_figure.add_subplot(111)
        self._style_axis(axis, "地形与飞行高度", "道号序列", "高程 / 飞行高度（m）")
        terrain = (self.synthesis or {}).get("terrain_points", [])
        for line_id in sorted({item["line_id"] for item in terrain}):
            values = [item for item in terrain if item["line_id"] == line_id]
            x = np.arange(len(values))
            ground = np.array([item["ground_elevation_m"] for item in values], dtype=float)
            height = np.array([item["height_agl_m"] for item in values], dtype=float)
            if np.isfinite(ground).any():
                axis.plot(x, ground, label=f"{line_id} 地面")
            if np.isfinite(height).any():
                axis.plot(x, height, linestyle="--", label=f"{line_id} 高度")
        if terrain:
            axis.legend(loc="best")
        else:
            axis.text(0.5, 0.5, "当前测线缺少地形或高度数据", transform=axis.transAxes, ha="center", color="#EAF4F4")
        self.terrain_figure.tight_layout()
        self.terrain_canvas.draw_idle()

    def _populate_summary(self) -> None:
        located = {
            item["line_id"]: item for item in (self.synthesis or {}).get("tracks", [])
        }
        unlocated = {
            item["line_id"]: item
            for item in (self.synthesis or {}).get("unlocated_lines", [])
        }
        line_ids = list(self.project.manifest.line_ids) if self.project else []
        self.summary_table.setRowCount(len(line_ids))
        for row, line_id in enumerate(line_ids):
            line = self.project.get_line(line_id)
            if line_id in located:
                values = (line.name, "已定位", located[line_id]["trace_count"])
            else:
                values = (line.name, unlocated.get(line_id, {}).get("reason", "无空间定位"), 0)
            for column, value in enumerate(values):
                self.summary_table.setItem(row, column, QTableWidgetItem(str(value)))

    @staticmethod
    def _style_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
        axis.set_facecolor("#101820")
        axis.tick_params(colors="#AFC2CF")
        axis.set_title(title, color="#EAF4F4")
        axis.set_xlabel(xlabel, color="#AFC2CF")
        axis.set_ylabel(ylabel, color="#AFC2CF")


__all__ = ["SpatialSynthesisPage"]
