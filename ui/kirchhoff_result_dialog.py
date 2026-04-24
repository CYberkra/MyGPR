#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kirchhoff 迁移结果独立显示窗口"""

import numpy as np
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtGui import QIcon


class KirchhoffResultDialog(QDialog):
    """Kirchhoff 迁移结果独立显示窗口"""

    def __init__(
        self, parent=None, data=None, header_info=None, title="Kirchhoff 迁移结果"
    ):
        super().__init__(parent)
        self.data = data
        self.header_info = header_info or {}
        self.setWindowTitle(title)

        # 设置窗口大小 - 宽屏比例
        self.setMinimumSize(1200, 700)
        self.resize(1400, 800)

        self._setup_ui()
        self._plot_result()

    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # 标题和信息
        info_text = self._build_info_text()
        self.info_label = QLabel(info_text)
        self.info_label.setProperty("class", "textSecondary")
        layout.addWidget(self.info_label)

        # Matplotlib 图形区域
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        # 工具栏
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_save = QPushButton("保存图像")
        self.btn_save.clicked.connect(self._on_save)
        btn_layout.addWidget(self.btn_save)

        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)

    def _build_info_text(self):
        """构建信息文本"""
        info_parts = []

        if self.data is not None:
            info_parts.append(f"数据: {self.data.shape[0]}×{self.data.shape[1]}")

        track_len = self.header_info.get("track_length_m")
        if track_len:
            info_parts.append(f"测线长度: {track_len:.1f}m")

        if self.header_info.get("is_elevation"):
            elev_top = self.header_info.get("elevation_top_m")
            elev_bottom = self.header_info.get("elevation_bottom_m")
            if elev_top is not None and elev_bottom is not None:
                info_parts.append(f"高程: {elev_bottom:.1f}~{elev_top:.1f}m")
        elif self.header_info.get("is_depth"):
            depth_max = self.header_info.get("depth_max_m")
            if depth_max:
                info_parts.append(f"深度: 0~{depth_max:.1f}m")

        return " | ".join(info_parts) if info_parts else "Kirchhoff 迁移结果"

    def _plot_result(self):
        """绘制结果"""
        if self.data is None:
            self.ax.text(
                0.5,
                0.5,
                "无数据",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            return

        # 计算显示范围
        extent = self._compute_extent()

        # 计算色标范围
        vmin, vmax = self._compute_vmin_vmax()

        # 绘制
        im = self.ax.imshow(
            self.data,
            cmap="seismic",
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
        )

        # 添加色标
        self.fig.colorbar(im, ax=self.ax, label="振幅")

        # 设置标签
        if self.header_info.get("is_elevation"):
            self.ax.set_ylabel("高程 (m)")
        elif self.header_info.get("is_depth"):
            self.ax.set_ylabel("深度 (m)")
        else:
            self.ax.set_ylabel("时间 (ns)")

        self.ax.set_xlabel("距离 (m)")
        self.ax.set_title("Kirchhoff Migration Profile")

        # 调整布局
        self.fig.tight_layout()
        self.canvas.draw()

    def _compute_extent(self):
        """计算显示范围"""
        n_samples, n_traces = self.data.shape

        # X轴：距离
        track_len = self.header_info.get("track_length_m")
        if track_len:
            x_range = [0, float(track_len)]
        else:
            trace_interval = self.header_info.get("trace_interval_m", 1.0)
            x_range = [0, n_traces * float(trace_interval)]

        # Y轴：高程或深度
        if self.header_info.get("is_elevation"):
            elev_top = self.header_info.get("elevation_top_m")
            elev_bottom = self.header_info.get("elevation_bottom_m")
            if elev_top is not None and elev_bottom is not None:
                # 高程轴：大的在上，小的在下
                y_range = [float(elev_bottom), float(elev_top)]
            else:
                depth_step = self.header_info.get("depth_step_m", 1.0)
                y_range = [n_samples * float(depth_step), 0]
        elif self.header_info.get("is_depth"):
            depth_max = self.header_info.get("depth_max_m")
            if depth_max:
                y_range = [float(depth_max), 0]  # 深度轴：0在上，大的在下
            else:
                depth_step = self.header_info.get("depth_step_m", 1.0)
                y_range = [n_samples * float(depth_step), 0]
        else:
            total_time = self.header_info.get("total_time_ns", n_samples)
            y_range = [float(total_time), 0]

        return [x_range[0], x_range[1], y_range[0], y_range[1]]

    def _compute_vmin_vmax(self):
        """计算色标范围"""
        # 使用百分比拉伸
        p_low, p_high = 0.5, 99.5
        vmin, vmax = np.percentile(self.data, [p_low, p_high])

        # 确保对称
        max_abs = max(abs(vmin), abs(vmax))
        if max_abs > 0:
            return -max_abs, max_abs
        return -1.0, 1.0

    def _on_save(self):
        """保存图像"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 Kirchhoff 迁移结果",
            "",
            "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg);;PDF文档 (*.pdf);;所有文件 (*)",
        )

        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches="tight")
                QMessageBox.information(self, "保存成功", f"图像已保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存图像时出错:\n{str(e)}")

    def showEvent(self, event):
        """显示时调整画布"""
        super().showEvent(event)
        self.canvas.draw()
