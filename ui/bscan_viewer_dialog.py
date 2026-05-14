#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matplotlib-backed B-scan viewer dialog for Workflow Studio previews."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QBuffer, QByteArray
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


def coerce_bscan_array(data: Any) -> np.ndarray | None:
    """Return a finite-compatible 2-D numeric array for B-scan viewing."""
    if data is None:
        return None
    try:
        array = np.asarray(data)
    except Exception:
        return None
    array = np.squeeze(array)
    if array.ndim != 2 or array.size == 0:
        return None
    if not np.issubdtype(array.dtype, np.number):
        return None
    return array


def downsample_for_view(
    array: np.ndarray,
    *,
    max_rows: int = 1800,
    max_cols: int = 2600,
) -> np.ndarray:
    """Bound the display array size while preserving B-scan geometry."""
    rows, cols = array.shape
    row_step = max(1, int(np.ceil(rows / max_rows)))
    col_step = max(1, int(np.ceil(cols / max_cols)))
    return array[::row_step, ::col_step]


class BscanViewerDialog(QDialog):
    """Stable large-view dialog for B-scan preview nodes.

    This dialog is always a top-level window (not embedded in graphics proxy widgets).
    """

    def __init__(self, data: Any, title: str = "B-scan Preview", parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._raw = coerce_bscan_array(data)
        self._view = downsample_for_view(self._raw) if self._raw is not None else None
        self._image = None
        self._colorbar = None
        self._title = title or "B-scan Preview"
        self.setWindowTitle(self._title)
        self.resize(1120, 780)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        title_label = QLabel(self._title)
        title_label.setProperty("class", "sectionTitle")
        header.addWidget(title_label, 1)

        self.btn_fit = QPushButton("适配")
        self.btn_100 = QPushButton("100%")
        self.btn_zoom_out = QPushButton("缩小")
        self.btn_zoom_in = QPushButton("放大")
        self.invert_check = QCheckBox("反相")
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItem("灰度", "gray")
        self.cmap_combo.addItem("伪彩", "turbo")
        self.percentile_low = QDoubleSpinBox()
        self.percentile_high = QDoubleSpinBox()
        for spin, value in [(self.percentile_low, 1.0), (self.percentile_high, 99.0)]:
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setValue(value)
            spin.setSuffix("%")
            spin.setMaximumWidth(76)

        for widget in [
            self.btn_fit,
            self.btn_100,
            self.btn_zoom_out,
            self.btn_zoom_in,
            self.invert_check,
            QLabel("色图"),
            self.cmap_combo,
            QLabel("幅值"),
            self.percentile_low,
            self.percentile_high,
        ]:
            header.addWidget(widget)
        header.addStretch()
        self.btn_save = QPushButton("保存截图")
        self.btn_copy = QPushButton("复制图像")
        for widget in [self.btn_save, self.btn_copy]:
            header.addWidget(widget)
        layout.addLayout(header)

        self.info_label = QLabel("")
        self.info_label.setProperty("class", "hintText")
        layout.addWidget(self.info_label)

        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, 1)

        self.cursor_label = QLabel("cursor: --")
        self.cursor_label.setProperty("class", "hintText")
        layout.addWidget(self.cursor_label)

        self.btn_fit.clicked.connect(self.fit_to_data)
        self.btn_100.clicked.connect(self.reset_100)
        self.btn_zoom_out.clicked.connect(lambda: self.zoom_by(1.25))
        self.btn_zoom_in.clicked.connect(lambda: self.zoom_by(0.80))
        self.invert_check.stateChanged.connect(lambda _=None: self.redraw())
        self.cmap_combo.currentIndexChanged.connect(lambda _=None: self.redraw())
        self.percentile_low.valueChanged.connect(lambda _=None: self.redraw())
        self.percentile_high.valueChanged.connect(lambda _=None: self.redraw())
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.btn_save.clicked.connect(self._save_screenshot)
        self.btn_copy.clicked.connect(self._copy_image)

        self.redraw()

    @property
    def has_data(self) -> bool:
        return self._view is not None

    def redraw(self) -> None:
        self.axes.clear()
        # Remove old colorbar if present
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:
                pass
            self._colorbar = None
        if self._view is None:
            self.info_label.setText("数据尺寸：--")
            self.axes.text(
                0.5,
                0.5,
                "没有可显示的 B-scan 数据\n请先运行工作流生成预览",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
                color="#667085",
                fontsize=13,
            )
            self.axes.set_axis_off()
            self.canvas.draw_idle()
            return

        display = np.asarray(self._view, dtype=np.float32)
        finite = display[np.isfinite(display)]
        low_pct = min(float(self.percentile_low.value()), float(self.percentile_high.value()))
        high_pct = max(float(self.percentile_low.value()), float(self.percentile_high.value()))
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, [low_pct, high_pct])
        else:
            vmin, vmax = 0.0, 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(display)), float(np.nanmax(display))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0

        cmap = str(self.cmap_combo.currentData() or "gray")
        if self.invert_check.isChecked() and cmap == "gray":
            cmap = "gray_r"
        elif self.invert_check.isChecked() and cmap == "turbo":
            display = -display
            vmin, vmax = -vmax, -vmin

        self._image = self.axes.imshow(
            display,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        # Add colorbar
        self._colorbar = self.figure.colorbar(self._image, ax=self.axes)
        self.axes.set_xlabel("轨迹")
        self.axes.set_ylabel("采样 / 时间索引")
        rows, cols = self._view.shape
        raw_rows, raw_cols = self._raw.shape if self._raw is not None else self._view.shape
        self.info_label.setText(
            f"数据尺寸：{raw_cols} 轨迹 × {raw_rows} 采样"
            f"    显示：{cols} × {rows}"
        )
        self.fit_to_data(redraw_canvas=False)
        self.canvas.draw_idle()

    def _save_screenshot(self) -> None:
        if self._view is None:
            QMessageBox.information(self, "无数据", "没有可保存的 B-scan 数据")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存截图",
            "",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg);;所有文件 (*.*)"
        )
        if not file_path:
            return
        # Save the figure
        try:
            self.figure.savefig(file_path, dpi=150, bbox_inches="tight")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"保存截图时出错：{str(e)}")
        else:
            QMessageBox.information(self, "保存成功", f"截图已保存至：{file_path}")

    def _copy_image(self) -> None:
        if self._view is None:
            QMessageBox.information(self, "无数据", "没有可复制的 B-scan 数据")
            return
        # Save figure to QPixmap
        try:
            buffer = QBuffer()
            buffer.open(QBuffer.OpenModeFlag.ReadWrite)
            self.figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.data(), "png")
            clipboard = QGuiApplication.clipboard()
            clipboard.setPixmap(pixmap)
            buffer.close()
        except Exception as e:
            QMessageBox.warning(self, "复制失败", f"复制图像时出错：{str(e)}")
        else:
            QMessageBox.information(self, "复制成功", "图像已复制到剪贴板")

    def fit_to_data(self, *, redraw_canvas: bool = True) -> None:
        if self._view is None:
            return
        rows, cols = self._view.shape
        self.axes.set_xlim(-0.5, cols - 0.5)
        self.axes.set_ylim(rows - 0.5, -0.5)
        if redraw_canvas:
            self.canvas.draw_idle()

    def reset_100(self) -> None:
        self.fit_to_data()

    def zoom_by(self, factor: float) -> None:
        if self._view is None:
            return
        x0, x1 = self.axes.get_xlim()
        y0, y1 = self.axes.get_ylim()
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        half_w = abs(x1 - x0) * float(factor) / 2.0
        half_h = abs(y1 - y0) * float(factor) / 2.0
        rows, cols = self._view.shape
        self.axes.set_xlim(max(-0.5, cx - half_w), min(cols - 0.5, cx + half_w))
        # Keep B-scan sample axis top-down.
        self.axes.set_ylim(min(rows - 0.5, cy + half_h), max(-0.5, cy - half_h))
        self.canvas.draw_idle()

    def _on_motion(self, event) -> None:
        if self._view is None or event.inaxes is not self.axes:
            self.cursor_label.setText("光标：--")
            return
        col = int(round(event.xdata)) if event.xdata is not None else -1
        row = int(round(event.ydata)) if event.ydata is not None else -1
        rows, cols = self._view.shape
        if 0 <= row < rows and 0 <= col < cols:
            value = self._view[row, col]
            self.cursor_label.setText(
                f"光标：轨迹={col}, 时间/采样={row}, 幅值={float(value):.6g}"
            )
        else:
            self.cursor_label.setText("光标：--")
