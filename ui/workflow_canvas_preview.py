#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B-scan preview cards for the workflow canvas."""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _coerce_bscan_array(data: Any) -> np.ndarray | None:
    """Return a numeric 2-D array suitable for image preview.

    Keep the original dtype here. Converting the full B-scan to float64 in the
    GUI thread can allocate hundreds of MB for large surveys and may terminate
    the process before Python can raise a normal exception.
    """
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


def _downsample_for_preview(
    array: np.ndarray,
    *,
    max_rows: int = 900,
    max_cols: int = 1400,
) -> np.ndarray:
    """Return a bounded view/copy for thumbnail and dialog rendering."""
    rows, cols = array.shape
    row_step = max(1, int(np.ceil(rows / max_rows)))
    col_step = max(1, int(np.ceil(cols / max_cols)))
    return array[::row_step, ::col_step]


def _array_to_pixmap(data: Any, *, width: int, height: int) -> QPixmap | None:
    """Convert a B-scan array to a grayscale QPixmap without full-size copies."""
    array = _coerce_bscan_array(data)
    if array is None:
        return None

    preview = _downsample_for_preview(array)
    try:
        preview_float = np.asarray(preview, dtype=np.float32)
    except Exception:
        return None

    finite = preview_float[np.isfinite(preview_float)]
    if finite.size == 0:
        return None

    try:
        low, high = np.nanpercentile(finite, [1.0, 99.0])
    except Exception:
        return None
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if high <= low:
        high = low + 1.0

    normalized = np.clip((preview_float - low) / (high - low), 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.5, posinf=1.0, neginf=0.0)
    # GPR B-scans are usually easier to read with strong positive amplitudes dark.
    pixels = np.ascontiguousarray(np.round((1.0 - normalized) * 255.0).astype(np.uint8))
    img_h, img_w = pixels.shape
    image = QImage(pixels.data, img_w, img_h, img_w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(image).scaled(
        max(1, int(width)),
        max(1, int(height)),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class BscanPreviewDialog(QDialog):
    """Large B-scan viewer opened from a workflow preview node."""

    def __init__(self, data: Any, title: str = "B-scan Preview", parent=None):
        super().__init__(parent)
        self._data = data
        self._title = title or "B-scan Preview"
        self._zoom = 1.0
        self._fit_mode = True
        self._base_pixmap: QPixmap | None = None
        self.setWindowTitle(self._title)
        self.resize(1080, 760)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        title_label = QLabel(self._title)
        title_label.setProperty("class", "sectionTitle")
        header_row.addWidget(title_label, 1)

        self.btn_fit = QPushButton("适配")
        self.btn_100 = QPushButton("100%")
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_in = QPushButton("+")
        for btn in [self.btn_fit, self.btn_100, self.btn_zoom_out, self.btn_zoom_in]:
            btn.setMinimumWidth(54)
            header_row.addWidget(btn)
        layout.addLayout(header_row)

        self.status_label = QLabel("")
        self.status_label.setProperty("class", "hintText")
        layout.addWidget(self.status_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self.scroll.setWidget(self.image_label)
        layout.addWidget(self.scroll, 1)

        self.btn_fit.clicked.connect(self.fit_to_window)
        self.btn_100.clicked.connect(self.show_100)
        self.btn_zoom_out.clicked.connect(lambda: self.set_zoom(self._zoom / 1.25))
        self.btn_zoom_in.clicked.connect(lambda: self.set_zoom(self._zoom * 1.25))

        self._load_pixmap()
        self.fit_to_window()

    def _load_pixmap(self) -> None:
        array = _coerce_bscan_array(self._data)
        if array is None:
            self.image_label.setText("没有可显示的 B-scan 数据")
            self.status_label.setText("shape: --")
            return

        self.status_label.setText(
            f"shape: {array.shape[1]} traces × {array.shape[0]} samples"
        )
        self._base_pixmap = _array_to_pixmap(array, width=1600, height=1200)
        if self._base_pixmap is None:
            self.image_label.setText("没有可显示的 B-scan 数据")

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if self._fit_mode:
            self._apply_pixmap()

    def fit_to_window(self) -> None:
        self._fit_mode = True
        self._apply_pixmap()

    def show_100(self) -> None:
        self._fit_mode = False
        self._zoom = 1.0
        self._apply_pixmap()

    def set_zoom(self, zoom: float) -> None:
        self._fit_mode = False
        self._zoom = max(0.1, min(8.0, float(zoom)))
        self._apply_pixmap()

    def _apply_pixmap(self) -> None:
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        if self._fit_mode:
            viewport = self.scroll.viewport().size()
            target = self._base_pixmap.scaled(
                max(1, viewport.width() - 8),
                max(1, viewport.height() - 8),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._zoom = target.width() / max(1, self._base_pixmap.width())
        else:
            target = self._base_pixmap.scaled(
                max(1, int(self._base_pixmap.width() * self._zoom)),
                max(1, int(self._base_pixmap.height() * self._zoom)),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.image_label.setPixmap(target)
        self.image_label.resize(target.size())


class BscanPreviewCard(QFrame):
    """Canvas card that previews the latest B-scan output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: Any = None
        self._label = "Workflow Output"
        self.compact = False
        self.source_label: QLabel | None = None
        self.thumbnail_label: QLabel | None = None
        self.setObjectName("bscanPreviewCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(300)
        self.setMaximumWidth(360)
        self.setStyleSheet(
            """
            QFrame#bscanPreviewCard {
                background: #fffdf7;
                border: 1px solid #e0c36a;
                border-radius: 10px;
            }
            QLabel#previewTitle {
                font-weight: 800;
                color: #5b4300;
            }
            QLabel#previewChip {
                background: #fff1b8;
                color: #765500;
                border: 1px solid #e0c36a;
                border-radius: 8px;
                padding: 2px 7px;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#previewMeta {
                color: #6b5a2a;
            }
            QLabel#previewImage {
                background: #111111;
                border-radius: 8px;
                padding: 4px;
            }
            """
        )
        self._build()

    @property
    def data_shape(self) -> tuple[int, int] | None:
        array = _coerce_bscan_array(self._data)
        if array is None:
            return None
        return int(array.shape[0]), int(array.shape[1])

    def set_preview_data(self, data: Any, label: str = "Workflow Output") -> None:
        self._data = data
        self._label = label or "Workflow Output"
        try:
            self._build()
        except Exception:
            # Preview rendering must never take down the whole GUI. A later
            # refresh can still recover when valid data is supplied.
            self._data = None
            self._build()

    def set_compact(self, compact: bool) -> None:
        compact = bool(compact)
        if self.compact == compact:
            return
        self.compact = compact
        self.setProperty("compact", compact)
        self._build()
        self.style().unpolish(self)
        self.style().polish(self)

    def mouseDoubleClickEvent(self, event):  # noqa: N802 - Qt override
        self.open_large_view()
        event.accept()

    def open_large_view(self) -> None:
        if self.data_shape is None:
            return
        parent = self.window() if isinstance(self.window(), QWidget) else None
        dialog = BscanPreviewDialog(self._data, title=self._label, parent=parent)
        dialog.exec()

    def _build(self) -> None:
        old_layout = self.layout()
        if old_layout is not None:
            while old_layout.count():
                item = old_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            QWidget().setLayout(old_layout)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(7)

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        title = QLabel("B-scan Preview")
        title.setObjectName("previewTitle")
        title_row.addWidget(title, 1)
        chip = QLabel("PREVIEW")
        chip.setObjectName("previewChip")
        title_row.addWidget(chip)
        root.addLayout(title_row)

        shape = self.data_shape
        shape_text = "--" if shape is None else f"{shape[1]} traces × {shape[0]} samples"
        self.source_label = QLabel(f"{self._label} · {shape_text}")
        self.source_label.setObjectName("previewMeta")
        self.source_label.setWordWrap(True)
        root.addWidget(self.source_label)

        self.thumbnail_label = QLabel()
        self.thumbnail_label.setObjectName("previewImage")
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumHeight(150)
        pixmap = _array_to_pixmap(self._data, width=320, height=180)
        if pixmap is None:
            self.thumbnail_label.setText("等待工作流运行结果")
        else:
            self.thumbnail_label.setPixmap(pixmap)
        root.addWidget(self.thumbnail_label)

        hint = QLabel("double click to open large" if shape is not None else "等待工作流运行结果")
        hint.setObjectName("previewMeta")
        root.addWidget(hint)
