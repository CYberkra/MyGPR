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
        self.setWindowTitle(title)
        self.resize(980, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QLabel(title)
        header.setProperty("class", "sectionTitle")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        frame = QFrame()
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(8, 8, 8, 8)

        pixmap = _array_to_pixmap(data, width=1200, height=860)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if pixmap is None:
            image_label.setText("没有可显示的 B-scan 数据")
        else:
            image_label.setPixmap(pixmap)
        frame_layout.addWidget(image_label)
        scroll.setWidget(frame)
        layout.addWidget(scroll, 1)


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
                border-radius: 12px;
            }
            QFrame#bscanPreviewCard[compact="true"] {
                background: #fff8df;
                border: 1px solid #d9a400;
            }
            QLabel#previewTitle {
                font-weight: 800;
                color: #5b4300;
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
        dialog = BscanPreviewDialog(self._data, title=self._label, parent=self)
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

        title = QLabel("B-scan Preview")
        title.setObjectName("previewTitle")
        root.addWidget(title)

        shape = self.data_shape
        shape_text = "--" if shape is None else f"{shape[1]} traces × {shape[0]} samples"
        self.source_label = QLabel(f"{self._label} · {shape_text}")
        self.source_label.setObjectName("previewMeta")
        self.source_label.setWordWrap(True)
        root.addWidget(self.source_label)

        if self.compact:
            hint = QLabel("双击打开大图" if shape is not None else "等待工作流运行结果")
            hint.setObjectName("previewMeta")
            root.addWidget(hint)
            return

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

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        open_button = QPushButton("大图")
        open_button.setEnabled(shape is not None)
        open_button.clicked.connect(self.open_large_view)
        action_row.addWidget(open_button)
        action_row.addStretch(1)
        root.addLayout(action_row)
