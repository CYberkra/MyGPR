#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B-scan preview cards for the workflow canvas."""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


PREVIEW_PORT_ROW_Y = 50.0


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


class BscanPreviewCard(QFrame):
    """Canvas card that previews the latest B-scan output."""

    large_view_requested = pyqtSignal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: Any = None
        self._label = "Workflow Output"
        self._is_stale = False
        self._is_failed = False
        self._has_data = False
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
            QFrame#bscanPreviewCard[status="failed"] {
                border: 2px solid #ef4444;
                background: #fff5f5;
            }
            QFrame#bscanPreviewCard[status="stale"] {
                border: 1px dashed #9ca3af;
                background: #f9fafb;
            }
            QFrame#bscanPreviewCard[status="fresh"] {
                border: 1px solid #36d399;
                background: #f0fdf4;
            }
            QLabel#previewTitle {
                font-weight: 800;
                color: #5b4300;
            }
            QLabel#previewPortLabel {
                color: #6b5a2a;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#previewChipFresh {
                background: #dcfce7;
                color: #166534;
                border: 1px solid #86efac;
                border-radius: 8px;
                padding: 2px 7px;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#previewChipStale {
                background: #f3f4f6;
                color: #6b7280;
                border: 1px dashed #9ca3af;
                border-radius: 8px;
                padding: 2px 7px;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#previewChipFailed {
                background: #fee2e2;
                color: #dc2626;
                border: 1px solid #fca5a5;
                border-radius: 8px;
                padding: 2px 7px;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#previewChipEmpty {
                background: #f3f4f6;
                color: #9ca3af;
                border: 1px solid #d1d5db;
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

    def set_preview_data(self, data: Any, label: str = "Workflow Output", success: bool = True) -> None:
        self._data = data
        self._label = label or "Workflow Output"
        self._has_data = data is not None
        self._is_stale = False
        self._is_failed = not success
        try:
            self._build()
        except Exception:
            self._data = None
            self._has_data = False
            self._build()

    def set_stale(self) -> None:
        """Mark the preview as stale (result from previous run)."""
        self._is_stale = True
        self._is_failed = False
        self._build()

    def set_failed(self, message: str = "运行失败") -> None:
        """Mark the preview as failed."""
        self._is_failed = True
        self._is_stale = False
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

    def set_lod_mode(self, mode: str) -> None:
        """Set level-of-detail display mode: full, compact, or mini."""
        mode = mode if mode in {"full", "compact", "mini"} else "full"
        self._lod_mode = mode
        
        if mode == "full":
            self.setMinimumWidth(300)
            self.setMaximumWidth(360)
            self._apply_full_mode()
        elif mode == "compact":
            self.setMinimumWidth(200)
            self.setMaximumWidth(280)
            self._apply_compact_mode()
        elif mode == "mini":
            self.setMinimumWidth(160)
            self.setMaximumWidth(220)
            self._apply_mini_mode()
        
        self.updateGeometry()

    def _apply_full_mode(self) -> None:
        """Apply full LOD mode with normal fonts."""
        main_layout = self.layout()
        if main_layout is not None:
            for i in range(main_layout.count()):
                item = main_layout.itemAt(i)
                if item and item.widget():
                    widget = item.widget()
                    widget.show()
                    widget.setStyleSheet("")
        self._simplify_layout(restore=True)

    def _apply_compact_mode(self) -> None:
        """Apply compact LOD mode with larger fonts."""
        main_layout = self.layout()
        if main_layout is not None:
            for i in range(main_layout.count()):
                item = main_layout.itemAt(i)
                if item and item.widget():
                    widget = item.widget()
                    if widget.objectName() == "previewImage":
                        widget.hide()
                    elif widget.objectName() == "previewMeta" and "双击" in widget.text():
                        widget.hide()
                    else:
                        widget.show()
                        if widget.objectName() == "previewTitle":
                            widget.setStyleSheet("font-size: 16px; font-weight: bold;")
                        elif widget.objectName() == "previewMeta":
                            widget.setStyleSheet("font-size: 14px;")
        self._simplify_layout(restore=False)

    def _apply_mini_mode(self) -> None:
        """Apply mini LOD mode with largest fonts."""
        main_layout = self.layout()
        if main_layout is not None:
            for i in range(main_layout.count()):
                item = main_layout.itemAt(i)
                if item and item.widget():
                    widget = item.widget()
                    if widget.objectName() in ("previewTitle",):
                        widget.show()
                        widget.setStyleSheet("font-size: 20px; font-weight: bold;")
                    elif widget.objectName() in ("previewMeta",):
                        widget.hide()
                    elif widget.objectName() == "previewImage":
                        widget.hide()
                    else:
                        widget.show()
        self._simplify_layout(restore=False)

    def _simplify_layout(self, restore: bool = False) -> None:
        """Simplify or restore layout margins."""
        layout = self.layout()
        if layout is not None:
            if restore:
                layout.setContentsMargins(10, 8, 10, 10)
            else:
                layout.setContentsMargins(8, 6, 8, 6)

    def mouseDoubleClickEvent(self, event):  # noqa: N802 - Qt override
        self.request_large_view()
        event.accept()

    def request_large_view(self) -> None:
        self.large_view_requested.emit(self._data, self._label)

    def open_large_view(self) -> None:
        self.request_large_view()

    def _build(self) -> None:
        old_layout = self.layout()
        if old_layout is not None:
            while old_layout.count():
                item = old_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            QWidget().setLayout(old_layout)

        if self._is_failed:
            self.setProperty("status", "failed")
        elif self._is_stale:
            self.setProperty("status", "stale")
        elif self._has_data:
            self.setProperty("status", "fresh")
        else:
            self.setProperty("status", "")
        self.style().unpolish(self)
        self.style().polish(self)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(5)

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        title = QLabel("B-scan Preview")
        title.setObjectName("previewTitle")
        title_row.addWidget(title, 1)

        if self._is_failed:
            chip_text = "失败"
            chip_class = "previewChipFailed"
        elif self._is_stale:
            chip_text = "上次"
            chip_class = "previewChipStale"
        elif self._has_data:
            chip_text = "最新"
            chip_class = "previewChipFresh"
        else:
            chip_text = "暂无"
            chip_class = "previewChipEmpty"

        chip = QLabel(chip_text)
        chip.setObjectName(chip_class)
        title_row.addWidget(chip)
        root.addLayout(title_row)

        port_row = QHBoxLayout()
        port_row.setContentsMargins(0, 4, 0, 4)
        left_port = QLabel("data")
        left_port.setObjectName("previewPortLabel")
        right_port = QLabel("preview")
        right_port.setObjectName("previewPortLabel")
        right_port.setAlignment(Qt.AlignmentFlag.AlignRight)
        port_row.addWidget(left_port)
        port_row.addStretch(1)
        port_row.addWidget(right_port)
        root.addLayout(port_row)
        self._port_row = port_row

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
            self.thumbnail_label.setText("暂无预览，请先运行工作流")
        else:
            self.thumbnail_label.setPixmap(pixmap)
        root.addWidget(self.thumbnail_label)

        hint = QLabel("双击打开大图" if shape is not None else "暂无预览，请先运行工作流")
        hint.setObjectName("previewMeta")
        root.addWidget(hint)

    def port_anchor_y(self) -> float:
        if hasattr(self, '_port_row') and self._port_row is not None:
            try:
                geom = self._port_row.geometry()
                if geom.isValid() and geom.height() > 0:
                    return float(geom.center().y())
            except Exception:
                pass
        return PREVIEW_PORT_ROW_Y
