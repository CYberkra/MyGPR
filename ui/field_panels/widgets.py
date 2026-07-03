#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable lightweight widgets for the MyGPR field workbench.

These classes used to live in ``field_workbench_window.py``.  Keeping them here
reduces the main window file and makes future page extraction less risky.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
try:  # Matplotlib versions used in some Windows envs expose this name.
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except Exception:  # pragma: no cover - toolbar is optional for the viewer.
    NavigationToolbar = None  # type: ignore[assignment]
from matplotlib.figure import Figure

import matplotlib.pyplot as plt


def _spacer(width: int = 0, height: int = 0) -> QWidget:
    w = QWidget()
    w.setFixedSize(width, height)
    return w


def _set_shadow(widget: QWidget, alpha: int = 22, blur: int = 18) -> None:
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, 4)
    shadow.setColor(QColor(25, 55, 85, alpha))
    widget.setGraphicsEffect(shadow)


class Card(QFrame):
    def __init__(self, *, title: str | None = None, object_name: str = "card") -> None:
        super().__init__()
        self.setObjectName(object_name)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 3, 4, 3)
        self.layout.setSpacing(2)
        self.title_bar: QWidget | None = None
        self.title_row: QHBoxLayout | None = None
        if title:
            self._create_title_bar(title)

    def _create_title_bar(self, title: str) -> None:
        title_bar = QWidget()
        title_bar.setObjectName("cardTitleBar")
        title_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        title_bar.setFixedHeight(13)
        row = QHBoxLayout(title_bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        label = QLabel(title)
        label.setObjectName("cardTitle")
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        row.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        self.title_bar = title_bar
        self.title_row = row
        self.layout.addWidget(title_bar)

    def add_title_button(self, button: QPushButton) -> None:
        """Add a compact action button to the card title bar.

        The method keeps page code from relying on the title bar internals.  If
        a legacy card has no title bar, a small empty title bar is created so
        the action remains visible without changing the card's public API.
        """
        if self.title_row is None:
            self._create_title_bar("")
        assert self.title_row is not None
        button.setFixedHeight(13)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.title_row.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)


class MetricCard(Card):
    def __init__(self, icon: str, title: str, value: str, suffix: str = "", note: str = "") -> None:
        super().__init__(object_name="metricCard")
        self.setFixedHeight(42)
        row = QHBoxLayout()
        row.setSpacing(6)
        icon_box = QLabel(icon)
        icon_box.setObjectName("metricIcon")
        icon_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_box.setFixedSize(20, 20)
        row.addWidget(icon_box)
        col = QVBoxLayout()
        col.setSpacing(1)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        self.title_label = title_label
        value_row = QHBoxLayout()
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        self.value_label = value_label
        suffix_label = QLabel(suffix)
        suffix_label.setObjectName("metricSuffix")
        self.suffix_label = suffix_label
        value_row.addWidget(value_label)
        value_row.addWidget(suffix_label)
        value_row.addStretch(1)
        col.addWidget(title_label)
        col.addLayout(value_row)
        self.note_label = None
        if note:
            note_label = QLabel(note)
            note_label.setObjectName("metricNote")
            self.note_label = note_label
            col.addWidget(note_label)
        row.addLayout(col, 1)
        self.layout.addLayout(row)


class PlotViewerDialog(QDialog):
    """Modal viewer for inspecting a plot without compressing the workspace."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        title: str,
        draw_callback: Callable[[FigureCanvas], None],
        size: tuple[int, int] = (1180, 760),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(*size)
        self.setMinimumSize(900, 600)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.figure = Figure(figsize=(9.5, 5.8), dpi=100, facecolor="white")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if NavigationToolbar is not None:
            layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        draw_callback(self.canvas)

    def closeEvent(self, event) -> None:
        self.canvas.close()
        plt.close(self.figure)
        super().closeEvent(event)


def open_plot_viewer(
    parent: QWidget | None,
    *,
    title: str,
    draw_callback: Callable[[FigureCanvas], None],
    size: tuple[int, int] = (1180, 760),
) -> None:
    # Rotate Figure resources every 5 opens to prevent unbounded memory growth
    if hasattr(open_plot_viewer, '_counter'):
        open_plot_viewer._counter += 1
    else:
        open_plot_viewer._counter = 0
    if open_plot_viewer._counter % 5 == 0:
        import gc; gc.collect()
    dialog = PlotViewerDialog(parent, title=title, draw_callback=draw_callback, size=size)
    dialog.exec()


class PlotCard(Card):
    _draw_count = 0

    def __init__(
        self,
        title: str | None = None,
        *,
        height: int = 240,
        expand_title: str | None = None,
        expand_callback: Callable[[FigureCanvas], None] | None = None,
        expand_parent: QWidget | None = None,
    ) -> None:
        super().__init__(title=title)
        self.figure = Figure(figsize=(4, 2.4), dpi=100, facecolor="white")
        self._draw_count = 0
        self.canvas = FigureCanvas(self.figure)
        target_height = max(48, int(height))
        # Keep plot previews from expanding to Matplotlib's default 240 px
        # height.  This is critical for 15.6-inch 1080P laptops with the
        # Windows taskbar visible.  Larger plot cards can still request a
        # larger explicit height via the constructor.
        self.canvas.setMinimumHeight(target_height)
        self.canvas.setMaximumHeight(target_height)
        frame_height = target_height + (24 if title else 10)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # The parent PlotCard must never be compressed below its fixed-height
        # canvas.  Without an explicit minimum height, compact bottom rows can
        # report a card geometry smaller than its canvas, causing small map
        # previews to be clipped on Windows 1080P/125% displays.
        self.setMinimumHeight(frame_height)
        self.setMaximumHeight(frame_height)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if expand_callback is not None:
            btn = QPushButton("⛶")
            btn.setObjectName("smallButton")
            btn.setFixedWidth(28)
            btn.setToolTip(f"放大查看：{expand_title or title or '图像'}")
            btn.clicked.connect(
                lambda _checked=False: open_plot_viewer(
                    expand_parent or self.window(),
                    title=expand_title or title or "图像预览",
                    draw_callback=expand_callback,
                )
            )
            self.add_title_button(btn)
        # QVBoxLayout centers a small fixed-height canvas inside an expanding
        # card when no stretch item is present.  Keep preview plots anchored at
        # the top of their cards so compact Windows captures do not show large
        # blank bands above B-scan/map previews.
        self.layout.addWidget(self.canvas, 0, Qt.AlignmentFlag.AlignTop)
        self.layout.addStretch(1)


class CollapsibleSidePanel(QWidget):
    """Narrow collapsible container for secondary side-panel content."""

    def __init__(
        self,
        *,
        title: str,
        content: QWidget,
        expanded_width: int,
        collapsed_width: int = 34,
        initially_expanded: bool = True,
    ) -> None:
        super().__init__()
        self.setObjectName("collapsibleSidePanel")
        self.setProperty("layoutKey", f"{title}CollapsibleSidePanel")
        self.title = title
        self.content = content
        self.expanded_width = max(int(expanded_width), 120)
        self.collapsed_width = max(int(collapsed_width), 28)
        self._expanded = bool(initially_expanded)
        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("smallButton")
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.clicked.connect(self.toggle)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toggle_button)
        layout.addWidget(content, 1)
        self._apply_state()

    def is_expanded(self) -> bool:
        return self._expanded

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._apply_state()

    def _apply_state(self) -> None:
        self.content.setVisible(self._expanded)
        width = self.expanded_width if self._expanded else self.collapsed_width
        self.setMinimumWidth(width)
        self.setMaximumWidth(width)
        self.toggle_button.setFixedWidth(width)
        if self._expanded:
            self.toggle_button.setText(f"◀ 收起{self.title}")
            self.toggle_button.setToolTip(f"收起{self.title}，扩大主工作区")
        else:
            self.toggle_button.setText("▶")
            self.toggle_button.setToolTip(f"展开{self.title}")


__all__ = [
    "Card",
    "CollapsibleSidePanel",
    "MetricCard",
    "PlotCard",
    "PlotViewerDialog",
    "open_plot_viewer",
    "_set_shadow",
    "_spacer",
]
