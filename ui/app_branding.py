# -*- coding: utf-8 -*-
"""Branding widgets and high-resolution Matplotlib toolbar for MyGPR.

This module keeps visual polish code out of the main window shell.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PyQt6.QtCore import Qt, QRect, QSize
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QFileDialog, QFrame, QMessageBox


class HiResNavigationToolbar(NavigationToolbar):
    """Matplotlib toolbar with theme-aware icons and high-resolution export.

    The stock NavigationToolbar save action uses Matplotlib defaults, which can
    produce soft PNGs on high-DPI screens.  MyGPR's B-scan export is evidence
    facing, so toolbar saves default to 600 DPI and tight bounds.  The toolbar
    icons are also redrawn with high-contrast Qt glyphs because the bundled
    Matplotlib icons can become very low contrast in the light qfluent theme.
    """

    EXPORT_DPI = 600

    _ACTION_SYMBOLS = {
        "home": "⌂",
        "back": "‹",
        "forward": "›",
        "pan": "✥",
        "zoom": "⌕",
        "subplots": "▦",
        "customize": "⚙",
        "save": "▣",
    }

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self._theme_key = "light"
        self.setIconSize(QSize(22, 22))
        self.apply_theme("light")

    def apply_theme(self, theme_key: str = "light") -> None:
        self._theme_key = "dark" if str(theme_key).lower() == "dark" else "light"
        fg = "#EAF0F8" if self._theme_key == "dark" else "#0F172A"
        disabled = "#7C8797" if self._theme_key == "dark" else "#64748B"
        for action in self.actions():
            if action.isSeparator():
                continue
            symbol = self._symbol_for_action(action)
            if not symbol:
                continue
            icon = QIcon()
            icon.addPixmap(self._make_symbol_pixmap(symbol, fg), QIcon.Mode.Normal)
            icon.addPixmap(self._make_symbol_pixmap(symbol, disabled), QIcon.Mode.Disabled)
            action.setIcon(icon)

    def save_figure(self, *args):  # noqa: D401 - Matplotlib toolbar override
        """Save the current B-scan figure at evidence-grade resolution."""
        default_path = str(Path.cwd() / "bscan_highres.png")
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "保存高清 B-scan 图像",
            default_path,
            "PNG 高清图像 (*.png);;PDF 矢量图 (*.pdf);;SVG 矢量图 (*.svg);;所有文件 (*)",
        )
        if not path:
            return
        out_path = Path(path)
        if not out_path.suffix:
            if "PDF" in selected_filter:
                out_path = out_path.with_suffix(".pdf")
            elif "SVG" in selected_filter:
                out_path = out_path.with_suffix(".svg")
            else:
                out_path = out_path.with_suffix(".png")
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.canvas.figure.savefig(
                str(out_path),
                dpi=self.EXPORT_DPI,
                bbox_inches="tight",
                facecolor=self.canvas.figure.get_facecolor(),
                edgecolor="none",
            )
        except Exception as exc:  # pragma: no cover - UI path
            QMessageBox.critical(self, "保存失败", f"高清 B-scan 图像保存失败：\n{exc}")
            return
        self.canvas.draw_idle()

    def _symbol_for_action(self, action) -> str | None:
        haystack = " ".join(
            part
            for part in [
                action.text() or "",
                action.toolTip() or "",
                action.statusTip() or "",
                action.iconText() or "",
            ]
            if part
        ).lower()
        for key, symbol in self._ACTION_SYMBOLS.items():
            if key in haystack:
                return symbol
        return None

    def _make_symbol_pixmap(self, symbol: str, color: str) -> QPixmap:
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setPen(QPen(QColor(color), 2))
        font = painter.font()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, symbol)
        painter.end()
        return pixmap


def make_mygpr_brand_pixmap(size: int = 64, *, dark: bool = False) -> QPixmap:
    """Create a small deterministic MyGPR brand mark without external assets."""
    size = max(32, int(size or 64))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    scale = size / 70.0

    def sx(value: float) -> int:
        return int(round(value * scale))

    rect = QRect(sx(5), sx(5), size - sx(10), size - sx(10))
    bg = QColor("#172A46" if dark else "#E7FAF7")
    border = QColor("#335B78" if dark else "#BDEBD1")
    primary = QColor("#93C5FD" if dark else "#13A6A4")
    muted = QColor("#A6B1C2" if dark else "#64748B")

    painter.setPen(QPen(border, max(1, sx(1.2))))
    painter.setBrush(bg)
    painter.drawRoundedRect(rect, sx(18), sx(18))

    cx = rect.center().x()
    top = rect.top() + sx(13)
    ground_y = rect.top() + sx(30)
    bottom = rect.bottom() - sx(11)

    painter.setPen(QPen(muted, max(1, sx(1.4))))
    painter.drawLine(rect.left() + sx(13), ground_y, rect.right() - sx(13), ground_y)

    painter.setPen(QPen(primary, max(1, sx(1.5))))
    painter.drawLine(cx, top, cx, ground_y - sx(3))
    for radius in (sx(12), sx(19)):
        wave_rect = QRect(cx - radius, ground_y - radius + sx(2), radius * 2, radius * 2)
        painter.drawArc(wave_rect, 205 * 16, 130 * 16)

    painter.setPen(QPen(primary, max(1, sx(1.7))))
    painter.drawArc(QRect(cx - sx(16), bottom - sx(15), sx(32), sx(18)), 0, 180 * 16)
    painter.setBrush(primary)
    painter.drawEllipse(cx - sx(3), bottom - sx(5), sx(6), sx(6))
    painter.end()
    return pixmap


class MyGPRMark(QFrame):
    """Small vector brand mark used by the empty state.

    It intentionally avoids external image assets: a ground line, a radar wave,
    and a buried target are enough to give MyGPR a stable visual signature while
    keeping the UI minimal.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MyGPRMark")
        self.setMinimumSize(70, 70)
        self.setMaximumSize(70, 70)

    def paintEvent(self, event):  # pragma: no cover - visual polish path
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(5, 5, -5, -5)
        is_dark = False
        try:
            from ui.theme import get_effective_theme_key
            from core.theme_manager import get_theme_manager

            is_dark = get_effective_theme_key(
                get_theme_manager().get_current_theme(), widget=self
            ) == "dark"
        except Exception:
            is_dark = False

        bg = QColor("#172A46" if is_dark else "#E7FAF7")
        border = QColor("#335B78" if is_dark else "#BDEBD1")
        primary = QColor("#93C5FD" if is_dark else "#13A6A4")
        muted = QColor("#A6B1C2" if is_dark else "#64748B")

        painter.setPen(QPen(border, 1.2))
        painter.setBrush(bg)
        painter.drawRoundedRect(rect, 18, 18)

        cx = rect.center().x()
        top = rect.top() + 13
        ground_y = rect.top() + 30
        bottom = rect.bottom() - 11

        painter.setPen(QPen(muted, 1.4))
        painter.drawLine(rect.left() + 13, ground_y, rect.right() - 13, ground_y)

        painter.setPen(QPen(primary, 1.5))
        painter.drawLine(cx, top, cx, ground_y - 3)
        for radius in (12, 19):
            wave_rect = QRect(cx - radius, ground_y - radius + 2, radius * 2, radius * 2)
            painter.drawArc(wave_rect, 205 * 16, 130 * 16)

        painter.setPen(QPen(primary, 1.7))
        painter.drawArc(QRect(cx - 16, bottom - 15, 32, 18), 0, 180 * 16)
        painter.setBrush(primary)
        painter.drawEllipse(cx - 3, bottom - 5, 6, 6)
        painter.end()
