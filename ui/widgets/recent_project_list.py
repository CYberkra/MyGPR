"""Recent project rows with real keyboard-accessible pin controls."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import TransparentToolButton, isDarkTheme
from qfluentwidgets import FluentIcon as FIF

from ui.design_tokens import color, radius
from ui.widgets.elided_caption import ElidedCaptionLabel


class RecentProjectRow(QWidget):
    selected = pyqtSignal(str)
    activated = pyqtSignal(str)
    pin_requested = pyqtSignal(str)

    def __init__(self, path: str, name: str, pinned: bool, parent=None):
        super().__init__(parent)
        self.path = path
        self._selected = False
        self._dark = isDarkTheme()
        self.setObjectName('recentProjectRow')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(0)
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 10, 10, 10)
        row.setSpacing(12)
        self.badge = QLabel((name[:1] or 'P').upper(), self)
        self.badge.setObjectName('recentBadge')
        self.badge.setFixedSize(38, 38)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.badge)
        text = QVBoxLayout()
        text.setSpacing(3)
        self.name_label = ElidedCaptionLabel(self)
        self.name_label.setObjectName('recentName')
        self.path_label = ElidedCaptionLabel(self)
        self.path_label.setObjectName('recentPath')
        for label, value in ((self.name_label, name), (self.path_label, path)):
            label.setText(value)
            label.setAccessibleName(value)
            text.addWidget(label)
        row.addLayout(text, 1)
        self.status = QLabel('不可用', self)
        self.status.setObjectName('recentStatus')
        self.status.hide()
        row.addWidget(self.status)
        self.pin_button = TransparentToolButton(FIF.PIN, self)
        self.pin_button.setFixedSize(32, 32)
        self.pin_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.pin_button.setCheckable(True)
        self.pin_button.setChecked(pinned)
        self.pin_button.setToolTip('取消固定' if pinned else '固定项目')
        self.pin_button.setAccessibleName(f'{self.pin_button.toolTip()}：{name}')
        self.pin_button.clicked.connect(lambda: self.pin_requested.emit(path))
        row.addWidget(self.pin_button)
        for label in (self.badge, self.name_label, self.path_label, self.status):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.apply_theme(self._dark)

    def set_selected(self, selected: bool) -> None:
        if self._selected == selected:
            return
        self._selected = selected
        self.apply_theme(self._dark)

    def set_error(self, error: str) -> None:
        self.status.setVisible(bool(error))
        self.setToolTip(f'{self.path}\n{error}' if error else self.path)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.path)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self.path)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def apply_theme(self, dark: bool) -> None:
        self._dark = dark
        # Fluent labels install their own QSS, which wins over ancestor QSS.
        for label, token, size, bold in (
                (self.name_label, 'text_primary', 15, True),
                (self.path_label, 'text_muted', 12, False)):
            label.setTextColor(color(token, False), color(token, True))
            font = label.font()
            font.setPixelSize(size)
            font.setBold(bold)
            label.setFont(font)
        bg = color('primary_subtle' if self._selected else 'bg_surface', dark)
        border = color('primary_border' if self._selected else 'bg_surface', dark)
        self.setStyleSheet(
            f'QWidget#recentProjectRow {{ background: {bg}; border: 1px solid {border};'
            f' border-radius: {radius("lg")}px; }}'
            f'QLabel {{ background: transparent; border: none; }}'
            f'#recentBadge {{ background: {color("bg_muted", dark)};'
            f' color: {color("text_secondary", dark)}; border-radius: 8px; font-size: 18px; }}'
            f'#recentName {{ color: {color("text_primary", dark)}; font-size: 15px; font-weight: 600; }}'
            f'#recentPath {{ color: {color("text_muted", dark)}; font-size: 12px; }}'
            f'#recentStatus {{ color: {color("warning", dark)}; font-size: 12px; }}')
