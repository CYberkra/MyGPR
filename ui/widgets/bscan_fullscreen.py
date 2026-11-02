"""Per-view, interruptible fullscreen transitions with an exact origin slot."""
from __future__ import annotations

from PyQt6.QtCore import (
    QByteArray, QEasingCurve, QPoint, QPropertyAnimation, QRect, Qt, pyqtSignal,
)
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QBoxLayout, QVBoxLayout, QWidget

from ui.motion import animations_enabled

__all__ = ['FullscreenHost']


class FullscreenHost(QWidget):
    closed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle('B-Scan 全屏浏览（Esc 返回）')
        self.state = 'embedded'
        self._target_state = ''
        self._guest: QWidget | None = None
        self._guest_parent: QWidget | None = None
        self._guest_layout: QBoxLayout | None = None
        self._guest_index = -1
        self._guest_stretch = 0
        self._placeholder: QWidget | None = None
        self._origin_rect = QRect()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._animation = QPropertyAnimation(self, b'geometry', self)
        self._animation.setDuration(220)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.finished.connect(self._finish_animation)
        self._escape_shortcut = QShortcut(QKeySequence('Escape'), self)
        self._escape_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._escape_shortcut.activated.connect(self._on_escape)

    def take(self, guest: QWidget) -> None:
        if self._guest is not None:
            self.release()
        self._guest = guest
        self._guest_parent = guest.parentWidget()
        self._guest_layout = None
        self._guest_index = -1
        self._guest_stretch = 0
        self._origin_rect = QRect(guest.mapToGlobal(QPoint(0, 0)), guest.size())
        parent = self._guest_parent
        layout = parent.layout() if parent is not None else None
        if isinstance(layout, QBoxLayout):
            index = layout.indexOf(guest)
            if index >= 0:
                self._guest_layout = layout
                self._guest_index = index
                self._guest_stretch = layout.stretch(index)
                self._placeholder = QWidget(parent)
                self._placeholder.setObjectName('bscanFullscreenPlaceholder')
                self._placeholder.setSizePolicy(guest.sizePolicy())
                self._placeholder.setMinimumSize(guest.minimumSize())
                layout.removeWidget(guest)
                layout.insertWidget(index, self._placeholder, self._guest_stretch)
                self._placeholder.show()
        self.layout().addWidget(guest)
        self.setGeometry(self._origin_rect)
        guest.show()
        guest.destroyed.connect(self._on_guest_destroyed)

    def _on_guest_destroyed(self):
        self._animation.stop()
        self._guest = None
        if self._placeholder is not None:
            self._placeholder.deleteLater()
            self._placeholder = None
        self.state = 'embedded'
        self.hide()
        self.deleteLater()

    def release(self) -> None:
        self._animation.stop()
        self.state = 'embedded'
        self._target_state = ''
        guest = self._guest
        if guest is not None:
            self.layout().removeWidget(guest)
            guest.setParent(self._guest_parent)
            if self._placeholder is not None and self._guest_layout is not None:
                self._guest_layout.removeWidget(self._placeholder)
                self._placeholder.deleteLater()
                self._placeholder = None
            if self._guest_layout is not None and self._guest_index >= 0:
                self._guest_layout.insertWidget(self._guest_index, guest, self._guest_stretch)
            guest.show()
        self._guest = None
        self._guest_layout = None
        self._guest_index = -1
        self.hide()

    def expand(self) -> None:
        if self._guest is None or self.state in ('fullscreen', 'expanding'):
            return
        self._animation.stop()
        screen = self._guest.screen() or self.screen()
        target = screen.availableGeometry() if screen is not None else self._origin_rect
        self.state = 'expanding'
        self.show()
        self.raise_()
        self.activateWindow()
        self._animate_to(target, 'fullscreen')

    def collapse(self) -> None:
        if self._guest is None or self.state in ('embedded', 'collapsing'):
            return
        self._animation.stop()
        self.state = 'collapsing'
        self._animate_to(self._embedded_rect(), 'embedded')

    def settle(self) -> None:
        self.release()
        self.closed.emit()

    def _embedded_rect(self) -> QRect:
        if self._placeholder is not None:
            return QRect(self._placeholder.mapToGlobal(QPoint(0, 0)),
                         self._placeholder.size())
        return self._origin_rect

    def _animate_to(self, target: QRect, state: str) -> None:
        self._target_state = state
        if not animations_enabled():
            self.setGeometry(target)
            self._finish_animation()
            return
        self._animation.stop()
        self._animation.setStartValue(self.geometry())
        self._animation.setEndValue(target)
        self._animation.start()

    def _finish_animation(self) -> None:
        if self._target_state == 'fullscreen':
            self.state = 'fullscreen'
        else:
            self.settle()

    def captured_geometry(self) -> QByteArray:
        return self.saveGeometry()

    def restore_saved_geometry(self, data: QByteArray | bytes | None) -> bool:
        if data is None:
            return False
        blob = data if isinstance(data, QByteArray) else QByteArray(data)
        if blob.isEmpty():
            return False
        return bool(self.restoreGeometry(blob))

    def _on_escape(self) -> None:
        selector = getattr(self._guest, '_selection_tool', None)
        if selector is not None and selector.active:
            selector.cancel()
        else:
            self.collapse()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._on_escape()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.settle()
        event.accept()
