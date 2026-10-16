"""Single-line captions keep their full identity available while resizing."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy
from qfluentwidgets import CaptionLabel


class ElidedCaptionLabel(CaptionLabel):
    def __init__(self, parent=None):
        self._full_text = ''
        super().__init__(parent=parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def setText(self, text):
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self._refresh_text()

    def _refresh_text(self):
        super().setText(self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideMiddle, max(0, self.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_text()

    def sizeHint(self):
        size = super().sizeHint()
        size.setWidth(min(320, self.fontMetrics().horizontalAdvance(self._full_text)))
        return size

    def minimumSizeHint(self):
        size = super().minimumSizeHint()
        size.setWidth(0)
        return size
