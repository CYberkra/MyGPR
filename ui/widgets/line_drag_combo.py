"""A normal Fluent combo whose popup rows can also be dragged as references."""
from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import ComboBox


class _LineDragFilter(QObject):
    def __init__(self, combo, menu, rows):
        super().__init__(menu)
        self.combo, self.menu, self.rows = combo, menu, rows
        self.press, self.row = None, -1
        menu.view.viewport().installEventFilter(self)

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self.press = event.position().toPoint()
            self.row = self.menu.view.indexAt(self.press).row()
        elif kind == QEvent.Type.MouseButtonRelease:
            self.press, self.row = None, -1
        elif (kind == QEvent.Type.MouseMove and self.press is not None
                and event.buttons() & Qt.MouseButton.LeftButton
                and (event.position().toPoint()-self.press).manhattanLength() >= QApplication.startDragDistance()):
            return self._drag()
        return False

    def _drag(self):
        row, self.row, self.press = self.row, -1, None
        if not 0 <= row < len(self.rows):
            return False
        mime = self.rows[row]
        if not mime.formats():
            return False
        pixmap = self.menu.view.viewport().grab(self.menu.view.visualItemRect(self.menu.view.item(row)))
        drag = QDrag(self.combo)
        drag.setMimeData(mime)
        drag.setPixmap(pixmap)
        # hide() releases the popup grab without WA_DeleteOnClose destroying the
        # list/filter during QDrag's nested event loop. Close after exec returns.
        self.menu.hide()
        drag.exec(Qt.DropAction.CopyAction)
        self.combo.drag_finished.emit()
        self.menu.close()
        return True


class LineDragComboBox(ComboBox):
    drag_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mime_factory = None
        self.setToolTip('单击切换测线；按住列表中的测线拖到 B 扫区域可添加对比')
        self.setAccessibleName('处理测线；列表支持拖拽对比')

    def _createComboMenu(self):
        menu = super()._createComboMenu()
        menu.destroyed.connect(lambda: self._forget_menu(menu))
        if self.mime_factory is not None:
            rows = [self.mime_factory(str(self.itemData(i) or '')) for i in range(self.count())]
            menu._line_drag_filter = _LineDragFilter(self, menu, rows)
        return menu

    def _forget_menu(self, menu):
        # Fluent keeps the popup during Windows' outside-click handling. Once
        # deleted, it must never be reused when this combo is opened again.
        if self.dropMenu is menu:
            self.dropMenu = None
