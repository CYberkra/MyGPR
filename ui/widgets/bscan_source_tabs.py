"""Tab interaction shared by B-scan workspace groups."""
from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRect, Qt
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import QApplication, QTabBar, QWidget
from qfluentwidgets import FluentIcon as FIF, ToolButton
from ui import motion
from ui.widgets.context_menus import bind_menu_keys


class _SourceTabBar(QTabBar):
    """Real Qt drag source; the workspace handles tab/edge drop destinations."""

    def __init__(self, group):
        super().__init__(group)
        self.group = group
        self._press = None
        self._drag_key = ''
        self._hover_index = -1
        self.setObjectName('bscanWorkspaceTabs')
        self.setDrawBase(False)
        self.setExpanding(False)
        self.setElideMode(Qt.TextElideMode.ElideRight)
        self.setUsesScrollButtons(True)
        self.setTabsClosable(True)
        self.setFixedHeight(32)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName('B-scan 数据标签')
        self._menu_shortcuts = bind_menu_keys(self, self._show_keyboard_menu)
        self.indicator = QWidget(self)
        self.indicator.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.indicator.hide()
        self.indicator_animation = QPropertyAnimation(self.indicator, b'geometry', self)
        self.indicator_animation.setDuration(160)
        self.indicator_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.currentChanged.connect(self._current_tab_changed)
        self.tabMoved.connect(lambda *_args: self.sync_indicator(animate=False))

    def _current_tab_changed(self, index):
        self._sync_close_visibility()
        self.sync_indicator(index)

    def set_indicator_color(self, value):
        self.indicator.setStyleSheet(f'background:{value};')

    def sync_indicator(self, index=None, *, animate=True):
        index = self.currentIndex() if index is None else index
        rect = self.tabRect(index) if index >= 0 else QRect()
        self.indicator_animation.stop()
        if not rect.isValid():
            self.indicator.hide()
            return
        target = QRect(rect.left(), rect.bottom() - 1, rect.width(), 2)
        if (not animate or self.indicator.isHidden()
                or not motion.animations_enabled() or not self.isVisible()):
            self.indicator.setGeometry(target)
            self.indicator.show()
            self.indicator.raise_()
            return
        self.indicator_animation.setStartValue(self.indicator.geometry())
        self.indicator_animation.setEndValue(target)
        self.indicator_animation.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.sync_indicator(animate=False)

    def showEvent(self, event):
        super().showEvent(event)
        self.sync_indicator(animate=False)

    def sync_close_button(self, index, key, title):
        side = QTabBar.ButtonPosition.RightSide
        button = self.tabButton(index, side)
        if not isinstance(button, ToolButton) or button.property('source_key') != key:
            button = ToolButton(FIF.CLOSE, self)
            button.setProperty('source_key', key)
            button.setFixedSize(20, 20)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _checked=False, key=key: self._close_key(key))
            self.setTabButton(index, side, button)
        button.setToolTip(f'关闭 {title}')
        button.setAccessibleName(f'关闭 {title}')
        button.setVisible(key not in self.group.pinned
                          and index in (self.currentIndex(), self._hover_index))

    def _sync_close_visibility(self):
        for index, key in enumerate(self.group.keys()):
            self.sync_close_button(index, key, self.tabText(index))

    def leaveEvent(self, event):
        self._hover_index = -1
        self._sync_close_visibility()
        super().leaveEvent(event)

    def _close_key(self, key):
        index = self.group.index_of(key)
        if index >= 0:
            self.tabCloseRequested.emit(index)

    def mousePressEvent(self, event):
        index = self.tabAt(event.position().toPoint())
        if event.button() == Qt.MouseButton.RightButton:
            event.accept()  # Opening a context menu never selects its target.
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            if index >= 0:
                self.group.workspace.close_tab(self.tabData(index), self.group.group_id)
            event.accept()
            return
        super().mousePressEvent(event)
        self._press = event.position().toPoint()
        self._drag_key = str(self.tabData(index) or '') if index >= 0 else ''

    def mouseMoveEvent(self, event):
        hover = self.tabAt(event.position().toPoint())
        if hover != self._hover_index:
            self._hover_index = hover
            self._sync_close_visibility()
        if (self._press is not None and self._drag_key
                and event.buttons() & Qt.MouseButton.LeftButton
                and (event.position().toPoint() - self._press).manhattanLength()
                >= QApplication.startDragDistance()):
            key, self._drag_key = self._drag_key, ''
            self._press = None
            drag = QDrag(self)
            drag.setMimeData(self.group.workspace.drag_mime_data(key, self.group.group_id))
            index = self.group.index_of(key)
            if index >= 0:
                drag.setPixmap(self.grab(self.tabRect(index)))
            drag.exec(Qt.DropAction.MoveAction)
            self.group.workspace.cancel_drag()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._press = None
        self._drag_key = ''
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        if event.reason() == event.Reason.Keyboard:
            self._show_keyboard_menu()
            event.accept()
            return
        index = self.tabAt(event.pos())
        key = str(self.tabData(index) or '') if index >= 0 else ''
        self.group.workspace.show_context_menu(self.group.group_id, key, event.globalPos())
        event.accept()

    def _show_keyboard_menu(self):
        index = self.currentIndex()
        key = str(self.tabData(index) or '') if index >= 0 else ''
        point = self.tabRect(index).bottomLeft() if index >= 0 else self.rect().bottomLeft()
        self.group.workspace.show_context_menu(self.group.group_id, key, self.mapToGlobal(point))
