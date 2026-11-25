"""Tab interaction shared by B-scan workspace groups."""
from PyQt6.QtCore import QAbstractAnimation, QEasingCurve, QPropertyAnimation, QRect, Qt, QVariantAnimation
from PyQt6.QtGui import QColor, QDrag, QPainter
from PyQt6.QtWidgets import QApplication, QLabel, QTabBar, QWidget
from qfluentwidgets import FluentIcon as FIF, ToolButton
from ui import motion
from ui.design_tokens import duration
from ui.widgets.context_menus import bind_menu_keys


class _DropHint(QLabel):
    """Paint the drop outline and action immediately; fade only the fill."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('bscanDropHint')
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._fill = QColor()
        self._opacity = 1.0
        self.animation = QVariantAnimation(self)
        self.animation.setDuration(duration('drop_hint'))
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.valueChanged.connect(self._set_opacity)
        self.hide()

    def set_fill(self, color):
        self._fill = QColor(color)
        self._fill.setAlphaF(0.16)
        self.update()

    def _set_opacity(self, value):
        self._opacity = value
        self.update()

    def paintEvent(self, event):
        with QPainter(self) as painter:
            painter.setOpacity(self._opacity)
            painter.fillRect(event.rect(), self._fill)
        super().paintEvent(event)

    def reveal(self):
        self.show()
        self.raise_()
        if self.isVisible() and motion.animations_enabled():
            self._set_opacity(0.0)
            self.animation.start()
        else:
            self.settle_motion()

    def dismiss(self):
        self.animation.stop()
        self.hide()

    def settle_motion(self):
        self.animation.stop()
        self._set_opacity(1.0)


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
        self.indicator_animation.setDuration(duration('data_tab'))
        self.indicator_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.tabMoved.connect(lambda *_args: self.settle_motion())

    def set_indicator_color(self, value):
        self.indicator.setStyleSheet(f'background:{value};')

    def sync_indicator(self, *, animate=False):
        index = self.currentIndex()
        rect = self.tabRect(index) if index >= 0 else QRect()
        animation = self.indicator_animation
        if not rect.isValid():
            animation.stop()
            self.indicator.hide()
            return
        target = QRect(rect.left(), rect.bottom() - 1, rect.width(), 2)
        if self.indicator.isVisible() and motion.animations_enabled():
            # Catalog refreshes can arrive while the selected source is loading.
            if animation.state() == QAbstractAnimation.State.Running and animation.endValue() == target:
                return
            if animate and self.indicator.geometry() != target:
                animation.stop()
                animation.setStartValue(self.indicator.geometry())
                animation.setEndValue(target)
                animation.start()
                return
        animation.stop()
        self.indicator.setGeometry(target)
        self.indicator.show()
        self.indicator.raise_()

    def settle_motion(self):
        self.indicator_animation.stop()
        self.sync_indicator()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.settle_motion()

    def showEvent(self, event):
        super().showEvent(event)
        self.settle_motion()

    def hideEvent(self, event):
        self.indicator_animation.stop()
        super().hideEvent(event)

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
