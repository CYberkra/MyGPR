"""One-shot, display-only rectangle zoom and local display-level scope."""
import math
import weakref

import numpy as np
from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, Qt, QTimer
from PyQt6.QtWidgets import QLabel, QRubberBand


class RectangleSelector(QObject):
    def __init__(self, view):
        super().__init__(view)
        self._view = weakref.ref(view)
        self.viewport = view._glw.viewport()
        self.viewport.installEventFilter(self)
        view._glw.installEventFilter(self)
        view.installEventFilter(self)
        self.active = False
        self.origin = None
        self._suppressed_releases = set()
        self.rubber = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport)
        self.hint = QLabel('框选查看 · 拖动鼠标，Esc / 右键取消', self.viewport)
        self.hint.setWordWrap(True)
        self.hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hint.setStyleSheet('QLabel { background: rgba(0,0,0,170); color: white; '
                               'padding: 5px 8px; border-radius: 3px; font-size: 9pt; }')
        self.hint.hide()

    def activate(self):
        self.cancel()
        self.active = True
        self._old_cursor = self.viewport.cursor()
        self.viewport.setCursor(Qt.CursorShape.CrossCursor)
        self.viewport.setFocus(Qt.FocusReason.OtherFocusReason)
        self._view()._hide_crosshair()
        rect = self._plot_rect()
        width = max(100, rect.width() - 16)
        self.hint.setMinimumWidth(min(280, width))
        self.hint.setMaximumWidth(width)
        self.hint.adjustSize()
        self.hint.move(rect.topLeft() + QPoint(8, 8))
        self.hint.show()
        self.hint.raise_()

    def cancel(self):
        if self.active:
            self.viewport.setCursor(self._old_cursor)
        self.active = False
        self.origin = None
        self.rubber.hide()
        self.hint.hide()

    def _plot_rect(self):
        view = self._view()
        return view._glw.mapFromScene(view._plot.vb.sceneBoundingRect()).boundingRect().intersected(self.viewport.rect())

    def _clamped(self, point):
        rect = self._plot_rect()
        return QPoint(min(rect.right(), max(rect.left(), point.x())),
                      min(rect.bottom(), max(rect.top(), point.y())))

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.Type.MouseButtonRelease and event.button() in self._suppressed_releases:
            self._suppressed_releases.discard(event.button())
            return True
        if not self.active:
            return False
        if kind == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            if self.origin is not None:
                self._suppressed_releases.add(Qt.MouseButton.LeftButton)
            self.cancel()
            return True
        if kind in (QEvent.Type.Hide, QEvent.Type.WindowDeactivate):
            self.cancel()
            return False
        if watched is not self.viewport:
            return False
        if kind == QEvent.Type.MouseButtonPress:
            return self._press(event)
        if kind == QEvent.Type.MouseMove:
            if self.origin is not None:
                self.rubber.setGeometry(QRect(self.origin, self._clamped(event.position().toPoint())).normalized())
                self.rubber.show()
            return True
        if kind == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            self._finish(event.position().toPoint())
            return True
        return False

    def _press(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._suppressed_releases.add(Qt.MouseButton.RightButton)
            if self.origin is not None:
                self._suppressed_releases.add(Qt.MouseButton.LeftButton)
            self.cancel()
            return True
        if event.button() == Qt.MouseButton.LeftButton:
            point = event.position().toPoint()
            if not self._plot_rect().contains(point):
                return True
            self.origin = point
            self.rubber.setGeometry(QRect(point, point))
            return True
        return True

    def _finish(self, end):
        start = self.origin
        end = self._clamped(end)
        self.cancel()
        if start is None or abs(end.x() - start.x()) < 5 or abs(end.y() - start.y()) < 5:
            return
        view = self._view()
        first = view._plot.vb.mapSceneToView(view._glw.mapToScene(start))
        last = view._plot.vb.mapSceneToView(view._glw.mapToScene(end))
        view.set_view_selection(QRectF(first, last).normalized())


class BScanSelectionMixin:
    def _init_selection(self):
        self._selection_tool = RectangleSelector(self)
        self._selection_timer = QTimer(self)
        self._selection_timer.setSingleShot(True)
        self._selection_timer.setInterval(40)
        self._selection_timer.timeout.connect(self._refresh_selection_levels)
        self._plot.vb.sigRangeChanged.connect(self._queue_selection_levels)

    def start_rectangle_zoom(self):
        if self._matrix is None or self._image_shape is None:
            return False
        self._selection_tool.activate()
        return True

    def _clear_view_selection(self):
        self._view_time_window = None
        self._view_selection = None
        if hasattr(self, '_selection_tool'):
            self._selection_tool.cancel()
            self._selection_timer.stop()

    def set_view_selection(self, rectangle):
        if self._image_shape is None or not isinstance(rectangle, QRectF):
            return False
        if not all(math.isfinite(v) for v in rectangle.getRect()):
            return False
        rectangle = rectangle.normalized().intersected(QRectF(0, 0, *self._image_shape))
        if rectangle.width() <= 1e-9 or rectangle.height() <= 1e-9:
            return False
        self._selection_tool.cancel()
        self._view_selection = rectangle
        self._restore_rectangle_view()
        self._update_selection_time()
        self._apply_levels_to_render()
        return True

    def _restore_rectangle_view(self):
        if self._view_selection is None:
            return False
        self._plot.vb.setAspectLocked(False)
        self._plot.vb.setRange(self._view_selection, padding=0)
        return True

    def _update_selection_time(self):
        axis = self._time_axis_for_view()
        self._view_time_window = None if axis is None else tuple(np.interp(
            self._plot.vb.viewRange()[1], np.arange(len(axis)) + .5, axis))

    def _queue_selection_levels(self, *_args):
        if self._view_selection is not None or self._view_time_window is not None:
            self._selection_timer.start()

    def _refresh_selection_levels(self):
        if self._view_selection is None and self._view_time_window is None:
            return
        if self._view_selection is not None:
            (left, right), (top, bottom) = self._plot.vb.viewRange()
            self._view_selection = QRectF(left, top, right-left, bottom-top)
        self._update_selection_time()
        self._apply_levels_to_render()

    def _levels_source_for_view(self, full_source):
        if (not self._selection_follows_levels
                or (self._view_selection is None and self._view_time_window is None)):
            return full_source
        source = self._image_item.image
        if source is None:
            return full_source
        (left, right), (top, bottom) = self._plot.vb.viewRange()
        height, width = source.shape
        x0, x1 = max(0, math.floor(left+1e-9)), min(width, math.ceil(right-1e-9))
        y0, y1 = max(0, math.floor(top+1e-9)), min(height, math.ceil(bottom-1e-9))
        return source[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else np.empty((0, 0))

    def _toggle_selection_levels(self, checked):
        self._selection_follows_levels = bool(checked)
        self._apply_levels_to_render()

    def _add_menu_selection(self, menu):
        from qfluentwidgets import Action
        from ui.widgets.context_menus import add_action
        add_action(menu, None, '框选放大', self.start_rectangle_zoom, enabled=self._matrix is not None)
        fixed = self._levels_override is not None
        label = '色阶跟随查看范围' if not fixed else '色阶跟随查看范围（统一色阶已锁定）'
        action = Action(label)
        action.setCheckable(True)
        action.setChecked(self._selection_follows_levels and not fixed)
        action.setEnabled(not fixed)
        action.triggered.connect(self._toggle_selection_levels)
        menu.addAction(action)
