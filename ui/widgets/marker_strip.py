"""A two-lane, pixel-clustered annotation strip sharing the B-scan transform."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget
from qfluentwidgets import isDarkTheme

from ui.design_tokens import color


class MarkerStrip(QWidget):
    selected = pyqtSignal(str)
    edit_requested = pyqtSignal(str)
    range_created = pyqtSignal(int, int, str)

    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.rows = ()
        self.selected_id = ''
        self._hits = []
        self._creating = False
        self._first = None
        self._adjusting = ''
        self._drag_side = None
        self.setFixedHeight(56)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName('测线标记条')
        self.setMouseTracking(True)
        view._plot.vb.sigRangeChanged.connect(lambda *_: self.update())
        view._plot.vb.sigResized.connect(lambda *_: self.update())
        self.hide()

    def set_rows(self, rows):
        self.rows = tuple(rows)
        self._first = None
        self._creating = False
        self._adjusting = ''
        self.setVisible(bool(rows) and self.view._acquisition_markers_visible)
        self.update()

    def begin_range(self):
        self._creating = True
        self._first = None
        self.show()
        self.setFocus()
        self.setToolTip('依次点击起止边界；Esc 取消')
        self.update()

    def begin_adjust(self, uid):
        self._adjusting = uid
        self.selected_id = uid
        self.show()
        self.setFocus()
        self.setToolTip('拖动选中区间边界；Esc 取消')
        self.update()

    def display_x(self, trace):
        grid = self.view._trace_indices
        if grid is None:
            return trace + .5
        return float(np.interp(trace, grid, np.arange(len(grid)))) + .5

    def pixel_x(self, trace):
        scene = self.view._plot.vb.mapViewToScene(QPointF(self.display_x(trace), 0))
        point = self.view._glw.mapFromScene(scene)
        return float(self._from_viewport(point).x())

    def _from_viewport(self, point):
        viewport = self.view._glw.viewport()
        return self.mapFromGlobal(viewport.mapToGlobal(point))

    def trace_at(self, pixel):
        point = self.view._glw.viewport().mapFromGlobal(self.mapToGlobal(pixel))
        scene = self.view._glw.mapToScene(point)
        x = self.view._plot.vb.mapSceneToView(scene).x() - .5
        grid = self.view._trace_indices
        value = np.interp(x, np.arange(len(grid)), grid) if grid is not None else x
        count = getattr(self.view, '_marker_view', None)
        return max(0, min((count.count if count else self.view._trace_count) - 1, int(round(value))))

    def paintEvent(self, event):
        painter = QPainter(self)
        dark = isDarkTheme()
        painter.fillRect(self.rect(), QColor(color('bg_surface', dark)))
        bounds = self.view._plot.vb.sceneBoundingRect()
        left = self._from_viewport(self.view._glw.mapFromScene(bounds.topLeft())).x()
        right = self._from_viewport(self.view._glw.mapFromScene(bounds.topRight())).x()
        painter.setClipRect(QRectF(left, 0, max(0, right - left), self.height()))
        painter.setPen(QPen(QColor(color('border_default', dark))))
        painter.drawLine(left, 0, right, 0)
        self._hits = []
        items = []
        for row in self.rows:
            if row['record'].excluded:
                continue
            projection = row['projection']
            spans = projection.spans or tuple((point, point) for point in projection.positions)
            for start, stop in spans:
                x1, x2 = self.pixel_x(start), self.pixel_x(max(start, stop - 1))
                lo, hi = min(x1, x2), max(x1, x2)
                if hi < left or lo > right:
                    continue
                items.append((lo, hi, row, bool(projection.spans)))
        # Selected records claim a lane first. Remaining collisions become count controls.
        items.sort(key=lambda item: (item[2]['uid'] != self.selected_id, item[0]))
        lanes = [[], []]
        clusters = {}
        for x1, x2, row, interval in items:
            lo, hi = max(float(left), x1), min(float(right), x2)
            width = max(24., min(100., right - lo), hi - lo)
            lane = next((i for i in range(2) if all(lo > end or lo + width < begin for begin, end in lanes[i])), None)
            if lane is None:
                cell = int((lo - left) // 32)
                clusters.setdefault(cell, []).append(row['uid'])
                continue
            lanes[lane].append((lo, lo + width))
            rect = QRectF(lo, 3 + lane * 26, width, 23)
            self._hits.append((rect, [row['uid']]))
            token = 'primary' if row['uid'] == self.selected_id else {
                '停车等待': 'warning', '外部干扰': 'error', '目标范围': 'success',
                '现场操作': 'info'}.get(row['purpose'], 'text_secondary')
            ink = QColor(color(token, dark))
            painter.setPen(QPen(ink, 2 if row['uid'] == self.selected_id else 1))
            y = rect.center().y()
            painter.drawLine(QPointF(x1, y - 7), QPointF(x1, y + 7))
            if interval:
                painter.drawLine(QPointF(x2, y - 7), QPointF(x2, y + 7))
                painter.drawLine(QPointF(x1, y + 7), QPointF(x2, y + 7))
            elif row['projection'].status == 'open':
                painter.drawText(QPointF(lo + 3, y + 4), '…')
            painter.setPen(QColor(color('text_primary', dark)))
            label = painter.fontMetrics().elidedText(row['name'], Qt.TextElideMode.ElideRight, int(width - 8))
            painter.drawText(rect.adjusted(5, 0, 0, -2), Qt.AlignmentFlag.AlignVCenter, label)
        for cell, ids in clusters.items():
            rect = QRectF(left + cell * 32, 30, 32, 23)
            # Include the lane members covered by the count control; none may become unreachable.
            covered = [(hit, members) for hit, members in self._hits if hit.intersects(rect)]
            ids = list(dict.fromkeys([*ids, *(uid for _, members in covered for uid in members)]))
            self._hits = [(hit, members) for hit, members in self._hits if not hit.intersects(rect)]
            self._hits.insert(0, (rect, ids))
            painter.fillRect(rect, QColor(color('bg_muted', dark)))
            painter.setPen(QColor(color('text_primary', dark)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f'+{len(ids)}')
        if self._creating:
            painter.setPen(QColor(color('primary', dark)))
            painter.drawText(left + 6, 18, '选择起点' if self._first is None else '选择终点')

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._creating:
            trace = self.trace_at(event.position().toPoint())
            if self._first is None:
                self._first = trace
            else:
                first = self._first
                self._creating, self._first = False, None
                self.range_created.emit(min(first, trace), max(first, trace), '')
            self.update()
            return
        if self._adjusting:
            row = next((item for item in self.rows if item['uid'] == self._adjusting), None)
            if row and row['projection'].spans:
                start, stop = row['projection'].spans[0][0], row['projection'].spans[-1][1] - 1
                self._drag_side = (start, stop, abs(event.position().x() - self.pixel_x(start))
                                  <= abs(event.position().x() - self.pixel_x(stop)))
                return
        for rect, ids in self._hits:
            if not rect.contains(event.position()):
                continue
            if len(ids) == 1:
                self.selected.emit(ids[0])
            else:
                from ui.widgets.marker_cluster import MarkerCluster
                lookup = {row['uid']: row for row in self.rows}
                self._cluster = MarkerCluster(tuple(lookup[uid] for uid in ids), self)
                self._cluster.selected.connect(self.selected)
                self._cluster.show()
            return

    def mouseReleaseEvent(self, event):
        if self._drag_side:
            start, end, first = self._drag_side
            value = self.trace_at(event.position().toPoint())
            start, end = (value, end) if first else (start, value)
            uid = self._adjusting
            self._drag_side, self._adjusting = None, ''
            self.range_created.emit(min(start, end), max(start, end), uid)

    def mouseDoubleClickEvent(self, event):
        for rect, ids in self._hits:
            if rect.contains(event.position()) and len(ids) == 1:
                self.edit_requested.emit(ids[0])
                return

    def mouseMoveEvent(self, event):
        if self._creating or self._adjusting:
            return
        for rect, ids in self._hits:
            if rect.contains(event.position()):
                row = next((row for row in self.rows if row['uid'] == ids[0]), None)
                self.setToolTip(f'{len(ids)} 条记录，点击展开' if len(ids) > 1 else
                               f"{row['name']} · {row['position']} · {row['status']}")
                return
        self.setToolTip('')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._creating, self._first, self._adjusting, self._drag_side = False, None, '', None
            self.setVisible(bool(self.rows) and self.view._acquisition_markers_visible)
            self.update()
            event.accept()
        else:
            super().keyPressEvent(event)
