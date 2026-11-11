"""A two-lane, pixel-clustered annotation strip sharing the B-scan transform."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget
from qfluentwidgets import isDarkTheme

from ui.constants import FONT_SIZE_SECONDARY
from ui.design_tokens import tokens
from ui.theme_helpers import ui_font


_PURPOSE_COLOR = {
    '停车等待': 'warning', '外部干扰': 'error', '目标范围': 'success', '现场操作': 'info',
}
_LABEL_INSET = 21


def _draw_pill(painter, rect, fill, border):
    painter.setBrush(QColor(fill))
    painter.setPen(QPen(QColor(border)))
    radius = rect.height() / 2
    painter.drawRoundedRect(rect.adjusted(.5, .5, -.5, -.5), radius, radius)
    painter.setBrush(Qt.BrushStyle.NoBrush)


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

    def pixel_x(self, trace, *, offset=0.):
        scene = self.view._plot.vb.mapViewToScene(QPointF(self.display_x(trace) + offset, 0))
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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(ui_font(FONT_SIZE_SECONDARY))
        metrics = painter.fontMetrics()
        theme = tokens(isDarkTheme())
        painter.fillRect(self.rect(), QColor(theme['bg_surface']))
        bounds = self.view._plot.vb.sceneBoundingRect()
        self._hits = []
        left = max(0, self._from_viewport(self.view._glw.mapFromScene(bounds.topLeft())).x())
        right = min(self.width(), self._from_viewport(self.view._glw.mapFromScene(bounds.topRight())).x())
        if right <= left:
            return
        painter.setClipRect(QRectF(left, 0, right - left, self.height()))
        painter.setPen(QPen(QColor(theme['border_default'])))
        painter.drawLine(left, 0, right, 0)
        items = []
        for row in self.rows:
            if row['record'].excluded:
                continue
            projection = row['projection']
            spans = projection.spans or tuple((point, point) for point in projection.positions)
            edge = .5 if projection.spans else 0.
            for start, stop in spans:
                x1 = self.pixel_x(start, offset=-edge)
                x2 = self.pixel_x(max(start, stop - 1), offset=edge)
                lo, hi = min(x1, x2), max(x1, x2)
                if hi < left or lo > right:
                    continue
                items.append((lo, hi, row, bool(projection.spans)))
        # Selected records claim a lane first. Remaining collisions become count controls.
        items.sort(key=lambda item: (item[2]['uid'] != self.selected_id, item[0]))
        lanes = [[], []]
        clusters = {}
        entries = []
        for x1, x2, row, interval in items:
            lo = max(float(left), x1 if interval else x1 - 10)
            label_width = min(152, max(36, metrics.horizontalAdvance(row['name']) + _LABEL_INSET + 8))
            width = min(right - left, max(label_width, min(float(right), x2) - lo))
            lo = min(lo, right - width)
            lane = next((i for i in range(2) if all(
                lo >= end + 3 or lo + width + 3 <= begin for begin, end in lanes[i])), None)
            if lane is None:
                cell = int((lo - left) // 56)
                clusters.setdefault(cell, []).append(row['uid'])
                continue
            lanes[lane].append((lo, lo + width))
            rect = QRectF(lo, 3 + lane * 26, width, 23)
            entries.append((rect, x1, x2, row, interval))
        # Reserve cluster space before drawing labels, so a count never obscures
        # half a name. The selected record owns lane zero and remains visible.
        for cell, ids in sorted(clusters.items()):
            width = min(52, right - left)
            rect = QRectF(min(left + cell * 56, right - width), 30, width, 22)
            overlapping = [(hit, members) for hit, members in self._hits if hit.intersects(rect)]
            for hit, members in overlapping:
                rect = rect.united(hit)
                ids = [*ids, *members]
            self._hits = [(hit, members) for hit, members in self._hits if not hit.intersects(rect)]
            covered = [entry for entry in entries if entry[0].intersects(rect)]
            ids = list(dict.fromkeys([*ids, *(entry[3]['uid'] for entry in covered)]))
            entries = [entry for entry in entries if not entry[0].intersects(rect)]
            self._hits.append((rect, ids))
        cluster_count = len(self._hits)
        for rect, x1, x2, row, interval in entries:
            self._hits.append((rect, [row['uid']]))
            self._draw_marker(painter, theme, rect, x1, x2, row, interval)
        for rect, ids in self._hits[:cluster_count]:
            _draw_pill(painter, rect, theme['bg_muted'], theme['border_default'])
            painter.setPen(QColor(theme['text_secondary']))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f'+{len(ids)}')
        if self._creating:
            label = '选择起点' if self._first is None else '选择终点'
            rect = QRectF(left, 3, min(right - left, metrics.horizontalAdvance(label) + 20), 20)
            _draw_pill(painter, rect, theme['primary_subtle'], theme['primary_border'])
            painter.setPen(QColor(theme['text_primary']))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_marker(self, painter, theme, rect, x1, x2, row, interval):
        selected = row['uid'] == self.selected_id
        ink = QColor(theme['primary' if selected else _PURPOSE_COLOR.get(row['purpose'], 'text_secondary')])
        pill = rect.adjusted(0, 0, 0, -3)
        fill = theme['primary_subtle' if selected else 'bg_subtle']
        border = theme['primary_border' if selected else 'border_default']
        _draw_pill(painter, pill, fill, border)
        x, y = pill.left() + 11, pill.center().y()
        painter.setPen(QPen(ink, 1.5, cap=Qt.PenCapStyle.RoundCap))
        if interval:
            painter.drawLine(QPointF(x - 4, y), QPointF(x + 4, y))
            rail_y = rect.bottom() - .5
            painter.drawLine(QPointF(x1, rail_y), QPointF(x2, rail_y))
            for endpoint in (x1, x2):
                painter.drawLine(QPointF(endpoint, rail_y - 3), QPointF(endpoint, rail_y))
        else:
            painter.drawLine(QPointF(x1, pill.top() - 2), QPointF(x1, pill.top()))
            if row['projection'].status == 'open':
                painter.drawText(QRectF(x - 6, pill.top(), 12, pill.height()), Qt.AlignmentFlag.AlignCenter, '…')
            else:
                painter.setBrush(ink)
                painter.drawEllipse(QPointF(x, y), 2.5, 2.5)
                painter.setBrush(Qt.BrushStyle.NoBrush)
        text_rect = pill.adjusted(_LABEL_INSET, 0, -8, 0)
        if text_rect.width() > 0:
            label = painter.fontMetrics().elidedText(row['name'], Qt.TextElideMode.ElideRight, int(text_rect.width()))
            painter.setPen(QColor(theme['text_primary']))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, label)

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
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                row = next((row for row in self.rows if row['uid'] == ids[0]), None)
                self.setToolTip(f'{len(ids)} 条记录，点击展开' if len(ids) > 1 else
                               f"{row['name']} · {row['position']} · {row['status']}")
                return
        self.setToolTip('')
        self.unsetCursor()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._creating, self._first, self._adjusting, self._drag_side = False, None, '', None
            self.setVisible(bool(self.rows) and self.view._acquisition_markers_visible)
            self.update()
            event.accept()
        else:
            super().keyPressEvent(event)
