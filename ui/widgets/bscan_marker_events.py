"""Event markers have along-line guides only; they never create depth picks."""
import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QColor
from qfluentwidgets import isDarkTheme

from ui.design_tokens import color
from ui.widgets.marker_strip import MarkerStrip


class BScanMarkerEventsMixin:
    def init_marker_events(self, layout):
        self._marker_view = None
        self._event_guides = []
        self.marker_strip = MarkerStrip(self)
        layout.addWidget(self.marker_strip)
        self.marker_strip.selected.connect(self.select_event_marker)

    def set_marker_view(self, view):
        self._marker_view = view
        self._clear_event_guides()
        self.marker_strip.set_rows(view.rows if view else ())
        if view is not None:
            self._marker_pins.hide()
        self.sig_marker_context_changed.emit()

    def _clear_event_guides(self):
        for item in self._event_guides:
            self._plot.removeItem(item)
        self._event_guides = []

    def select_event_marker(self, uid, *, locate=False):
        self.marker_strip.selected_id = uid
        self.marker_strip.update()
        self._clear_event_guides()
        view = self._marker_view
        row = next((item for item in view.rows if item['uid'] == uid), None) if view else None
        if row is None or row['record'].excluded:
            return
        projection = row['projection']
        ink = QColor(color('primary', isDarkTheme()))
        shade = QColor(ink)
        shade.setAlpha(24)
        for start, stop in projection.spans:
            a, b = self.marker_strip.display_x(start) - .5, self.marker_strip.display_x(stop - 1) + .5
            region = pg.LinearRegionItem((a, b), movable=False, brush=shade, pen=pg.mkPen(ink))
            region.setZValue(12)
            self._plot.addItem(region, ignoreBounds=True)
            self._event_guides.append(region)
        for point in projection.positions:
            guide = pg.InfiniteLine(self.marker_strip.display_x(point), angle=90, movable=False, pen=pg.mkPen(ink, width=2))
            self._plot.addItem(guide, ignoreBounds=True)
            self._event_guides.append(guide)
        points = projection.positions or tuple(a for a, _ in projection.spans)
        if locate and points:
            x = self.marker_strip.display_x(points[0])
            self._reveal_marker(x)
            self._sync_ascan_to_marker(int(np.floor(x)))
        self.sig_event_marker_selected.emit(uid)
