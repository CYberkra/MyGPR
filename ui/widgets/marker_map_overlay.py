"""Reference points and disjoint trajectory fragments on the existing map."""
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, pyqtSignal
from qfluentwidgets import isDarkTheme

from ui.design_tokens import color
from ui.widgets.map_view import _track_to_mercator_uncached
from ui.widgets.map_tiles import is_gcj02_source


class MarkerMapOverlay(QObject):
    selected = pyqtSignal(str, str)

    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.rows = ()
        self.current = ''
        self._line = pg.PlotDataItem(connect='finite')
        self._points = pg.ScatterPlotItem(size=8)
        self._highlight = pg.PlotDataItem(connect='finite')
        for item in (self._line, self._points, self._highlight):
            item.setZValue(25)
            view._plot.addItem(item, ignoreBounds=True)
        self._points.sigClicked.connect(self._clicked)
        self._coordinates = {}

    def _clicked(self, item, spots):
        if spots:
            line_id, uid = spots[0].data()
            self.selected.emit(line_id, uid)

    def set_rows(self, rows):
        self.rows = tuple(rows)
        self._coordinates = {}
        xs, ys, spots = [], [], []
        gcj = is_gcj02_source(self.view._layer.source_key())
        for row in self.rows:
            native = row.get('geometry')
            if not native or row['record'].excluded:
                continue
            segments = []
            for part in native['parts']:
                track = SimpleNamespace(points=tuple(SimpleNamespace(x=x, y=y, elevation_m=0.) for x, y in part),
                                        coordinate_system=native['crs'], source=native['source'], line_id=row['line_id'], name=row['name'])
                converted = _track_to_mercator_uncached(track, gcj02=gcj)
                if not converted['mapped']:
                    continue
                x, y = converted['xs'], converted['ys']
                segments.append((x, y))
                if len(part) > 1:
                    xs.extend([*x, np.nan])
                    ys.extend([*y, np.nan])
                for index in set((0, len(x) - 1)):
                    spots.append({'pos': (x[index], y[index]), 'data': (row['line_id'], row['uid'])})
            self._coordinates[row['uid']] = segments
        ink = color('primary', isDarkTheme())
        self._line.setData(xs, ys, pen=pg.mkPen(ink, width=2), connect='finite')
        self._points.setData(spots, brush=pg.mkBrush(ink), pen=pg.mkPen(ink))
        self.select(self.current)

    def select(self, uid, *, locate=False):
        self.current = uid
        segments = self._coordinates.get(uid, ())
        xs, ys = [], []
        for x, y in segments:
            xs.extend([*x, np.nan])
            ys.extend([*y, np.nan])
        self._highlight.setData(xs, ys, pen=pg.mkPen(color('warning', isDarkTheme()), width=4), connect='finite')
        if locate and xs:
            self.view._plot.vb.setRange(xRange=(np.nanmin(xs) - 2, np.nanmax(xs) + 2),
                                       yRange=(np.nanmin(ys) - 2, np.nanmax(ys) + 2), padding=.05)
        return bool(segments)
