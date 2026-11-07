"""Selected B-scan alongside the 3-D curtain view."""
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import ComboBox

from ui.widgets.bscan_view import BScanView


class CurtainBScanPanel(QWidget):
    line_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._line = ComboBox(self)
        layout.addWidget(self._line)
        self._bscan = BScanView(self)
        layout.addWidget(self._bscan, 1)
        self._window = pg.LinearRegionItem((0, 1), orientation='horizontal', movable=False,
                                           brush=pg.mkBrush(50, 125, 230, 55))
        self._trace = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#397fff', width=1))
        self._time = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#397fff', width=1))
        self._bscan._plot.addItem(self._window)
        self._bscan._plot.addItem(self._trace)
        self._bscan._plot.addItem(self._time)
        self._window.hide()
        self._trace.hide()
        self._time.hide()
        self._line.currentIndexChanged.connect(self._selected)

    def set_lines(self, tracks, current_line):
        self._line.blockSignals(True)
        self._line.clear()
        for track in tracks:
            self._line.addItem(str(track.name or track.line_id), userData=str(track.line_id))
        self._line.setCurrentIndex(max(0, self._line.findData(str(current_line))))
        self._line.blockSignals(False)

    def select_line(self, line_id):
        index = self._line.findData(str(line_id))
        if index >= 0:
            self._line.blockSignals(True)
            self._line.setCurrentIndex(index)
            self._line.blockSignals(False)

    def _selected(self, *_args):
        line_id = str(self._line.currentData() or '')
        if line_id:
            self.line_selected.emit(line_id)

    def set_bundle(self, bundle, marker, start_ns, end_ns):
        self._window.hide()
        self._trace.hide()
        self._time.hide()
        if bundle is None:
            self._bscan.clear()
            return
        self._bscan.set_bundle(bundle)
        axis = getattr(bundle, 'sample_axis', None)
        label = str(getattr(bundle, 'sample_axis_label', '') or '')
        trace_marker, picked_time = marker if isinstance(marker, tuple) else (marker, None)
        if axis is not None and 'ns' in label.lower():
            axis = np.asarray(axis, dtype=float)
            if len(axis) >= 2 and np.isfinite(axis).all() and np.all(np.diff(axis) > 0):
                if axis[0] <= start_ns < end_ns <= axis[-1]:
                    self._window.setRegion(np.interp([start_ns, end_ns], axis, np.arange(len(axis)) + .5))
                    self._window.show()
                if picked_time is not None and axis[0] <= picked_time <= axis[-1]:
                    self._time.setValue(float(np.interp(picked_time, axis, np.arange(len(axis)) + .5)))
                    self._time.show()
        if trace_marker is not None:
            self._trace.setValue(float(trace_marker))
            self._trace.show()

    def apply_theme(self, dark):
        self._bscan.apply_theme(dark)
