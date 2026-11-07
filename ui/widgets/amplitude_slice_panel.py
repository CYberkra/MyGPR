"""Controls and linked views for observation-only amplitude slices."""
from __future__ import annotations

from uuid import uuid4

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QSplitter, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, ComboBox, DoubleSpinBox, PushButton, Slider

from ui.widgets.bscan_view import BScanView
from ui.widgets.depth_slice_view import DepthSliceView


class AmplitudeSlicePanel(QWidget):
    preview_requested = pyqtSignal(object, object, str, float, float, float, str)
    cancel_requested = pyqtSignal()
    artifacts_requested = pyqtSignal(str)
    bscan_requested = pyqtSignal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks = []
        self._source_ids = {}
        self._artifact_lists = {}
        self._positions = ()
        self._grid = None
        self._bundle = None
        self._bscan_request = ''
        self._selected_raw_trace = None  # (line_id, original trace index)
        self._levels = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self._line = ComboBox(self)
        self._source = ComboBox(self)
        self._line.setMinimumWidth(105)
        self._source.setMinimumWidth(130)
        self._start = self._spin(0., ' ns')
        self._end = self._spin(10., ' ns')
        self._cell = self._spin(1., ' m', minimum=0.05)
        self._attribute = ComboBox(self)
        for label, value in (('RMS', 'rms'), ('均值', 'mean'), ('最大值', 'max')):
            self._attribute.addItem(label, userData=value)
        self._component = ComboBox(self)
        for label, value in (('自动分量', ''), ('时域实部', 'real'), ('时域幅值', 'mag')):
            self._component.addItem(label, userData=value)
        self._run = PushButton('生成切片', self)
        self._cancel = PushButton('取消', self)
        self._cancel.setEnabled(False)
        for widget in (self._line, self._source, self._component, self._attribute):
            toolbar.addWidget(widget)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)
        parameters = QHBoxLayout()
        for widget in (CaptionLabel('开始', self), self._start, CaptionLabel('结束', self),
                       self._end, CaptionLabel('网格', self), self._cell, self._run, self._cancel):
            parameters.addWidget(widget)
        parameters.addStretch(1)
        layout.addLayout(parameters)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._map = DepthSliceView(splitter)
        self._bscan = BScanView(splitter)
        splitter.addWidget(self._map)
        splitter.addWidget(self._bscan)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        bottom = QHBoxLayout()
        self._slider = Slider(Qt.Orientation.Horizontal, self)
        self._slider.setRange(0, 10000)  # tenths of a nanosecond; spins allow longer windows
        self._slider.setValue(50)
        self._status = CaptionLabel('时间窗：采集时间', self)
        bottom.addWidget(self._status)
        bottom.addWidget(self._slider, 1)
        layout.addLayout(bottom)
        self._line.currentIndexChanged.connect(self._line_changed)
        self._source.currentIndexChanged.connect(self._source_changed)
        self._run.clicked.connect(self._request)
        self._cancel.clicked.connect(self._cancel_preview)
        self._attribute.currentIndexChanged.connect(self._reset_levels)
        self._component.currentIndexChanged.connect(self._reset_levels)
        self._start.valueChanged.connect(self._sync_slider)
        self._end.valueChanged.connect(self._sync_slider)
        self._slider.sliderReleased.connect(self._slider_released)
        self._map.scene().sigMouseClicked.connect(self._map_clicked)
        self._window_band = pg.LinearRegionItem((0, 1), movable=False,
                                                brush=pg.mkBrush(50, 125, 230, 55))
        self._window_band.hide()
        self._bscan._plot.addItem(self._window_band)
        self._trace_marker = pg.InfiniteLine(angle=90, movable=False,
                                             pen=pg.mkPen('#397fff', width=1))
        self._trace_marker.hide()
        self._bscan._plot.addItem(self._trace_marker)

    def _spin(self, value, suffix, minimum=0.):
        control = DoubleSpinBox(self)
        control.setRange(minimum, 1_000_000.)
        control.setDecimals(2)
        control.setValue(value)
        control.setSuffix(suffix)
        return control

    def set_tracks(self, tracks, _colors):
        self._tracks = list(tracks or ())
        self.cancel_requested.emit()
        self._run.setEnabled(True)
        self._cancel.setEnabled(False)
        if not self._tracks:
            self._source_ids.clear()
            self._artifact_lists.clear()
            self._bundle = None
            self._selected_raw_trace = None
            self._trace_marker.hide()
            self._window_band.hide()
        old = str(self._line.currentData() or '')
        self._line.blockSignals(True)
        self._line.clear()
        for track in self._tracks:
            self._line.addItem(str(track.name or track.line_id), userData=str(track.line_id))
        index = self._line.findData(old)
        self._line.setCurrentIndex(max(0, index))
        self._line.blockSignals(False)
        self._positions = ()
        self._grid = None
        self._levels = None
        self._map.clear_grid()
        self._line_changed()

    def set_artifacts(self, line_id, artifacts):
        self._artifact_lists[str(line_id)] = list(artifacts or ())
        if str(self._line.currentData() or '') == str(line_id):
            self._refresh_source()

    def _line_changed(self, *_args):
        line_id = str(self._line.currentData() or '')
        self._refresh_source()
        if line_id:
            self.artifacts_requested.emit(line_id)
            self._request_bscan()

    def _refresh_source(self):
        line_id = str(self._line.currentData() or '')
        selected = str(self._source_ids.get(line_id, ''))
        self._source.blockSignals(True)
        self._source.clear()
        self._source.addItem('原始输入', userData='')
        for artifact in self._artifact_lists.get(line_id, ()):
            self._source.addItem(str(artifact.name), userData=str(artifact.artifact_id))
        index = self._source.findData(selected)
        if index < 0:
            self._source_ids.pop(line_id, None)
        self._source.setCurrentIndex(max(index, 0))
        self._source.blockSignals(False)

    def _source_changed(self, *_args):
        line_id = str(self._line.currentData() or '')
        if line_id:
            self._source_ids[line_id] = str(self._source.currentData() or '')
            self._levels = None
            self._request_bscan()

    def _request(self):
        if not self._tracks:
            self._status.setText('请先勾选带定位信息的测线')
            return
        start, end = float(self._start.value()), float(self._end.value())
        if start >= end:
            self._status.setText('开始时间必须小于结束时间')
            return
        self._run.setEnabled(False)
        self._cancel.setEnabled(True)
        self._status.setText('正在生成切片…')
        self.preview_requested.emit(self._tracks, dict(self._source_ids),
                                    str(self._component.currentData() or ''), start, end,
                                    float(self._cell.value()), str(self._attribute.currentData()))

    def _cancel_preview(self):
        self.cancel_requested.emit()
        self._run.setEnabled(True)
        self._cancel.setEnabled(False)
        self._status.setText('已取消')

    def _reset_levels(self, *_args):
        self._levels = None

    def _sync_slider(self, *_args):
        self._slider.setValue(round((self._start.value() + self._end.value()) * 5))

    def _slider_released(self):
        width = float(self._end.value()) - float(self._start.value())
        if width <= 0:
            return
        center = max(width / 2., self._slider.value() / 10.)
        self._start.setValue(center - width / 2.)
        self._end.setValue(center + width / 2.)
        self._request()

    def set_result(self, result, positions):
        self._run.setEnabled(True)
        self._cancel.setEnabled(False)
        self._positions = tuple(positions)
        grid = result.grid
        self._grid = grid
        self._map.set_grid(grid.values, x_origin_m=grid.x_origin_m,
                           y_origin_m=grid.y_origin_m, cell_size_m=grid.cell_size_m,
                           attribute=f'{result.attribute.upper()} · {result.coordinate_system}')
        if self._levels is None:
            self._levels = self._map.value_range()
        if self._levels is not None and self._levels[0] < self._levels[1]:
            self._map._image.setLevels(self._levels)
            self._map._colorbar.setLevels(self._levels)
        self._status.setText(f'采集时间 {result.start_ns:g}–{result.end_ns:g} ns')
        self._show_window()

    def set_failure(self, message):
        self._run.setEnabled(True)
        self._cancel.setEnabled(False)
        self._status.setText(str(message or '切片失败'))

    def _map_clicked(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self._positions or self._grid is None:
            return
        if not self._map._plot_item.sceneBoundingRect().contains(event.scenePos()):
            return
        pos = self._map._plot_item.vb.mapSceneToView(event.scenePos())
        grid = self._grid
        cell = grid.cell_size_m
        col = int(np.floor((pos.x() - (grid.x_origin_m - cell / 2)) / cell))
        row = int(np.floor(((grid.y_origin_m + cell / 2) - pos.y()) / cell))
        if not (0 <= row < grid.values.shape[0] and 0 <= col < grid.values.shape[1]
                and np.isfinite(grid.values[row, col])):
            return
        nearest = None
        for line in self._positions:
            if len(line.x) == 0:
                continue
            d2 = (line.x - pos.x()) ** 2 + (line.y - pos.y()) ** 2
            in_cell = ((np.floor((line.x - (grid.x_origin_m - cell / 2)) / cell) == col)
                       & (np.floor(((grid.y_origin_m + cell / 2) - line.y) / cell) == row))
            d2 = np.where(in_cell, d2, np.inf)
            index = int(np.argmin(d2))
            if not np.isfinite(d2[index]):
                continue
            candidate = (float(d2[index]), line.source.line_id, int(line.trace_indices[index]))
            if nearest is None or candidate[0] < nearest[0]:
                nearest = candidate
        if nearest is None:
            return
        self._selected_raw_trace = (nearest[1], nearest[2])
        index = self._line.findData(nearest[1])
        if index != self._line.currentIndex():
            self._line.setCurrentIndex(index)
        else:
            self._request_bscan()

    def _request_bscan(self):
        line_id = str(self._line.currentData() or '')
        if not line_id:
            return
        self._bscan_request = uuid4().hex
        self.bscan_requested.emit(line_id, str(self._source_ids.get(line_id, '')), self._bscan_request)

    def deliver_bscan(self, line_id, artifact_id, request_id, bundle, error):
        if request_id != self._bscan_request or str(line_id) != str(self._line.currentData() or ''):
            return
        if error or bundle is None:
            self._status.setText(str(error or '无法加载剖面'))
            return
        self._bundle = bundle
        self._bscan.set_bundle(bundle)
        self._trace_marker.hide()
        # Raw track indices are not necessarily artifact indices after processing.
        # Until an explicit mapping is supplied, never draw a misleading marker.
        indices = getattr(bundle, 'trace_indices', None)
        if (not artifact_id and self._selected_raw_trace is not None and indices is not None
                and self._selected_raw_trace[0] == str(line_id)):
            indices = np.asarray(indices)
            if indices.size:
                self._trace_marker.setValue(int(np.argmin(np.abs(indices - self._selected_raw_trace[1]))) + 0.5)
                self._trace_marker.show()
        self._show_window()

    def _show_window(self):
        axis = getattr(self._bundle, 'sample_axis', None) if self._bundle is not None else None
        label = str(getattr(self._bundle, 'sample_axis_label', '') or '') if self._bundle is not None else ''
        if axis is None or 'ns' not in label.lower():
            self._window_band.hide()
            return
        axis = np.asarray(axis, dtype=float)
        if len(axis) < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            self._window_band.hide()
            return
        start, end = self._start.value(), self._end.value()
        if start < axis[0] or end > axis[-1]:
            self._window_band.hide()
            return
        self._window_band.setRegion(np.interp([start, end], axis, np.arange(len(axis)) + .5))
        self._window_band.show()

    def apply_theme(self, dark):
        self._map.apply_theme(dark)
        self._bscan.apply_theme(dark)
