"""AScanView — A-Scan 单道波形显示控件（SPEC §5.2）。

简单 pg.PlotWidget 包装，复刻 style_spec §3.3 A-Scan 区：
pen 宽 2、Y 范围按有限数据自动留边、轴标签 bottom='采样点' / left='幅度'。

缩放/导出/轴主题继承 GraphicsViewBase（pg_view_base 统一收敛）。
"""

from PyQt6.QtWidgets import QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon as FIF

import pyqtgraph as pg

from ui.widgets.empty_state import EmptyStateOverlay
from ui.widgets.pg_view_base import GraphicsViewBase, style_plot_item


class AScanView(GraphicsViewBase, QWidget):
    """A-Scan 时域波形视图。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot = pg.PlotWidget(self, title='A-Scan时域波形')
        self._plot_item = self._plot.getPlotItem()   # GraphicsViewBase 约定属性
        self._plot.setLabel('bottom', '采样点')
        self._plot.setLabel('left', '幅度')
        self._curve = self._plot.plot(pen=pg.mkPen('b', width=2))
        self._curve.setData([], [])
        self._cursors = [pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('r')) for _ in range(2)]
        for cursor in self._cursors:
            self._plot.addItem(cursor)
            cursor.sigPositionChanged.connect(self._update_cursor_label)
        self._unit = '采样点'
        self._cursor_label = QLabel(self)
        self._axis = None
        self._samples = None
        self._spectrum_window = None
        self._spectrum_button = QPushButton('当前单道幅度谱', self)
        self._spectrum_button.clicked.connect(self._show_spectrum)
        self._selection_button = QPushButton('游标区段幅度谱', self)
        self._selection_button.clicked.connect(lambda: self._show_spectrum(selection=True))
        self._spectrum_button.setEnabled(False)
        self._selection_button.setEnabled(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)
        layout.addWidget(self._cursor_label)
        buttons = QHBoxLayout()
        buttons.addWidget(self._spectrum_button)
        buttons.addWidget(self._selection_button)
        layout.addLayout(buttons)

        # 空态引导浮层：零数据时替代"只剩坐标轴"的空画布
        self._empty_overlay = EmptyStateOverlay(
            self._plot, icon=FIF.PHOTO, title='暂无波形',
            hint='在测线上选取一道数据后，此处显示 A-Scan 时域波形')

        self.apply_theme(False)

    def set_trace(self, samples, *, title="A-Scan时域波形", axis=None, axis_label='') -> None:
        """绘制单道波形，按有限值范围留边；不改变样值。"""
        import numpy as np

        data = np.asarray(samples, dtype=float).ravel()
        self._plot.setTitle(title)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            self.clear()
            self._plot.setTitle(title)
            return
        dark_pen = self._curve.opts.get('pen')
        x = np.asarray(axis).ravel() if axis is not None else np.arange(data.size)
        if x.size != data.size or not np.all(np.isfinite(x)) or not axis_label:
            x = np.arange(data.size)
            axis_label = ''
        self._axis, self._samples = x, data
        self._spectrum_button.setEnabled(True)
        self._selection_button.setEnabled(True)
        if self._spectrum_window is not None:
            self._spectrum_window.close()
            self._spectrum_window = None
        self._unit = 'ns' if 'ns' in axis_label.lower() else ('m' if '(m)' in axis_label.lower() else '采样点')
        self._plot.setLabel('bottom', axis_label or '采样点')
        self._curve.setData(x, data, pen=dark_pen)
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        span = y_max - y_min
        padding = span * 0.05 if span > 0 else max(abs(y_min) * 0.05, 1e-12)
        self._plot.setYRange(y_min - padding, y_max + padding, padding=0)
        self._plot.setXRange(float(x[0]), float(x[-1]) if x[-1] != x[0] else float(x[0]) + 1)
        self._cursors[0].setValue(float(x[0]))
        self._cursors[1].setValue(float(x[-1]))
        self._update_cursor_label()
        self._empty_overlay.setVisible(False)

    def _update_cursor_label(self, *_args) -> None:
        delta = abs(self._cursors[1].value() - self._cursors[0].value())
        self._cursor_label.setText(f'游标差: {delta:.4g} {self._unit}')

    def _show_spectrum(self, _checked=False, *, selection=False) -> None:
        from PyQt6.QtWidgets import QMessageBox
        from ui.widgets.ascan_spectrum import amplitude_spectrum

        bounds = sorted(cursor.value() for cursor in self._cursors) if selection else None
        try:
            frequency, amplitude = amplitude_spectrum(self._samples, self._axis, self._unit, bounds)
        except ValueError as exc:
            QMessageBox.information(self, '频谱不可用', str(exc))
            return
        if self._spectrum_window is not None:
            self._spectrum_window.close()
        window = pg.PlotWidget(title='当前矩阵单道信号幅度谱')
        window.plot(frequency, amplitude)
        window.setLabel('bottom', '频率', units='MHz')
        window.setLabel('left', '幅度')
        self._spectrum_window = window
        window.show()

    def clear(self) -> None:
        self._curve.setData([], [])
        self._axis = self._samples = None
        self._spectrum_button.setEnabled(False)
        self._selection_button.setEnabled(False)
        if self._spectrum_window is not None:
            self._spectrum_window.close()
            self._spectrum_window = None
        self._empty_overlay.setVisible(True)

    def apply_theme(self, dark: bool) -> None:
        """深色 bg 'k'/曲线 'w'；浅色 bg 'w'/曲线 'b'；轴 pen/textPen/标签/标题同步。"""
        self._dark = bool(dark)
        self._plot.setBackground('k' if dark else 'w')
        self._curve.setPen(pg.mkPen('w' if dark else 'b', width=2))
        # AScan 是单道波形曲线，淡网格有助于读数（图像类视图不适用）
        style_plot_item(self._plot_item, dark, grid=True)
