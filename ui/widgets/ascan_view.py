"""AScanView — A-Scan 单道波形显示控件（SPEC §5.2）。

简单 pg.PlotWidget 包装，复刻 style_spec §3.3 A-Scan 区：
pen 宽 2、Y 范围 min-0.1 ~ max+0.1、轴标签 bottom='采样点' / left='幅度'。

缩放/导出/轴主题继承 GraphicsViewBase（pg_view_base 统一收敛）。
"""

from PyQt6.QtWidgets import QVBoxLayout, QWidget
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

        # 空态引导浮层：零数据时替代"只剩坐标轴"的空画布
        self._empty_overlay = EmptyStateOverlay(
            self._plot, icon=FIF.PHOTO, title='暂无波形',
            hint='在测线上选取一道数据后，此处显示 A-Scan 时域波形')

        self.apply_theme(False)

    def set_trace(self, samples, *, title="A-Scan时域波形") -> None:
        """绘制单道波形；pen 宽 2，Y 范围 min-0.1 ~ max+0.1。"""
        import numpy as np

        data = np.asarray(samples, dtype=float).ravel()
        self._plot.setTitle(title)
        if data.size == 0:
            self.clear()
            self._plot.setTitle(title)
            return
        dark_pen = self._curve.opts.get('pen')
        self._curve.setData(data, pen=dark_pen)
        y_min = float(np.nanmin(data)) - 0.1
        y_max = float(np.nanmax(data)) + 0.1
        self._plot.setYRange(y_min, y_max)
        self._plot.setXRange(0, max(data.size - 1, 1))
        self._empty_overlay.setVisible(False)

    def clear(self) -> None:
        self._curve.setData([], [])
        self._empty_overlay.setVisible(True)

    def apply_theme(self, dark: bool) -> None:
        """深色 bg 'k'/曲线 'w'；浅色 bg 'w'/曲线 'b'；轴 pen/textPen/标签/标题同步。"""
        self._dark = bool(dark)
        self._plot.setBackground('k' if dark else 'w')
        self._curve.setPen(pg.mkPen('w' if dark else 'b', width=2))
        # AScan 是单道波形曲线，淡网格有助于读数（图像类视图不适用）
        style_plot_item(self._plot_item, dark, grid=True)
