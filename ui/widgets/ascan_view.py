"""AScanView — A-Scan 单道波形显示控件（SPEC §5.2）。

简单 pg.PlotWidget 包装，复刻 style_spec §3.3 A-Scan 区：
pen 宽 2、Y 范围 min-0.1 ~ max+0.1、轴标签 bottom='采样点' / left='幅度'。
"""

from PyQt6.QtWidgets import QVBoxLayout, QWidget

import pyqtgraph as pg


class AScanView(QWidget):
    """A-Scan 时域波形视图。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot = pg.PlotWidget(self, title='A-Scan时域波形')
        self._plot.setLabel('bottom', '采样点')
        self._plot.setLabel('left', '幅度')
        self._curve = self._plot.plot(pen=pg.mkPen('b', width=2))
        self._curve.setData([], [])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)
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

    def clear(self) -> None:
        self._curve.setData([], [])

    def apply_theme(self, dark: bool) -> None:
        """深色 bg 'k'/曲线 'w'；浅色 bg 'w'/曲线 'b'；轴 pen/textPen/标签同步。"""
        bg = 'k' if dark else 'w'
        fg = 'w' if dark else 'k'
        curve_color = 'w' if dark else 'b'
        self._plot.setBackground(bg)
        self._curve.setPen(pg.mkPen(curve_color, width=2))
        # 不能用 QColor(fg)：Qt 颜色名不含 'w'/'k'，非法色会变黑导致深色下轴字不可见
        pen = pg.mkPen(fg)
        for name in ('bottom', 'left'):
            axis = self._plot.getAxis(name)
            axis.setPen(pen)
            axis.setTextPen(pen)
            # 轴标题（采样点/幅度）是独立 label，不随 textPen 变色，需显式同步
            axis.setLabel(text=axis.labelText, color=fg)
        title_item = self._plot.getPlotItem().titleLabel
        title_item.setText(title_item.text, color=fg)
