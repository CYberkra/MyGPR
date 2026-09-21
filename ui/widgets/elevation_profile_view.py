# -*- coding: utf-8 -*-
"""ElevationProfileView — 选中测线的里程-高程剖面视图。

从高程剖面页（ui/pages/spatial_page.py）移入的重型 pyqtgraph 视图：
里程由相邻点平面距离累积，曲线颜色与平面地图一致。
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QColor
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import isDarkTheme

from ui import constants
from ui.widgets.empty_state import EmptyStateOverlay
from ui.widgets.pg_view_base import style_plot_item

__all__ = ['ElevationProfileView']


class ElevationProfileView(pg.PlotWidget):
    """高程剖面：选中测线的里程-高程曲线（里程由相邻点距离累积）。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._plot_item = self.getPlotItem()
        self._plot_item.setLabel('bottom', '里程', units='m')
        self._plot_item.setLabel('left', '高程', units='m')
        # 网格透明度在 apply_theme 里按主题取 token（浅 0.28 / 深 0.18）：
        # 构造期写死 0.3 会让深色主题的网格明显抢过数据。
        # 空态引导浮层：零曲线时替代"只剩坐标轴"的空画布
        self._empty_overlay = EmptyStateOverlay(
            self, icon=FIF.UP,
            title='暂无高程剖面',
            hint='勾选含高程数据的测线后，此处显示里程-高程曲线')
        self.apply_theme(isDarkTheme())

    def set_tracks(self, tracks, colors: dict) -> None:
        """重绘选中测线的里程-高程曲线。"""
        self._plot_item.clear()
        legend = self._plot_item.legend
        if legend is not None:
            legend.clear()
        else:
            legend = self._plot_item.addLegend(
                offset=(8, 8),
                labelTextColor='w' if self._dark else 'k')
        colors = dict(colors or {})
        plotted = False
        for track in tracks or []:
            points = list(getattr(track, 'points', ()) or ())
            if len(points) < 2:
                continue
            xs = np.asarray([float(getattr(p, 'x', 0.0)) for p in points])
            ys = np.asarray([float(getattr(p, 'y', 0.0)) for p in points])
            zs = np.asarray([float(getattr(p, 'elevation_m', 0.0)) for p in points])
            finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
            if np.count_nonzero(finite) < 2:
                continue
            steps = np.hypot(np.diff(xs[finite]), np.diff(ys[finite]))
            mileage = np.concatenate(([0.0], np.cumsum(steps)))
            line_id = str(getattr(track, 'line_id', '') or '')
            name = str(getattr(track, 'name', '') or line_id)
            pen = pg.mkPen(QColor(colors.get(line_id, constants.CHART_TRACK_DEFAULT)), width=2)
            self._plot_item.plot(mileage, zs[finite], pen=pen, name=name)
            plotted = True
        self._empty_overlay.setVisible(not plotted)

    def apply_theme(self, dark: bool) -> None:
        """深色 bg 'k'/文字 'w'；浅色 bg 'w'/文字 'k'；轴 pen/textPen/标签/图例同步。"""
        self._dark = bool(dark)
        self.setBackground('k' if dark else 'w')
        # 高程剖面是折线曲线，淡网格有助于读数（图像类视图不适用）；
        # alpha 随主题取 token（深底更淡），由 style_plot_item(grid=True) 统一。
        fg = style_plot_item(self._plot_item, dark, grid=True)
        # 已有图例的条目文字颜色不随主题更新，逐条同步
        legend = self._plot_item.legend
        if legend is not None:
            for _sample, label in legend.items:
                label.setText(label.text, color=fg)
