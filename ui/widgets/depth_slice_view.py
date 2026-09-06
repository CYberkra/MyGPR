#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DepthSliceView — 深度切片视图（通用标量场：ImageItem + 等深线 + 轨迹）。

数据契约（与 application 层 `interface_depth_preview` payload 对齐）：
- matrix 形状 (nrows, ncols)，行序 = y 降序、列序 = x 升序（GIS 惯例）；
- x/y_origin 是首 cell 中心坐标（y_origin 为最高行的 y）；
- cell_size_m 为格网步长。

视图按通用标量场设计（Phase 3.1 数据源 = 界面深度场；Phase 4 换能量场
只需换 payload，不改本组件）。等值线由 pyqtgraph IsocurveItem 在视图层
直接吃矩阵，切片 level 经 :meth:`set_isoline` 切换。
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QTransform

from qfluentwidgets import isDarkTheme

__all__ = ["DepthSliceView"]

_ISOLINE_PEN_DARK = (255, 210, 90)
_ISOLINE_PEN_LIGHT = (176, 108, 0)


class DepthSliceView(pg.PlotWidget):
    """平面标量场视图：色块网格 + 等值线 + 测线轨迹叠加。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._plot_item = self.getPlotItem()
        self._plot_item.setLabel('bottom', '东向坐标', units='m')
        self._plot_item.setLabel('left', '北向坐标', units='m')
        self._plot_item.showGrid(x=True, y=True, alpha=0.3)
        self._plot_item.setAspectLocked(True)
        self._image = pg.ImageItem(axisOrder='row-major')
        self._plot_item.addItem(self._image)
        self._isocurve = pg.IsocurveItem(axisOrder='row-major')
        self._isocurve.setPen(pg.mkPen(_ISOLINE_PEN_DARK, width=2))
        self._isocurve.setZValue(5)
        self._plot_item.addItem(self._isocurve)
        self._track_scatter = pg.ScatterPlotItem(pen=None)
        self._track_scatter.setZValue(10)
        self._plot_item.addItem(self._track_scatter)
        self._grid_extent = None    # (x0, y0, x1, y1) 有效网格范围（米）
        self._matrix = None
        self.apply_theme(isDarkTheme())

    # ------------------------------------------------------------ 网格
    def set_grid(
        self,
        matrix,
        *,
        x_origin_m: float,
        y_origin_m: float,
        cell_size_m: float,
        attribute: str = "",
    ) -> None:
        """载入标量场矩阵（行序 y 降序；origin 为首 cell 中心）。"""
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.size == 0:
            self.clear_grid()
            return
        if not np.isfinite(values).any():
            self.clear_grid()
            return
        self._matrix = values
        nrows, ncols = values.shape
        cell = float(cell_size_m)
        # row 0 是最高 y。pyqtgraph 像素/等值线坐标都是 cell 中心空间
        # （像素 i 的中心在 i+0.5）：item 点 (col+0.5, row+0.5) 必须映射到
        # 数据点米坐标 (x0 + col*cell, y0 - row*cell)，故平移补偿半格。
        transform = QTransform()
        transform.scale(cell, -cell)
        transform.translate(
            float(x_origin_m) / cell - 0.5, -(float(y_origin_m) / cell + 0.5))
        self._image.setImage(values, autoLevels=False)
        self._image.setTransform(transform)
        levels = self._value_range()
        if levels is not None:
            self._image.setLevels(levels)
        self._image.setLookupTable(pg.colormap.get('viridis').getLookupTable())
        self._isocurve.setData(values, level=self._isocurve.level)
        self._isocurve.setTransform(transform)
        # 网格覆盖范围含半格边距（cell 中心语义：图像边沿在中心 ± 0.5 cell）
        x0 = float(x_origin_m) - cell / 2.0
        y0 = float(y_origin_m) + cell / 2.0
        x1 = x0 + ncols * cell
        y1 = y0 - nrows * cell
        self._grid_extent = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self._plot_item.setTitle(attribute or None)
        self._auto_range()

    def clear_grid(self) -> None:
        """清空网格与等值线（轨迹保留，由 set_tracks 单独管理）。"""
        self._image.clear()
        self._isocurve.setData(np.zeros((1, 1)), level=0.0)
        self._matrix = None
        self._grid_extent = None
        self._plot_item.setTitle(None)

    def value_range(self) -> tuple[float, float] | None:
        """当前矩阵的 (min, max)，无有效数据时 None（供滑条定界）。"""
        return self._value_range()

    def _value_range(self) -> tuple[float, float] | None:
        if self._matrix is None:
            return None
        finite = self._matrix[np.isfinite(self._matrix)]
        if finite.size == 0:
            return None
        return float(finite.min()), float(finite.max())

    # ------------------------------------------------------------ 等值线
    def set_isoline(self, value: float) -> None:
        """切换等值线 level（切片深度，单位 = 场值单位，即米）。"""
        if self._matrix is None:
            return
        self._isocurve.setData(self._matrix, level=float(value))

    def isoline_value(self) -> float | None:
        """当前等值线 level；未设网格时 None。"""
        if self._matrix is None:
            return None
        return float(self._isocurve.level)

    # ------------------------------------------------------------ 轨迹
    def set_tracks(self, tracks, colors: dict) -> None:
        """叠加选中测线轨迹点（与 ElevationProfileView 同一勾选数据源）。"""
        spots = []
        for track in tracks or []:
            line_id = str(getattr(track, 'line_id', '') or '')
            color = str(colors.get(line_id, '#808080'))
            for point in getattr(track, 'points', ()) or ():
                x = float(getattr(point, 'x', 0.0))
                y = float(getattr(point, 'y', 0.0))
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                spots.append({'pos': QPointF(x, y), 'size': 3,
                              'brush': pg.mkBrush(color)})
        self._track_scatter.clear()
        if spots:
            self._track_scatter.setData(spots)
        self._auto_range()

    def clear_tracks(self) -> None:
        self._track_scatter.clear()

    # ------------------------------------------------------------ 内部
    def _auto_range(self) -> None:
        """网格/轨迹变化后重设视野：以网格范围并集轨迹点为准。"""
        if self._grid_extent is not None:
            x0, y0, x1, y1 = self._grid_extent
            data = self._track_scatter.points()
            if data.size:
                xs = [float(pt.pos().x()) for pt in data]
                ys = [float(pt.pos().y()) for pt in data]
                x0, x1 = min(x0, min(xs)), max(x1, max(xs))
                y0, y1 = min(y0, min(ys)), max(y1, max(ys))
            self._plot_item.setXRange(x0, x1, padding=0.05)
            self._plot_item.setYRange(y0, y1, padding=0.05)

    # ------------------------------------------------------------ 主题
    def apply_theme(self, dark: bool) -> None:
        """深色 bg 'k'/文字 'w'；浅色 bg 'w'/文字 'k'（与剖面视图一致）。"""
        self._dark = bool(dark)
        bg = 'k' if dark else 'w'
        fg = 'w' if dark else 'k'
        self.setBackground(bg)
        pen = pg.mkPen(fg)
        for name in ('bottom', 'left'):
            axis = self._plot_item.getAxis(name)
            axis.setPen(pen)
            axis.setTextPen(pen)
            axis.setLabel(text=axis.labelText, color=fg)
        self._isocurve.setPen(pg.mkPen(
            _ISOLINE_PEN_DARK if dark else _ISOLINE_PEN_LIGHT, width=2))
