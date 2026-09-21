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
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTransform

from qfluentwidgets import isDarkTheme

from ui.widgets.pg_view_base import GraphicsViewBase, style_plot_item

__all__ = ["DepthSliceView"]

_ISOLINE_PEN_DARK = (255, 210, 90)
_ISOLINE_PEN_LIGHT = (176, 108, 0)


class DepthSliceView(GraphicsViewBase, pg.PlotWidget):
    """平面标量场视图：色块网格 + 等值线 + 测线轨迹叠加。

    缩放/导出/轴主题继承 GraphicsViewBase；右键菜单（导出 PNG/自适应/
    复制图像）随本类补齐（UI 收敛轮能力补齐）。
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._plot_item = self.getPlotItem()
        self._plot_item.setLabel('bottom', '东向坐标', units='m')
        self._plot_item.setLabel('left', '北向坐标', units='m')
        self._plot_item.showGrid(x=True, y=True, alpha=0.3)
        self._plot_item.setAspectLocked(True)
        self._image = pg.ImageItem(axisOrder='row-major')
        self._plot_item.addItem(self._image)
        # 色标：viridis 默认保持（SPEC 深度切片色表），ColorBarItem 提供
        # level 范围读出（值域刻度随 set_grid 同步）
        self._cmap = pg.colormap.get('viridis')
        self._colorbar = pg.ColorBarItem(label='场值', interactive=False)
        self._colorbar.setImageItem(self._image, insert_in=self._plot_item)
        self._isocurve = pg.IsocurveItem(axisOrder='row-major')
        self._isocurve.setPen(pg.mkPen(_ISOLINE_PEN_DARK, width=2))
        self._isocurve.setZValue(5)
        self._plot_item.addItem(self._isocurve)
        # 轨迹散点：**按颜色分组的一个 scatter 池**，而不是单个 scatter
        # 吃逐点颜色。pyqtgraph 的 ScatterPlotItem 一旦收到「逐点颜色」的
        # (N,4) 刷子数组，内部要逐点建 QBrush 变体，实测 10,394 点要 426ms；
        # 同一个 item 只用一个 QBrush 时只要 35ms（12×）。测线数 = 颜色数
        # （每条测线一个色），所以按色分组的 item 数 = 测线数（个位数），
        # 总点数不变但每个 item 走单色快路径。
        self._track_scatter = pg.ScatterPlotItem(pen=None)
        self._track_scatter.setZValue(10)
        self._plot_item.addItem(self._track_scatter)
        self._track_scatters: dict[str, pg.ScatterPlotItem] = {}   # color -> item
        # 轨迹点 (N,2) float 缓存：_auto_range 取视野范围用，避免走
        # scatter.points() 逐点 pt.pos()（10k 次 Python 循环 ~79ms）
        self._track_xy = None
        self._grid_extent = None    # (x0, y0, x1, y1) 有效网格范围（米）
        self._matrix = None
        # 关闭 pyqtgraph 原生英文右键菜单，右键由统一 RoundMenu 接管
        self._plot_item.vb.setMenuEnabled(False)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)
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
            self._colorbar.setLevels(levels)
        self._image.setLookupTable(self._cmap.getLookupTable())
        self._colorbar.setColorMap(self._cmap)
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
        self._colorbar.setLevels((0.0, 1.0))

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
        """叠加选中测线轨迹点（与 ElevationProfileView 同一勾选数据源）。

        性能（实测，10,394 点 / 6 条测线）：
        - pyqtgraph ``ScatterPlotItem`` 收「逐点 dict 列表」= 425ms，
          收「逐点颜色 (N,4) 数组」= 426ms，收「单色标量」= 35ms。
          故按**颜色分组**：每条测线一个 scatter（只带一个 QBrush），
          颜色数 = 测线数（个位数），全部走单色快路径。
        - 视野范围取自 ``self._track_xy`` 缓存而非 ``scatter.points()``
          逐点 ``pt.pos()``（后者 10k 次 Python 循环 ~79ms）。

        合计从 ~700ms 降到 ~40ms 量级。轨迹点是 UAV-GPR 全采样（每条
        测线上千点），旧实现在「打开项目」的全量刷新里会把主线程冻住
        数秒，Windows 合成器拿不到新帧 → 整窗「消失再出现」。勿改回
        逐点颜色。
        """
        groups: dict[str, tuple[list[float], list[float]]] = {}
        for track in tracks or []:
            line_id = str(getattr(track, 'line_id', '') or '')
            color = str(colors.get(line_id, '#808080'))
            xs, ys = groups.setdefault(color, ([], []))
            for point in getattr(track, 'points', ()) or ():
                x = float(getattr(point, 'x', 0.0))
                y = float(getattr(point, 'y', 0.0))
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                xs.append(x)
                ys.append(y)

        total = sum(len(v[0]) for v in groups.values())
        if total:
            xs_all = np.concatenate(
                [np.asarray(v[0], dtype=float) for v in groups.values()])
            ys_all = np.concatenate(
                [np.asarray(v[1], dtype=float) for v in groups.values()])
            self._track_xy = np.column_stack((xs_all, ys_all))
        else:
            self._track_xy = None

        self._sync_track_scatters(groups)
        self._auto_range()

    def _sync_track_scatters(self, groups) -> None:
        """按颜色分组增量同步散点 item（复用池，避免反复 add/remove）。"""
        # 不再使用的颜色：隐藏（保留实例供复用，避免 item 创建/销毁抖动）
        for color, item in self._track_scatters.items():
            if color not in groups:
                item.setData(x=[], y=[])
                item.setVisible(False)
        for color, (xs, ys) in groups.items():
            item = self._track_scatters.get(color)
            if item is None:
                item = pg.ScatterPlotItem(pen=None)
                item.setZValue(10)
                item.setBrush(pg.mkBrush(color))
                self._plot_item.addItem(item)
                self._track_scatters[color] = item
            item.setData(x=np.asarray(xs, dtype=float),
                         y=np.asarray(ys, dtype=float),
                         size=3, symbol='o', pen=None,
                         brush=pg.mkBrush(color))
            item.setVisible(True)
        # 兜底：非分组路径残留的旧 item 一并清空
        self._track_scatter.clear()

    def clear_tracks(self) -> None:
        self._track_scatter.clear()
        for item in self._track_scatters.values():
            item.setData(x=[], y=[])
            item.setVisible(False)
        self._track_xy = None

    # ------------------------------------------------------------ 内部
    def _auto_range(self) -> None:
        """网格/轨迹变化后重设视野：有网格按网格（并集轨迹点），
        否则按已有轨迹数据 fit；两者都没有才 no-op。

        无网格阶段（未请求深度预览时）也要能自适应，否则视图永远停在
        初始 0~1 视野，轨迹点挤在角落看不见。

        性能：视野范围取自**缓存的 numpy 数组**，不走
        ``scatter.points()`` 再逐点 ``pt.pos().x()``（那是一条 10k 次的
        Python 级循环，实测 ~79ms）。
        """
        if self._track_xy is None or self._track_xy.size == 0:
            if self._grid_extent is not None:
                x0, y0, x1, y1 = self._grid_extent
                self._plot_item.setXRange(x0, x1, padding=0.05)
                self._plot_item.setYRange(y0, y1, padding=0.05)
            return
        xs = self._track_xy[:, 0]
        ys = self._track_xy[:, 1]
        tx0, tx1 = float(xs.min()), float(xs.max())
        ty0, ty1 = float(ys.min()), float(ys.max())
        if self._grid_extent is not None:
            x0, y0, x1, y1 = self._grid_extent
            tx0, tx1 = min(x0, tx0), max(x1, tx1)
            ty0, ty1 = min(y0, ty0), max(y1, ty1)
        self._plot_item.setXRange(tx0, tx1, padding=0.05)
        self._plot_item.setYRange(ty0, ty1, padding=0.05)

    # ------------------------------------------------------------ 缩放 / 右键菜单
    def _fit_view(self) -> None:
        """自适应视野（网格 ∪ 轨迹范围；皆空时 no-op）。"""
        self._auto_range()

    def _on_mouse_clicked(self, event) -> None:
        """右键 → 统一 RoundMenu：缩放组 + 标准项（自适应/复制图像/导出 PNG）。"""
        if event.button() != Qt.MouseButton.RightButton:
            return
        if not self._plot_item.sceneBoundingRect().contains(event.scenePos()):
            return
        from qfluentwidgets import FluentIcon as FIF
        from ui.widgets.context_menus import add_action, make_menu
        menu = make_menu(self)
        add_action(menu, FIF.ZOOM_IN, '放大', self.zoom_in)
        add_action(menu, FIF.ZOOM_OUT, '缩小', self.zoom_out)
        self._add_standard_menu_actions(menu, can_export=self._matrix is not None,
                                        export_prefix='depth_slice')
        menu.exec(event.screenPos().toPoint())

    # ------------------------------------------------------------ 主题
    def apply_theme(self, dark: bool) -> None:
        """深色 bg 'k'/文字 'w'；浅色 bg 'w'/文字 'k'（与剖面视图一致）。

        轴/标题/色标轴统一走 style_plot_item（色标轴随主题同步）。
        """
        self._dark = bool(dark)
        self.setBackground('k' if dark else 'w')
        style_plot_item(self._plot_item, dark,
                        colorbar_axis=self._colorbar.axis)
        self._isocurve.setPen(pg.mkPen(
            _ISOLINE_PEN_DARK if dark else _ISOLINE_PEN_LIGHT, width=2))
