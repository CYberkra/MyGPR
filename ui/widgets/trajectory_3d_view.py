# -*- coding: utf-8 -*-
"""Trajectory3DView — 三维测线轨迹视图。

pyqtgraph.opengl.GLViewWidget：GLGridItem 地面网格 + 每条测线一条
GLLinePlotItem（颜色与平面地图一致）。坐标归一化到局部原点（全体点减均值），
z 取 elevation_m。import pyqtgraph.opengl 失败（缺 PyOpenGL）时降级为
QLabel 提示，不影响页面其余部分。
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    import pyqtgraph.opengl as _gl
    from PyQt6.QtGui import QVector3D as _Vector
except Exception:  # noqa: BLE001 - PyOpenGL 缺失等任意导入失败 → 降级
    _gl = None
    _Vector = None


class Trajectory3DView(QWidget):
    """三维轨迹视图容器（内部按需创建 GLViewWidget 或降级 QLabel）。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._gl_view = None
        self._grid = None
        self._line_items: list = []
        self._fallback_label = None
        if _gl is not None:
            self._gl_view = _gl.GLViewWidget(self)
            self._gl_view.setBackgroundColor('w')
            self._grid = _gl.GLGridItem()
            self._grid.setSize(500.0, 500.0)
            self._grid.setSpacing(20.0, 20.0)
            self._gl_view.addItem(self._grid)
            layout.addWidget(self._gl_view, 1)
        else:
            self._fallback_label = QLabel('三维视图需要 PyOpenGL', self)
            self._fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._fallback_label, 1)

    # ------------------------------------------------------------ 数据
    def set_tracks(self, tracks, colors: dict | None = None) -> None:
        """设置测线轨迹：颜色与平面图一致，坐标归一化到局部原点。

        tracks: SpatialTrack 列表（鸭子类型取属性）；colors: {line_id: '#rrggbb'}。
        """
        if self._gl_view is None:
            return
        for item in self._line_items:
            self._gl_view.removeItem(item)
        self._line_items = []
        colors = dict(colors or {})

        arrays = []
        for track in tracks or []:
            points = list(getattr(track, 'points', ()) or ())
            if not points:
                continue
            xyz = np.asarray([[float(getattr(p, 'x', 0.0)),
                               float(getattr(p, 'y', 0.0)),
                               float(getattr(p, 'elevation_m', 0.0))]
                              for p in points], dtype=float)
            finite = np.isfinite(xyz).all(axis=1)
            if np.count_nonzero(finite) < 1:
                continue
            arrays.append((str(getattr(track, 'line_id', '') or ''), xyz[finite]))
        if not arrays:
            return

        # 全体点减均值 → 局部原点，z 同样归一化避免相机被大高程数值拉远
        origin = np.mean(np.concatenate([xyz for _lid, xyz in arrays]), axis=0)
        extent = 0.0
        for line_id, xyz in arrays:
            local = xyz - origin
            extent = max(extent, float(np.abs(local[:, :2]).max()), 1.0)
            color = colors.get(line_id, '#1f77b4')
            item = _gl.GLLinePlotItem(
                pos=local, color=color, width=2.0,
                antialias=True, mode='line_strip')
            self._gl_view.addItem(item)
            self._line_items.append(item)

        # 网格随数据范围调整，相机距离/中心自动适配
        grid_size = max(extent * 2.0, 100.0)
        self._grid.setSize(grid_size, grid_size)
        self._grid.setSpacing(grid_size / 20.0, grid_size / 20.0)
        self._gl_view.setCameraPosition(
            distance=max(extent * 2.5, 100.0), elevation=30.0, azimuth=45.0)
        self._gl_view.opts['center'] = _Vector(0.0, 0.0, 0.0)

    # ------------------------------------------------------------ 主题
    def apply_theme(self, dark: bool) -> None:
        """深色黑底 / 浅色白底。"""
        if self._gl_view is not None:
            self._gl_view.setBackgroundColor('k' if dark else 'w')


__all__ = ['Trajectory3DView']
