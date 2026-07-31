# -*- coding: utf-8 -*-
"""Trajectory3DView — 三维测线轨迹视图（可叠加真实地形）。

pyqtgraph.opengl.GLViewWidget：GLGridItem 地面网格 + 每条测线一条
GLLinePlotItem（颜色与平面地图一致）。坐标归一化到局部原点（全体点减均值），
z 取 elevation_m。import pyqtgraph.opengl 失败（缺 PyOpenGL）时降级为
QLabel 提示，不影响页面其余部分。

真实地形：测线坐标系可识别（EPSG 或经纬度启发）时，后台线程按测线包围盒
下载 AWS Terrarium 高程瓦片（磁盘缓存 ~/MyGPR/tile_cache/terrarium/），
解码拼接后双线性重采样到测线坐标系规则网格，以 GLSurfacePlotItem
（heightColor 着色）铺地；网络失败静默降级为平面网格。
瓦片解码 / 采样数学均为 ui.widgets.terrain_tiles 中的纯函数。
"""
from __future__ import annotations

import logging
import os
import urllib.request

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ui.widgets.map_tiles import extract_epsg
from ui.widgets.terrain_tiles import (decode_terrarium, mosaic_from_tiles,
                                      sample_bilinear, terrarium_cache_path,
                                      terrarium_url, tile_grid_for_bbox)

try:
    import pyqtgraph.opengl as _gl
    from PyQt6.QtGui import QVector3D as _Vector
except Exception:  # noqa: BLE001 - PyOpenGL 缺失等任意导入失败 → 降级
    _gl = None
    _Vector = None

_LOGGER = logging.getLogger(__name__)

_TERRAIN_CACHE_ROOT = os.path.join(os.path.expanduser('~'), 'MyGPR', 'tile_cache')
_USER_AGENT = 'MyGPR/1.0 (https://mygpr.local; terrain client)'
_TERRAIN_GRID = 96               # 地形网格每边最大采样数


class _TerrainSignals(QObject):
    """QRunnable 无法自带信号，用独立 QObject 回主线程。"""

    finished = pyqtSignal(int, object)   # generation, payload(dict|None)


class _TerrainWorker(QRunnable):
    """后台地形构建：下载高程瓦片 → 马赛克 → 重采样到测线坐标系网格。"""

    def __init__(self, generation: int, epsg: int, bbox: tuple,
                 cache_root: str, signals: _TerrainSignals) -> None:
        super().__init__()
        self._generation = int(generation)
        self._epsg = int(epsg)
        self._bbox = bbox          # (x_min, y_min, x_max, y_max) 源坐标系
        self._cache_root = cache_root
        self._signals = signals

    def _fetch_tile(self, z: int, x: int, y: int) -> np.ndarray | None:
        path = terrarium_cache_path(self._cache_root, z, x, y)
        try:
            if not os.path.exists(path):
                request = urllib.request.Request(
                    terrarium_url(z, x, y), headers={'User-Agent': _USER_AGENT})
                with urllib.request.urlopen(request, timeout=15) as response:
                    if getattr(response, 'status', 200) != 200:
                        raise OSError(f'HTTP {response.status}')
                    data = response.read()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp_path = path + '.tmp'
                with open(tmp_path, 'wb') as fh:
                    fh.write(data)
                os.replace(tmp_path, path)
            with open(path, 'rb') as fh:
                return decode_terrarium(fh.read())
        except Exception as exc:  # noqa: BLE001 - 离线/超时静默
            _LOGGER.debug('高程瓦片获取失败 %s/%s/%s: %s', z, x, y, exc)
            return None

    def run(self) -> None:
        payload = None
        try:
            from pyproj import Transformer
            x_min, y_min, x_max, y_max = self._bbox
            to_lonlat = Transformer.from_crs(
                f'EPSG:{self._epsg}', 'EPSG:4326', always_xy=True)
            corners_x = [x_min, x_min, x_max, x_max]
            corners_y = [y_min, y_min, y_max, y_max]
            lons, lats = to_lonlat.transform(corners_x, corners_y)
            zoom, x0, x1, y0, y1 = tile_grid_for_bbox(
                min(lons), min(lats), max(lons), max(lats))

            tiles = {}
            for tx in range(x0, x1 + 1):
                for ty in range(y0, y1 + 1):
                    block = self._fetch_tile(zoom, tx, ty)
                    if block is not None:
                        tiles[(tx, ty)] = block
            if tiles:
                elev, xs_m, ys_m = mosaic_from_tiles(tiles, zoom, x0, x1, y0, y1)
                # 源坐标系规则网格 → 经纬度 → Mercator 米 → 双线性采样
                steps_x = min(_TERRAIN_GRID, 96)
                steps_y = min(_TERRAIN_GRID, 96)
                gx = np.linspace(x_min, x_max, steps_x)
                gy = np.linspace(y_min, y_max, steps_y)
                mesh_x, mesh_y = np.meshgrid(gx, gy)
                lon, lat = to_lonlat.transform(mesh_x.ravel(), mesh_y.ravel())
                to_merc = Transformer.from_crs('EPSG:4326', 'EPSG:3857',
                                               always_xy=True)
                mx, my = to_merc.transform(lon, lat)
                z = sample_bilinear(elev, xs_m, ys_m, mx, my)
                z = z.reshape(mesh_x.shape)
                if np.isfinite(z).any():
                    mean = float(np.nanmean(z))
                    z = np.where(np.isfinite(z), z, np.float32(mean))
                    payload = {'gx': gx, 'gy': gy, 'z': z.astype(np.float32)}
        except Exception as exc:  # noqa: BLE001 - 任意失败静默降级平面网格
            _LOGGER.debug('地形构建失败: %s', exc)
            payload = None
        self._signals.finished.emit(self._generation, payload)


class Trajectory3DView(QWidget):
    """三维轨迹视图容器（内部按需创建 GLViewWidget 或降级 QLabel）。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._gl_view = None
        self._grid = None
        self._line_items: list = []
        self._terrain_item = None
        self._fallback_label = None
        self._terrain_pool = None
        self._terrain_signals = None
        self._terrain_generation = 0
        self._origin = None
        if _gl is not None:
            self._gl_view = _gl.GLViewWidget(self)
            self._gl_view.setBackgroundColor('w')
            self._grid = _gl.GLGridItem()
            self._grid.setSize(500.0, 500.0)
            self._grid.setSpacing(20.0, 20.0)
            self._gl_view.addItem(self._grid)
            layout.addWidget(self._gl_view, 1)
            self._terrain_pool = QThreadPool(self)
            self._terrain_pool.setMaxThreadCount(1)
            self._terrain_signals = _TerrainSignals(self)
            self._terrain_signals.finished.connect(self._on_terrain_finished)
        else:
            self._fallback_label = QLabel('三维视图需要 PyOpenGL', self)
            self._fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._fallback_label, 1)

    # ------------------------------------------------------------ 数据
    def set_tracks(self, tracks, colors: dict | None = None) -> None:
        """设置测线轨迹：颜色与平面图一致，坐标归一化到局部原点。

        tracks: SpatialTrack 列表（鸭子类型取属性）；colors: {line_id: '#rrggbb'}。
        坐标系可识别时后台构建真实地形，完成后自动替换平面网格。
        """
        if self._gl_view is None:
            return
        for item in self._line_items:
            self._gl_view.removeItem(item)
        self._line_items = []
        self._clear_terrain()
        self._terrain_generation += 1   # 立即使上一批地形任务失效
        colors = dict(colors or {})

        arrays = []
        epsg = None
        lonlat_like = True
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
            if epsg is None:
                epsg = extract_epsg(getattr(track, 'coordinate_system', ''))
        if not arrays:
            return

        # 全体点减均值 → 局部原点，z 同样归一化避免相机被大高程数值拉远
        origin = np.mean(np.concatenate([xyz for _lid, xyz in arrays]), axis=0)
        self._origin = origin
        extent = 0.0
        for line_id, xyz in arrays:
            local = xyz - origin
            extent = max(extent, float(np.abs(local[:, :2]).max()), 1.0)
            if np.nanmax(np.abs(xyz[:, :2])) >= 1000.0:
                lonlat_like = False
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
        self._grid.setVisible(True)
        self._gl_view.setCameraPosition(
            distance=max(extent * 2.5, 100.0), elevation=30.0, azimuth=45.0)
        self._gl_view.opts['center'] = _Vector(0.0, 0.0, 0.0)

        # 坐标系可识别 → 后台构建真实地形
        if epsg is None and lonlat_like:
            epsg = 4326
        if epsg is not None:
            all_xyz = np.concatenate([xyz for _lid, xyz in arrays])
            pad_x = max((all_xyz[:, 0].max() - all_xyz[:, 0].min()) * 0.05, 1e-6)
            pad_y = max((all_xyz[:, 1].max() - all_xyz[:, 1].min()) * 0.05, 1e-6)
            bbox = (float(all_xyz[:, 0].min() - pad_x),
                    float(all_xyz[:, 1].min() - pad_y),
                    float(all_xyz[:, 0].max() + pad_x),
                    float(all_xyz[:, 1].max() + pad_y))
            self._terrain_generation += 1
            self._terrain_pool.start(_TerrainWorker(
                self._terrain_generation, epsg, bbox,
                _TERRAIN_CACHE_ROOT, self._terrain_signals))

    # ------------------------------------------------------------ 地形
    def _clear_terrain(self) -> None:
        if self._terrain_item is not None and self._gl_view is not None:
            self._gl_view.removeItem(self._terrain_item)
        self._terrain_item = None

    def _on_terrain_finished(self, generation: int, payload) -> None:
        """地形网格就绪：替换平面网格（过期代次直接丢弃）。"""
        if generation != self._terrain_generation or not payload:
            return
        if self._gl_view is None or _gl is None or self._origin is None:
            return
        try:
            origin = self._origin
            x = np.asarray(payload['gx'], dtype=float) - origin[0]
            y = np.asarray(payload['gy'], dtype=float) - origin[1]
            z = np.asarray(payload['z'], dtype=float)
            # 高程基准对齐：DEM（正高）与测线 GPS 高程常有系统性偏差（大地水准面
            # 差距可达数十米），按各自均值对齐保证测线贴地显示
            z = z - float(np.mean(z))
            self._clear_terrain()
            # z 形状 (len(x), len(y))：pyqtgraph 期望 x 为第一维
            z = np.ascontiguousarray(z.T)
            item = _gl.GLSurfacePlotItem(
                x=x, y=y, z=z, shader='heightColor', smooth=True)
            self._gl_view.addItem(item)
            self._terrain_item = item
            self._grid.setVisible(False)   # 地形就位后隐藏平面网格
        except Exception as exc:  # noqa: BLE001 - 构建失败保留平面网格
            _LOGGER.debug('地形渲染失败: %s', exc)

    # ------------------------------------------------------------ 主题
    def apply_theme(self, dark: bool) -> None:
        """深色黑底 / 浅色白底。"""
        if self._gl_view is not None:
            self._gl_view.setBackgroundColor('k' if dark else 'w')


__all__ = ['Trajectory3DView']
