# -*- coding: utf-8 -*-
"""MapView — 在线瓦片地图 + 测线轨迹叠加视图。

视图坐标系为 Web Mercator 米（EPSG:3857），基于 pyqtgraph GraphicsLayoutWidget：

- TileLayer（pg.GraphicsObject）：按当前可视范围/缩放级别计算所需 XYZ 瓦片，
  只绘制磁盘缓存已有的；缺失瓦片入队 QThreadPool 后台下载（urllib + UA 头，
  失败静默），下载完成后 update() 重绘。磁盘缓存
  ``~/MyGPR/tile_cache/{source_key}/{z}/{x}/{y}.png``，命中缓存不再联网。
- 轨迹叠加：每条测线一条彩色折线 + 端点散点。坐标系字符串经
  ``map_tiles.extract_epsg`` 提取 EPSG 后用 pyproj 转到 3857；无 EPSG 且
  坐标绝对值 <1000（像经纬度）按 EPSG:4326 处理；仍不行则按原始坐标绘制
  （不套底图，坐标系信息由页面投影卡提示）。

瓦片数学 / CRS 解析均为 ``ui.widgets.map_tiles`` 中的纯函数，可独立测试。
"""
from __future__ import annotations

import logging
import os
import urllib.request

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QRectF, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtGui import QColor, QImage

from ui.widgets.map_tiles import (DEFAULT_TILE_SOURCE, TILE_SOURCES, WORLD_SIZE_M,
                                  extract_epsg, lonlat_to_mercator,
                                  lonlat_to_tile, mercator_to_lonlat,
                                  tile_bounds_mercator, tile_url,
                                  zoom_for_resolution)

_LOGGER = logging.getLogger(__name__)

_TILE_CACHE_ROOT = os.path.join(os.path.expanduser('~'), 'MyGPR', 'tile_cache')
_USER_AGENT = 'MyGPR/1.0 (https://mygpr.local; tile client)'
_HALF_WORLD = WORLD_SIZE_M / 2.0


def _track_to_mercator(track) -> dict:
    """把 SpatialTrack（鸭子类型）转换为展示用结构（Web Mercator 米）。

    返回 dict(line_id, name, xs, ys, zs, crs, source, mapped, epsg)。
    mapped=False 时 xs/ys 为原始坐标（不套底图）。
    """
    points = list(getattr(track, 'points', ()) or ())
    xs = np.asarray([float(getattr(p, 'x', 0.0)) for p in points], dtype=float)
    ys = np.asarray([float(getattr(p, 'y', 0.0)) for p in points], dtype=float)
    zs = np.asarray([float(getattr(p, 'elevation_m', 0.0)) for p in points], dtype=float)
    crs = str(getattr(track, 'coordinate_system', '') or '')
    source = str(getattr(track, 'source', '') or '')
    epsg = extract_epsg(crs)
    mapped = False
    if epsg is not None:
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:3857',
                                               always_xy=True)
            xs, ys = (np.asarray(a, dtype=float)
                      for a in transformer.transform(xs, ys))
            mapped = True
        except Exception as exc:  # noqa: BLE001 - 单条测线失败不影响其它
            _LOGGER.debug('测线坐标转换失败 EPSG:%s: %s', epsg, exc)
    if not mapped and epsg is None and xs.size:
        # 无 EPSG 且坐标绝对值像经纬度（<1000）→ 按 EPSG:4326 处理
        if (np.nanmax(np.abs(xs)) < 1000.0 and np.nanmax(np.abs(ys)) < 1000.0
                and np.isfinite(xs).all() and np.isfinite(ys).all()):
            converted = [lonlat_to_mercator(lon, lat) for lon, lat in zip(xs, ys)]
            xs = np.asarray([c[0] for c in converted], dtype=float)
            ys = np.asarray([c[1] for c in converted], dtype=float)
            mapped = True
    return {
        'line_id': str(getattr(track, 'line_id', '') or ''),
        'name': str(getattr(track, 'name', '') or ''),
        'xs': xs, 'ys': ys, 'zs': zs,
        'crs': crs, 'source': source, 'mapped': mapped, 'epsg': epsg,
    }


class _TileWorkerSignals(QObject):
    """QRunnable 无法自带信号，用独立 QObject 回主线程。"""

    finished = pyqtSignal(int, int, int, str)   # z, x, y, 缓存文件路径（失败为空串）


class _TileWorker(QRunnable):
    """后台瓦片获取：磁盘缓存命中直接返回；否则 urllib 下载并原子落盘。"""

    def __init__(self, source_key: str, z: int, x: int, y: int,
                 cache_path: str, signals: _TileWorkerSignals) -> None:
        super().__init__()
        self._source_key = source_key
        self._key = (int(z), int(x), int(y))
        self._cache_path = cache_path
        self._signals = signals

    def run(self) -> None:
        z, x, y = self._key
        path = self._cache_path
        try:
            if not os.path.exists(path):
                url = tile_url(self._source_key, z, x, y)
                request = urllib.request.Request(
                    url, headers={'User-Agent': _USER_AGENT})
                with urllib.request.urlopen(request, timeout=10) as response:
                    if getattr(response, 'status', 200) != 200:
                        raise OSError(f'HTTP {response.status}')
                    data = response.read()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp_path = path + '.tmp'
                with open(tmp_path, 'wb') as fh:
                    fh.write(data)
                os.replace(tmp_path, path)
        except Exception as exc:  # noqa: BLE001 - 离线/超时静默，地图空白但轨迹照画
            _LOGGER.debug('瓦片下载失败 %s/%s/%s/%s: %s',
                          self._source_key, z, x, y, exc)
            path = ''
        self._signals.finished.emit(z, x, y, path)


class TileLayer(pg.GraphicsObject):
    """XYZ 在线瓦片图层（GraphicsObject，画在 ViewBox 世界坐标内）。"""

    prefetch_progress = pyqtSignal(int, int)    # 已完成, 总数

    def __init__(self, source_key: str = DEFAULT_TILE_SOURCE,
                 cache_root: str = _TILE_CACHE_ROOT, parent=None) -> None:
        super().__init__(parent)
        self._source_key = source_key if source_key in TILE_SOURCES else DEFAULT_TILE_SOURCE
        self._cache_root = cache_root
        self._images: dict[tuple[int, int, int], QImage] = {}
        self._inflight: set[tuple[int, int, int]] = set()
        self._failed: set[tuple[int, int, int]] = set()
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(4)
        self._signals = _TileWorkerSignals(self)
        self._signals.finished.connect(self._on_tile_finished)
        self._prefetch_total = 0
        self._prefetch_done = 0
        self.setZValue(-100.0)   # 底图压在最底层

    # ------------------------------------------------------------ 配置
    def source_key(self) -> str:
        return self._source_key

    def set_source(self, source_key: str) -> None:
        """切换瓦图源：清空内存图与失败集（磁盘缓存按源分目录，互不影响）。"""
        source_key = str(source_key)
        if source_key not in TILE_SOURCES or source_key == self._source_key:
            return
        self._source_key = source_key
        self._images.clear()
        self._failed.clear()
        self.update()

    def _cache_path(self, z: int, x: int, y: int) -> str:
        return os.path.join(self._cache_root, self._source_key,
                            str(z), str(x), f'{y}.png')

    # ------------------------------------------------------------ GraphicsObject
    def boundingRect(self) -> QRectF:
        return QRectF(-_HALF_WORLD, -_HALF_WORLD, WORLD_SIZE_M, WORLD_SIZE_M)

    def paint(self, painter, option, widget=None) -> None:
        view_box = self.getViewBox()
        if view_box is None:
            return
        rect = view_box.viewRect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        dpr = painter.device().devicePixelRatioF() if painter.device() else 1.0
        mpp = rect.width() / max(view_box.width() * dpr, 1.0)
        zoom = zoom_for_resolution(mpp)
        n = 2 ** zoom

        # 可视范围（Web Mercator 米，y 向上）→ 瓦片编号范围（y 向下）
        lon_min, lat_min = mercator_to_lonlat(rect.x(), rect.y())
        lon_max, lat_max = mercator_to_lonlat(rect.x() + rect.width(),
                                              rect.y() + rect.height())
        x0f, _ = lonlat_to_tile(lon_min, 0.0, zoom)
        x1f, _ = lonlat_to_tile(lon_max, 0.0, zoom)
        _, y0f = lonlat_to_tile(0.0, lat_max, zoom)   # 北 → 较小瓦片 y
        _, y1f = lonlat_to_tile(0.0, lat_min, zoom)
        x0 = max(0, min(n - 1, int(x0f)))
        x1 = max(0, min(n - 1, int(x1f)))
        y0 = max(0, min(n - 1, int(y0f)))
        y1 = max(0, min(n - 1, int(y1f)))

        # 内存瓦片过多时丢弃其它级别的，防无限增长
        if len(self._images) > 800:
            for key in [k for k in self._images if k[0] != zoom]:
                self._images.pop(key, None)

        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                key = (zoom, tx, ty)
                image = self._images.get(key)
                if image is None:
                    self._enqueue(zoom, tx, ty)
                    continue
                bx0, by0, bx1, by1 = tile_bounds_mercator(zoom, tx, ty)
                painter.drawImage(QRectF(bx0, by0, bx1 - bx0, by1 - by0), image)

    # ------------------------------------------------------------ 下载队列
    def _enqueue(self, z: int, x: int, y: int) -> None:
        key = (int(z), int(x), int(y))
        if key in self._images or key in self._inflight or key in self._failed:
            return
        self._inflight.add(key)
        self._pool.start(_TileWorker(self._source_key, z, x, y,
                                     self._cache_path(z, x, y), self._signals))

    def _on_tile_finished(self, z: int, x: int, y: int, path: str) -> None:
        key = (int(z), int(x), int(y))
        self._inflight.discard(key)
        image = None
        if path:
            loaded = QImage(path)
            if not loaded.isNull():
                # 视图 y 轴向上，drawImage 会被世界变换纵向翻转，先预镜像
                image = loaded.mirrored(False, True)
        if image is not None:
            self._images[key] = image
        else:
            self._failed.add(key)
        if self._prefetch_total > 0:
            self._prefetch_done += 1
            self.prefetch_progress.emit(self._prefetch_done, self._prefetch_total)
            if self._prefetch_done >= self._prefetch_total:
                self._prefetch_total = 0
        self.update()

    # ------------------------------------------------------------ 离线预下载
    def prefetch_current_view(self, max_extra_zoom: int = 2) -> int:
        """把当前可视范围在 zoom..zoom+max_extra_zoom 的瓦片全部入队下载。

        返回入队总瓦片数；完成进度经 prefetch_progress 信号回报。
        """
        view_box = self.getViewBox()
        if view_box is None:
            return 0
        rect = view_box.viewRect()
        if rect.width() <= 0:
            return 0
        mpp = rect.width() / max(view_box.width(), 1.0)
        base_zoom = zoom_for_resolution(mpp)
        self._failed.clear()
        queued = 0
        for zoom in range(base_zoom, base_zoom + max(0, int(max_extra_zoom)) + 1):
            n = 2 ** zoom
            lon_min, lat_min = mercator_to_lonlat(rect.x(), rect.y())
            lon_max, lat_max = mercator_to_lonlat(rect.x() + rect.width(),
                                                  rect.y() + rect.height())
            x0f, _ = lonlat_to_tile(lon_min, 0.0, zoom)
            x1f, _ = lonlat_to_tile(lon_max, 0.0, zoom)
            _, y0f = lonlat_to_tile(0.0, lat_max, zoom)
            _, y1f = lonlat_to_tile(0.0, lat_min, zoom)
            for tx in range(max(0, int(x0f)), min(n - 1, int(x1f)) + 1):
                for ty in range(max(0, int(y0f)), min(n - 1, int(y1f)) + 1):
                    key = (zoom, tx, ty)
                    if key in self._images or key in self._inflight:
                        continue
                    self._inflight.add(key)
                    self._pool.start(_TileWorker(
                        self._source_key, zoom, tx, ty,
                        self._cache_path(zoom, tx, ty), self._signals))
                    queued += 1
        self._prefetch_total = queued
        self._prefetch_done = 0
        if queued:
            self.prefetch_progress.emit(0, queued)
        return queued


class MapView(pg.GraphicsLayoutWidget):
    """平面地图视图：瓦片底图 + 测线轨迹折线。"""

    prefetch_progress = pyqtSignal(int, int)    # 已完成, 总数

    def __init__(self, source_key: str = DEFAULT_TILE_SOURCE, parent=None) -> None:
        super().__init__(parent)
        self._plot = self.addPlot()
        self._plot.hideAxis('bottom')
        self._plot.hideAxis('left')
        self._plot.setAspectLocked(True)
        self._plot.showGrid(False, False)

        self._layer = TileLayer(source_key)
        self._layer.prefetch_progress.connect(self.prefetch_progress)
        self._plot.addItem(self._layer)

        self._track_items: list = []
        self._track_summaries: list[dict] = []
        self.setBackground('w')

    # ------------------------------------------------------------ 底图
    def source_key(self) -> str:
        return self._layer.source_key()

    def set_source(self, source_key: str) -> None:
        self._layer.set_source(source_key)

    def prefetch_current_view(self, max_extra_zoom: int = 2) -> int:
        """预下载当前可视区域瓦片（离线包效果），返回入队瓦片数。"""
        return self._layer.prefetch_current_view(max_extra_zoom)

    # ------------------------------------------------------------ 轨迹
    def set_tracks(self, tracks, colors: dict | None = None) -> None:
        """设置测线轨迹并自动 fit 到全部轨迹范围。

        tracks: SpatialTrack 列表（鸭子类型取属性）；colors: {line_id: '#rrggbb'}。
        """
        for item in self._track_items:
            self._plot.removeItem(item)
        self._track_items = []
        self._track_summaries = []
        colors = dict(colors or {})

        all_x = []
        all_y = []
        for track in tracks or []:
            info = _track_to_mercator(track)
            self._track_summaries.append(info)
            xs, ys = info['xs'], info['ys']
            finite = np.isfinite(xs) & np.isfinite(ys)
            if np.count_nonzero(finite) < 1:
                continue
            xs, ys = xs[finite], ys[finite]
            color = QColor(colors.get(info['line_id'], '#1f77b4'))
            pen = pg.mkPen(color, width=2)
            line_item = self._plot.plot(xs, ys, pen=pen)
            endpoint_item = pg.ScatterPlotItem(
                [xs[0], xs[-1]], [ys[0], ys[-1]], symbol='o', size=9,
                pen=pg.mkPen(color, width=1), brush=pg.mkBrush(color))
            self._plot.addItem(endpoint_item)
            self._track_items.extend([line_item, endpoint_item])
            all_x.append(xs)
            all_y.append(ys)

        if all_x:
            xs = np.concatenate(all_x)
            ys = np.concatenate(all_y)
            margin_x = max((xs.max() - xs.min()) * 0.08, 50.0)
            margin_y = max((ys.max() - ys.min()) * 0.08, 50.0)
            self._plot.setRange(
                xRange=(xs.min() - margin_x, xs.max() + margin_x),
                yRange=(ys.min() - margin_y, ys.max() + margin_y),
                padding=0.0)

    def track_summaries(self) -> list[dict]:
        """最近一次 set_tracks 的坐标转换摘要（投影信息卡用）。"""
        return list(self._track_summaries)

    # ------------------------------------------------------------ 主题
    def apply_theme(self, dark: bool) -> None:
        """深色黑底 / 浅色白底（瓦片覆盖不到的空区可见）。"""
        self.setBackground('k' if dark else 'w')


__all__ = ['MapView', 'TileLayer']
