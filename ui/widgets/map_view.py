# -*- coding: utf-8 -*-
"""MapView — 在线瓦片地图 + 测线轨迹叠加视图。

视图坐标系为 Web Mercator 米（EPSG:3857），基于 pyqtgraph GraphicsLayoutWidget：

- TileLayer（pg.GraphicsObject）：按当前可视范围/缩放级别计算所需 XYZ 瓦片，
  只绘制磁盘缓存已有的；缺失瓦片入队 QThreadPool 后台下载（urllib + UA 头，
  失败静默），下载完成后 update() 重绘。磁盘缓存
  ``ui.desktop_backend_facade.tile_cache_dir()/{source_key}/{z}/{x}/{y}.png``，
  命中缓存不再联网。
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
from PyQt6.QtCore import QObject, QRectF, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import ToolButton

from ui.desktop_backend_facade import tile_cache_dir
from ui.widgets.context_menus import add_action, make_menu
from ui.widgets.map_tiles import (DEFAULT_TILE_SOURCE, TILE_SOURCE_MAX_ZOOM,
                                  TILE_SOURCES, WORLD_SIZE_M,
                                  choose_prefetch_zooms, extract_epsg,
                                  gcj02_to_wgs84, is_gcj02_source,
                                  lonlat_to_mercator, lonlat_to_tile,
                                  mercator_to_lonlat, resolve_basemap,
                                  tile_bounds_mercator,
                                  tile_range_for_bbox, tile_url,
                                  wgs84_to_gcj02, zoom_for_resolution)

_LOGGER = logging.getLogger(__name__)

_TILE_CACHE_ROOT = tile_cache_dir()
_USER_AGENT = 'MyGPR/1.0 (https://mygpr.local; tile client)'
_HALF_WORLD = WORLD_SIZE_M / 2.0


def _track_to_mercator(track, gcj02: bool = False) -> dict:
    """把 SpatialTrack（鸭子类型）转换为展示用结构（Web Mercator 米）。

    返回 dict(line_id, name, xs, ys, zs, crs, source, mapped, epsg)。
    mapped=False 时 xs/ys 为原始坐标（不套底图）。

    gcj02=True 时（底图为高德等火星坐标瓦片），WGS84/CGCS2000 经纬度
    先做 GCJ-02 加密转换再转 Mercator，否则中国境内有 300~600 m 偏移。
    """
    points = list(getattr(track, 'points', ()) or ())
    xs = np.asarray([float(getattr(p, 'x', 0.0)) for p in points], dtype=float)
    ys = np.asarray([float(getattr(p, 'y', 0.0)) for p in points], dtype=float)
    zs = np.asarray([float(getattr(p, 'elevation_m', 0.0)) for p in points], dtype=float)
    crs = str(getattr(track, 'coordinate_system', '') or '')
    source = str(getattr(track, 'source', '') or '')
    epsg = extract_epsg(crs)
    mapped = False
    if gcj02 and epsg is not None:
        try:
            # 先转 WGS84 经纬度（always_xy → lon,lat），再 GCJ-02 → Mercator
            from ui.widgets.proj_safe import transform_coordinates
            xs, ys = (np.asarray(a, dtype=float)
                      for a in transform_coordinates(epsg, 4326, xs, ys))
            converted = [wgs84_to_gcj02(lon, lat) for lon, lat in zip(xs, ys)]
            converted = [lonlat_to_mercator(glon, glat) for glon, glat in converted]
            xs = np.asarray([c[0] for c in converted], dtype=float)
            ys = np.asarray([c[1] for c in converted], dtype=float)
            mapped = True
        except Exception as exc:  # noqa: BLE001 - 单条测线失败不影响其它
            _LOGGER.debug('测线 GCJ-02 坐标转换失败 EPSG:%s: %s', epsg, exc)
    if not mapped and epsg is not None:
        try:
            # 经 proj_safe 串行化：与地形构建工作线程并行调 PROJ 会段错误
            from ui.widgets.proj_safe import transform_coordinates
            xs, ys = (np.asarray(a, dtype=float)
                      for a in transform_coordinates(epsg, 3857, xs, ys))
            mapped = True
        except Exception as exc:  # noqa: BLE001 - 单条测线失败不影响其它
            _LOGGER.debug('测线坐标转换失败 EPSG:%s: %s', epsg, exc)
    if not mapped and epsg is None and xs.size:
        # 无 EPSG 且坐标绝对值像经纬度（<1000）→ 按 EPSG:4326 处理
        if (np.nanmax(np.abs(xs)) < 1000.0 and np.nanmax(np.abs(ys)) < 1000.0
                and np.isfinite(xs).all() and np.isfinite(ys).all()):
            if gcj02:
                converted = [wgs84_to_gcj02(lon, lat) for lon, lat in zip(xs, ys)]
                xs = np.asarray([c[0] for c in converted], dtype=float)
                ys = np.asarray([c[1] for c in converted], dtype=float)
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

    def _max_zoom(self) -> int:
        """该瓦图源的有效最大级别（超出会返回占位图）。"""
        return TILE_SOURCE_MAX_ZOOM.get(self._source_key, 19)

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
        zoom = min(zoom_for_resolution(mpp), self._max_zoom())
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
    def prefetch_current_view(self, max_extra_zoom: int = 2,
                              report_progress: bool = True) -> int:
        """把当前可视范围在 zoom..zoom+max_extra_zoom 的瓦片全部入队下载。

        返回入队总瓦片数；report_progress=True 时完成进度经
        prefetch_progress 信号回报（叠加层用 False，避免进度重复计数）。
        """
        view_box = self.getViewBox()
        if view_box is None:
            return 0
        rect = view_box.viewRect()
        if rect.width() <= 0:
            return 0
        mpp = rect.width() / max(view_box.width(), 1.0)
        base_zoom = min(zoom_for_resolution(mpp), self._max_zoom())
        self._failed.clear()
        queued = 0
        zoom_top = min(base_zoom + max(0, int(max_extra_zoom)), self._max_zoom())
        for zoom in range(base_zoom, zoom_top + 1):
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
        self._prefetch_total = queued if report_progress else 0
        self._prefetch_done = 0
        if queued and report_progress:
            self.prefetch_progress.emit(0, queued)
        return queued

    def prefetch_region(self, lon_min: float, lat_min: float,
                        lon_max: float, lat_max: float,
                        *, detail_zoom: int = 16,
                        max_tiles: int = 400,
                        report_progress: bool = True) -> int:
        """预下载经纬度包围盒区域的瓦片（多级别，受 max_tiles 预算约束）。

        级别范围由 map_tiles.choose_prefetch_zooms 决定：默认精细到 z16，
        向上 4 级概览；区域过大时自动降级别保证总量不超预算。
        返回入队总瓦片数；report_progress=True 时完成进度经
        prefetch_progress 信号回报（叠加层用 False，避免进度重复计数）。
        """
        zoom_min, zoom_max = choose_prefetch_zooms(
            lon_min, lat_min, lon_max, lat_max,
            detail_zoom=detail_zoom, max_tiles=max_tiles)
        zoom_max = min(zoom_max, self._max_zoom())
        zoom_min = min(zoom_min, zoom_max)
        self._failed.clear()
        queued = 0
        for zoom in range(zoom_min, zoom_max + 1):
            x0, x1, y0, y1 = tile_range_for_bbox(
                lon_min, lat_min, lon_max, lat_max, zoom)
            for tx in range(x0, x1 + 1):
                for ty in range(y0, y1 + 1):
                    key = (zoom, tx, ty)
                    if key in self._images or key in self._inflight:
                        continue
                    self._inflight.add(key)
                    self._pool.start(_TileWorker(
                        self._source_key, zoom, tx, ty,
                        self._cache_path(zoom, tx, ty), self._signals))
                    queued += 1
        self._prefetch_total = queued if report_progress else 0
        self._prefetch_done = 0
        if queued and report_progress:
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
        # 关闭 pyqtgraph 原生英文右键菜单，右键由自定义 RoundMenu 接管
        self._plot.vb.setMenuEnabled(False)

        self._basemap_key, base_key, overlay_key = resolve_basemap(source_key)
        self._layer = TileLayer(base_key)
        self._layer.prefetch_progress.connect(self.prefetch_progress)
        self._plot.addItem(self._layer)
        # 叠加层（如影像注记）：惰性创建，压在基础层之上、测线之下
        self._overlay_layer: TileLayer | None = None
        if overlay_key is not None:
            self._ensure_overlay_layer(overlay_key)

        self._track_items: list = []
        self._track_summaries: list[dict] = []
        self._track_colors: dict = {}
        self._raw_tracks: list = []
        self._dark = False
        self.setBackground('w')
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # ---------------- 浮动覆盖层（子控件，resizeEvent 中定位） ----------------
        # 左上：测线图例（色块 + 名称），set_tracks 时重建
        self._legend = QFrame(self)
        self._legend.setObjectName('mapLegend')
        self._legend_layout = QVBoxLayout(self._legend)
        self._legend_layout.setContentsMargins(10, 8, 10, 8)
        self._legend_layout.setSpacing(4)
        self._legend_labels: list = []
        self._legend.hide()

        # 右上：缩放按钮列（放大 / 缩小 / 适应全部测线）
        self._zoom_panel = QFrame(self)
        self._zoom_panel.setObjectName('mapZoomPanel')
        zoom_layout = QVBoxLayout(self._zoom_panel)
        zoom_layout.setContentsMargins(4, 4, 4, 4)
        zoom_layout.setSpacing(4)
        for icon, tip, slot in (
                (FIF.ZOOM_IN, '放大', self._zoom_in),
                (FIF.ZOOM_OUT, '缩小', self._zoom_out),
                (FIF.FIT_PAGE, '适应全部测线', self.fit_to_tracks)):
            btn = ToolButton(icon, self._zoom_panel)
            btn.setFixedSize(30, 30)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            zoom_layout.addWidget(btn)

        # 右下：鼠标位置经纬度实时读出（纬度,经度，与"复制中心坐标"格式一致）
        self._coord_label = QLabel(self)
        self._coord_label.setObjectName('mapCoordLabel')
        self._coord_label.hide()

        self._restyle_overlays()

    # ------------------------------------------------------------ 底图
    def source_key(self) -> str:
        return self._basemap_key

    def _ensure_overlay_layer(self, overlay_key: str) -> 'TileLayer':
        """惰性创建叠加瓦片层（Z 值高于基础层、低于测线轨迹）。"""
        if self._overlay_layer is None:
            self._overlay_layer = TileLayer(overlay_key)
            self._overlay_layer.setZValue(-99.0)
            self._plot.addItem(self._overlay_layer)
        return self._overlay_layer

    def set_source(self, source_key: str) -> None:
        """切换底图（BASEMAP_LAYERS 预设 key；兼容直接传瓦图源 key）。"""
        key, base_key, overlay_key = resolve_basemap(source_key)
        if key == self._basemap_key and (
                self._overlay_layer is None
                or self._overlay_layer.source_key() == overlay_key):
            return
        self._basemap_key = key
        self._layer.set_source(base_key)
        if overlay_key is not None:
            self._ensure_overlay_layer(overlay_key).set_source(overlay_key)
            self._overlay_layer.setVisible(True)
        elif self._overlay_layer is not None:
            self._overlay_layer.setVisible(False)
        # 底图坐标系可能变化（高德 GCJ-02 ↔ OSM WGS84），测线需重新转换
        if self._raw_tracks:
            self._render_tracks()

    def prefetch_current_view(self, max_extra_zoom: int = 2) -> int:
        """预下载当前可视区域瓦片（离线包效果），返回基础层入队瓦片数。"""
        queued = self._layer.prefetch_current_view(max_extra_zoom)
        if self._overlay_layer is not None and self._overlay_layer.isVisible():
            self._overlay_layer.prefetch_current_view(
                max_extra_zoom, report_progress=False)
        return queued

    def tracks_bbox_lonlat(self) -> tuple | None:
        """已配准轨迹的经纬度联合包围盒 (lon_min, lat_min, lon_max, lat_max)。

        没有任何已配准（mapped）轨迹时返回 None。
        """
        xs, ys = [], []
        for info in self._track_summaries:
            if not info.get('mapped'):
                continue
            tx = info.get('xs')
            ty = info.get('ys')
            if tx is None or ty is None or not len(tx):
                continue
            finite = np.isfinite(tx) & np.isfinite(ty)
            if np.count_nonzero(finite):
                xs.append(np.asarray(tx)[finite])
                ys.append(np.asarray(ty)[finite])
        if not xs:
            return None
        mx = np.concatenate(xs)
        my = np.concatenate(ys)
        lon_min, lat_min = mercator_to_lonlat(mx.min(), my.min())
        lon_max, lat_max = mercator_to_lonlat(mx.max(), my.max())
        # 外扩约 8%（最小约 0.002° ≈ 200m），保证轨迹边缘不在瓦片边界上
        pad_lon = max((lon_max - lon_min) * 0.08, 0.002)
        pad_lat = max((lat_max - lat_min) * 0.08, 0.002)
        return (lon_min - pad_lon, lat_min - pad_lat,
                lon_max + pad_lon, lat_max + pad_lat)

    def prefetch_tracks(self, *, detail_zoom: int = 16,
                        max_tiles: int = 400) -> int:
        """按测线包围盒预下载对应地理区域瓦片；无已配准轨迹返回 0。"""
        bbox = self.tracks_bbox_lonlat()
        if bbox is None:
            return 0
        queued = self._layer.prefetch_region(
            *bbox, detail_zoom=detail_zoom, max_tiles=max_tiles)
        if self._overlay_layer is not None and self._overlay_layer.isVisible():
            self._overlay_layer.prefetch_region(
                *bbox, detail_zoom=detail_zoom, max_tiles=max_tiles,
                report_progress=False)
        return queued

    # ------------------------------------------------------------ 轨迹
    def set_tracks(self, tracks, colors: dict | None = None) -> None:
        """设置测线轨迹并自动 fit 到全部轨迹范围。

        tracks: SpatialTrack 列表（鸭子类型取属性）；colors: {line_id: '#rrggbb'}。
        原始轨迹留存：切换底图（WGS84 ↔ GCJ-02）时按新坐标系重新转换渲染。
        """
        self._raw_tracks = list(tracks or [])
        self._track_colors = dict(colors or {})
        self._render_tracks()

    def _render_tracks(self) -> None:
        """按当前底图坐标系（WGS84 / GCJ-02）转换并绘制测线。"""
        for item in self._track_items:
            self._plot.removeItem(item)
        self._track_items = []
        self._track_summaries = []
        colors = self._track_colors
        gcj02 = is_gcj02_source(self._layer.source_key())

        all_x = []
        all_y = []
        for track in self._raw_tracks:
            info = _track_to_mercator(track, gcj02=gcj02)
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

        self.fit_to_tracks()
        self._rebuild_legend()

    # ------------------------------------------------------------ 浮动覆盖层
    def _zoom_in(self) -> None:
        self._plot.vb.scaleBy((0.75, 0.75))

    def _zoom_out(self) -> None:
        self._plot.vb.scaleBy((1.0 / 0.75, 1.0 / 0.75))

    def _on_mouse_moved(self, pos) -> None:
        """鼠标位置 → 经纬度实时读出（视图世界坐标恒为 Web Mercator 米）。"""
        if not self._plot.sceneBoundingRect().contains(pos):
            self._coord_label.hide()
            return
        point = self._plot.vb.mapSceneToView(pos)
        lon, lat = mercator_to_lonlat(point.x(), point.y())
        if not (np.isfinite(lon) and np.isfinite(lat)) or abs(lat) > 85.06:
            self._coord_label.hide()
            return
        # 高德底图的世界坐标是 GCJ-02，读出统一回 WGS84 与测线数据一致
        if is_gcj02_source(self._layer.source_key()):
            lon, lat = gcj02_to_wgs84(lon, lat)
        self._coord_label.setText(f'{lat:.6f}, {lon:.6f}')
        self._coord_label.show()
        self._layout_overlays()

    def _rebuild_legend(self) -> None:
        """按当前测线重建左上图例（色块 + 名称）；无测线时隐藏。"""
        while self._legend_layout.count():
            child = self._legend_layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.setParent(None)   # 立刻摘除，避免与重建内容重叠一帧
                widget.deleteLater()
        self._legend_labels = []
        entries = []
        for info in self._track_summaries:
            label = info.get('name') or info.get('line_id')
            if not label:
                continue
            color = self._track_colors.get(info.get('line_id'), '#1f77b4')
            entries.append((label, color))
        if not entries:
            self._legend.hide()
            return
        text_color = '#f0f0f0' if self._dark else '#202020'
        for label, color in entries:
            row = QWidget(self._legend)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            swatch = QFrame(row)
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f'background-color: {color}; border-radius: 3px;'
                f' border: 1px solid rgba(0,0,0,80);')
            name = QLabel(str(label), row)
            name.setStyleSheet(f'color: {text_color}; background: transparent;')
            self._legend_labels.append(name)
            row_layout.addWidget(swatch)
            row_layout.addWidget(name)
            row_layout.addStretch(1)
            self._legend_layout.addWidget(row)
        self._legend.show()
        self._layout_overlays()

    def _restyle_overlays(self) -> None:
        """覆盖层样式跟随深浅主题（图例行不重建，只刷新文字颜色）。"""
        if self._dark:
            panel = ('background-color: rgba(32,32,32,200);'
                     ' border: 1px solid rgba(255,255,255,45); border-radius: 8px;')
            text = '#f0f0f0'
        else:
            panel = ('background-color: rgba(255,255,255,220);'
                     ' border: 1px solid rgba(0,0,0,45); border-radius: 8px;')
            text = '#202020'
        self._legend.setStyleSheet(f'QFrame#mapLegend {{ {panel} }}')
        self._zoom_panel.setStyleSheet(f'QFrame#mapZoomPanel {{ {panel} }}')
        self._coord_label.setStyleSheet(
            f'QLabel#mapCoordLabel {{ {panel} color: {text}; padding: 3px 8px; }}')
        for name in self._legend_labels:
            name.setStyleSheet(f'color: {text}; background: transparent;')

    def _layout_overlays(self) -> None:
        """把浮动覆盖层定位到四角（resize / 内容尺寸变化时调用）。"""
        # pyqtgraph addPlot() 内部会提前触发 resizeEvent，此时覆盖层尚未创建
        if not hasattr(self, '_legend'):
            return
        margin = 10
        if self._legend.isVisible():
            self._legend.adjustSize()
            self._legend.move(margin, margin)
        self._zoom_panel.adjustSize()
        self._zoom_panel.move(
            self.width() - self._zoom_panel.width() - margin, margin)
        if self._coord_label.isVisible():
            self._coord_label.adjustSize()
            self._coord_label.move(
                self.width() - self._coord_label.width() - margin,
                self.height() - self._coord_label.height() - margin)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_overlays()

    def fit_to_tracks(self) -> None:
        """视野自适应到全部轨迹范围（右键菜单"适应全部测线"同用）。"""
        all_x, all_y = [], []
        for info in self._track_summaries:
            xs, ys = info.get('xs'), info.get('ys')
            if xs is None or ys is None or not len(xs):
                continue
            finite = np.isfinite(xs) & np.isfinite(ys)
            if np.count_nonzero(finite):
                all_x.append(np.asarray(xs)[finite])
                all_y.append(np.asarray(ys)[finite])
        if all_x:
            xs = np.concatenate(all_x)
            ys = np.concatenate(all_y)
            margin_x = max((xs.max() - xs.min()) * 0.08, 50.0)
            margin_y = max((ys.max() - ys.min()) * 0.08, 50.0)
            self._plot.setRange(
                xRange=(xs.min() - margin_x, xs.max() + margin_x),
                yRange=(ys.min() - margin_y, ys.max() + margin_y),
                padding=0.0)

    # ------------------------------------------------------------ 右键菜单
    def _on_mouse_clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.RightButton:
            return
        if not self._plot.sceneBoundingRect().contains(event.scenePos()):
            return
        menu = make_menu(self)
        add_action(menu, FIF.FIT_PAGE, '适应全部测线', self.fit_to_tracks,
                   enabled=bool(self._track_summaries))
        add_action(menu, FIF.DOWNLOAD, '下载当前区域瓦片',
                   self.prefetch_current_view)
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制中心坐标', self._copy_center_lonlat)
        menu.exec(event.screenPos().toPoint())

    def _copy_center_lonlat(self) -> None:
        """视图中心 → WGS84 经纬度（'纬度,经度'，便于粘贴到奥维等软件对照）。"""
        center = self._plot.vb.viewRect().center()
        lon, lat = mercator_to_lonlat(center.x(), center.y())
        if is_gcj02_source(self._layer.source_key()):
            lon, lat = gcj02_to_wgs84(lon, lat)
        QApplication.clipboard().setText(f'{lat:.6f},{lon:.6f}')

    def track_summaries(self) -> list[dict]:
        """最近一次 set_tracks 的坐标转换摘要（投影信息卡用）。"""
        return list(self._track_summaries)

    # ------------------------------------------------------------ 主题
    def apply_theme(self, dark: bool) -> None:
        """深色黑底 / 浅色白底（瓦片覆盖不到的空区可见）。"""
        self._dark = bool(dark)
        self.setBackground('k' if dark else 'w')
        self._restyle_overlays()


__all__ = ['MapView', 'TileLayer']
