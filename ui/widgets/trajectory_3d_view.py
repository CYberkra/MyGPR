# -*- coding: utf-8 -*-
"""Trajectory3DView — 三维测线轨迹视图（可叠加真实地形）。

pyqtgraph.opengl.GLViewWidget：GLGridItem 地面网格 + 每条测线一条
GLLinePlotItem（颜色与平面地图一致）。EPSG:4326（经纬度）轨迹先转
Web Mercator 米再显示——否则 xy（度，跨度 ~0.01）与 z（米，百米级）
尺度悬殊，测线会塌缩成一根竖线。坐标归一化到局部原点（全体点减均值），
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
from qfluentwidgets import FluentIcon as FIF

from ui.widgets.context_menus import add_action, make_menu
from ui.widgets.map_tiles import WORLD_SIZE_M, extract_epsg
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


def _lonlat_xyz_to_mercator(xyz: np.ndarray) -> np.ndarray:
    """经纬度（度）xyz → Web Mercator（米）xyz，z 原样保留。"""
    out = np.array(xyz, dtype=float, copy=True)
    radius = WORLD_SIZE_M / (2.0 * np.pi)
    out[:, 0] = radius * np.radians(out[:, 0])
    lat = np.radians(np.clip(out[:, 1], -85.0, 85.0))
    out[:, 1] = radius * np.log(np.tan(np.pi / 4.0 + lat / 2.0))
    return out


_TERRAIN_CMAP = None


def _terrain_cmap():
    """gist_earth 地形色表（绿谷→棕山→白雪顶），惰性构建。"""
    global _TERRAIN_CMAP
    if _TERRAIN_CMAP is None:
        import pyqtgraph as pg
        _TERRAIN_CMAP = pg.colormap.getFromMatplotlib('gist_earth')
    return _TERRAIN_CMAP


_TERRAIN_SHADER = 'mygprTerrain'


def _register_terrain_shader() -> None:
    """注册自定义地形着色器（幂等）。

    pyqtgraph 内置 'shaded' 的光照方向为 (1,-1,-1)，朝上的地表
    dot(normal, light) 恒为负 → 只剩 0.2 环境光，地形渲成近全黑。
    这里改为顶光 (0.35,-0.35,0.87) + abs() 双面受光 + 0.45 环境光。
    """
    from pyqtgraph.opengl import shaders
    if _TERRAIN_SHADER in shaders.ShaderProgram.names:
        return
    shaders.ShaderProgram(_TERRAIN_SHADER, [
        shaders.VertexShader("""
            uniform mat4 u_mvp;
            uniform mat3 u_normal;
            attribute vec4 a_position;
            attribute vec3 a_normal;
            attribute vec4 a_color;
            varying vec4 v_color;
            varying vec3 v_normal;
            void main() {
                v_normal = normalize(u_normal * a_normal);
                v_color = a_color;
                gl_Position = u_mvp * a_position;
            }
        """),
        shaders.FragmentShader("""
            #ifdef GL_ES
            precision mediump float;
            #endif
            varying vec4 v_color;
            varying vec3 v_normal;
            void main() {
                vec3 n = normalize(v_normal);
                float p = abs(dot(n, normalize(vec3(0.35, -0.35, 0.87))));
                vec3 rgb = v_color.rgb * (0.45 + 0.55 * p);
                gl_FragColor = vec4(rgb, v_color.a);
            }
        """)])


class _TerrainSignals(QObject):
    """QRunnable 无法自带信号，用独立 QObject 回主线程。"""

    finished = pyqtSignal(int, object)   # generation, payload(dict|None)


class _TerrainWorker(QRunnable):
    """后台地形构建：下载高程瓦片 → 马赛克 → 双线性重采样。

    只做 urllib 下载与 numpy 运算——pyproj/PROJ 坐标换算全部在 GUI
    线程由 ``_prepare_terrain`` 预先完成。PROJ C 库在工作线程内创建
    Transformer 会在 proj.dll 段错误（native-crash.log 实证），即使
    Python 层加锁串行化也无法避免，因此工作线程完全不碰 pyproj。
    """

    def __init__(self, generation: int, prep: dict,
                 cache_root: str, signals: _TerrainSignals) -> None:
        super().__init__()
        self._generation = int(generation)
        self._prep = prep            # _prepare_terrain 的预计算结果
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
            prep = self._prep
            zoom = prep['zoom']
            x0, x1, y0, y1 = prep['tile_range']
            tiles = {}
            for tx in range(x0, x1 + 1):
                for ty in range(y0, y1 + 1):
                    block = self._fetch_tile(zoom, tx, ty)
                    if block is not None:
                        tiles[(tx, ty)] = block
            if tiles:
                elev, xs_m, ys_m = mosaic_from_tiles(tiles, zoom, x0, x1, y0, y1)
                # 预计算的 Mercator 采样点 → 双线性采样 → 源坐标系规则网格
                z = sample_bilinear(elev, xs_m, ys_m, prep['mx'], prep['my'])
                z = z.reshape(prep['shape'])
                if np.isfinite(z).any():
                    mean = float(np.nanmean(z))
                    z = np.where(np.isfinite(z), z, np.float32(mean))
                    payload = {'gx': prep['gx'], 'gy': prep['gy'],
                               'z': z.astype(np.float32)}
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
        self._extent = 1.0   # set_tracks 记录的数据范围（重置视角用）
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
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
            if np.nanmax(np.abs(xyz[:, :2])) >= 1000.0:
                lonlat_like = False
            if epsg is None:
                epsg = extract_epsg(getattr(track, 'coordinate_system', ''))
        if not arrays:
            return

        # 坐标系启发：无 EPSG 且数值像经纬度 → 按 EPSG:4326
        if epsg is None and lonlat_like:
            epsg = 4326
        # 地形包围盒用原始源坐标计算（必须在 4326→Mercator 显示转换之前，
        # 否则会把 Mercator 米当经纬度再转一遍，地形整个失效）
        terrain_bbox = None
        if epsg is not None:
            raw_xyz = np.concatenate([xyz for _lid, xyz in arrays])
            pad_x = max((raw_xyz[:, 0].max() - raw_xyz[:, 0].min()) * 0.05, 1e-6)
            pad_y = max((raw_xyz[:, 1].max() - raw_xyz[:, 1].min()) * 0.05, 1e-6)
            terrain_bbox = (float(raw_xyz[:, 0].min() - pad_x),
                            float(raw_xyz[:, 1].min() - pad_y),
                            float(raw_xyz[:, 0].max() + pad_x),
                            float(raw_xyz[:, 1].max() + pad_y))
        if epsg == 4326:
            # 经纬度（度）→ Web Mercator（米）。否则 xy 跨度仅 ~0.01°，
            # 在高程（百米级）/网格尺度下测线塌缩成一根竖线（营山数据实测）
            arrays = [(lid, _lonlat_xyz_to_mercator(xyz))
                      for lid, xyz in arrays]

        # 全体点减均值 → 局部原点，z 同样归一化避免相机被大高程数值拉远
        origin = np.mean(np.concatenate([xyz for _lid, xyz in arrays]), axis=0)
        self._origin = origin
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
        self._extent = extent
        self._reset_camera()

        # 坐标系可识别 → 后台构建真实地形
        if terrain_bbox is not None:
            self._terrain_generation += 1
            # pyproj 坐标换算在 GUI 线程预计算（工作线程碰 PROJ 会段错误），
            # worker 只拿预计算结果做下载与采样
            prep = self._prepare_terrain(epsg, terrain_bbox)
            if prep is not None:
                self._terrain_pool.start(_TerrainWorker(
                    self._terrain_generation, prep,
                    _TERRAIN_CACHE_ROOT, self._terrain_signals))

    # ------------------------------------------------------------ 视角 / 右键菜单
    def _reset_camera(self) -> None:
        """按记录的数据范围重置网格尺寸与相机（右键"重置视角"同用）。"""
        if self._gl_view is None or self._grid is None:
            return
        extent = self._extent
        grid_size = max(extent * 2.0, 100.0)
        self._grid.setSize(grid_size, grid_size)
        self._grid.setSpacing(grid_size / 20.0, grid_size / 20.0)
        # 有地形时网格保持隐藏（旧逻辑：set_tracks 先清地形再显网格，等价）
        self._grid.setVisible(self._terrain_item is None)
        self._gl_view.setCameraPosition(
            distance=max(extent * 2.5, 100.0), elevation=30.0, azimuth=45.0)
        self._gl_view.opts['center'] = _Vector(0.0, 0.0, 0.0)

    def _show_context_menu(self, pos) -> None:
        if self._gl_view is None:
            return
        from qfluentwidgets import Action
        menu = make_menu(self)
        add_action(menu, FIF.ROTATE, '重置视角', self._reset_camera)
        menu.addSeparator()
        terrain_action = Action('显示地形')
        terrain_action.setCheckable(True)
        terrain_action.setChecked(
            self._terrain_item is not None and self._terrain_item.visible())
        terrain_action.triggered.connect(self._toggle_terrain)
        menu.addAction(terrain_action)
        grid_action = Action('显示网格')
        grid_action.setCheckable(True)
        grid_action.setChecked(self._grid is not None and self._grid.visible())
        grid_action.triggered.connect(self._toggle_grid)
        menu.addAction(grid_action)
        menu.exec(self.mapToGlobal(pos))

    def _toggle_terrain(self, checked: bool) -> None:
        if self._terrain_item is not None:
            self._terrain_item.setVisible(bool(checked))

    def _toggle_grid(self, checked: bool) -> None:
        if self._grid is not None:
            self._grid.setVisible(bool(checked))

    # ------------------------------------------------------------ 地形
    def _prepare_terrain(self, epsg: int, bbox: tuple) -> dict | None:
        """GUI 线程预计算地形任务的全部 pyproj 坐标换算。

        PROJ C 库在 QThreadPool 工作线程内创建 Transformer 会在 proj.dll
        段错误（native-crash.log 实证），故所有换算集中在此处（GUI 线程，
        与地图换算同线程、天然无并发）；worker 只做 urllib 下载与 numpy。
        返回 dict(zoom, tile_range, gx, gy, mx, my, shape)；失败返回 None
        （降级为平面网格）。
        """
        try:
            from ui.widgets.proj_safe import LockedTransformer
            x_min, y_min, x_max, y_max = bbox
            to_lonlat = LockedTransformer(epsg, 4326)
            lons, lats = to_lonlat.transform(
                [x_min, x_min, x_max, x_max], [y_min, y_min, y_max, y_max])
            zoom, x0, x1, y0, y1 = tile_grid_for_bbox(
                min(lons), min(lats), max(lons), max(lats))
            # 源坐标系规则网格 → 经纬度 → Mercator 米（worker 内双线性采样用）
            steps_x = min(_TERRAIN_GRID, 96)
            steps_y = min(_TERRAIN_GRID, 96)
            gx = np.linspace(x_min, x_max, steps_x)
            gy = np.linspace(y_min, y_max, steps_y)
            mesh_x, mesh_y = np.meshgrid(gx, gy)
            lon, lat = to_lonlat.transform(mesh_x.ravel(), mesh_y.ravel())
            mx, my = LockedTransformer(4326, 3857).transform(lon, lat)
            gx_out, gy_out = gx, gy
            if int(epsg) == 4326:
                # 显示坐标已转 Mercator 米（见 set_tracks），地形网格同步：
                # mx 仅随 lon 变、my 仅随 lat 变，可取 1D 网格轴
                mx2 = np.asarray(mx, dtype=float).reshape(mesh_x.shape)
                my2 = np.asarray(my, dtype=float).reshape(mesh_x.shape)
                gx_out = mx2[0, :]
                gy_out = my2[:, 0]
            return {'zoom': zoom, 'tile_range': (x0, x1, y0, y1),
                    'gx': gx_out, 'gy': gy_out,
                    'mx': np.asarray(mx, dtype=float),
                    'my': np.asarray(my, dtype=float),
                    'shape': mesh_x.shape}
        except Exception as exc:  # noqa: BLE001 - 换算失败降级平面网格
            _LOGGER.debug('地形预计算失败: %s', exc)
            return None
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
            # 均值对齐后局部地形仍可能高出测线几十米把线埋住：
            # 整体下移地形使其最高点始终低于测线最低点（留 1% 间距），
            # 保证测线永远浮在地形表面之上可见
            if self._line_items:
                line_z_min = min(float(li.pos[:, 2].min())
                                 for li in self._line_items)
                margin = max(self._extent * 0.01, 1.0)
                sink = line_z_min - margin - float(z.max())
                if sink < 0.0:
                    z = z + sink
            self._clear_terrain()
            # z 形状 (len(x), len(y))：pyqtgraph 期望 x 为第一维
            z = np.ascontiguousarray(z.T)
            # heightColor shader 按 z∈[0,1] 绝对值取色（实测米级高程渲成全黑），
            # 改用 gist_earth 地形色表逐顶点着色 + shaded 光照
            z_min = float(z.min())
            z_span = max(float(z.max()) - z_min, 1e-6)
            z_norm = (z - z_min) / z_span
            lut = _terrain_cmap().getLookupTable(0.0, 1.0, 256)
            # gist_earth 低端是深海深蓝（陆地区域渲成发黑），只取陆地段 0.38~1.0
            z_land = 0.38 + 0.62 * z_norm
            colors = lut[np.clip(
                (z_land * 255.0).astype(np.int64), 0, 255)]
            if colors.shape[-1] == 3:
                # getLookupTable 只回 RGB；GLSurfacePlotItem 顶点色必须 4 通道，
                # 否则 VBO 按 4 字节/顶点错行读取，颜色全乱（实证渲成近全黑）
                alpha = np.full(colors.shape[:-1] + (1,), 255, dtype=np.uint8)
                colors = np.concatenate([colors, alpha], axis=-1)
            # MeshData 原生支持 uint8 顶点色（0-255），转 float32 会把色值放大 255 倍
            _register_terrain_shader()
            item = _gl.GLSurfacePlotItem(
                x=x, y=y, z=z, colors=np.ascontiguousarray(colors, np.uint8),
                shader=_TERRAIN_SHADER, smooth=True)
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
