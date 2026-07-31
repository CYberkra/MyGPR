# -*- coding: utf-8 -*-
"""在线瓦片地图的纯数学 / 纯解析工具（无 Qt 依赖，可独立导入测试）。

包含：
- 经纬度 ↔ Web Mercator 米（EPSG:3857）换算
- 经纬度 → XYZ 瓦片编号（z/x/y）计算
- 瓦片范围 ↔ Web Mercator 米范围换算
- 坐标系字符串 EPSG 编码提取
- 内置瓦图源定义（OSM / 高德矢量 / 高德影像）
"""
from __future__ import annotations

import math
import re

# Web Mercator 常量
EARTH_RADIUS_M = 6378137.0
MAX_LATITUDE = 85.05112878          # EPSG:3857 有效纬度范围
WORLD_SIZE_M = 2 * math.pi * EARTH_RADIUS_M   # 世界宽度（米）
TILE_SIZE_PX = 256

# 内置瓦图源：key -> (显示名, URL 模板)。{s} 为高德子域占位。
TILE_SOURCES = {
    'osm': ('OpenStreetMap',
            'https://tile.openstreetmap.org/{z}/{x}/{y}.png'),
    'gaode_vec': ('高德矢量',
                  'https://webrd0{s}.is.autonavi.com/appmaptile'
                  '?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}'),
    'gaode_img': ('高德影像',
                  'https://webst0{s}.is.autonavi.com/appmaptile'
                  '?style=6&x={x}&y={y}&z={z}'),
    # 高德影像注记（透明 PNG：路网/地名），叠加在 gaode_img 之上
    'gaode_img_lbl': ('高德影像注记',
                      'https://webst0{s}.is.autonavi.com/appmaptile'
                      '?style=8&x={x}&y={y}&z={z}'),
}
DEFAULT_TILE_SOURCE = 'gaode_img'

# 各瓦图源有效最大级别：高德 z19 起只返回灰色占位图（实证，非真实影像）
TILE_SOURCE_MAX_ZOOM = {
    'osm': 19,
    'gaode_vec': 18,
    'gaode_img': 18,
    'gaode_img_lbl': 18,
}

# 使用 GCJ-02（火星坐标）的瓦图源：WGS84 测线叠加前需先做坐标加密转换，
# 否则在中国境内有 300~600 m 系统偏移（营山测区实测约 463 m）
_GCJ02_SOURCES = frozenset({'gaode_vec', 'gaode_img', 'gaode_img_lbl'})


def is_gcj02_source(source_key) -> bool:
    """瓦图源是否 GCJ-02 坐标（高德全系；OSM 为 WGS84）。"""
    return str(source_key or '') in _GCJ02_SOURCES


_GCJ_A = 6378245.0               # Krasovsky 椭球长半轴
_GCJ_EE = 0.00669342162296594323  # 偏心率平方


def _gcj_transform_lat(x: float, y: float) -> float:
    ret = (-100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y
           + 0.1 * x * y + 0.2 * math.sqrt(abs(x)))
    ret += (20.0 * math.sin(6.0 * x * math.pi)
            + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi)
            + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi)
            + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _gcj_transform_lon(x: float, y: float) -> float:
    ret = (300.0 + x + 2.0 * y + 0.1 * x * x
           + 0.1 * x * y + 0.1 * math.sqrt(abs(x)))
    ret += (20.0 * math.sin(6.0 * x * math.pi)
            + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi)
            + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi)
            + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lon: float, lat: float) -> tuple[float, float]:
    """WGS84 → GCJ-02（火星坐标加密）。中国境外坐标原样返回。"""
    lon, lat = float(lon), float(lat)
    if not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271):
        return lon, lat
    dlat = _gcj_transform_lat(lon - 105.0, lat - 35.0)
    dlon = _gcj_transform_lon(lon - 105.0, lat - 35.0)
    rad_lat = math.radians(lat)
    magic = 1.0 - _GCJ_EE * math.sin(rad_lat) ** 2
    sqrt_magic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_GCJ_A * (1.0 - _GCJ_EE))
                             / (magic * sqrt_magic) * math.pi)
    dlon = (dlon * 180.0) / (_GCJ_A / sqrt_magic
                             * math.cos(rad_lat) * math.pi)
    return lon + dlon, lat + dlat


def gcj02_to_wgs84(lon: float, lat: float) -> tuple[float, float]:
    """GCJ-02 → WGS84（迭代反解，3 次迭代收敛到亚米级）。"""
    lon, lat = float(lon), float(lat)
    if not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271):
        return lon, lat
    w_lon, w_lat = lon, lat
    for _ in range(3):
        g_lon, g_lat = wgs84_to_gcj02(w_lon, w_lat)
        w_lon -= g_lon - lon
        w_lat -= g_lat - lat
    return w_lon, w_lat

# 底图预设（用户可切换项）：key -> (显示名, 基础瓦图源, 叠加瓦图源|None)。
# key 与基础瓦图源同名，保证旧配置（spatial_basemap_source）仍然有效。
BASEMAP_LAYERS = {
    'gaode_img': ('卫星影像（高德·含路网标注）', 'gaode_img', 'gaode_img_lbl'),
    'gaode_vec': ('矢量地图（高德）', 'gaode_vec', None),
    'osm': ('OpenStreetMap', 'osm', None),
}


def resolve_basemap(key) -> tuple[str, str, str | None]:
    """底图 key → (底图 key, 基础瓦图源 key, 叠加瓦图源 key|None)。

    兼容直接传瓦图源 key（无叠加层）；未知 key 回退默认底图。
    """
    key = str(key or '')
    if key in BASEMAP_LAYERS:
        _display, base, overlay = BASEMAP_LAYERS[key]
        return key, base, overlay
    if key in TILE_SOURCES:
        return key, key, None
    _display, base, overlay = BASEMAP_LAYERS[DEFAULT_TILE_SOURCE]
    return DEFAULT_TILE_SOURCE, base, overlay

_EPSG_RE = re.compile(r'EPSG:(\d+)')


def extract_epsg(text) -> int | None:
    """从坐标系描述字符串中提取 EPSG 编码；无则返回 None。

    例: "CGCS2000 / 3-degree GK Zone 36 (auto) (EPSG:4524, zone 36)" -> 4524
    """
    match = _EPSG_RE.search(str(text or ''))
    return int(match.group(1)) if match else None


def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    """WGS84 经纬度（度）→ Web Mercator 米。"""
    lat = max(-MAX_LATITUDE, min(MAX_LATITUDE, float(lat)))
    x = EARTH_RADIUS_M * math.radians(float(lon))
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    """Web Mercator 米 → WGS84 经纬度（度）。"""
    lon = math.degrees(float(x) / EARTH_RADIUS_M)
    lat = math.degrees(2.0 * math.atan(math.exp(float(y) / EARTH_RADIUS_M)) - math.pi / 2.0)
    return lon, lat


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """经纬度 → XYZ 瓦片坐标（浮点，整数部分即瓦片编号 z/x/y）。

    纬度钳制到 Web Mercator 有效范围。
    """
    lat = max(-MAX_LATITUDE, min(MAX_LATITUDE, float(lat)))
    n = 2 ** int(zoom)
    x = (float(lon) + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_bounds_mercator(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """瓦片 (z,x,y) 的 Web Mercator 米范围 (x_min, y_min, x_max, y_max)。

    瓦片 y 轴向下（北在上），故 y_max 为北边界。
    """
    n = 2 ** int(z)
    tile_w = WORLD_SIZE_M / n
    x_min = -WORLD_SIZE_M / 2.0 + int(x) * tile_w
    y_max = WORLD_SIZE_M / 2.0 - int(y) * tile_w
    return x_min, y_max - tile_w, x_min + tile_w, y_max


def tile_url(source_key: str, z: int, x: int, y: int) -> str:
    """按瓦图源模板生成下载 URL；高德 {s} 子域按 (x+y) 轮询 1-4。"""
    template = TILE_SOURCES[source_key][1]
    return template.format(z=int(z), x=int(x), y=int(y), s=(int(x) + int(y)) % 4 + 1)


def zoom_for_resolution(meters_per_pixel: float, *, min_zoom: int = 1,
                        max_zoom: int = 19) -> int:
    """按当前分辨率（米/像素）选合适瓦片级别。"""
    mpp = max(float(meters_per_pixel), 1e-9)
    zoom = math.log2(WORLD_SIZE_M / (TILE_SIZE_PX * mpp))
    return max(min_zoom, min(max_zoom, int(round(zoom))))


def tile_range_for_bbox(lon_min: float, lat_min: float,
                        lon_max: float, lat_max: float,
                        zoom: int) -> tuple[int, int, int, int]:
    """经纬度包围盒在某级别下覆盖的瓦片编号范围 (x0, x1, y0, y1)（含端点）。

    纬度自动钳制到 Web Mercator 有效范围；经度按 [-180, 180] 截断。
    """
    n = 2 ** int(zoom)
    lon_min = max(-180.0, min(180.0, float(lon_min)))
    lon_max = max(-180.0, min(180.0, float(lon_max)))
    if lon_max < lon_min:
        lon_min, lon_max = lon_max, lon_min
    if lat_max < lat_min:
        lat_min, lat_max = lat_max, lat_min
    x0f, _ = lonlat_to_tile(lon_min, 0.0, zoom)
    x1f, _ = lonlat_to_tile(lon_max, 0.0, zoom)
    _, y0f = lonlat_to_tile(0.0, lat_max, zoom)   # 北 → 较小瓦片 y
    _, y1f = lonlat_to_tile(0.0, lat_min, zoom)
    return (max(0, min(n - 1, int(x0f))), max(0, min(n - 1, int(x1f))),
            max(0, min(n - 1, int(y0f))), max(0, min(n - 1, int(y1f))))


def count_tiles_for_bbox(lon_min: float, lat_min: float,
                         lon_max: float, lat_max: float,
                         zoom_min: int, zoom_max: int) -> int:
    """zoom_min..zoom_max 范围内覆盖包围盒的瓦片总数。"""
    total = 0
    for zoom in range(int(zoom_min), int(zoom_max) + 1):
        x0, x1, y0, y1 = tile_range_for_bbox(
            lon_min, lat_min, lon_max, lat_max, zoom)
        total += (x1 - x0 + 1) * (y1 - y0 + 1)
    return total


def choose_prefetch_zooms(lon_min: float, lat_min: float,
                          lon_max: float, lat_max: float,
                          *, detail_zoom: int = 16, overview_span: int = 4,
                          max_tiles: int = 400) -> tuple[int, int]:
    """为包围盒预下载选 (zoom_min, zoom_max)。

    detail_zoom 为目标最精细级别；zoom_min = detail_zoom - overview_span。
    若瓦片总数超过 max_tiles，则整体下调级别直到满足预算（不低于级别 3）。
    """
    detail_zoom = max(3, min(19, int(detail_zoom)))
    while detail_zoom > 3:
        zoom_min = max(1, detail_zoom - int(overview_span))
        if count_tiles_for_bbox(lon_min, lat_min, lon_max, lat_max,
                                zoom_min, detail_zoom) <= int(max_tiles):
            break
        detail_zoom -= 1
    zoom_min = max(1, detail_zoom - int(overview_span))
    return zoom_min, detail_zoom


__all__ = [
    'EARTH_RADIUS_M', 'MAX_LATITUDE', 'WORLD_SIZE_M', 'TILE_SIZE_PX',
    'TILE_SOURCES', 'DEFAULT_TILE_SOURCE', 'TILE_SOURCE_MAX_ZOOM',
    'BASEMAP_LAYERS', 'resolve_basemap', 'is_gcj02_source',
    'wgs84_to_gcj02', 'gcj02_to_wgs84',
    'extract_epsg', 'lonlat_to_mercator', 'mercator_to_lonlat',
    'lonlat_to_tile', 'tile_bounds_mercator', 'tile_url',
    'zoom_for_resolution', 'tile_range_for_bbox', 'count_tiles_for_bbox',
    'choose_prefetch_zooms',
]
