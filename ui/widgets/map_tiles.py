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
}
DEFAULT_TILE_SOURCE = 'gaode_img'

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


__all__ = [
    'EARTH_RADIUS_M', 'MAX_LATITUDE', 'WORLD_SIZE_M', 'TILE_SIZE_PX',
    'TILE_SOURCES', 'DEFAULT_TILE_SOURCE',
    'extract_epsg', 'lonlat_to_mercator', 'mercator_to_lonlat',
    'lonlat_to_tile', 'tile_bounds_mercator', 'tile_url',
    'zoom_for_resolution',
]
