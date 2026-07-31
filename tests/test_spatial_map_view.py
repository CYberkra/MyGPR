# -*- coding: utf-8 -*-
"""空间信息页瓦片数学 / CRS 解析纯函数测试（无 Qt 依赖）。"""
from __future__ import annotations

import math

from ui.widgets.map_tiles import (choose_prefetch_zooms, count_tiles_for_bbox,
                                  extract_epsg, lonlat_to_mercator,
                                  lonlat_to_tile, mercator_to_lonlat,
                                  tile_bounds_mercator, tile_range_for_bbox)


# ------------------------------------------------------------------ 瓦片数学
def test_zoom0_world_is_single_tile():
    """z=0 全球一张瓦片：任意经纬度瓦片编号均为 (0, 0)。"""
    for lon, lat in ((0.0, 0.0), (120.0, 30.0), (-75.0, -45.0), (179.9, 85.0)):
        x, y = lonlat_to_tile(lon, lat, 0)
        assert int(x) == 0 and int(y) == 0


def test_zoom1_eastern_longitude_x_at_least_one():
    """z=1 时 lon>0（东半球）瓦片 x >= 1。"""
    x, _ = lonlat_to_tile(100.0, 30.0, 1)
    assert x >= 1.0
    assert int(x) == 1


def test_equator_y_is_half_world():
    """lat=0 时瓦片 y 恰为 2^(z-1)（世界纵向中点）。"""
    for zoom in (1, 3, 10, 18):
        _, y = lonlat_to_tile(106.0, 0.0, zoom)
        assert math.isclose(y, 2.0 ** (zoom - 1), rel_tol=0.0, abs_tol=1e-9)


def test_mercator_roundtrip():
    """经纬度 ↔ Web Mercator 米互换算闭环。"""
    lon, lat = 106.55, 31.08
    x, y = lonlat_to_mercator(lon, lat)
    lon2, lat2 = mercator_to_lonlat(x, y)
    assert math.isclose(lon2, lon, abs_tol=1e-9)
    assert math.isclose(lat2, lat, abs_tol=1e-9)


def test_tile_bounds_consistent_with_tile_math():
    """瓦片米范围与其中心点反算的瓦片编号一致。"""
    z, tx, ty = 12, 3309, 1687
    x0, y0, x1, y1 = tile_bounds_mercator(z, tx, ty)
    assert x1 > x0 and y1 > y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    lon, lat = mercator_to_lonlat(cx, cy)
    fx, fy = lonlat_to_tile(lon, lat, z)
    assert int(fx) == tx and int(fy) == ty


# ------------------------------------------------------------------ CRS 解析
def test_extract_epsg_from_cgcs2000_auto_string():
    text = "CGCS2000 / 3-degree GK Zone 36 (auto) (EPSG:4524, zone 36)"
    assert extract_epsg(text) == 4524


def test_extract_epsg_returns_none_without_code():
    assert extract_epsg("") is None
    assert extract_epsg(None) is None
    assert extract_epsg("WGS 84 地理坐标系") is None


def test_extract_epsg_simple():
    assert extract_epsg("EPSG:4326") == 4326
    assert extract_epsg("prefix EPSG:3857 suffix") == 3857


# ------------------------------------------------------------------ 包围盒预下载
def test_tile_range_for_bbox_small_area_single_tile():
    """小范围（百米级测区）在低级别下应收敛到单张瓦片。"""
    x0, x1, y0, y1 = tile_range_for_bbox(106.55, 31.08, 106.56, 31.09, 12)
    assert x0 == x1 and y0 == y1


def test_tile_range_for_bbox_covers_all_points():
    """包围盒内任意采样点的瓦片编号都落在返回范围内。"""
    bbox = (106.5, 31.0, 106.7, 31.2)
    zoom = 13
    x0, x1, y0, y1 = tile_range_for_bbox(*bbox, zoom)
    for i in range(5):
        lon = bbox[0] + (bbox[2] - bbox[0]) * i / 4.0
        lat = bbox[1] + (bbox[3] - bbox[1]) * i / 4.0
        fx, fy = lonlat_to_tile(lon, lat, zoom)
        assert x0 <= int(fx) <= x1
        assert y0 <= int(fy) <= y1


def test_tile_range_for_bbox_swapped_and_clamped():
    """经纬度顺序颠倒自动纠正；纬度越界钳制不抛异常。"""
    x0, x1, y0, y1 = tile_range_for_bbox(106.7, 95.0, 106.5, -95.0, 10)
    assert x1 >= x0 and y1 >= y0
    assert 0 <= x0 <= x1 < 2 ** 10
    assert 0 <= y0 <= y1 < 2 ** 10


def test_count_tiles_for_bbox_monotonic_with_zoom():
    """级别越高瓦片数越多。"""
    bbox = (106.5, 31.0, 106.6, 31.1)
    low = count_tiles_for_bbox(*bbox, 10, 11)
    high = count_tiles_for_bbox(*bbox, 14, 15)
    assert high > low >= 1


def test_choose_prefetch_zooms_respects_tile_budget():
    """选定级别范围后瓦片总数不超预算（小测区直接给足 detail_zoom）。"""
    bbox = (106.55, 31.05, 106.57, 31.07)   # 约 2km 测区
    z0, z1 = choose_prefetch_zooms(*bbox, detail_zoom=16, max_tiles=400)
    assert z1 == 16
    assert count_tiles_for_bbox(*bbox, z0, z1) <= 400


def test_choose_prefetch_zooms_large_area_drops_zoom():
    """超大区域（全省）必须降级别满足预算，且不低于级别 3。"""
    bbox = (97.0, 26.0, 108.0, 34.0)   # 四川量级
    z0, z1 = choose_prefetch_zooms(*bbox, detail_zoom=16, max_tiles=400)
    assert z1 < 16
    assert z1 >= 3
    assert count_tiles_for_bbox(*bbox, z0, z1) <= 400
