# -*- coding: utf-8 -*-
"""真实地形（Terrarium 高程瓦片）纯函数测试（无 Qt 依赖）。"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from ui.widgets.terrain_tiles import (decode_terrarium, mosaic_from_tiles,
                                      sample_bilinear, terrarium_url,
                                      tile_grid_for_bbox)


def _png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), 'RGB').save(buffer, format='PNG')
    return buffer.getvalue()


# ------------------------------------------------------------------ 解码
def test_terrarium_url_format():
    assert terrarium_url(12, 3309, 1687).endswith('/terrarium/12/3309/1687.png')


def test_decode_terrarium_sea_level_is_zero():
    """R*256+G+B/256 = 32768 → 高程 0（海平面）。"""
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :, 0] = 128     # 128*256 = 32768
    elev = decode_terrarium(_png_bytes(rgb))
    assert elev.shape == (4, 4)
    np.testing.assert_allclose(elev, 0.0, atol=1e-3)


def test_decode_terrarium_positive_elevation():
    """128*256 + 100 → 100m。"""
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[:, :, 0] = 128
    rgb[:, :, 1] = 100
    elev = decode_terrarium(_png_bytes(rgb))
    np.testing.assert_allclose(elev, 100.0, atol=1e-3)


# ------------------------------------------------------------------ 瓦片选级
def test_tile_grid_for_bbox_respects_budget():
    zoom, x0, x1, y0, y1 = tile_grid_for_bbox(
        106.5, 31.0, 106.7, 31.2, max_tiles=64, preferred_zoom=13)
    assert 3 <= zoom <= 13
    assert (x1 - x0 + 1) * (y1 - y0 + 1) <= 64


def test_tile_grid_for_bbox_huge_area_drops_zoom():
    zoom, x0, x1, y0, y1 = tile_grid_for_bbox(
        97.0, 26.0, 108.0, 34.0, max_tiles=64, preferred_zoom=13)
    assert zoom < 13
    assert (x1 - x0 + 1) * (y1 - y0 + 1) <= 64


# ------------------------------------------------------------------ 马赛克
def test_mosaic_from_tiles_places_blocks_and_flips_y():
    """2×1 瓦片（东西相邻）：列拼接；北瓦片行应位于马赛克上半（y 升序）。"""
    zoom = 5
    x0, x1, y0, y1 = 10, 11, 20, 20
    west = np.full((256, 256), 100.0, dtype=np.float32)
    east = np.full((256, 256), 200.0, dtype=np.float32)
    elev, xs, ys = mosaic_from_tiles({(10, 20): west, (11, 20): east},
                                     zoom, x0, x1, y0, y1)
    assert elev.shape == (256, 512)
    assert np.all(elev[:, :256] == 100.0)
    assert np.all(elev[:, 256:] == 200.0)
    assert xs.size == 512 and ys.size == 256
    assert np.all(np.diff(xs) > 0) and np.all(np.diff(ys) > 0)


def test_mosaic_from_tiles_missing_tile_is_nan():
    zoom = 5
    elev, _xs, _ys = mosaic_from_tiles({}, zoom, 10, 10, 20, 20)
    assert np.isnan(elev).all()


def test_mosaic_from_tiles_flips_rows_within_tile():
    """瓦片内部行序必须翻转：瓦片行 0 为北缘，马赛克行 0 为南缘。

    回归：漏掉内部翻转会把瓦片尺度内的地形南北镜像（营山实测
    下坡地形显示成上坡，与 GPS 轨迹坡向相反）。
    """
    zoom = 5
    # 单瓦片：行 0（北）= 500m，逐行向南递减到行 255（南）= 245m
    block = (500.0 - np.arange(256, dtype=np.float32)).reshape(256, 1)
    block = np.repeat(block, 256, axis=1)
    elev, xs, ys = mosaic_from_tiles({(10, 20): block}, zoom, 10, 10, 20, 20)
    assert np.all(np.diff(ys) > 0)
    # 马赛克行 0（最南）= 瓦片最后一行 245m；行 255（最北）= 500m
    assert elev[0, 0] == pytest.approx(245.0)
    assert elev[-1, 0] == pytest.approx(500.0)
    # 沿 y 升序（南→北）采样应得到递增高程（避开边缘一行，双线性需在格点内）
    qy = np.array([ys[8], ys[-8]])
    qx = np.array([xs[8], xs[8]])
    profile = sample_bilinear(elev, xs, ys, qx, qy)
    assert np.isfinite(profile).all()
    assert profile[0] < profile[1]


# ------------------------------------------------------------------ 双线性采样
def test_sample_bilinear_linear_field():
    """线性场 z = x + 2y 上采样误差应≈0。"""
    xs = np.linspace(0.0, 10.0, 11)
    ys = np.linspace(0.0, 20.0, 21)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    elev = (mesh_x + 2.0 * mesh_y).astype(np.float32)
    qx = np.array([2.5, 7.3])
    qy = np.array([5.5, 11.1])
    got = sample_bilinear(elev, xs, ys, qx, qy)
    np.testing.assert_allclose(got, qx + 2.0 * qy, atol=1e-4)


def test_sample_bilinear_outside_is_nan():
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 2.0])
    elev = np.ones((3, 3), dtype=np.float32)
    got = sample_bilinear(elev, xs, ys, np.array([-1.0, 5.0]),
                          np.array([0.5, 0.5]))
    assert np.isnan(got).all()


def test_sample_bilinear_nan_neighborhood_is_nan():
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 2.0])
    elev = np.ones((3, 3), dtype=np.float32)
    elev[0, 0] = np.nan
    got = sample_bilinear(elev, xs, ys, np.array([0.4, 1.5]),
                          np.array([0.4, 1.5]))
    assert np.isnan(got[0])
    assert np.isfinite(got[1])


# ------------------------------------------------------------------ 影像采样
def test_sample_imagery_pixels_single_tile():
    """单色瓦片：块内采样回该色；全局像素坐标以瓦片左上角为原点。"""
    from ui.widgets.terrain_tiles import sample_imagery_pixels
    tile = np.zeros((256, 256, 3), dtype=np.uint8)
    tile[:, :, 0] = 200     # 纯红 200
    got = sample_imagery_pixels({(5, 7): tile}, 5, 7, 5, 7,
                                np.array([5 * 256 + 128.0]),
                                np.array([7 * 256 + 64.0]))
    assert got.shape == (1, 3)
    np.testing.assert_allclose(got[0], [200.0, 0.0, 0.0], atol=1.0)


def test_sample_imagery_pixels_two_tiles_bilinear_blend():
    """跨瓦片边界采样：两色之间应得中间色。"""
    from ui.widgets.terrain_tiles import sample_imagery_pixels
    red = np.zeros((256, 256, 3), dtype=np.uint8)
    red[:, :, 0] = 200
    blue = np.zeros((256, 256, 3), dtype=np.uint8)
    blue[:, :, 2] = 200
    tiles = {(5, 7): red, (6, 7): blue}
    # 边界正中（x=6*256 处）：红蓝各半
    got = sample_imagery_pixels(tiles, 5, 7, 6, 7,
                                np.array([6.0 * 256]),
                                np.array([7.5 * 256]))
    np.testing.assert_allclose(got[0], [100.0, 0.0, 100.0], atol=2.0)


def test_sample_imagery_pixels_missing_tile_is_nan():
    from ui.widgets.terrain_tiles import sample_imagery_pixels
    tile = np.full((256, 256, 3), 120, dtype=np.uint8)
    got = sample_imagery_pixels({(5, 7): tile}, 5, 7, 6, 8,
                                np.array([5.5 * 256, 8.5 * 256]),
                                np.array([7.5 * 256, 8.5 * 256]))
    assert np.isfinite(got[0]).all()
    assert np.isnan(got[1]).all()


def test_sample_imagery_pixels_empty_tiles_returns_none():
    from ui.widgets.terrain_tiles import sample_imagery_pixels
    assert sample_imagery_pixels({}, 5, 7, 6, 8,
                                 np.array([1.0]), np.array([1.0])) is None


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
