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


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
