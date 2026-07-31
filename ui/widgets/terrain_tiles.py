# -*- coding: utf-8 -*-
"""真实地形高程（Terrarium PNG 瓦片）的纯数学 / 纯解析工具（无 Qt 依赖）。

数据源：AWS Open Data Terrarium 高程瓦片
    https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png
解码公式：elevation_m = R * 256 + G + B / 256 - 32768

包含：
- 瓦片 URL / 磁盘缓存路径
- Terrarium PNG 字节流 → 高程矩阵（PIL + numpy）
- 按经纬度包围盒选瓦片级别与编号范围（数量预算约束）
- 瓦片矩阵拼接为单幅高程马赛克（Web Mercator 米坐标，x/y 均升序）
- 纯 numpy 双线性采样（规则网格 → 任意查询点）
"""
from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image

from ui.widgets.map_tiles import (MAX_LATITUDE, WORLD_SIZE_M,
                                  lonlat_to_tile, tile_bounds_mercator)

TERRARIUM_URL = ('https://s3.amazonaws.com/elevation-tiles-prod'
                 '/terrarium/{z}/{x}/{y}.png')
TERRARIUM_MAX_ZOOM = 15          # Terrarium 数据最细级别


def terrarium_url(z: int, x: int, y: int) -> str:
    """Terrarium 高程瓦片 URL。"""
    return TERRARIUM_URL.format(z=int(z), x=int(x), y=int(y))


def terrarium_cache_path(cache_root: str, z: int, x: int, y: int) -> str:
    """磁盘缓存路径（{cache_root}/terrarium/{z}/{x}/{y}.png）。"""
    return os.path.join(cache_root, 'terrarium', str(z), str(x), f'{y}.png')


def decode_terrarium(data: bytes) -> np.ndarray:
    """Terrarium PNG 字节流 → float32 高程矩阵（H×W，行向下为南）。"""
    image = Image.open(io.BytesIO(data)).convert('RGB')
    rgb = np.asarray(image, dtype=np.float32)
    return rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0


def tile_grid_for_bbox(lon_min: float, lat_min: float,
                       lon_max: float, lat_max: float,
                       *, max_tiles: int = 64,
                       preferred_zoom: int = 13) -> tuple[int, int, int, int, int]:
    """为经纬度包围盒选高程瓦片 (zoom, x0, x1, y0, y1)。

    从 preferred_zoom（不超过 TERRARIUM_MAX_ZOOM）往下减，
    直到瓦片总数 <= max_tiles（不低于级别 3）。
    """
    lat_min = max(-MAX_LATITUDE, min(MAX_LATITUDE, float(lat_min)))
    lat_max = max(-MAX_LATITUDE, min(MAX_LATITUDE, float(lat_max)))
    import math as _math
    if not all(_math.isfinite(v) for v in (lon_min, lon_max, lat_min, lat_max)):
        raise ValueError('经纬度包围盒包含非有限值')
    zoom = max(3, min(TERRARIUM_MAX_ZOOM, int(preferred_zoom)))
    while True:
        n = 2 ** zoom
        x0f, _ = lonlat_to_tile(lon_min, 0.0, zoom)
        x1f, _ = lonlat_to_tile(lon_max, 0.0, zoom)
        _, y0f = lonlat_to_tile(0.0, lat_max, zoom)
        _, y1f = lonlat_to_tile(0.0, lat_min, zoom)
        x0 = max(0, min(n - 1, int(x0f)))
        x1 = max(0, min(n - 1, int(x1f)))
        y0 = max(0, min(n - 1, int(y0f)))
        y1 = max(0, min(n - 1, int(y1f)))
        if (x1 - x0 + 1) * (y1 - y0 + 1) <= int(max_tiles) or zoom <= 3:
            return zoom, x0, x1, y0, y1
        zoom -= 1


def mosaic_from_tiles(tiles: dict, zoom: int,
                      x0: int, x1: int, y0: int, y1: int,
                      tile_px: int = 256,
                      fill: float = np.nan) -> tuple:
    """把 {(x, y): 高程矩阵} 拼成单幅马赛克。

    返回 (elev, xs, ys)：
    - elev: float32 二维数组，行对应 y 升序（南→北）、列对应 x 升序（西→东）；
      缺失瓦片区域填 ``fill``。
    - xs / ys: 每列 / 每行中心的 Web Mercator 米坐标（一维升序）。
    """
    cols = int(x1) - int(x0) + 1
    rows = int(y1) - int(y0) + 1
    elev = np.full((rows * tile_px, cols * tile_px), fill, dtype=np.float32)
    for (tx, ty), block in tiles.items():
        if not (x0 <= tx <= x1 and y0 <= ty <= y1):
            continue
        block = np.asarray(block, dtype=np.float32)
        # 瓦片行向下为南 → 马赛克行向上为北（y 升序），行序翻转
        row = (int(y1) - int(ty)) * tile_px
        col = (int(tx) - int(x0)) * tile_px
        elev[row:row + block.shape[0], col:col + block.shape[1]] = block

    bx0, by0, _, _ = tile_bounds_mercator(zoom, x0, y1)   # 西南角（最小 x/y）
    step = WORLD_SIZE_M / (2 ** int(zoom)) / tile_px
    xs = bx0 + (np.arange(cols * tile_px) + 0.5) * step
    ys = by0 + (np.arange(rows * tile_px) + 0.5) * step
    return elev, xs, ys


def sample_bilinear(elev: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                    qx: np.ndarray, qy: np.ndarray,
                    fill: float = np.nan) -> np.ndarray:
    """规则网格双线性采样。

    elev: (len(ys), len(xs)) 高程；xs/ys 一维升序；qx/qy 任意形状查询点。
    网格外或落点邻域含 NaN 的查询返回 ``fill``。返回形状与 qx 相同。
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    if elev.size == 0 or xs.size < 2 or ys.size < 2:
        return np.full(qx.shape, fill, dtype=np.float32)

    fx = (qx - xs[0]) / (xs[1] - xs[0])
    fy = (qy - ys[0]) / (ys[1] - ys[0])
    ix = np.floor(fx).astype(np.int64)
    iy = np.floor(fy).astype(np.int64)
    inside = (ix >= 0) & (iy >= 0) & (ix < xs.size - 1) & (iy < ys.size - 1)

    out = np.full(qx.shape, fill, dtype=np.float32)
    if not np.any(inside):
        return out
    ix = ix[inside]
    iy = iy[inside]
    tx = fx[inside] - ix
    ty = fy[inside] - iy
    z00 = elev[iy, ix].astype(float)
    z10 = elev[iy, ix + 1].astype(float)
    z01 = elev[iy + 1, ix].astype(float)
    z11 = elev[iy + 1, ix + 1].astype(float)
    valid = np.isfinite(z00) & np.isfinite(z10) & np.isfinite(z01) & np.isfinite(z11)
    values = ((z00 * (1 - tx) + z10 * tx) * (1 - ty)
              + (z01 * (1 - tx) + z11 * tx) * ty)
    values = np.where(valid, values, np.nan)
    flat = out.ravel()
    flat[np.flatnonzero(inside.ravel())] = values.astype(np.float32)
    return out


__all__ = [
    'TERRARIUM_URL', 'TERRARIUM_MAX_ZOOM',
    'terrarium_url', 'terrarium_cache_path', 'decode_terrarium',
    'tile_grid_for_bbox', 'mosaic_from_tiles', 'sample_bilinear',
]
