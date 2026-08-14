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
        # 瓦片行向下为南 → 马赛克行向上为北（y 升序）：
        # 既要翻转瓦片间的行序（ty=y1 南瓦片在 row 0），
        # 也要翻转瓦片内部行序（block 行 0 是该瓦片北缘，
        # 应落在马赛克该块区的上沿）——漏掉内部翻转会把
        # 瓦片尺度（z14 约 2km）内的地形南北镜像（营山实测复现）
        row = (int(y1) - int(ty)) * tile_px
        col = (int(tx) - int(x0)) * tile_px
        elev[row:row + block.shape[0], col:col + block.shape[1]] = block[::-1, :]

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


def sample_imagery_pixels(tiles: dict, x0: int, y0: int, x1: int, y1: int,
                          qpx: np.ndarray, qpy: np.ndarray,
                          tile_px: int = 256) -> np.ndarray | None:
    """影像瓦片块按全局像素坐标双线性采样 RGB。

    tiles: {(tx, ty): (tile_px, tile_px, 3)}（瓦片原始朝向：行向下为南）。
    qpx/qpy: 该级别全局像素坐标（= lonlat_to_tile 返回值 × tile_px），
    任意形状、二者同形状。采样逐通道复用 sample_bilinear。
    返回 (..., 3) float32；瓦片缺失或块外的顶点为 NaN；tiles 为空返回 None。
    """
    if not tiles:
        return None
    cols = int(x1) - int(x0) + 1
    rows = int(y1) - int(y0) + 1
    mosaic = np.full((rows * tile_px, cols * tile_px, 3), np.nan,
                     dtype=np.float32)
    for (tx, ty), block in tiles.items():
        if not (x0 <= tx <= x1 and y0 <= ty <= y1):
            continue
        block = np.asarray(block, dtype=np.float32)
        row = (int(ty) - int(y0)) * tile_px   # 原始朝向：行 0 为北，向下为南
        col = (int(tx) - int(x0)) * tile_px
        mosaic[row:row + block.shape[0], col:col + block.shape[1]] = block[..., :3]
    xs = np.arange(cols * tile_px, dtype=float)
    ys = np.arange(rows * tile_px, dtype=float)
    fx = np.asarray(qpx, dtype=float) - int(x0) * tile_px - 0.5
    fy = np.asarray(qpy, dtype=float) - int(y0) * tile_px - 0.5
    channels = [sample_bilinear(mosaic[:, :, c], xs, ys, fx, fy)
                for c in range(3)]
    return np.stack(channels, axis=-1)


def idw_grid(px: np.ndarray, py: np.ndarray, pz: np.ndarray,
             qx: np.ndarray, qy: np.ndarray,
             *, k: int = 12, power: float = 2.0,
             min_points: int = 4) -> np.ndarray:
    """散点反距离加权（IDW）插值到任意查询点。

    px/py/pz: 一维等长散点（非有限值已在外部过滤）；
    qx/qy: 同形状查询点，返回同形状 float32。
    散点数少于 ``min_points`` 抛 ValueError（调用方据此回退其他数据源）。
    单条直线测线等退化分布也能工作（interpolate_scatter 的兜底）。
    """
    from scipy.spatial import cKDTree

    px = np.asarray(px, dtype=float).ravel()
    py = np.asarray(py, dtype=float).ravel()
    pz = np.asarray(pz, dtype=float).ravel()
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    if px.size < int(min_points):
        raise ValueError(f'散点数 {px.size} 少于下限 {min_points}，无法 IDW 插值')
    tree = cKDTree(np.column_stack([px, py]))
    kk = min(int(k), px.size)
    dist, idx = tree.query(np.column_stack([qx.ravel(), qy.ravel()]), k=kk)
    dist = np.maximum(dist, 1e-9)
    weights = 1.0 / dist ** float(power)
    values = np.sum(weights * pz[idx], axis=1) / np.sum(weights, axis=1)
    return values.astype(np.float32).reshape(qx.shape)


def interpolate_scatter(px: np.ndarray, py: np.ndarray, pz: np.ndarray,
                        qx: np.ndarray, qy: np.ndarray,
                        *, min_points: int = 4) -> np.ndarray:
    """散点 → 规则/任意查询网格的主插值入口。

    优先线性三角剖分（griddata linear，凸包内对平面/缓变地形精确），
    凸包外的查询点用最近邻补齐；三角剖分失败（共线单测线等退化分布）
    回退 IDW。散点数少于 ``min_points`` 抛 ValueError。
    """
    from scipy.interpolate import griddata
    from scipy.spatial import QhullError

    px = np.asarray(px, dtype=float).ravel()
    py = np.asarray(py, dtype=float).ravel()
    pz = np.asarray(pz, dtype=float).ravel()
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    if px.size < int(min_points):
        raise ValueError(f'散点数 {px.size} 少于下限 {min_points}，无法插值')
    points = np.column_stack([px, py])
    queries = np.column_stack([qx.ravel(), qy.ravel()])
    try:
        values = griddata(points, pz, queries, method='linear')
        if np.isfinite(values).any():
            holes = ~np.isfinite(values)
            if holes.any():   # 凸包外：最近邻补齐（边缘不外推坡度）
                nearest = griddata(points, pz, queries[holes], method='nearest')
                values[holes] = nearest
            return values.astype(np.float32).reshape(qx.shape)
    except QhullError:
        pass   # 共线/退化分布 → IDW
    return idw_grid(px, py, pz, qx, qy, min_points=min_points)


__all__ = [
    'TERRARIUM_URL', 'TERRARIUM_MAX_ZOOM',
    'terrarium_url', 'terrarium_cache_path', 'decode_terrarium',
    'tile_grid_for_bbox', 'mosaic_from_tiles', 'sample_bilinear',
    'sample_imagery_pixels', 'idw_grid', 'interpolate_scatter',
]
