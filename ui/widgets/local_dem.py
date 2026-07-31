# -*- coding: utf-8 -*-
"""本地 DEM 格网解析（Global Mapper 等 GIS 软件导出的 XYZ 文本格网）。

约定：三列数值（经度, 纬度, 高程米），WGS84 经纬度（度）；逗号/分号/
制表符/空白分隔均可；无法解析为三个浮点数的行视为表头/注释跳过。
数据点必须构成（近似完整的）规则格网，否则抛 ValueError —— 散点
请先在 GIS 软件里格网化再导出。
"""
from __future__ import annotations

import numpy as np

# 规则格网完整度下限：低于此比例按"散点/缺块严重"拒绝
_MIN_FILL_RATIO = 0.9
# 防御性上限：单文件点数（约 2000x2000 格网），超出按误选文件拒绝
_MAX_POINTS = 4_000_000


def load_xyz_grid(path: str) -> dict:
    """解析 XYZ 文本格网文件。

    返回 ``{'elev': (ny, nx) float32, 'lons': (nx,), 'lats': (ny,)}``，
    lons/lats 均为升序一维轴，elev 行序与 lats 对应；格网内缺数为 NaN。
    文件无有效数据、点数超限、或不构成规则格网时抛 ValueError（中文信息）。
    """
    xs: list = []
    ys: list = []
    zs: list = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            parts = line.replace(',', ' ').replace(';', ' ').split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue  # 表头/注释行
            xs.append(x)
            ys.append(y)
            zs.append(z)
            if len(xs) > _MAX_POINTS:
                raise ValueError(
                    f'数据点超过 {_MAX_POINTS}，请先在 GIS 软件中裁剪测区范围')
    if not xs:
        raise ValueError('文件中没有可解析的"经度 纬度 高程"数据行')

    xs_a = np.asarray(xs, dtype=float)
    ys_a = np.asarray(ys, dtype=float)
    zs_a = np.asarray(zs, dtype=float)
    lons = np.unique(xs_a)
    lats = np.unique(ys_a)
    nx, ny = lons.size, lats.size
    if nx < 2 or ny < 2:
        raise ValueError('有效网格不足 2x2，无法构建地形')

    # 规则格网还原：点 → (iy, ix) 索引；重复点保留首个
    ix = np.searchsorted(lons, xs_a)
    iy = np.searchsorted(lats, ys_a)
    exact = (lons[np.clip(ix, 0, nx - 1)] == xs_a) & (
        lats[np.clip(iy, 0, ny - 1)] == ys_a)
    if not np.all(exact):
        raise ValueError('数据点不构成规则格网（含非格网点），请先格网化再导出')
    fill = np.zeros((ny, nx), dtype=bool)
    fill[iy, ix] = True
    ratio = float(np.count_nonzero(fill)) / float(nx * ny)
    if ratio < _MIN_FILL_RATIO:
        raise ValueError(
            f'格网完整度仅 {ratio:.0%}（疑似散点或严重缺块），请先格网化再导出')

    elev = np.full((ny, nx), np.nan, dtype=np.float32)
    elev[iy, ix] = zs_a.astype(np.float32)
    return {'elev': elev, 'lons': lons, 'lats': lats}
