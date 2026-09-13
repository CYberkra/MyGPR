# -*- coding: utf-8 -*-
"""纯地理/覆盖统计数学（无 Qt import，可独立单测）。

空间信息页"项目覆盖统计"卡的计算内核：轨迹点数、段数与总里程。
空间轨迹通常为投影米坐标；地理经纬度轨迹（EPSG:4326 等）用 haversine
公式测距，避免把度当米。非法坐标点跳过，不阻断整项目的统计渲染。
"""
from __future__ import annotations

import numpy as np

__all__ = ['coverage_statistics', 'format_distance']

# 平均地球半径（IUGG）：haversine 距离用
_EARTH_RADIUS_M = 6_371_008.8


def _is_geographic(crs: str, points: list[tuple[float, float]]) -> bool:
    """判断是否按地理坐标（经纬度）测距。

    SpatialPersistenceMixin 将经纬度轨迹标注 EPSG:4326；仅当 CRS 缺失时
    才用坐标幅值作遗留兜底——局部米坐标完全可能合法地落在原点附近。
    """
    crs = str(crs or '').lower()
    if ('4326' in crs or '4490' in crs or 'wgs84' in crs
            or 'geographic' in crs or '经纬' in crs):
        return True
    return not crs and all(abs(x) <= 180.0 and abs(y) <= 90.0
                           for x, y in points)


def _haversine_m(x0: float, y0: float, x1: float, y1: float) -> float:
    """两经纬度点（度）间的大圆距离（米）。"""
    lat0, lat1 = np.radians((y0, y1))
    dlat = lat1 - lat0
    dlon = np.radians(x1 - x0)
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2.0) ** 2)
    return float(_EARTH_RADIUS_M * 2.0 * np.arctan2(
        np.sqrt(a), np.sqrt(max(0.0, 1.0 - a))))


def coverage_statistics(tracks: list) -> dict[str, object]:
    """Return project coverage figures derived solely from spatial tracks.

    :param tracks: SpatialTrack 列表（鸭子类型：points[].x/y、
        coordinate_system）。
    :return: {'track_count', 'point_count', 'segment_count', 'length_m'}。
    """
    track_count = 0
    point_count = 0
    segment_count = 0
    length_m = 0.0

    for track in tracks or []:
        points: list[tuple[float, float]] = []
        for point in getattr(track, 'points', ()) or ():
            try:
                x = float(getattr(point, 'x', float('nan')))
                y = float(getattr(point, 'y', float('nan')))
            except (TypeError, ValueError):
                continue
            if np.isfinite(x) and np.isfinite(y):
                points.append((x, y))

        if not points:
            continue
        track_count += 1
        point_count += len(points)

        geographic = _is_geographic(
            str(getattr(track, 'coordinate_system', '') or ''), points)
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            segment_count += 1
            if geographic:
                length_m += _haversine_m(x0, y0, x1, y1)
            else:
                length_m += float(np.hypot(x1 - x0, y1 - y0))

    return {
        'track_count': track_count,
        'point_count': point_count,
        'segment_count': segment_count,
        'length_m': length_m,
    }


def format_distance(distance_m: float) -> str:
    """Format a distance compactly for the spatial coverage card."""
    if distance_m >= 1000.0:
        return f'{distance_m / 1000.0:.2f} km'
    return f'{distance_m:.0f} m'
