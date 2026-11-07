"""Project amplitude observations into a common metre-based coordinate system.

The desktop calls this from its GUI thread, matching the existing PROJ
initialization restriction in Trajectory3DView. Waveform I/O stays in workers.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from pyproj import CRS, Transformer

from mygpr.application.spatial.amplitude import LineObservations


def _metric_projected(crs: CRS) -> bool:
    return bool(crs.is_projected and len(crs.axis_info) >= 2 and all(
        abs(float(axis.unit_conversion_factor) - 1.) < 1e-12 for axis in crs.axis_info[:2]))


def validate_metric_crs(value: str) -> None:
    try:
        crs = CRS.from_user_input(value)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ValueError('坐标系无效') from exc
    if not _metric_projected(crs):
        raise ValueError('目标坐标系必须是米制投影坐标系')


def project_observations(
    observations: list[LineObservations] | tuple[LineObservations, ...],
    *, target_crs: str = '',
) -> tuple[tuple[LineObservations, ...], str]:
    if not observations:
        raise ValueError('没有可投影的测线')
    try:
        source_crs = [CRS.from_user_input(item.coordinate_system) for item in observations]
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ValueError('测线坐标系未知或无效，请先确认坐标系') from exc
    if target_crs:
        try:
            target = CRS.from_user_input(target_crs)
        except (ValueError, TypeError, RuntimeError) as exc:
            raise ValueError('目标坐标系无效') from exc
    elif all(crs == source_crs[0] for crs in source_crs) and _metric_projected(source_crs[0]):
        target = source_crs[0]
    else:
        # Select one UTM zone for the complete site, rather than projecting
        # each flight line around a different local origin.
        lons, lats = [], []
        for item, crs in zip(observations, source_crs):
            if not len(item.x):
                continue
            lon, lat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(item.x, item.y)
            lons.extend(np.asarray(lon, dtype=float))
            lats.extend(np.asarray(lat, dtype=float))
        if not lons or not np.isfinite(lons).all() or not np.isfinite(lats).all():
            raise ValueError('无法将测线坐标转换为经纬度')
        lon, lat = np.asarray(lons), np.asarray(lats)
        if np.ptp(lon) > 6 or np.any(np.abs(lon) > 180) or np.any((lat < -80) | (lat > 84)):
            raise ValueError('测线范围不适合自动选择 UTM，请指定米制投影坐标系')
        centre_lon, centre_lat = float((lon.min() + lon.max()) / 2), float((lat.min() + lat.max()) / 2)
        zone = min(60, max(1, int((centre_lon + 180) // 6) + 1))
        target = CRS.from_epsg((32600 if centre_lat >= 0 else 32700) + zone)
    if not _metric_projected(target):
        raise ValueError('目标坐标系必须是米制投影坐标系')
    projected = []
    for item, source in zip(observations, source_crs):
        if source == target:
            x, y = item.x.copy(), item.y.copy()
        else:
            x, y = Transformer.from_crs(source, target, always_xy=True).transform(item.x, item.y)
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('坐标投影产生无效位置')
        projected.append(replace(item, coordinate_system=target.to_string(), x=x, y=y))
    return tuple(projected), target.to_string()
