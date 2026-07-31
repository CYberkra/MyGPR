#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coordinate projection helpers for MyGPR field projects.

The field workflow stores raw GNSS longitude/latitude from airborne GPR CSV
files, but spatial deliverables need engineering planar coordinates.  This
module keeps projection policy out of UI callbacks and line import code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from core.app_errors import MyGPRError

try:  # pyproj is the formal runtime dependency for projection work.
    from pyproj import CRS, Transformer  # type: ignore
except ImportError:  # pragma: no cover - reported through ProjectionError
    CRS = None  # type: ignore
    Transformer = None  # type: ignore


CGCS2000_GEOGRAPHIC_EPSG = 4490
WGS84_GEOGRAPHIC_EPSG = 4326


class ProjectionError(MyGPRError):
    """Raised when project coordinates cannot be transformed safely."""


@dataclass(frozen=True)
class ProjectionSpec:
    """Resolved project projection specification."""

    name: str
    epsg: int
    zone: int | None = None
    source_epsg: int = WGS84_GEOGRAPHIC_EPSG
    is_auto: bool = False

    @property
    def description(self) -> str:
        if self.zone is not None:
            return f"{self.name} (EPSG:{self.epsg}, zone {self.zone})"
        return f"{self.name} (EPSG:{self.epsg})"


def _zone_to_cgcs2000_3deg_epsg(zone: int) -> int:
    """Return EPSG for CGCS2000 / 3-degree Gauss-Kruger *zone*.

    EPSG distinguishes ``zone`` CRS from ``CM`` (central-meridian) CRS.
    For example, zone 39 is EPSG:4527, while EPSG:4547 is the CM 114E CRS.
    MyGPR project settings that say ``Zone 39`` must therefore resolve to
    ``4488 + zone`` rather than the CM-series code.
    """
    if not 25 <= int(zone) <= 45:
        raise ProjectionError(f"CGCS2000 3-degree GK zone out of expected range: {zone}")
    return 4488 + int(zone)


def infer_3deg_zone_from_longitude(longitude: float) -> int:
    """Infer 3-degree Gauss-Kruger zone from longitude.

    For 3-degree GK, central meridian is ``zone * 3`` degrees.  Chinese mapping
    practice commonly uses round(lon / 3).  The result is clamped to a practical
    China zone range to avoid invalid EPSG codes for bad input.
    """
    lon = float(longitude)
    if not np.isfinite(lon):
        raise ProjectionError("Cannot infer projection zone from non-finite longitude")
    zone = int(round(lon / 3.0))
    return max(25, min(45, zone))


def resolve_projection_spec(coordinate_system: str | None, *, mean_longitude: float | None = None) -> ProjectionSpec:
    """Resolve a project coordinate-system string to a pyproj-ready spec.

    Supported examples:
    - ``CGCS2000 / 3-degree GK Zone 39``
    - ``CGCS2000 3-degree Gauss-Kruger zone 36``
    - ``EPSG:4544``
    - empty/auto strings, if ``mean_longitude`` is provided.
    """
    text = (coordinate_system or "").strip()
    lower = text.lower()

    epsg_match = re.search(r"epsg\s*[:：]?\s*(\d{4,6})", lower)
    if epsg_match:
        epsg = int(epsg_match.group(1))
        return ProjectionSpec(name=text or f"EPSG:{epsg}", epsg=epsg, zone=None, is_auto=False)

    zone_match = re.search(r"(?:zone|带|分带)\s*([0-9]{1,2})", lower)
    if "cgcs2000" in lower and ("3" in lower or "gk" in lower or "gauss" in lower or "高斯" in lower):
        auto_zone = False
        if zone_match:
            zone = int(zone_match.group(1))
        elif mean_longitude is not None:
            zone = infer_3deg_zone_from_longitude(mean_longitude)
            auto_zone = True
        else:
            raise ProjectionError(f"坐标系统缺少 3-degree GK 分带号：{text!r}")
        epsg = _zone_to_cgcs2000_3deg_epsg(zone)
        suffix = " (auto)" if auto_zone else ""
        return ProjectionSpec(name=f"CGCS2000 / 3-degree GK Zone {zone}{suffix}", epsg=epsg, zone=zone, is_auto=auto_zone)

    if mean_longitude is not None:
        zone = infer_3deg_zone_from_longitude(mean_longitude)
        epsg = _zone_to_cgcs2000_3deg_epsg(zone)
        return ProjectionSpec(name=f"CGCS2000 / 3-degree GK Zone {zone} (auto)", epsg=epsg, zone=zone, is_auto=True)

    raise ProjectionError(f"暂不支持或无法识别坐标系统：{text or '未填写'}")


def project_lonlat_to_xy(
    longitude: Iterable[float],
    latitude: Iterable[float],
    *,
    coordinate_system: str | None,
) -> tuple[np.ndarray, np.ndarray, ProjectionSpec]:
    """Project longitude/latitude arrays to engineering x/y coordinates."""
    lon = np.asarray(list(longitude), dtype=np.float64)
    lat = np.asarray(list(latitude), dtype=np.float64)
    if lon.shape != lat.shape:
        raise ProjectionError(f"Longitude/latitude shape mismatch: {lon.shape} vs {lat.shape}")
    if lon.size == 0:
        raise ProjectionError("Cannot project an empty trajectory")
    finite = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(finite):
        raise ProjectionError("Trajectory has no finite longitude/latitude values")
    mean_lon = float(np.nanmean(lon[finite]))
    spec = resolve_projection_spec(coordinate_system, mean_longitude=mean_lon)
    if Transformer is None or CRS is None:
        raise ProjectionError("pyproj is not installed; cannot perform formal coordinate projection")
    transformer = Transformer.from_crs(CRS.from_epsg(spec.source_epsg), CRS.from_epsg(spec.epsg), always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), spec


__all__ = [
    "ProjectionError",
    "ProjectionSpec",
    "infer_3deg_zone_from_longitude",
    "project_lonlat_to_xy",
    "resolve_projection_spec",
]
