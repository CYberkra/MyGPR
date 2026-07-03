#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RTK/IMU trajectory model for interpolating target coordinates by mileage."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class TrajectoryPoint:
    distance_m: float
    x: float
    y: float
    z: float = 0.0
    yaw_deg: float = 0.0
    quality: str = "固定解"
    longitude: float = 0.0
    latitude: float = 0.0
    coordinate_system: str = ""


class TrajectoryModel:
    def __init__(self, points: Iterable[TrajectoryPoint]) -> None:
        pts = sorted(points, key=lambda p: p.distance_m)
        if not pts:
            raise ValueError("TrajectoryModel requires at least one point")
        self.points = pts
        self.distance = np.asarray([p.distance_m for p in pts], dtype=np.float64)
        self.x = np.asarray([p.x for p in pts], dtype=np.float64)
        self.y = np.asarray([p.y for p in pts], dtype=np.float64)
        self.z = np.asarray([p.z for p in pts], dtype=np.float64)
        self.longitude = np.asarray([p.longitude for p in pts], dtype=np.float64)
        self.latitude = np.asarray([p.latitude for p in pts], dtype=np.float64)

    @classmethod
    def demo(cls, *, length_m: float = 212.35, count: int = 160) -> "TrajectoryModel":
        distance = np.linspace(0.0, float(length_m), int(count))
        # Slightly curved urban road segment in projected coordinates.
        x = 451110.0 + distance * 1.82 + 4.5 * np.sin(distance / 58.0)
        y = 3487832.0 + distance * 1.62 + 6.0 * np.cos(distance / 74.0)
        z = 22.4 + 0.6 * np.sin(distance / 36.0)
        points = [TrajectoryPoint(float(d), float(xx), float(yy), float(zz), quality="浮动解" if 85 < d < 135 else "固定解") for d, xx, yy, zz in zip(distance, x, y, z)]
        return cls(points)

    @classmethod
    def from_csv(cls, path: str | Path) -> "TrajectoryModel":
        src = Path(path)
        points: list[TrajectoryPoint] = []
        with src.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                def get_float(*keys: str, default: float = 0.0) -> float:
                    for key in keys:
                        value = row.get(key)
                        if value not in (None, ""):
                            try:
                                return float(value)
                            except ValueError:
                                pass
                    return default
                d = get_float("distance_m", "mileage", "里程", "distance")
                x = get_float("x", "X", "x_m", "东坐标")
                y = get_float("y", "Y", "y_m", "北坐标")
                z = get_float("z", "Z", "elevation", "高程")
                quality = row.get("quality") or row.get("rtk_status") or row.get("定位状态") or "未知"
                lon = get_float("longitude", "lon", "经度", default=x)
                lat = get_float("latitude", "lat", "纬度", default=y)
                coord = row.get("coordinate_system") or row.get("坐标系统") or ""
                points.append(TrajectoryPoint(d, x, y, z, quality=quality, longitude=lon, latitude=lat, coordinate_system=coord))
        return cls(points)

    def to_csv(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["distance_m", "x", "y", "z", "longitude", "latitude", "yaw_deg", "quality", "coordinate_system"],
            )
            writer.writeheader()
            for p in self.points:
                writer.writerow({
                    "distance_m": f"{p.distance_m:.3f}",
                    "x": f"{p.x:.3f}",
                    "y": f"{p.y:.3f}",
                    "z": f"{p.z:.3f}",
                    "longitude": f"{p.longitude:.10f}",
                    "latitude": f"{p.latitude:.10f}",
                    "yaw_deg": f"{p.yaw_deg:.3f}",
                    "quality": p.quality,
                    "coordinate_system": p.coordinate_system,
                })
        return out

    def interpolate(self, distance_m: float) -> TrajectoryPoint:
        d = float(np.clip(distance_m, self.distance[0], self.distance[-1]))
        x = float(np.interp(d, self.distance, self.x))
        y = float(np.interp(d, self.distance, self.y))
        z = float(np.interp(d, self.distance, self.z))
        nearest_idx = int(np.argmin(np.abs(self.distance - d)))
        return TrajectoryPoint(d, x, y, z, quality=self.points[nearest_idx].quality)


__all__ = ["TrajectoryModel", "TrajectoryPoint"]
