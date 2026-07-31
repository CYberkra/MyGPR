#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Engineering trajectory model with RTK/IMU and trace-time fields."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from core.tabular_security import safe_tabular_value


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
    timestamp_s: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    satellites: int = 0
    hdop: float = 0.0
    pdop: float = 0.0
    flight_height_m: float = 0.0
    alignment_status: str = "aligned"
    trace_index: int = -1


class TrajectoryModel:
    CSV_FIELDS = [
        "trace_index",
        "timestamp_s",
        "distance_m",
        "x",
        "y",
        "z",
        "longitude",
        "latitude",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "flight_height_m",
        "quality",
        "satellites",
        "hdop",
        "pdop",
        "alignment_status",
        "coordinate_system",
    ]

    def __init__(self, points: Iterable[TrajectoryPoint]) -> None:
        pts = list(points)
        if not pts:
            raise ValueError("TrajectoryModel requires at least one point")
        self.points = pts
        self.distance = np.asarray([p.distance_m for p in pts], dtype=np.float64)
        self.x = np.asarray([p.x for p in pts], dtype=np.float64)
        self.y = np.asarray([p.y for p in pts], dtype=np.float64)
        self.z = np.asarray([p.z for p in pts], dtype=np.float64)
        self.longitude = np.asarray([p.longitude for p in pts], dtype=np.float64)
        self.latitude = np.asarray([p.latitude for p in pts], dtype=np.float64)
        self.timestamp = np.asarray([p.timestamp_s for p in pts], dtype=np.float64)
        self.roll = np.asarray([p.roll_deg for p in pts], dtype=np.float64)
        self.pitch = np.asarray([p.pitch_deg for p in pts], dtype=np.float64)
        self.yaw = np.asarray([p.yaw_deg for p in pts], dtype=np.float64)
        self.flight_height = np.asarray([p.flight_height_m for p in pts], dtype=np.float64)
        self._validate_distance_axis()
        self._distance_order = np.argsort(self.distance, kind="stable")
        self.distance_index = self.distance[self._distance_order]
        self._x_by_distance = self.x[self._distance_order]
        self._y_by_distance = self.y[self._distance_order]
        self._z_by_distance = self.z[self._distance_order]
        self._longitude_by_distance = self.longitude[self._distance_order]
        self._latitude_by_distance = self.latitude[self._distance_order]
        self._timestamp_by_distance = self.timestamp[self._distance_order]
        self._roll_by_distance = self.roll[self._distance_order]
        self._pitch_by_distance = self.pitch[self._distance_order]
        self._yaw_by_distance = self.yaw[self._distance_order]
        self._flight_height_by_distance = self.flight_height[self._distance_order]

    def _validate_distance_axis(self) -> None:
        if not np.isfinite(self.distance).all():
            raise ValueError("Trajectory distance contains non-finite values")
        # Trace order is authoritative and may contain distance reversals on
        # turn-backs. A separate distance-indexed view is used for interpolation.

    @classmethod
    def demo(cls, *, length_m: float = 212.35, count: int = 160) -> "TrajectoryModel":
        distance = np.linspace(0.0, float(length_m), int(count))
        x = 451110.0 + distance * 1.82 + 4.5 * np.sin(distance / 58.0)
        y = 3487832.0 + distance * 1.62 + 6.0 * np.cos(distance / 74.0)
        z = 22.4 + 0.6 * np.sin(distance / 36.0)
        points = [
            TrajectoryPoint(
                float(d), float(xx), float(yy), float(zz),
                quality="浮动解" if 85 < d < 135 else "固定解",
                timestamp_s=float(i) * 0.1,
                satellites=24,
                hdop=0.8,
                trace_index=i,
            )
            for i, (d, xx, yy, zz) in enumerate(zip(distance, x, y, z))
        ]
        return cls(points)

    @classmethod
    def from_csv(cls, path: str | Path) -> "TrajectoryModel":
        src = Path(path)
        points: list[TrajectoryPoint] = []
        with src.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError(f"Trajectory file has no header: {src}")
            for row_index, row in enumerate(reader):
                def get_float(*keys: str, default: float = 0.0) -> float:
                    for key in keys:
                        value = row.get(key)
                        if value not in (None, ""):
                            try:
                                return float(value)
                            except ValueError:
                                pass
                    return default

                def get_int(*keys: str, default: int = 0) -> int:
                    return int(round(get_float(*keys, default=float(default))))

                d = get_float("distance_m", "mileage", "里程", "distance")
                x = get_float("x", "X", "x_m", "local_x_m", "东坐标", default=float("nan"))
                y = get_float("y", "Y", "y_m", "local_y_m", "北坐标", default=float("nan"))
                z = get_float("z", "Z", "local_z_m", "elevation", "高程", default=float("nan"))
                quality = row.get("quality") or row.get("rtk_status") or row.get("rtk_fix_type") or row.get("定位状态") or "未知"
                lon = get_float("longitude", "lon", "经度", default=float("nan"))
                lat = get_float("latitude", "lat", "纬度", default=float("nan"))
                coord = row.get("coordinate_system") or row.get("crs") or row.get("坐标系统") or ""
                points.append(
                    TrajectoryPoint(
                        distance_m=d,
                        x=x,
                        y=y,
                        z=z,
                        yaw_deg=get_float("yaw_deg", "yaw", "heading_deg", "航向角"),
                        quality=str(quality),
                        longitude=lon,
                        latitude=lat,
                        coordinate_system=str(coord),
                        timestamp_s=get_float("timestamp_s", "trace_timestamp_s", "timestamp", "time_s"),
                        roll_deg=get_float("roll_deg", "roll"),
                        pitch_deg=get_float("pitch_deg", "pitch"),
                        satellites=get_int("satellites", "sat", "num_satellites"),
                        hdop=get_float("hdop"),
                        pdop=get_float("pdop"),
                        flight_height_m=get_float("flight_height_m", "height_agl_m", "agl_m"),
                        alignment_status=str(row.get("alignment_status") or "aligned"),
                        trace_index=get_int("trace_index", default=row_index),
                    )
                )
        if not points:
            raise ValueError(f"Trajectory file contains no points: {src}")
        return cls(points)

    def to_csv(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for p in self.points:
                writer.writerow({
                    "trace_index": p.trace_index,
                    "timestamp_s": f"{p.timestamp_s:.9f}",
                    "distance_m": f"{p.distance_m:.4f}",
                    "x": _format_float(p.x, 4),
                    "y": _format_float(p.y, 4),
                    "z": _format_float(p.z, 4),
                    "longitude": _format_float(p.longitude, 10),
                    "latitude": _format_float(p.latitude, 10),
                    "roll_deg": _format_float(p.roll_deg, 4),
                    "pitch_deg": _format_float(p.pitch_deg, 4),
                    "yaw_deg": _format_float(p.yaw_deg, 4),
                    "flight_height_m": _format_float(p.flight_height_m, 4),
                    "quality": safe_tabular_value(p.quality),
                    "satellites": p.satellites,
                    "hdop": _format_float(p.hdop, 3),
                    "pdop": _format_float(p.pdop, 3),
                    "alignment_status": safe_tabular_value(p.alignment_status),
                    "coordinate_system": safe_tabular_value(p.coordinate_system),
                })
        return out

    @staticmethod
    def _interp_angle(distance: float, axis: np.ndarray, values_deg: np.ndarray) -> float:
        valid = np.isfinite(axis) & np.isfinite(values_deg)
        if np.count_nonzero(valid) < 2:
            return float("nan")
        values = np.unwrap(np.radians(values_deg[valid]))
        return float(np.degrees(np.interp(distance, axis[valid], values)) % 360.0)

    @staticmethod
    def _interp_finite(distance: float, axis: np.ndarray, values: np.ndarray) -> float:
        valid = np.isfinite(axis) & np.isfinite(values)
        if np.count_nonzero(valid) == 0:
            return float("nan")
        if np.count_nonzero(valid) == 1:
            return float(values[valid][0])
        return float(np.interp(distance, axis[valid], values[valid]))

    def interpolate(self, distance_m: float, *, extrapolation: str = "nearest") -> TrajectoryPoint:
        requested = float(distance_m)
        status_suffix = ""
        axis = self.distance_index
        if requested < axis[0]:
            if extrapolation == "error":
                raise ValueError(f"距离 {requested:.3f} m 小于轨迹起点 {axis[0]:.3f} m")
            d = float(axis[0])
            status_suffix = "clamped_start"
        elif requested > axis[-1]:
            if extrapolation == "error":
                raise ValueError(f"距离 {requested:.3f} m 超过轨迹终点 {axis[-1]:.3f} m")
            d = float(axis[-1])
            status_suffix = "clamped_end"
        else:
            d = requested
        nearest_idx = int(np.argmin(np.abs(self.distance - d)))
        nearest = self.points[nearest_idx]
        status = status_suffix or nearest.alignment_status
        return TrajectoryPoint(
            distance_m=requested,
            x=self._interp_finite(d, axis, self._x_by_distance),
            y=self._interp_finite(d, axis, self._y_by_distance),
            z=self._interp_finite(d, axis, self._z_by_distance),
            yaw_deg=self._interp_angle(d, axis, self._yaw_by_distance),
            quality=nearest.quality,
            longitude=self._interp_finite(d, axis, self._longitude_by_distance),
            latitude=self._interp_finite(d, axis, self._latitude_by_distance),
            coordinate_system=nearest.coordinate_system,
            timestamp_s=self._interp_finite(d, axis, self._timestamp_by_distance),
            roll_deg=self._interp_finite(d, axis, self._roll_by_distance),
            pitch_deg=self._interp_finite(d, axis, self._pitch_by_distance),
            satellites=nearest.satellites,
            hdop=nearest.hdop,
            pdop=nearest.pdop,
            flight_height_m=self._interp_finite(d, axis, self._flight_height_by_distance),
            alignment_status=status,
            trace_index=nearest.trace_index,
        )

    def diagnostics(self) -> dict[str, float | int]:
        axis = self.distance_index
        steps = np.hypot(np.diff(self.x), np.diff(self.y))
        fixed = np.asarray(["固定" in p.quality for p in self.points], dtype=bool)
        aligned = np.asarray([p.alignment_status == "aligned" for p in self.points], dtype=bool)
        return {
            "point_count": len(self.points),
            "length_m": float(axis[-1] - axis[0]),
            "fixed_solution_ratio": float(np.mean(fixed)),
            "aligned_ratio": float(np.mean(aligned)),
            "median_step_m": float(np.median(steps)) if steps.size else 0.0,
            "max_step_m": float(np.max(steps)) if steps.size else 0.0,
            "duplicate_distance_count": int(np.count_nonzero(np.diff(self.distance) == 0)),
        }


def _format_float(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}" if np.isfinite(value) else ""


__all__ = ["TrajectoryModel", "TrajectoryPoint"]
