#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial export persistence mixin for field projects."""

from __future__ import annotations

import csv
import uuid
from pathlib import Path
from typing import Any

from core.field_project_models import validate_line_id


class FieldSpatialStoreMixin:
    """Manage derived spatial exports under ``spatial/``.

    The field workbench treats spatial deliverables as auditable project files,
    not just UI previews.  These helpers deliberately write plain CSV files so
    that coordinate deliverables can be inspected, reloaded and attached to the
    report package without requiring a GIS stack.
    """

    def export_spatial_targets_xy(self, line_id: str) -> Path:
        safe_line_id = validate_line_id(line_id)
        targets = self.load_targets(safe_line_id)
        path = self.root / "spatial" / f"{safe_line_id}_targets_xy.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
            fieldnames = ["target_id", "line_id", "x", "y", "distance_m", "depth_m", "type", "status", "confidence"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            try:
                trajectory = self.load_trajectory(safe_line_id)
            except Exception:
                trajectory = None
            for target in targets:
                distance_m = float(target.get("mileage", target.get("distance_m", 0.0)))
                if trajectory is not None:
                    point = trajectory.interpolate(distance_m)
                    x_val, y_val = point.x, point.y
                else:
                    raw_x = target.get("x", "")
                    raw_y = target.get("y", "")
                    x_val = float(raw_x) if raw_x not in (None, "") else ""
                    y_val = float(raw_y) if raw_y not in (None, "") else ""
                writer.writerow(
                    {
                        "target_id": target.get("target_id", target.get("name", "")),
                        "line_id": target.get("line_id", safe_line_id),
                        "x": f"{float(x_val):.3f}" if x_val != "" else "",
                        "y": f"{float(y_val):.3f}" if y_val != "" else "",
                        "distance_m": f"{distance_m:.3f}",
                        "depth_m": f"{float(target.get('depth', target.get('depth_m', 0.0))):.3f}",
                        "type": target.get("type", ""),
                        "status": target.get("status", ""),
                        "confidence": target.get("confidence", ""),
                    }
                )
        tmp.replace(path)
        return path

    def export_project_spatial_coordinates(self, *, filename: str | None = None) -> Path:
        """Export all available trajectories and target coordinates to one CSV.

        Rows with ``record_type=trajectory`` represent measured RTK/IMU points.
        Rows with ``record_type=target`` represent interpreted targets projected
        onto a trajectory by mileage.  For targets, ``z`` is an estimated buried
        elevation computed as trajectory elevation minus interpreted depth.
        """
        name = filename or "project_spatial_coordinates.csv"
        if Path(name).suffix.lower() != ".csv":
            name = f"{name}.csv"
        path = self.root / "spatial" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        fieldnames = [
            "record_type",
            "line_id",
            "point_id",
            "target_id",
            "distance_m",
            "x",
            "y",
            "z",
            "elevation_m",
            "depth_m",
            "longitude",
            "latitude",
            "quality",
            "coordinate_system",
            "target_type",
            "status",
            "confidence",
            "note",
        ]
        with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for line in self.list_lines():
                line_id = validate_line_id(line.line_id)
                try:
                    trajectory = self.load_trajectory(line_id)
                except Exception:
                    trajectory = None
                if trajectory is not None:
                    for idx, point in enumerate(trajectory.points, start=1):
                        writer.writerow(
                            {
                                "record_type": "trajectory",
                                "line_id": line_id,
                                "point_id": f"{line_id}-P{idx:05d}",
                                "target_id": "",
                                "distance_m": f"{float(point.distance_m):.3f}",
                                "x": f"{float(point.x):.3f}",
                                "y": f"{float(point.y):.3f}",
                                "z": f"{float(point.z):.3f}",
                                "elevation_m": f"{float(point.z):.3f}",
                                "depth_m": "",
                                "longitude": f"{float(point.longitude):.10f}",
                                "latitude": f"{float(point.latitude):.10f}",
                                "quality": point.quality,
                                "coordinate_system": point.coordinate_system,
                                "target_type": "",
                                "status": getattr(line, "rtk_status", ""),
                                "confidence": "",
                                "note": "trajectory",
                            }
                        )
                targets: list[dict[str, Any]] = []
                try:
                    targets = self.load_targets(line_id)
                except Exception:
                    targets = []
                for idx, target in enumerate(targets, start=1):
                    distance_m = _to_float(target.get("distance_m", target.get("mileage", 0.0)), 0.0)
                    depth_m = _to_float(target.get("depth_m", target.get("depth", 0.0)), 0.0)
                    x_val = _to_float(target.get("x", ""), None)
                    y_val = _to_float(target.get("y", ""), None)
                    elevation = None
                    lon = lat = 0.0
                    quality = ""
                    coord = ""
                    if trajectory is not None:
                        point = trajectory.interpolate(distance_m)
                        # Project targets onto the active trajectory by mileage.
                        # This prevents stale demo/manual x/y fields from
                        # stretching the project map outside the measured line.
                        x_val = point.x
                        y_val = point.y
                        elevation = point.z
                        lon = point.longitude
                        lat = point.latitude
                        quality = point.quality
                        coord = point.coordinate_system
                    z_val = (float(elevation) - depth_m) if elevation is not None else ""
                    writer.writerow(
                        {
                            "record_type": "target",
                            "line_id": line_id,
                            "point_id": f"{line_id}-T{idx:04d}",
                            "target_id": target.get("target_id") or target.get("name") or f"T-{idx:04d}",
                            "distance_m": f"{distance_m:.3f}",
                            "x": f"{float(x_val):.3f}" if x_val is not None else "",
                            "y": f"{float(y_val):.3f}" if y_val is not None else "",
                            "z": f"{float(z_val):.3f}" if z_val != "" else "",
                            "elevation_m": f"{float(elevation):.3f}" if elevation is not None else "",
                            "depth_m": f"{depth_m:.3f}",
                            "longitude": f"{float(lon):.10f}" if lon else "",
                            "latitude": f"{float(lat):.10f}" if lat else "",
                            "quality": quality,
                            "coordinate_system": coord,
                            "target_type": target.get("type", ""),
                            "status": target.get("status", ""),
                            "confidence": target.get("confidence", ""),
                            "note": target.get("note", ""),
                        }
                    )
        tmp.replace(path)
        self.append_log(f"导出项目空间坐标成果：{path.relative_to(self.root).as_posix()}")
        return path


def _to_float(value: Any, default: float | None = 0.0) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["FieldSpatialStoreMixin"]
