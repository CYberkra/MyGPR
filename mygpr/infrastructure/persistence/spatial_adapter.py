#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial-track and versioned-result persistence mixin."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from core.spatial_result_versions import SpatialResultVersionService
from mygpr.domain.spatial.models import SpatialResult, SpatialTrack, SpatialTrackPoint

def _spatial_record(value: Any) -> SpatialResult:
    return SpatialResult(
        result_id=str(value.result_id), name=str(value.name), revision=int(value.revision),
        status=str(value.status), line_ids=tuple(value.line_ids), created_at=str(value.created_at),
        coordinate_system=str(value.coordinate_system), vertical_datum=str(value.vertical_datum),
        stale=bool(value.stale), summary=dict(value.summary or {}), files=dict(value.files or {}),
    )


class SpatialPersistenceMixin:
    """Project-session spatial operations using the formal result-version store."""

    def load_spatial_tracks(self) -> Sequence[SpatialTrack]:
        coordinate_system = str(self._store.manifest.coordinate_system or "")
        tracks: list[SpatialTrack] = []
        for line in self.list_lines():
            metadata = self.read_trace_metadata(line.line_id)
            def first(*names: str):
                for name in names:
                    if name in metadata:
                        return np.asarray(metadata[name], dtype=float)
                return None
            x = first("local_x_m", "easting_m", "x")
            y = first("local_y_m", "northing_m", "y")
            source = "projected trace metadata"
            crs = coordinate_system
            if x is None or y is None:
                x = first("longitude", "lon")
                y = first("latitude", "lat")
                source = "geographic trace metadata"
                crs = "EPSG:4326"
            z = first("local_z_m", "elevation_m", "altitude_m", "height_m", "z")
            fh = first("flight_height_m", "height_agl_m", "agl_m")
            ge = first("ground_elevation_m")
            if x is None or y is None:
                # CSV 导入的测线 trace metadata 常无坐标键，回退到轨迹文件
                track = self._spatial_track_from_trajectory(line, coordinate_system)
                if track is not None:
                    tracks.append(track)
                continue
            if z is None:
                z = np.zeros_like(x)
                # z 为合成零值时「海拔 − 离地高度」无意义，禁用该估算回退
                # （ground_elevation_m 仍存在，不受影响）
                fh = None
            count = min(len(x), len(y), len(z))

            def _at(values, index: int):
                if values is None or index >= len(values) or not np.isfinite(values[index]):
                    return None
                return float(values[index])

            points = tuple(
                SpatialTrackPoint(
                    index, float(x[index]), float(y[index]), float(z[index]),
                    flight_height_m=_at(fh, index),
                    ground_elevation_m=_at(ge, index),
                )
                for index in range(count)
                if np.isfinite(x[index]) and np.isfinite(y[index])
            )
            tracks.append(SpatialTrack(line.line_id, line.name, points, crs, source))
        return tuple(tracks)

    def _spatial_track_from_trajectory(self, line, coordinate_system: str) -> SpatialTrack | None:
        """轨迹文件回退：store.load_trajectory(line_id) → SpatialTrack。

        x/y/z 与 coordinate_system 照抄 TrajectoryModel（投影米坐标）；
        无轨迹文件或全部坐标非有限值时返回 None。
        """
        try:
            trajectory = self._store.load_trajectory(line.line_id)
        except Exception:  # noqa: BLE001 - 无轨迹文件等 → 该测线无空间轨迹
            return None
        points = tuple(
            SpatialTrackPoint(
                int(point.trace_index) if point.trace_index >= 0 else index,
                float(point.x), float(point.y), float(point.z),
                flight_height_m=(
                    float(point.flight_height_m)
                    if np.isfinite(point.flight_height_m) else None
                ),
            )
            for index, point in enumerate(trajectory.points)
            if np.isfinite(point.x) and np.isfinite(point.y)
        )
        if not points:
            return None
        crs = next((str(p.coordinate_system) for p in trajectory.points
                    if p.coordinate_system), coordinate_system)
        return SpatialTrack(line.line_id, line.name, points, crs, "trajectory file")

    def list_spatial_results(self) -> Sequence[SpatialResult]:
        with self._lock:
            return tuple(_spatial_record(item) for item in SpatialResultVersionService(self._store).list_results())

    def spatial_preflight(
        self, *, line_ids: Sequence[str] | None, generate_surface: bool
    ) -> Mapping[str, Any]:
        safe_ids = None if line_ids is None else tuple(self._validated_line(value) for value in line_ids)
        with self._lock:
            return dict(SpatialResultVersionService(self._store).preflight(
                line_ids=safe_ids, generate_surface=generate_surface
            ))

    def create_spatial_result(
        self, *, name: str, line_ids: Sequence[str] | None,
        velocity_m_per_ns: float | None, generate_surface: bool
    ) -> SpatialResult:
        safe_ids = tuple(self._validated_line(value) for value in (line_ids or [line.line_id for line in self.list_lines()]))
        velocity = velocity_m_per_ns
        if velocity is None:
            values = []
            for line_id in safe_ids:
                epsilon = float(self.get_dataset_info(line_id).dielectric_constant)
                if epsilon > 0:
                    values.append(0.299792458 / np.sqrt(epsilon))
            if not values:
                raise ValueError("无法从测线介电常数推导电磁波速度")
            velocity = float(np.median(values))
        with self._lock:
            self._store.assert_writable()
            value = SpatialResultVersionService(self._store).create_result(
                name=name, line_ids=safe_ids, velocity_m_per_ns=float(velocity),
                generate_surface=generate_surface,
            )
        return _spatial_record(value)

    def set_current_spatial_result(self, result_id: str) -> None:
        with self._lock:
            self._store.assert_writable()
            SpatialResultVersionService(self._store).set_current(str(result_id))



__all__ = ["SpatialPersistenceMixin"]
