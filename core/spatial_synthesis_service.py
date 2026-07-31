#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-level multi-line spatial synthesis without invented coordinates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.interpretation_service import InterpretationService
from core.processing_session import ProcessingSessionService
from core.project_service import ProjectService
from core.sidecar_parsers import parse_sidecar_csv


class SpatialSynthesisService:
    def __init__(self, project: ProjectService):
        self.project = project

    def build(self) -> dict[str, Any]:
        tracks: list[dict[str, Any]] = []
        terrain_points: list[dict[str, Any]] = []
        interpretation_features: list[dict[str, Any]] = []
        unlocated_lines: list[dict[str, Any]] = []
        interpretation_service = InterpretationService(self.project)

        for line in self.project.list_lines():
            if not line.raw_files:
                unlocated_lines.append(
                    {
                        "line_id": line.line_id,
                        "name": line.name,
                        "reason": "缺少主雷达数据",
                    }
                )
                continue
            try:
                metadata = self._metadata_from_sidecars(line)
                if metadata is None:
                    session = ProcessingSessionService.open_line(
                        self.project,
                        line.line_id,
                        enforce_processing_gate=False,
                    )
                    metadata = session.original_trace_metadata or {}
            except Exception as exc:
                unlocated_lines.append(
                    {
                        "line_id": line.line_id,
                        "name": line.name,
                        "reason": f"空间定位数据无法读取：{exc}",
                    }
                )
                continue
            longitude = np.asarray(metadata.get("longitude", []), dtype=np.float64)
            latitude = np.asarray(metadata.get("latitude", []), dtype=np.float64)
            count = min(longitude.size, latitude.size)
            if count <= 0:
                unlocated_lines.append(
                    {
                        "line_id": line.line_id,
                        "name": line.name,
                        "reason": "无空间定位逐道元数据",
                    }
                )
                continue
            longitude = longitude[:count]
            latitude = latitude[:count]
            track = {
                "line_id": line.line_id,
                "name": line.name,
                "longitude": longitude.tolist(),
                "latitude": latitude.tolist(),
                "trace_count": int(count),
            }
            tracks.append(track)
            ground = np.asarray(
                metadata.get("ground_elevation_m", np.full(count, np.nan)),
                dtype=np.float64,
            )
            height = np.asarray(
                metadata.get("height_agl_m", metadata.get("flight_height_m", np.full(count, np.nan))),
                dtype=np.float64,
            )
            for index in range(count):
                terrain_points.append(
                    {
                        "line_id": line.line_id,
                        "trace": index,
                        "longitude": float(longitude[index]),
                        "latitude": float(latitude[index]),
                        "ground_elevation_m": _finite_or_none(ground, index),
                        "height_agl_m": _finite_or_none(height, index),
                    }
                )

            for feature in interpretation_service.list_features(line.line_id):
                trace = _feature_trace(feature.geometry)
                if trace is None:
                    continue
                trace_index = int(round(trace))
                if trace_index < 0 or trace_index >= count:
                    continue
                interpretation_features.append(
                    {
                        "feature_id": feature.feature_id,
                        "feature_type": feature.feature_type,
                        "line_id": line.line_id,
                        "trace": trace,
                        "longitude": float(longitude[trace_index]),
                        "latitude": float(latitude[trace_index]),
                        "properties": {
                            **feature.properties,
                            "confidence": feature.confidence,
                            "result_id": feature.result_id,
                        },
                    }
                )

        return {
            "schema": "mygpr.spatial_synthesis.v1",
            "project_id": self.project.manifest.project_id,
            "tracks": tracks,
            "terrain_points": terrain_points,
            "interpretation_features": interpretation_features,
            "unlocated_lines": unlocated_lines,
            "summary": {
                "line_count": len(self.project.manifest.line_ids),
                "located_line_count": len(tracks),
                "unlocated_line_count": len(unlocated_lines),
                "track_point_count": sum(item["trace_count"] for item in tracks),
                "terrain_point_count": len(terrain_points),
                "interpretation_feature_count": len(interpretation_features),
            },
        }

    def _metadata_from_sidecars(self, line) -> dict[str, np.ndarray] | None:
        """Build spatial metadata from sidecar files without loading the full B-scan.

        Spatial synthesis and delivery packaging only need per-trace position/height.
        Loading the full primary radar CSV for every line makes large field projects
        unnecessarily slow, especially when primary CSV rows are stored sample-by-sample.
        """
        rtk_path = self._resolve_sidecar_path(line, "rtk")
        if rtk_path is None or not rtk_path.exists():
            return None
        rtk = parse_sidecar_csv(rtk_path, kind="rtk")
        longitude = np.asarray(rtk.get("longitude", []), dtype=np.float64)
        latitude = np.asarray(rtk.get("latitude", []), dtype=np.float64)
        if longitude.size == 0 or latitude.size == 0:
            return None
        count = min(longitude.size, latitude.size)
        metadata: dict[str, np.ndarray] = {
            "trace_index": np.arange(count, dtype=np.int32),
            "longitude": longitude[:count],
            "latitude": latitude[:count],
        }
        for key in ("ground_elevation_m", "flight_height_m"):
            value = rtk.get(key)
            if value is not None:
                metadata[key] = np.asarray(value)[:count]
        alt_path = self._resolve_sidecar_path(line, "altimeter")
        if alt_path is not None and alt_path.exists():
            try:
                alt = parse_sidecar_csv(alt_path, kind="altimeter")
                height = np.asarray(alt.get("height_agl_m", []), dtype=np.float64)
                if height.size:
                    metadata["height_agl_m"] = height[:count]
            except (OSError, UnicodeError, ValueError, TypeError, KeyError):
                # RTK coordinates are still usable; keep spatial synthesis available.
                pass
        if "height_agl_m" not in metadata and "flight_height_m" in metadata:
            metadata["height_agl_m"] = np.asarray(metadata["flight_height_m"], dtype=np.float64)[:count]
        return metadata

    def _resolve_sidecar_path(self, line, kind: str) -> Path | None:
        value = line.sidecars.get(kind)
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.project.resolve_relative_path(path)
        return path


def _finite_or_none(values: np.ndarray, index: int) -> float | None:
    if index >= values.size or not np.isfinite(values[index]):
        return None
    return float(values[index])


def _feature_trace(geometry: dict[str, Any]) -> float | None:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind == "Point" and coordinates:
        return float(coordinates[0])
    if kind == "LineString" and coordinates:
        return float(np.mean([item[0] for item in coordinates]))
    if kind == "Polygon" and coordinates and coordinates[0]:
        return float(np.mean([item[0] for item in coordinates[0]]))
    return None


__all__ = ["SpatialSynthesisService"]
