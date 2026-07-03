#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Versioned GeoJSON interpretation storage for project lines."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from core.project_models import InterpretationFeatureV1
from core.project_service import ProjectService, atomic_write_json, utc_now


class InterpretationService:
    def __init__(self, project: ProjectService):
        self.project = project

    def list_features(self, line_id: str) -> list[InterpretationFeatureV1]:
        payload = self._read(line_id)
        return [
            self._from_geojson_feature(line_id, item)
            for item in payload.get("features", [])
            if isinstance(item, dict)
        ]

    def add_point(
        self,
        line_id: str,
        *,
        trace: float,
        sample: float,
        confidence: float,
        result_id: str | None = None,
        label: str = "解释点",
        properties: dict[str, Any] | None = None,
    ) -> InterpretationFeatureV1:
        return self._add(
            line_id,
            feature_type="point",
            geometry={"type": "Point", "coordinates": [float(trace), float(sample)]},
            confidence=confidence,
            result_id=result_id,
            label=label,
            properties=properties,
        )

    def add_interface_line(
        self,
        line_id: str,
        *,
        points: list[tuple[float, float]],
        confidence: float,
        result_id: str | None = None,
        label: str = "界面线",
        properties: dict[str, Any] | None = None,
    ) -> InterpretationFeatureV1:
        if len(points) < 2:
            raise ValueError("界面线至少需要两个点")
        return self._add(
            line_id,
            feature_type="interface_line",
            geometry={
                "type": "LineString",
                "coordinates": [[float(x), float(y)] for x, y in points],
            },
            confidence=confidence,
            result_id=result_id,
            label=label,
            properties=properties,
        )

    def add_interval(
        self,
        line_id: str,
        *,
        trace_start: float,
        trace_end: float,
        sample_start: float,
        sample_end: float,
        confidence: float,
        result_id: str | None = None,
        label: str = "异常区间",
        properties: dict[str, Any] | None = None,
    ) -> InterpretationFeatureV1:
        x0, x1 = sorted((float(trace_start), float(trace_end)))
        y0, y1 = sorted((float(sample_start), float(sample_end)))
        if x0 == x1 or y0 == y1:
            raise ValueError("异常区间必须具有非零宽度和高度")
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        return self._add(
            line_id,
            feature_type="interval",
            geometry={"type": "Polygon", "coordinates": [ring]},
            confidence=confidence,
            result_id=result_id,
            label=label,
            properties=properties,
        )

    def update_feature(
        self,
        line_id: str,
        feature_id: str,
        *,
        confidence: float | None = None,
        result_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> InterpretationFeatureV1:
        features = self.list_features(line_id)
        for feature in features:
            if feature.feature_id != feature_id:
                continue
            if confidence is not None:
                feature.confidence = self._confidence(confidence)
            if result_id is not None:
                feature.result_id = result_id
            if properties is not None:
                feature.properties.update(dict(properties))
            feature.properties["updated_at"] = utc_now()
            self._write(line_id, features)
            return feature
        raise KeyError(feature_id)

    def delete_feature(self, line_id: str, feature_id: str) -> bool:
        features = self.list_features(line_id)
        remaining = [item for item in features if item.feature_id != feature_id]
        if len(remaining) == len(features):
            return False
        self._write(line_id, remaining)
        return True

    def _add(
        self,
        line_id: str,
        *,
        feature_type: str,
        geometry: dict[str, Any],
        confidence: float,
        result_id: str | None,
        label: str,
        properties: dict[str, Any] | None,
    ) -> InterpretationFeatureV1:
        now = utc_now()
        feature = InterpretationFeatureV1(
            feature_id=f"I-{uuid.uuid4().hex[:12]}",
            line_id=line_id,
            feature_type=feature_type,
            geometry=geometry,
            confidence=self._confidence(confidence),
            result_id=result_id,
            properties={
                "label": str(label).strip() or feature_type,
                "created_at": now,
                "updated_at": now,
                **dict(properties or {}),
            },
        )
        features = self.list_features(line_id)
        features.append(feature)
        self._write(line_id, features)
        return feature

    @staticmethod
    def _confidence(value: float) -> float:
        confidence = float(value)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("置信度必须位于 0 到 1")
        return confidence

    def _path(self, line_id: str) -> Path:
        return self.project.resolve_relative_path(f"interpretations/{line_id}.geojson")

    def _read(self, line_id: str) -> dict[str, Any]:
        path = self._path(line_id)
        if not path.exists():
            return {
                "type": "FeatureCollection",
                "schema": "mygpr.interpretations.v1",
                "line_id": line_id,
                "features": [],
            }
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(
        self,
        line_id: str,
        features: list[InterpretationFeatureV1],
    ) -> None:
        atomic_write_json(
            self._path(line_id),
            {
                "type": "FeatureCollection",
                "schema": "mygpr.interpretations.v1",
                "line_id": line_id,
                "updated_at": utc_now(),
                "features": [self._to_geojson_feature(item) for item in features],
            },
        )

    @staticmethod
    def _to_geojson_feature(feature: InterpretationFeatureV1) -> dict[str, Any]:
        return {
            "type": "Feature",
            "id": feature.feature_id,
            "geometry": feature.geometry,
            "properties": {
                **feature.properties,
                "feature_type": feature.feature_type,
                "line_id": feature.line_id,
                "confidence": feature.confidence,
                "result_id": feature.result_id,
                "schema": feature.schema,
            },
        }

    @staticmethod
    def _from_geojson_feature(
        line_id: str,
        payload: dict[str, Any],
    ) -> InterpretationFeatureV1:
        properties = dict(payload.get("properties") or {})
        return InterpretationFeatureV1(
            feature_id=str(payload.get("id")),
            line_id=str(properties.pop("line_id", line_id)),
            feature_type=str(properties.pop("feature_type")),
            geometry=dict(payload.get("geometry") or {}),
            confidence=float(properties.pop("confidence", 0.0)),
            result_id=properties.pop("result_id", None),
            properties=properties,
            schema=str(properties.pop("schema", "mygpr.interpretation_feature.v1")),
        )


__all__ = ["InterpretationService"]
