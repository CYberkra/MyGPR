"""Project-scoped interpretation use cases.

The service owns validation and scientific depth conversion. Persistence remains
behind the project session port, so presentations never touch project paths.
"""
from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from mygpr.application.project.service import ProjectService
from mygpr.domain.interpretation.models import (
    BoreholeComparison,
    BoreholeRecord,
    InterfaceAnnotation,
    InterpretationFeature,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class InterpretationService:
    def __init__(self, projects: ProjectService) -> None:
        self._projects = projects

    def list_features(self, project_id: str, line_id: str) -> tuple[InterpretationFeature, ...]:
        return self._projects.list_interpretation_features(project_id, line_id)

    def _add(
        self,
        project_id: str,
        line_id: str,
        *,
        feature_type: str,
        geometry: Mapping[str, Any],
        confidence: float,
        label: str,
        result_id: str = "",
        properties: Mapping[str, Any] | None = None,
    ) -> InterpretationFeature:
        value = float(confidence)
        if not 0.0 <= value <= 1.0:
            raise ValueError("置信度必须位于 0 到 1")
        now = _utc_now()
        feature = InterpretationFeature(
            feature_id=f"I-{uuid.uuid4().hex[:12]}",
            line_id=line_id,
            feature_type=feature_type,
            label=str(label or feature_type),
            confidence=value,
            geometry=dict(geometry),
            status="draft",
            result_id=str(result_id or ""),
            created_at=now,
            updated_at=now,
            properties=dict(properties or {}),
        )
        items = list(self.list_features(project_id, line_id))
        items.append(feature)
        self._projects.replace_interpretation_features(project_id, line_id, tuple(items))
        return feature

    def add_point(self, project_id: str, line_id: str, *, trace: float, sample: float, confidence: float, label: str = "解释点", result_id: str = "", properties: Mapping[str, Any] | None = None) -> InterpretationFeature:
        return self._add(project_id, line_id, feature_type="point", geometry={"type": "Point", "coordinates": [float(trace), float(sample)]}, confidence=confidence, label=label, result_id=result_id, properties=properties)

    def add_interface(self, project_id: str, line_id: str, *, points: Sequence[tuple[float, float]], confidence: float, label: str = "基覆界面", result_id: str = "", properties: Mapping[str, Any] | None = None) -> InterpretationFeature:
        coords = [[float(x), float(y)] for x, y in points]
        if len(coords) < 2:
            raise ValueError("界面线至少需要两个点")
        return self._add(project_id, line_id, feature_type="interface_line", geometry={"type": "LineString", "coordinates": coords}, confidence=confidence, label=label, result_id=result_id, properties=properties)

    def add_zone(self, project_id: str, line_id: str, *, trace_start: float, trace_end: float, sample_start: float, sample_end: float, confidence: float, label: str = "解释区间", properties: Mapping[str, Any] | None = None) -> InterpretationFeature:
        x0, x1 = sorted((float(trace_start), float(trace_end)))
        y0, y1 = sorted((float(sample_start), float(sample_end)))
        if x0 == x1 or y0 == y1:
            raise ValueError("异常区间必须具有非零宽度和高度")
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        return self._add(project_id, line_id, feature_type="interval", geometry={"type": "Polygon", "coordinates": [ring]}, confidence=confidence, label=label, properties=properties)

    def delete_feature(self, project_id: str, line_id: str, feature_id: str) -> bool:
        items = list(self.list_features(project_id, line_id))
        remaining = tuple(item for item in items if item.feature_id != str(feature_id))
        if len(remaining) == len(items):
            return False
        self._projects.replace_interpretation_features(project_id, line_id, remaining)
        return True

    def load_interface(self, project_id: str, line_id: str, *, create: bool = True) -> InterfaceAnnotation | None:
        return self._projects.load_interface_annotation(project_id, line_id, create=create)

    def save_interface(self, project_id: str, annotation: InterfaceAnnotation) -> InterfaceAnnotation:
        if annotation.status == "confirmed" and len(annotation.points) < 2:
            raise ValueError("确认界面至少需要两个关键点")
        if annotation.status == "draft" and not annotation.points and not annotation.zones:
            raise ValueError("草稿至少需要一个关键点或一个语义区间")
        confidence = float(annotation.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("置信度必须位于 0 到 1")
        return self._projects.save_interface_annotation(project_id, replace(annotation, updated_at=_utc_now()))

    def list_boreholes(self, project_id: str) -> tuple[BoreholeRecord, ...]:
        return self._projects.list_boreholes(project_id)

    def save_borehole(self, project_id: str, borehole: BoreholeRecord) -> BoreholeRecord:
        if not str(borehole.borehole_id).strip():
            raise ValueError("钻孔编号不能为空")
        return self._projects.save_borehole(project_id, borehole)

    def delete_borehole(self, project_id: str, borehole_id: str) -> bool:
        return self._projects.delete_borehole(project_id, borehole_id)

    def compare_boreholes(self, project_id: str, line_id: str, *, threshold_m: float = 1.0, velocity_m_per_ns: float | None = None) -> tuple[BoreholeComparison, ...]:
        annotation = self.load_interface(project_id, line_id, create=False)
        if annotation is None or len(annotation.points) < 2:
            return ()
        points = sorted(annotation.points, key=lambda item: item.trace_index)
        trace_axis = np.asarray([item.trace_index for item in points], dtype=float)
        sample_axis = np.asarray([item.sample_index for item in points], dtype=float)
        holes = [item for item in self.list_boreholes(project_id) if item.line_id == line_id and item.trace_index >= 0]
        if not holes:
            return ()
        samples = np.asarray([float(np.interp(item.trace_index, trace_axis, sample_axis)) for item in holes], dtype=float)
        if velocity_m_per_ns is None:
            depths = self._projects.depth_at_samples(project_id, line_id, samples)
        else:
            info = self._projects.get_dataset_info(project_id, line_id)
            dt_ns = float(info.time_window_ns) / max(1, int(info.shape[0]) - 1)
            depths = samples * dt_ns * float(velocity_m_per_ns) / 2.0
        limit = max(0.0, float(threshold_m))
        return tuple(
            BoreholeComparison(
                borehole_id=hole.borehole_id,
                line_id=line_id,
                measured_depth_m=float(hole.basal_depth_m),
                interpreted_depth_m=float(depth),
                absolute_error_m=abs(float(depth) - float(hole.basal_depth_m)),
                passed=abs(float(depth) - float(hole.basal_depth_m)) <= limit,
            )
            for hole, depth in zip(holes, depths)
        )
