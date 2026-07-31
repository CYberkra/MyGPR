#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed interpretation persistence mixin for hybrid field projects."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from core.field_project_models import validate_line_id
from core.basal_interface_annotations import (
    BasalInterfaceAnnotation, InterfaceKeyPoint, InterfaceSegment,
)
from core.security_paths import resolve_managed_path
from mygpr.domain.interpretation.models import (
    BoreholeLayer, BoreholeRecord, InterfaceAnnotation, InterpretationFeature,
    InterpretationPoint, InterpretationZone,
)

_BOREHOLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _validate_borehole_id(value: str) -> str:
    text = str(value or "").strip()
    if not _BOREHOLE_ID_RE.fullmatch(text):
        raise ValueError("钻孔编号仅允许字母、数字、下划线、点和连字符，长度 1–64。")
    return text


def _feature_payload(feature: InterpretationFeature) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": feature.feature_id,
        "geometry": dict(feature.geometry),
        "properties": {
            **dict(feature.properties),
            "feature_type": feature.feature_type,
            "line_id": feature.line_id,
            "label": feature.label,
            "confidence": feature.confidence,
            "status": feature.status,
            "result_id": feature.result_id,
            "created_at": feature.created_at,
            "updated_at": feature.updated_at,
        },
    }


def _feature_from_payload(line_id: str, payload: Mapping[str, Any]) -> InterpretationFeature:
    props = dict(payload.get("properties") or {})
    return InterpretationFeature(
        feature_id=str(payload.get("id") or ""),
        line_id=str(props.pop("line_id", line_id)),
        feature_type=str(props.pop("feature_type", "")),
        label=str(props.pop("label", "解释对象")),
        confidence=float(props.pop("confidence", 0.0)),
        geometry=dict(payload.get("geometry") or {}),
        status=str(props.pop("status", "draft")),
        result_id=str(props.pop("result_id", "") or ""),
        created_at=str(props.pop("created_at", "")),
        updated_at=str(props.pop("updated_at", "")),
        properties=props,
    )


def _interface_from_core(value: BasalInterfaceAnnotation) -> InterfaceAnnotation:
    confidence_map = {"low": 0.45, "medium": 0.7, "high": 0.9}
    points = tuple(
        InterpretationPoint(
            float(point.trace_index), float(point.sample_index),
            confidence_map.get(str(point.confidence), 0.7), str(point.note),
        )
        for point in value.keypoints
    )
    zones = tuple(
        InterpretationZone(
            float(segment.start_trace), float(segment.end_trace), 0.0,
            float(max(value.sample_count - 1, 0)), str(segment.kind), str(segment.note),
        )
        for segment in value.segments
    )
    confidence = float(np.mean([point.confidence for point in points])) if points else 0.0
    return InterfaceAnnotation(
        annotation_id=f"B-{value.line_id}", line_id=str(value.line_id),
        name=f"{value.line_id} 基覆界面", version=int(value.version),
        status=str(value.status), points=points, zones=zones, confidence=confidence,
        processing_result=str(value.source_result_id or ""), created_at=str(value.created_at),
        updated_at=str(value.updated_at), note=str(value.note or ""),
    )


def _interface_payload(value: InterfaceAnnotation) -> dict[str, Any]:
    return {
        "schema": "mygpr.interface_annotation.v2",
        "annotation_id": value.annotation_id,
        "line_id": value.line_id,
        "name": value.name,
        "version": int(value.version),
        "status": value.status,
        "points": [asdict(item) for item in value.points],
        "zones": [asdict(item) for item in value.zones],
        "confidence": float(value.confidence),
        "processing_result": value.processing_result,
        "created_at": value.created_at,
        "updated_at": value.updated_at,
        "note": value.note,
        "uncertainty_samples": float(value.uncertainty_samples),
        "edit_metadata": dict(value.edit_metadata),
    }


def _interface_from_payload(payload: Mapping[str, Any]) -> InterfaceAnnotation:
    return InterfaceAnnotation(
        annotation_id=str(payload.get("annotation_id") or ""),
        line_id=str(payload.get("line_id") or ""),
        name=str(payload.get("name") or "基覆界面"),
        version=int(payload.get("version") or 1),
        status=str(payload.get("status") or "draft"),
        points=tuple(
            InterpretationPoint(**dict(item))
            for item in payload.get("points", ()) if isinstance(item, Mapping)
        ),
        zones=tuple(
            InterpretationZone(**dict(item))
            for item in payload.get("zones", ()) if isinstance(item, Mapping)
        ),
        confidence=float(payload.get("confidence") or 0.0),
        processing_result=str(payload.get("processing_result") or ""),
        created_at=str(payload.get("created_at") or ""),
        updated_at=str(payload.get("updated_at") or ""),
        note=str(payload.get("note") or ""),
        uncertainty_samples=float(payload.get("uncertainty_samples") or 0.0),
        edit_metadata=dict(payload.get("edit_metadata") or {}),
    )




class InterpretationPersistenceMixin:
    """Project-session interpretation operations with validated managed paths."""

    def _interpretation_path(self, line_id: str) -> Path:
        safe = self._validated_line(line_id)
        return resolve_managed_path(
            self._store.root, f"metadata/interpretations/features/{safe}.geojson"
        )

    def list_interpretation_features(self, line_id: str) -> Sequence[InterpretationFeature]:
        safe = self._validated_line(line_id)
        path = self._interpretation_path(safe)
        legacy = resolve_managed_path(self._store.root, f"interpretations/{safe}.geojson")
        source = path if path.exists() else legacy
        if not source.exists():
            return ()
        payload = json.loads(source.read_text(encoding="utf-8"))
        return tuple(
            _feature_from_payload(safe, item)
            for item in payload.get("features", [])
            if isinstance(item, Mapping)
        )

    def replace_interpretation_features(
        self, line_id: str, features: Sequence[InterpretationFeature]
    ) -> None:
        safe = self._validated_line(line_id)
        for item in features:
            if validate_line_id(item.line_id) != safe:
                raise ValueError("解释对象与目标测线不一致")
        payload = {
            "type": "FeatureCollection",
            "schema": "mygpr.interpretations.v1",
            "line_id": safe,
            "updated_at": self._store.now(),
            "features": [_feature_payload(item) for item in features],
        }
        with self._lock:
            self._store.write_json(self._interpretation_path(safe), payload)
            self._store.storage.catalog.append_audit(
                "interpretation_features_replaced", object_type="line", object_id=safe,
                payload={"feature_count": len(features)},
            )

    def _interface_contract_path(self, line_id: str) -> Path:
        safe = self._validated_line(line_id)
        return resolve_managed_path(
            self._store.root, f"metadata/interpretations/interfaces/{safe}.json"
        )

    def _interface_contract_history_path(
        self, line_id: str, version: int, updated_at: str
    ) -> Path:
        safe = self._validated_line(line_id)
        stamp = re.sub(r"[^0-9A-Za-z._-]+", "_", str(updated_at or self._store.now()))
        return resolve_managed_path(
            self._store.root,
            f"metadata/interpretations/interfaces/history/{safe}/{safe}_v{int(version):04d}_{stamp}.json",
        )

    def load_interface_annotation(
        self, line_id: str, *, create: bool
    ) -> InterfaceAnnotation | None:
        safe = self._validated_line(line_id)
        contract_path = self._interface_contract_path(safe)
        with self._lock:
            if contract_path.exists():
                payload = json.loads(contract_path.read_text(encoding="utf-8"))
                value = _interface_from_payload(payload)
                if validate_line_id(value.line_id) != safe:
                    raise ValueError("界面标注与测线标识不一致")
                return value
            legacy = self._store.load_basal_interface_annotation(safe, create=create)
        return None if legacy is None else _interface_from_core(legacy)

    def save_interface_annotation(self, annotation: InterfaceAnnotation) -> InterfaceAnnotation:
        safe = self._validated_line(annotation.line_id)
        info = self.get_dataset_info(safe)
        max_sample = float(info.shape[0] - 1)
        max_trace = float(info.shape[1] - 1)
        for point in annotation.points:
            if not (0.0 <= point.trace_index <= max_trace and 0.0 <= point.sample_index <= max_sample):
                raise ValueError(
                    f"界面点超出数据范围: trace={point.trace_index}, sample={point.sample_index}"
                )
        for zone in annotation.zones:
            if not (
                0.0 <= zone.start_trace <= max_trace
                and 0.0 <= zone.end_trace <= max_trace
                and 0.0 <= zone.start_sample <= max_sample
                and 0.0 <= zone.end_sample <= max_sample
            ):
                raise ValueError("界面语义区间超出数据范围")

        contract_path = self._interface_contract_path(safe)
        with self._lock:
            self._store.assert_writable()
            existing: InterfaceAnnotation | None = None
            if contract_path.exists():
                old_payload = json.loads(contract_path.read_text(encoding="utf-8"))
                existing = _interface_from_payload(old_payload)
                self._store.write_json(
                    self._interface_contract_history_path(
                        safe, existing.version, existing.updated_at
                    ),
                    old_payload,
                )
            else:
                legacy = self._store.load_basal_interface_annotation(safe, create=False)
                if legacy is not None:
                    existing = _interface_from_core(legacy)

            next_version = max(1, int(existing.version) + 1 if existing else int(annotation.version))
            saved = InterfaceAnnotation(
                annotation_id=str(annotation.annotation_id or f"B-{safe}"),
                line_id=safe,
                name=str(annotation.name or f"{safe} 基覆界面"),
                version=next_version,
                status=str(annotation.status),
                points=tuple(sorted(annotation.points, key=lambda item: item.trace_index)),
                zones=tuple(annotation.zones),
                confidence=float(annotation.confidence),
                processing_result=str(annotation.processing_result or ""),
                created_at=str((existing.created_at if existing else annotation.created_at) or self._store.now()),
                updated_at=self._store.now(),
                note=str(annotation.note or ""),
                uncertainty_samples=float(annotation.uncertainty_samples),
                edit_metadata=dict(annotation.edit_metadata),
            )

            confidence_name = lambda value: "high" if value >= 0.82 else ("medium" if value >= 0.58 else "low")
            legacy_value = BasalInterfaceAnnotation(
                line_id=safe, trace_count=int(info.shape[1]), sample_count=int(info.shape[0]),
                source_result_id=saved.processing_result or f"{safe}_raw",
                source_mode="processed" if saved.processing_result else "raw",
                status=saved.status,
                version=max(1, saved.version - 1),
                keypoints=[
                    InterfaceKeyPoint(
                        int(round(point.trace_index)), float(point.sample_index),
                        confidence_name(point.confidence), point.note,
                    )
                    for point in saved.points
                ],
                segments=[
                    InterfaceSegment(
                        int(round(zone.start_trace)), int(round(zone.end_trace)),
                        zone.kind, zone.note,
                    )
                    for zone in saved.zones
                ],
                note=saved.note, created_at=saved.created_at, updated_at=saved.updated_at,
            )
            self._store.save_basal_interface_annotation(
                safe, legacy_value, export_labels=True
            )
            self._store.write_json(contract_path, _interface_payload(saved))
            self._store.storage.catalog.append_audit(
                "basal_interface_saved", object_type="line", object_id=safe,
                payload={
                    "version": saved.version, "status": saved.status,
                    "point_count": len(saved.points), "zone_count": len(saved.zones),
                    "processing_result": saved.processing_result,
                },
            )
        return saved

    def _borehole_path(self) -> Path:
        return resolve_managed_path(
            self._store.root, "metadata/interpretations/boreholes.json"
        )

    def list_boreholes(self) -> Sequence[BoreholeRecord]:
        path = self._borehole_path()
        legacy = resolve_managed_path(self._store.root, "interpretations/boreholes.json")
        source = path if path.exists() else legacy
        if not source.exists():
            return ()
        payload = json.loads(source.read_text(encoding="utf-8"))
        result: list[BoreholeRecord] = []
        for item in payload.get("boreholes", []):
            if not isinstance(item, Mapping):
                continue
            layers = tuple(BoreholeLayer(**layer) for layer in item.get("layers", []) if isinstance(layer, Mapping))
            clean = {key: value for key, value in item.items() if key != "layers"}
            result.append(BoreholeRecord(**clean, layers=layers))
        return tuple(sorted(result, key=lambda item: item.borehole_id))

    def save_borehole(self, borehole: BoreholeRecord) -> BoreholeRecord:
        borehole_id = _validate_borehole_id(borehole.borehole_id)
        if borehole.line_id:
            self._validated_line(borehole.line_id)
        normalized = BoreholeRecord(
            borehole_id=borehole_id, name=str(borehole.name or borehole_id),
            x=float(borehole.x), y=float(borehole.y),
            surface_elevation_m=float(borehole.surface_elevation_m),
            line_id=str(borehole.line_id), trace_index=float(borehole.trace_index),
            basal_depth_m=float(borehole.basal_depth_m), layers=tuple(borehole.layers),
            note=str(borehole.note),
        )
        items = {item.borehole_id: item for item in self.list_boreholes()}
        items[borehole_id] = normalized
        with self._lock:
            self._store.write_json(
                self._borehole_path(),
                {
                    "schema": "mygpr.boreholes.v1", "updated_at": self._store.now(),
                    "boreholes": [asdict(item) for item in sorted(items.values(), key=lambda value: value.borehole_id)],
                },
            )
            self._store.storage.catalog.append_audit(
                "borehole_saved", object_type="borehole", object_id=borehole_id,
                payload={"line_id": normalized.line_id},
            )
        return normalized

    def delete_borehole(self, borehole_id: str) -> bool:
        safe = _validate_borehole_id(borehole_id)
        items = {item.borehole_id: item for item in self.list_boreholes()}
        removed = items.pop(safe, None)
        if removed is None:
            return False
        with self._lock:
            self._store.write_json(
                self._borehole_path(),
                {
                    "schema": "mygpr.boreholes.v1", "updated_at": self._store.now(),
                    "boreholes": [asdict(item) for item in sorted(items.values(), key=lambda value: value.borehole_id)],
                },
            )
            self._store.storage.catalog.append_audit(
                "borehole_deleted", object_type="borehole", object_id=safe
            )
        return True

    def depth_at_samples(self, line_id: str, samples: np.ndarray) -> np.ndarray:
        safe = self._validated_line(line_id)
        with self._lock:
            dataset = self._store.load_gpr_dataset(safe)
            axis = np.asarray(dataset.depth_axis_m, dtype=float)
        if axis.size == 0:
            raise ValueError(f"测线 {safe} 缺少深度轴")
        sample_axis = np.arange(axis.size, dtype=float)
        return np.interp(np.asarray(samples, dtype=float), sample_axis, axis)



__all__ = ["InterpretationPersistenceMixin"]
