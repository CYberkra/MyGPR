"""Immutable interpretation and borehole contracts used by backend services."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
import math


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class InterpretationPoint:
    trace_index: float
    sample_index: float
    confidence: float = 0.8
    note: str = ""

    def __post_init__(self) -> None:
        values = (float(self.trace_index), float(self.sample_index), float(self.confidence))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("interpretation point values must be finite")
        if not 0.0 <= values[2] <= 1.0:
            raise ValueError("interpretation point confidence must be in [0, 1]")
        object.__setattr__(self, "trace_index", values[0])
        object.__setattr__(self, "sample_index", values[1])
        object.__setattr__(self, "confidence", values[2])


@dataclass(frozen=True, slots=True)
class InterpretationZone:
    start_trace: float
    end_trace: float
    start_sample: float
    end_sample: float
    kind: str = "weak"
    note: str = ""

    def __post_init__(self) -> None:
        x0, x1 = sorted((float(self.start_trace), float(self.end_trace)))
        y0, y1 = sorted((float(self.start_sample), float(self.end_sample)))
        if not all(math.isfinite(value) for value in (x0, x1, y0, y1)):
            raise ValueError("interpretation zone values must be finite")
        if x0 == x1 or y0 == y1:
            raise ValueError("interpretation zone must have non-zero extent")
        kind = str(self.kind)
        if kind not in {"weak", "ignore", "no_interface"}:
            raise ValueError(f"unsupported interpretation zone kind: {kind}")
        object.__setattr__(self, "start_trace", x0)
        object.__setattr__(self, "end_trace", x1)
        object.__setattr__(self, "start_sample", y0)
        object.__setattr__(self, "end_sample", y1)
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True, slots=True)
class InterpretationFeature:
    feature_id: str
    line_id: str
    feature_type: str
    label: str
    confidence: float
    geometry: Mapping[str, Any]
    status: str = "draft"
    result_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry", _mapping(self.geometry))
        object.__setattr__(self, "properties", _mapping(self.properties))


@dataclass(frozen=True, slots=True)
class InterfaceAnnotation:
    annotation_id: str
    line_id: str
    name: str
    version: int
    status: str
    points: tuple[InterpretationPoint, ...] = ()
    zones: tuple[InterpretationZone, ...] = ()
    confidence: float = 0.0
    processing_result: str = ""
    created_at: str = ""
    updated_at: str = ""
    note: str = ""
    uncertainty_samples: float = 0.0
    edit_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.annotation_id).strip() or not str(self.line_id).strip():
            raise ValueError("annotation_id and line_id are required")
        if int(self.version) < 1:
            raise ValueError("annotation version must be >= 1")
        if str(self.status) not in {"draft", "confirmed", "superseded"}:
            raise ValueError(f"unsupported annotation status: {self.status}")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("annotation confidence must be in [0, 1]")
        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "zones", tuple(self.zones))
        uncertainty = float(self.uncertainty_samples)
        if not math.isfinite(uncertainty) or uncertainty < 0.0:
            raise ValueError("annotation uncertainty_samples must be finite and non-negative")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "uncertainty_samples", uncertainty)
        object.__setattr__(self, "edit_metadata", _mapping(self.edit_metadata))


@dataclass(frozen=True, slots=True)
class InterfaceTraceConfig:
    search_half_window: int = 18
    max_step_samples: int = 8
    smooth_radius: int = 2
    anchor_weight: float = 0.08
    continuity_weight: float = 0.12
    min_sample: int = 0
    max_sample: int | None = None
    max_points: int = 160

    def __post_init__(self) -> None:
        if int(self.search_half_window) < 1:
            raise ValueError("search_half_window must be >= 1")
        if int(self.max_step_samples) < 0 or int(self.smooth_radius) < 0:
            raise ValueError("trace step and smoothing values must be non-negative")
        if int(self.max_points) < 2:
            raise ValueError("max_points must be >= 2")


@dataclass(frozen=True, slots=True)
class InterfaceEditSnapshot:
    session_id: str
    project_id: str
    line_id: str
    annotation: InterfaceAnnotation
    input_artifact_id: str = ""
    source_shape: tuple[int, int] = (0, 0)
    undo_depth: int = 0
    redo_depth: int = 0
    dirty: bool = False
    audit: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_shape", tuple(int(v) for v in self.source_shape))
        object.__setattr__(self, "audit", _mapping(self.audit))


@dataclass(frozen=True, slots=True)
class InterpretationLabelPackage:
    package_id: str
    project_id: str
    line_id: str
    root_path: str
    manifest_path: str
    files: Mapping[str, str] = field(default_factory=dict)
    sha256: Mapping[str, str] = field(default_factory=dict)
    summary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", _mapping(self.files))
        object.__setattr__(self, "sha256", _mapping(self.sha256))
        object.__setattr__(self, "summary", _mapping(self.summary))


@dataclass(frozen=True, slots=True)
class BoreholeLayer:
    name: str
    top_depth_m: float
    bottom_depth_m: float
    lithology: str = ""
    color_hint: str = ""


@dataclass(frozen=True, slots=True)
class BoreholeRecord:
    borehole_id: str
    name: str
    x: float = 0.0
    y: float = 0.0
    surface_elevation_m: float = 0.0
    line_id: str = ""
    trace_index: float = -1.0
    basal_depth_m: float = 0.0
    layers: tuple[BoreholeLayer, ...] = ()
    note: str = ""


@dataclass(frozen=True, slots=True)
class BoreholeComparison:
    borehole_id: str
    line_id: str
    measured_depth_m: float
    interpreted_depth_m: float
    absolute_error_m: float
    passed: bool


__all__ = [
    "BoreholeComparison", "BoreholeLayer", "BoreholeRecord", "InterfaceAnnotation",
    "InterfaceEditSnapshot", "InterfaceTraceConfig", "InterpretationLabelPackage",
    "InterpretationFeature", "InterpretationPoint", "InterpretationZone",
]
