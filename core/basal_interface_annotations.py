#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basal/overburden interface annotation data contract.

The annotation is a continuous trace-wise path, not a collection of point or
hyperbola targets.  Sparse key points define the path; interval semantics mark
weakly visible, ignored and explicit no-interface ranges.  Training exports are
anchored to the raw B-scan trace/sample coordinate system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable

import numpy as np

BASAL_INTERFACE_SCHEMA = "mygpr.basal_interface_annotation.v1"
BASAL_LABEL_SCHEMA = "mygpr.basal_interface_labels.v1"

SEGMENT_KINDS = {"weak", "ignore", "no_interface"}
VISIBILITY_UNKNOWN = np.uint8(0)
VISIBILITY_CLEAR = np.uint8(1)
VISIBILITY_WEAK = np.uint8(2)
VISIBILITY_NO_INTERFACE = np.uint8(3)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True, order=True)
class InterfaceKeyPoint:
    trace_index: int
    sample_index: float
    confidence: str = "medium"
    note: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterfaceKeyPoint":
        return cls(
            trace_index=int(payload.get("trace_index", 0)),
            sample_index=float(payload.get("sample_index", 0.0)),
            confidence=str(payload.get("confidence", "medium")),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True, order=True)
class InterfaceSegment:
    start_trace: int
    end_trace: int
    kind: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in SEGMENT_KINDS:
            raise ValueError(f"Unsupported interface segment kind: {self.kind}")
        if self.end_trace < self.start_trace:
            start, end = self.end_trace, self.start_trace
            object.__setattr__(self, "start_trace", start)
            object.__setattr__(self, "end_trace", end)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterfaceSegment":
        return cls(
            start_trace=int(payload.get("start_trace", 0)),
            end_trace=int(payload.get("end_trace", 0)),
            kind=str(payload.get("kind", "weak")),
            note=str(payload.get("note", "")),
        )


@dataclass
class BasalInterfaceAnnotation:
    line_id: str
    trace_count: int
    sample_count: int
    source_result_id: str = ""
    source_mode: str = "raw"
    status: str = "draft"
    version: int = 1
    interpolation: str = "linear"
    soft_sigma_samples: float = 3.0
    interface_type: str = "basal_interface"
    snap_mode: str = "envelope_peak"
    smoothing_strength: int = 35
    uncertainty_width: int = 58
    keypoints: list[InterfaceKeyPoint] = field(default_factory=list)
    segments: list[InterfaceSegment] = field(default_factory=list)
    note: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    schema: str = BASAL_INTERFACE_SCHEMA

    def normalize(self) -> "BasalInterfaceAnnotation":
        unique: dict[int, InterfaceKeyPoint] = {}
        for point in self.keypoints:
            trace = int(np.clip(point.trace_index, 0, max(self.trace_count - 1, 0)))
            sample = float(np.clip(point.sample_index, 0.0, max(float(self.sample_count - 1), 0.0)))
            unique[trace] = InterfaceKeyPoint(trace, sample, point.confidence, point.note)
        self.keypoints = sorted(unique.values(), key=lambda p: p.trace_index)
        normalized_segments: list[InterfaceSegment] = []
        for segment in self.segments:
            start = int(np.clip(segment.start_trace, 0, max(self.trace_count - 1, 0)))
            end = int(np.clip(segment.end_trace, 0, max(self.trace_count - 1, 0)))
            normalized_segments.append(InterfaceSegment(min(start, end), max(start, end), segment.kind, segment.note))
        self.segments = sorted(normalized_segments, key=lambda s: (s.start_trace, s.end_trace, s.kind))
        return self

    def clone(self) -> "BasalInterfaceAnnotation":
        return BasalInterfaceAnnotation.from_dict(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        self.normalize()
        payload = asdict(self)
        payload["keypoints"] = [asdict(point) for point in self.keypoints]
        payload["segments"] = [asdict(segment) for segment in self.segments]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BasalInterfaceAnnotation":
        annotation = cls(
            line_id=str(payload.get("line_id", "")),
            trace_count=int(payload.get("trace_count", 0)),
            sample_count=int(payload.get("sample_count", 0)),
            source_result_id=str(payload.get("source_result_id", "")),
            source_mode=str(payload.get("source_mode", "raw")),
            status=str(payload.get("status", "draft")),
            version=int(payload.get("version", 1)),
            interpolation=str(payload.get("interpolation", "linear")),
            soft_sigma_samples=float(payload.get("soft_sigma_samples", 3.0)),
            interface_type=str(payload.get("interface_type", "basal_interface")),
            snap_mode=str(payload.get("snap_mode", "envelope_peak")),
            smoothing_strength=int(payload.get("smoothing_strength", 35)),
            uncertainty_width=int(payload.get("uncertainty_width", 58)),
            keypoints=[InterfaceKeyPoint.from_dict(item) for item in payload.get("keypoints", []) if isinstance(item, dict)],
            segments=[InterfaceSegment.from_dict(item) for item in payload.get("segments", []) if isinstance(item, dict)],
            note=str(payload.get("note", "")),
            created_at=str(payload.get("created_at", _now())),
            updated_at=str(payload.get("updated_at", _now())),
            schema=str(payload.get("schema", BASAL_INTERFACE_SCHEMA)),
        )
        return annotation.normalize()

    def set_keypoint(self, trace_index: int, sample_index: float, *, confidence: str = "medium", note: str = "") -> None:
        trace = int(np.clip(trace_index, 0, max(self.trace_count - 1, 0)))
        self.keypoints = [point for point in self.keypoints if point.trace_index != trace]
        self.keypoints.append(InterfaceKeyPoint(trace, float(sample_index), confidence, note))
        self.updated_at = _now()
        self.normalize()

    def remove_nearest_keypoint(self, trace_index: int, *, max_distance: int | None = None) -> InterfaceKeyPoint | None:
        if not self.keypoints:
            return None
        nearest = min(self.keypoints, key=lambda point: abs(point.trace_index - int(trace_index)))
        if max_distance is not None and abs(nearest.trace_index - int(trace_index)) > int(max_distance):
            return None
        self.keypoints.remove(nearest)
        self.updated_at = _now()
        return nearest

    def set_segment(self, start_trace: int, end_trace: int, kind: str, note: str = "") -> None:
        if kind not in SEGMENT_KINDS:
            raise ValueError(kind)
        start, end = sorted((int(start_trace), int(end_trace)))
        # Replace overlapping intervals of the same semantic kind.  Other kinds
        # remain explicit and precedence is resolved in ``trace_semantics``.
        kept = [
            seg for seg in self.segments
            if seg.kind != kind or seg.end_trace < start or seg.start_trace > end
        ]
        kept.append(InterfaceSegment(start, end, kind, note))
        self.segments = kept
        self.updated_at = _now()
        self.normalize()

    def clear_segment_kind(self, kind: str) -> None:
        self.segments = [seg for seg in self.segments if seg.kind != kind]
        self.updated_at = _now()

    def curve_samples(self) -> np.ndarray:
        """Return trace-wise sample coordinates with NaN outside picked extent."""
        curve = np.full(max(self.trace_count, 0), np.nan, dtype=np.float32)
        points = sorted(self.keypoints, key=lambda point: point.trace_index)
        if not points:
            return curve
        if len(points) == 1:
            curve[points[0].trace_index] = np.float32(points[0].sample_index)
            return curve
        traces = np.asarray([point.trace_index for point in points], dtype=np.float64)
        samples = np.asarray([point.sample_index for point in points], dtype=np.float64)
        target = np.arange(points[0].trace_index, points[-1].trace_index + 1, dtype=np.int64)
        curve[target] = np.interp(target, traces, samples).astype(np.float32)
        return curve

    def trace_semantics(self) -> dict[str, np.ndarray]:
        trace_count = max(self.trace_count, 0)
        visibility = np.full(trace_count, VISIBILITY_UNKNOWN, dtype=np.uint8)
        curve = self.curve_samples()
        visibility[np.isfinite(curve)] = VISIBILITY_CLEAR
        ignore = np.zeros(trace_count, dtype=bool)
        no_interface = np.zeros(trace_count, dtype=bool)
        weak = np.zeros(trace_count, dtype=bool)
        for segment in self.segments:
            sl = slice(segment.start_trace, segment.end_trace + 1)
            if segment.kind == "weak":
                weak[sl] = True
            elif segment.kind == "ignore":
                ignore[sl] = True
            elif segment.kind == "no_interface":
                no_interface[sl] = True
        visibility[weak & np.isfinite(curve)] = VISIBILITY_WEAK
        visibility[no_interface] = VISIBILITY_NO_INTERFACE
        # Ignore has highest precedence and is represented through a separate
        # mask; visibility remains unknown to avoid creating a false class.
        visibility[ignore] = VISIBILITY_UNKNOWN
        valid = np.isfinite(curve) & ~ignore & ~no_interface
        return {
            "visibility_gt": visibility,
            "valid_trace_mask": valid,
            "no_interface_mask": no_interface,
            "ignore_trace_mask": ignore,
            "weak_trace_mask": weak,
        }

    def statistics(self) -> dict[str, float | int | str]:
        semantics = self.trace_semantics()
        total = max(self.trace_count, 1)
        curve = self.curve_samples()
        picked = np.isfinite(curve)
        judged = picked | semantics["no_interface_mask"] | semantics["ignore_trace_mask"]
        clear = semantics["visibility_gt"] == VISIBILITY_CLEAR
        weak = semantics["visibility_gt"] == VISIBILITY_WEAK
        return {
            "keypoint_count": len(self.keypoints),
            "segment_count": len(self.segments),
            "coverage_ratio": float(picked.sum() / total),
            "judged_ratio": float(judged.sum() / total),
            "clear_ratio": float(clear.sum() / total),
            "weak_ratio": float(weak.sum() / total),
            "ignore_ratio": float(semantics["ignore_trace_mask"].sum() / total),
            "no_interface_ratio": float(semantics["no_interface_mask"].sum() / total),
            "status": self.status,
        }

    def build_1d_labels(
        self,
        *,
        time_axis_ns: np.ndarray | None = None,
        depth_axis_m: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        curve = self.curve_samples()
        semantics = self.trace_semantics()
        time_gt = np.full(self.trace_count, np.nan, dtype=np.float32)
        depth_gt = np.full(self.trace_count, np.nan, dtype=np.float32)
        finite = np.isfinite(curve)
        if finite.any() and time_axis_ns is not None and len(time_axis_ns):
            axis = np.asarray(time_axis_ns, dtype=np.float32)
            positions = np.arange(len(axis), dtype=np.float32)
            time_gt[finite] = np.interp(curve[finite], positions, axis).astype(np.float32)
        if finite.any() and depth_axis_m is not None and len(depth_axis_m):
            axis = np.asarray(depth_axis_m, dtype=np.float32)
            positions = np.arange(len(axis), dtype=np.float32)
            depth_gt[finite] = np.interp(curve[finite], positions, axis).astype(np.float32)
        return {
            "curve_gt": curve,
            "time_gt_ns": time_gt,
            "depth_gt": depth_gt,
            **semantics,
        }

    def iter_soft_mask_chunks(self, *, chunk_traces: int = 512) -> Iterable[tuple[int, int, np.ndarray, np.ndarray]]:
        labels = self.build_1d_labels()
        curve = labels["curve_gt"]
        valid = labels["valid_trace_mask"]
        ignore_trace = labels["ignore_trace_mask"]
        sigma = max(float(self.soft_sigma_samples), 0.25)
        samples = np.arange(self.sample_count, dtype=np.float32)[:, None]
        for start in range(0, self.trace_count, max(int(chunk_traces), 1)):
            end = min(self.trace_count, start + max(int(chunk_traces), 1))
            positions = curve[start:end][None, :]
            soft = np.exp(-0.5 * ((samples - positions) / sigma) ** 2).astype(np.float32)
            soft[:, ~valid[start:end]] = 0.0
            ignore_mask = np.broadcast_to(ignore_trace[start:end][None, :], soft.shape).copy()
            yield start, end, soft, ignore_mask


__all__ = [
    "BASAL_INTERFACE_SCHEMA",
    "BASAL_LABEL_SCHEMA",
    "SEGMENT_KINDS",
    "InterfaceKeyPoint",
    "InterfaceSegment",
    "BasalInterfaceAnnotation",
    "VISIBILITY_UNKNOWN",
    "VISIBILITY_CLEAR",
    "VISIBILITY_WEAK",
    "VISIBILITY_NO_INTERFACE",
]
