#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive processing-workbench contracts.

These types model a mutable user session without exposing storage handles,
Qt objects, or concrete processing implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class WorkbenchStep:
    step_id: str
    method_id: str
    label: str
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    output_shape: tuple[int, int] = (0, 0)
    output_dtype: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.step_id).strip():
            raise ValueError("step_id must not be empty")
        if not str(self.method_id).strip():
            raise ValueError("method_id must not be empty")
        object.__setattr__(self, "step_id", str(self.step_id))
        object.__setattr__(self, "method_id", str(self.method_id))
        object.__setattr__(self, "label", str(self.label or self.method_id))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params or {})))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))
        shape = tuple(int(value) for value in self.output_shape)
        if shape != (0, 0) and (len(shape) != 2 or min(shape) <= 0):
            raise ValueError(f"output_shape must be empty or non-empty 2D, got {shape!r}")
        object.__setattr__(self, "output_shape", shape)
        object.__setattr__(self, "output_dtype", str(self.output_dtype))


@dataclass(frozen=True, slots=True)
class WorkbenchSessionSnapshot:
    session_id: str
    project_id: str
    line_id: str
    input_artifact_id: str = ""
    branch_id: str = ""
    name: str = ""
    steps: Sequence[WorkbenchStep] = ()
    current_shape: tuple[int, int] = (0, 0)
    current_dtype: str = ""
    can_undo: bool = False
    can_redo: bool = False
    dirty: bool = False
    revision: int = 0

    def __post_init__(self) -> None:
        for field_name in ("session_id", "project_id", "line_id"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty")
        object.__setattr__(self, "steps", tuple(self.steps or ()))
        shape = tuple(int(value) for value in self.current_shape)
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"current_shape must be non-empty 2D, got {shape!r}")
        object.__setattr__(self, "current_shape", shape)
        object.__setattr__(self, "revision", max(0, int(self.revision)))


@dataclass(slots=True)
class WorkbenchPreview:
    session_id: str
    method_id: str
    params: dict[str, Any]
    data: np.ndarray
    sample_indices: np.ndarray
    trace_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.data)
        samples = np.asarray(self.sample_indices)
        traces = np.asarray(self.trace_indices)
        if matrix.ndim != 2 or matrix.shape != (samples.size, traces.size):
            raise ValueError("preview axes do not match 2D data")
        self.data = matrix
        self.sample_indices = samples
        self.trace_indices = traces
        self.params = dict(self.params or {})
        self.metadata = dict(self.metadata or {})


@dataclass(frozen=True, slots=True)
class ProcessingTemplate:
    template_id: str
    name: str
    description: str = ""
    steps: Sequence[WorkbenchStep] = ()
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not str(self.template_id).strip():
            raise ValueError("template_id must not be empty")
        if not str(self.name).strip():
            raise ValueError("template name must not be empty")
        object.__setattr__(self, "template_id", str(self.template_id))
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "description", str(self.description or ""))
        object.__setattr__(self, "steps", tuple(self.steps or ()))


@dataclass(slots=True)
class SignalAnalysis:
    line_id: str
    trace_index: int
    sample_axis: np.ndarray
    amplitude: np.ndarray
    frequency_axis: np.ndarray
    magnitude: np.ndarray
    statistics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trace_index = int(self.trace_index)
        self.sample_axis = np.asarray(self.sample_axis, dtype=np.float64)
        self.amplitude = np.asarray(self.amplitude, dtype=np.float64)
        self.frequency_axis = np.asarray(self.frequency_axis, dtype=np.float64)
        self.magnitude = np.asarray(self.magnitude, dtype=np.float64)
        if self.sample_axis.shape != self.amplitude.shape:
            raise ValueError("sample axis and amplitude shape mismatch")
        if self.frequency_axis.shape != self.magnitude.shape:
            raise ValueError("frequency axis and magnitude shape mismatch")
        self.statistics = {str(key): float(value) for key, value in self.statistics.items()}


@dataclass(slots=True)
class DatasetComparison:
    line_id: str
    left_label: str
    right_label: str
    left_data: np.ndarray
    right_data: np.ndarray
    difference: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        left = np.asarray(self.left_data)
        right = np.asarray(self.right_data)
        diff = np.asarray(self.difference)
        if left.ndim != 2 or right.shape != left.shape or diff.shape != left.shape:
            raise ValueError("comparison datasets must share one 2D shape")
        self.left_data = left
        self.right_data = right
        self.difference = diff
        self.metrics = {str(key): float(value) for key, value in self.metrics.items()}


@dataclass(frozen=True, slots=True)
class ProcessingStepDiagnostic:
    step_index: int
    step_id: str
    method_id: str
    label: str
    enabled: bool
    status: str
    score: float
    metrics: Mapping[str, float] = field(default_factory=dict)
    warnings: Sequence[str] = ()
    output_shape: tuple[int, int] = (0, 0)
    output_dtype: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_index", max(0, int(self.step_index)))
        object.__setattr__(self, "step_id", str(self.step_id))
        object.__setattr__(self, "method_id", str(self.method_id))
        object.__setattr__(self, "label", str(self.label or self.method_id))
        object.__setattr__(self, "status", str(self.status or "unknown"))
        object.__setattr__(self, "score", float(np.clip(float(self.score), 0.0, 100.0)))
        object.__setattr__(self, "metrics", MappingProxyType({str(k): float(v) for k, v in dict(self.metrics or {}).items()}))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings or ()))
        shape = tuple(int(value) for value in self.output_shape)
        if shape != (0, 0) and (len(shape) != 2 or min(shape) <= 0):
            raise ValueError(f"output_shape must be empty or non-empty 2D, got {shape!r}")
        object.__setattr__(self, "output_shape", shape)
        object.__setattr__(self, "output_dtype", str(self.output_dtype))


@dataclass(frozen=True, slots=True)
class ProcessingEvidencePackage:
    package_id: str
    project_id: str
    line_id: str
    session_id: str
    path: str
    sha256: str
    created_at: str
    files: Sequence[str] = ()
    manifest: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("package_id", "project_id", "line_id", "session_id", "path", "sha256", "created_at"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty")
        object.__setattr__(self, "files", tuple(str(item) for item in self.files or ()))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest or {})))


__all__ = [
    "DatasetComparison",
    "ProcessingTemplate",
    "ProcessingEvidencePackage",
    "ProcessingStepDiagnostic",
    "SignalAnalysis",
    "WorkbenchPreview",
    "WorkbenchSessionSnapshot",
    "WorkbenchStep",
]
