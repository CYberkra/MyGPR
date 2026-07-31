#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable, UI-independent processing contracts.

The domain types intentionally avoid Qt, SQLite, HDF5 and concrete algorithm
implementations.  They are safe to use from a desktop UI, a CLI, tests or a
future service process.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    copied: dict[str, Any] = {}
    for key, item in value.items():
        copied[str(key)] = np.array(item, copy=True) if isinstance(item, np.ndarray) else item
    return copied


@dataclass(frozen=True, slots=True)
class ProcessingMethodDescriptor:
    """Public metadata describing one processing method."""

    method_id: str
    name: str
    category: str = "experimental"
    auto_tune_enabled: bool = False
    auto_tune_family: str = ""
    auto_tune_stage: str = ""
    visibility: str = "public"
    parameter_schema: Mapping[str, Any] = field(default_factory=dict)
    capabilities: frozenset[str] = field(default_factory=frozenset)
    implementation_version: str = "legacy"

    def __post_init__(self) -> None:
        method_id = str(self.method_id).strip()
        if not method_id:
            raise ValueError("method_id must not be empty")
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "name", str(self.name or method_id))
        object.__setattr__(self, "category", str(self.category or "experimental"))
        object.__setattr__(self, "visibility", str(self.visibility or "public"))
        object.__setattr__(
            self,
            "parameter_schema",
            MappingProxyType(dict(self.parameter_schema or {})),
        )
        object.__setattr__(self, "capabilities", frozenset(self.capabilities or ()))


@dataclass(slots=True)
class ProcessingRequest:
    """Input to one processing method invocation."""

    data: np.ndarray
    method_id: str
    params: dict[str, Any] = field(default_factory=dict)
    header_info: dict[str, Any] = field(default_factory=dict)
    trace_metadata: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.data)
        if array.ndim != 2 or array.size == 0:
            raise ValueError("processing data must be a non-empty 2D array")
        method_id = str(self.method_id).strip()
        if not method_id:
            raise ValueError("method_id must not be empty")
        self.data = array
        self.method_id = method_id
        self.params = dict(self.params or {})
        self.header_info = _copy_mapping(self.header_info)
        self.trace_metadata = {
            str(key): np.array(value, copy=True)
            for key, value in (self.trace_metadata or {}).items()
        }


@dataclass(slots=True)
class ProcessingResult:
    """Output of one processing method invocation."""

    data: np.ndarray
    method_id: str
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    header_info: dict[str, Any] = field(default_factory=dict)
    trace_metadata: dict[str, np.ndarray] = field(default_factory=dict)
    runtime_warnings: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        array = np.asarray(self.data)
        if array.ndim != 2 or array.size == 0:
            raise ValueError("processing result data must be a non-empty 2D array")
        self.data = array
        self.method_id = str(self.method_id)
        self.params = dict(self.params or {})
        self.metadata = dict(self.metadata or {})
        self.header_info = _copy_mapping(self.header_info)
        self.trace_metadata = {
            str(key): np.array(value, copy=True)
            for key, value in (self.trace_metadata or {}).items()
        }
        self.runtime_warnings = [
            dict(item) for item in (self.runtime_warnings or []) if isinstance(item, dict)
        ]


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Best-effort resource estimate produced before method execution."""

    memory_bytes: int = 0
    temporary_disk_bytes: int = 0
    relative_cost: str = "unknown"
    supports_cancellation: bool = True
    supports_chunking: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_bytes", max(0, int(self.memory_bytes)))
        object.__setattr__(
            self, "temporary_disk_bytes", max(0, int(self.temporary_disk_bytes))
        )
        object.__setattr__(self, "relative_cost", str(self.relative_cost or "unknown"))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))


@dataclass(frozen=True, slots=True)
class PipelineStep:
    """One immutable processing-pipeline step."""

    method_id: str
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        method_id = str(self.method_id).strip()
        if not method_id:
            raise ValueError("pipeline step method_id must not be empty")
        object.__setattr__(self, "method_id", method_id)
        object.__setattr__(self, "params", MappingProxyType(dict(self.params or {})))
        object.__setattr__(self, "label", str(self.label or method_id))


@dataclass(frozen=True, slots=True)
class PipelineDefinition:
    """Versioned sequence of processing steps."""

    steps: Sequence[PipelineStep]
    schema_version: str = "mygpr.processing_pipeline.v1"
    name: str = ""

    def __post_init__(self) -> None:
        normalized = tuple(
            step if isinstance(step, PipelineStep) else PipelineStep(**dict(step))
            for step in self.steps
        )
        if not normalized:
            raise ValueError("pipeline must contain at least one step")
        object.__setattr__(self, "steps", normalized)
        object.__setattr__(self, "name", str(self.name or "Processing pipeline"))


@dataclass(slots=True)
class PipelineExecutionResult:
    """Final data and lineage emitted by a pipeline execution."""

    data: np.ndarray
    header_info: dict[str, Any]
    trace_metadata: dict[str, np.ndarray]
    step_results: list[ProcessingResult]

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.data.ndim != 2 or self.data.size == 0:
            raise ValueError("pipeline result data must be a non-empty 2D array")
        self.header_info = _copy_mapping(self.header_info)
        self.trace_metadata = {
            str(key): np.array(value, copy=True)
            for key, value in (self.trace_metadata or {}).items()
        }
        self.step_results = list(self.step_results or [])


@dataclass(frozen=True, slots=True)
class ProcessingStepRecord:
    """Serializable lineage record for one pipeline step."""

    method_id: str
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    implementation_version: str = ""
    output_shape: tuple[int, int] = (0, 0)
    output_dtype: str = ""
    output_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", str(self.method_id).strip())
        object.__setattr__(self, "params", MappingProxyType(dict(self.params or {})))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))
        shape = tuple(int(value) for value in self.output_shape)
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"output_shape must be non-empty 2D, got {shape!r}")
        object.__setattr__(self, "output_shape", shape)
        object.__setattr__(self, "output_dtype", str(self.output_dtype))
        object.__setattr__(self, "output_sha256", str(self.output_sha256))


@dataclass(slots=True)
class BlockPipelineSummary:
    """Lineage emitted by a file-backed pipeline without exposing its matrix."""

    shape: tuple[int, int]
    dtype: str
    header_info: dict[str, Any]
    trace_metadata: dict[str, np.ndarray]
    step_records: list[ProcessingStepRecord]
    input_sha256: str = ""
    output_sha256: str = ""

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"block pipeline shape must be non-empty 2D, got {shape!r}")
        self.shape = shape
        self.dtype = str(self.dtype)
        self.header_info = _copy_mapping(self.header_info)
        self.trace_metadata = {
            str(key): np.array(value, copy=True)
            for key, value in (self.trace_metadata or {}).items()
        }
        self.step_records = list(self.step_records or [])
        self.input_sha256 = str(self.input_sha256)
        self.output_sha256 = str(self.output_sha256)
