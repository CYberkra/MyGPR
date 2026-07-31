#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapters around the historical processing registry and ndarray engine.

All imports from ``core`` and ``PythonModule`` are contained in infrastructure,
so application services can remain UI- and implementation-independent.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from core.methods_registry import PROCESSING_METHODS, get_auto_tune_stage, is_public_method
from core.processing_engine import (
    clone_header_info,
    clone_trace_metadata,
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.processing.ports import ProcessingCatalogPort, ProcessingExecutorPort
from mygpr.domain.processing.models import (
    ProcessingMethodDescriptor,
    ProcessingRequest,
    ProcessingResult,
    ResourceEstimate,
)


class LegacyProcessingCatalog(ProcessingCatalogPort):
    """Read method metadata from the current legacy registry."""

    def get(self, method_id: str) -> ProcessingMethodDescriptor | None:
        raw = PROCESSING_METHODS.get(str(method_id))
        if raw is None:
            return None
        capabilities = {"ndarray"}
        if raw.get("auto_tune_enabled"):
            capabilities.add("auto_tune")
        if str(raw.get("auto_tune_stage") or raw.get("auto_tune_family") or "") == "motion_comp":
            capabilities.add("trace_metadata")
        if str(method_id) in {"kirchhoff_migration", "stolt_migration", "fk_filter"}:
            capabilities.add("global_transform")
        raw_params = raw.get("parameter_schema") or raw.get("params") or {}
        if isinstance(raw_params, dict):
            parameter_schema = dict(raw_params)
        elif isinstance(raw_params, list):
            parameter_schema = {
                str(item.get("name", index)): dict(item)
                for index, item in enumerate(raw_params)
                if isinstance(item, dict)
            }
        else:
            parameter_schema = {}
        return ProcessingMethodDescriptor(
            method_id=str(method_id),
            name=str(raw.get("display_name") or raw.get("name") or method_id),
            category=str(raw.get("category", "experimental")),
            auto_tune_enabled=bool(raw.get("auto_tune_enabled", False)),
            auto_tune_family=str(raw.get("auto_tune_family") or ""),
            auto_tune_stage=get_auto_tune_stage(str(method_id)),
            visibility=str(raw.get("visibility", "public")),
            parameter_schema=parameter_schema,
            capabilities=frozenset(capabilities),
            implementation_version=str(raw.get("implementation_version", "legacy-core")),
        )

    def list(self, *, public_only: bool = False) -> Sequence[ProcessingMethodDescriptor]:
        descriptors: list[ProcessingMethodDescriptor] = []
        for method_id in PROCESSING_METHODS:
            if public_only and not is_public_method(method_id):
                continue
            descriptor = self.get(method_id)
            if descriptor is not None:
                descriptors.append(descriptor)
        return tuple(descriptors)

    def auto_tune_stage(self, method_id: str) -> str:
        return get_auto_tune_stage(str(method_id))

    def raw_metadata(self, method_id: str) -> dict[str, Any]:
        return dict(PROCESSING_METHODS.get(str(method_id), {}) or {})


class LegacyProcessingExecutor(ProcessingExecutorPort):
    """Execute methods through the current ndarray processing engine."""

    _HIGH_MEMORY_METHODS = {
        "svd_bg",
        "svd_subspace",
        "wavelet_svd",
        "hankel_svd",
        "rpca_background",
        "fk_filter",
        "stolt_migration",
        "kirchhoff_migration",
    }
    _CHUNKABLE_METHODS = {
        "dewow",
        "agcGain",
        "compensatingGain",
        "sec_gain",
        "frequency_filter_1d",
        "running_average_2D",
        "trace_median_filter",
        "trace_savgol_filter",
    }

    def execute(
        self,
        request: ProcessingRequest,
        context: ExecutionContext | None = None,
    ) -> ProcessingResult:
        execution_context = context or ExecutionContext.null()
        execution_context.raise_if_cancelled()
        prepared = prepare_runtime_params(
            request.method_id,
            request.params,
            clone_header_info(request.header_info),
            clone_trace_metadata(request.trace_metadata),
            request.data.shape,
        )
        output, metadata = run_processing_method(
            request.data,
            request.method_id,
            prepared,
            cancel_checker=execution_context.is_cancelled,
        )
        execution_context.raise_if_cancelled()
        result_meta = dict(metadata or {})
        merged_header = merge_result_header_info(
            request.header_info,
            result_meta,
            data_shape=np.asarray(output).shape,
        )
        merged_trace_metadata = merge_result_trace_metadata(
            request.trace_metadata,
            result_meta,
        )
        warnings = list(result_meta.get("runtime_warnings", []) or [])
        for warning in warnings:
            if isinstance(warning, dict):
                execution_context.emit_warning(warning)
        return ProcessingResult(
            data=np.asarray(output),
            method_id=request.method_id,
            params=dict(request.params),
            metadata=result_meta,
            header_info=merged_header,
            trace_metadata=merged_trace_metadata,
            runtime_warnings=warnings,
        )

    def estimate(self, request: ProcessingRequest) -> ResourceEstimate:
        array = np.asarray(request.data)
        base_bytes = int(array.nbytes)
        method_id = request.method_id
        if method_id in self._HIGH_MEMORY_METHODS:
            multiplier = 8 if method_id in {"rpca_background", "kirchhoff_migration"} else 5
            relative_cost = "high"
        elif method_id in {"wavelet_2d", "median_background_2D", "motion_compensation_v2"}:
            multiplier = 4
            relative_cost = "medium"
        else:
            multiplier = 3
            relative_cost = "low"
        temporary = base_bytes * (2 if method_id in self._HIGH_MEMORY_METHODS else 1)
        notes: list[str] = []
        if method_id in self._HIGH_MEMORY_METHODS:
            notes.append("legacy implementation may materialize the full B-scan in memory")
        if method_id not in self._CHUNKABLE_METHODS:
            notes.append("chunked execution is not yet guaranteed")
        return ResourceEstimate(
            memory_bytes=base_bytes * multiplier,
            temporary_disk_bytes=temporary,
            relative_cost=relative_cost,
            supports_cancellation=True,
            supports_chunking=method_id in self._CHUNKABLE_METHODS,
            notes=tuple(notes),
        )
