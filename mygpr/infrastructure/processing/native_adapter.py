"""Native processing catalog/executor (converged single implementation).

自任务 F 候选 2 收敛起，native 目录/执行器是唯一实现：
``NATIVE_ALGORITHMS`` 覆盖全部历史方法（36/36），展示元数据经
:mod:`metadata_bridge` 取自 core 单一事实来源；旧 Legacy/Composite 适配器已拆除。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.processing.ports import ProcessingCatalogPort, ProcessingExecutorPort
from mygpr.domain.processing.models import (
    ProcessingMethodDescriptor,
    ProcessingRequest,
    ProcessingResult,
    ResourceEstimate,
)
from mygpr.infrastructure.processing.algorithms.methods import NATIVE_ALGORITHMS
from mygpr.infrastructure.processing.metadata_bridge import legacy_overlay


class NativeProcessingCatalog(ProcessingCatalogPort):
    """Catalog for methods migrated from the historical engine.

    ``NATIVE_ALGORITHMS`` 覆盖全部历史方法（36/36）。展示元数据（中文
    display_name、category、visibility、auto_tune_*）经
    :mod:`metadata_bridge` 取自 core 单一事实来源，与旧 Composite 目录输出等价
    （由 ``tests/test_native_convergence_baseline.py`` 的 descriptor 基线断言）。
    """

    def get(self, method_id: str) -> ProcessingMethodDescriptor | None:
        algorithm = NATIVE_ALGORITHMS.get(str(method_id))
        if algorithm is None:
            return None
        overlay = legacy_overlay(str(method_id))
        capabilities = {"ndarray", "native", "cancellable"}
        if algorithm.supports_chunking:
            capabilities.update({"chunked", f"block-axis-{algorithm.block_axis}"})
        elif algorithm.block_axis == "global":
            capabilities.update({"global_transform", "file_backed_staging"})
        elif algorithm.block_axis == "loaded_global":
            capabilities.update({"global_transform", "loaded_global"})
        if algorithm.auto_tune_family:
            capabilities.add("auto_tune")
        capabilities.update(overlay["legacy_capabilities"])
        auto_tune_family = overlay["auto_tune_family"] or algorithm.auto_tune_family
        auto_tune_stage = overlay["auto_tune_stage"] or algorithm.auto_tune_stage or auto_tune_family
        return ProcessingMethodDescriptor(
            method_id=algorithm.method_id,
            name=overlay["name"] or algorithm.name,
            category=overlay["category"] or algorithm.category,
            auto_tune_enabled=bool(auto_tune_family),
            auto_tune_family=auto_tune_family,
            auto_tune_stage=auto_tune_stage,
            visibility=overlay["visibility"],
            parameter_schema=dict(algorithm.parameter_schema or {}),
            capabilities=frozenset(capabilities),
            implementation_version=algorithm.implementation_version,
        )

    def list(self, *, public_only: bool = False) -> Sequence[ProcessingMethodDescriptor]:
        descriptors: list[ProcessingMethodDescriptor] = []
        for method_id in NATIVE_ALGORITHMS:
            if public_only and legacy_overlay(str(method_id))["visibility"] != "public":
                continue
            descriptor = self.get(method_id)
            if descriptor is not None:
                descriptors.append(descriptor)
        return tuple(descriptors)

    def auto_tune_stage(self, method_id: str) -> str:
        descriptor = self.get(method_id)
        return descriptor.auto_tune_stage if descriptor else ""

    def raw_metadata(self, method_id: str) -> dict[str, Any]:
        descriptor = self.get(method_id)
        if descriptor is None:
            return {}
        overlay = legacy_overlay(str(method_id))
        return {
            "method_id": descriptor.method_id,
            "name": descriptor.name,
            "display_name": overlay["name"],
            "category": descriptor.category,
            "maturity": overlay.get("maturity") or "experimental",
            "auto_tune_enabled": descriptor.auto_tune_enabled,
            "auto_tune_family": descriptor.auto_tune_family,
            "auto_tune_stage": descriptor.auto_tune_stage,
            "visibility": descriptor.visibility,
            "parameter_schema": dict(descriptor.parameter_schema),
            "implementation_version": descriptor.implementation_version,
        }


class NativeProcessingExecutor(ProcessingExecutorPort):
    """Execute migrated methods without importing ``core.processing_engine``."""

    def supports(self, method_id: str) -> bool:
        return str(method_id) in NATIVE_ALGORITHMS

    def execute(
        self,
        request: ProcessingRequest,
        context: ExecutionContext | None = None,
    ) -> ProcessingResult:
        algorithm = NATIVE_ALGORITHMS.get(request.method_id)
        if algorithm is None:
            raise KeyError(f"native processing method not found: {request.method_id}")
        execution_context = context or ExecutionContext.null()
        execution_context.raise_if_cancelled()
        params = prepare_native_params(request)
        params["_execution_context"] = execution_context
        params.setdefault("cancel_checker", execution_context.is_cancelled)
        output, metadata = algorithm.function(request.data, params)
        execution_context.raise_if_cancelled()
        metadata = {**metadata, "implementation_version": algorithm.implementation_version}
        warnings = [dict(item) for item in metadata.get("runtime_warnings", []) if isinstance(item, dict)]
        for item in warnings:
            execution_context.emit_warning(item)
        header = clone_mapping(request.header_info)
        header_updates = metadata.get("header_info_updates")
        if isinstance(header_updates, dict):
            header.update(clone_mapping(header_updates))
        header.update(a_scan_length=int(output.shape[0]), num_traces=int(output.shape[1]))
        trace_out = metadata.get("trace_metadata_out")
        if isinstance(trace_out, dict):
            trace_metadata = clone_trace_metadata(trace_out)
        else:
            trace_metadata = clone_trace_metadata(request.trace_metadata)
            trace_updates = metadata.get("trace_metadata_updates")
            if isinstance(trace_updates, dict):
                trace_metadata.update(clone_trace_metadata(trace_updates))
        return ProcessingResult(
            data=output,
            method_id=request.method_id,
            params=dict(request.params),
            metadata=metadata,
            header_info=header,
            trace_metadata=trace_metadata,
            runtime_warnings=warnings,
        )

    def estimate(self, request: ProcessingRequest) -> ResourceEstimate:
        algorithm = NATIVE_ALGORITHMS.get(request.method_id)
        if algorithm is None:
            raise KeyError(f"native processing method not found: {request.method_id}")
        if algorithm.resource_estimator is not None:
            return algorithm.resource_estimator(
                tuple(int(value) for value in request.data.shape),
                request.data.dtype,
                dict(request.params),
                dict(request.header_info),
            )
        base = int(np.asarray(request.data).nbytes)
        multiplier = float(algorithm.memory_multiplier)
        return ResourceEstimate(
            memory_bytes=int(base * multiplier),
            temporary_disk_bytes=int(base * float(algorithm.temporary_multiplier)),
            relative_cost=algorithm.relative_cost,
            supports_cancellation=True,
            supports_chunking=algorithm.supports_file_backed,
            notes=(
                f"native implementation {algorithm.implementation_version}",
                f"execution axis: {algorithm.block_axis}",
                "global methods use file-backed staging but may allocate spectral/factor workspaces"
                if algorithm.block_axis == "global" else "bounded block execution",
            ),
        )


def prepare_native_params(request: ProcessingRequest) -> dict[str, Any]:
    params = dict(request.params or {})
    params.setdefault("_header_info", clone_mapping(request.header_info))
    params.setdefault("_trace_metadata", clone_trace_metadata(request.trace_metadata))
    samples = max(1, int(request.data.shape[0]))
    total_time_ns = float(request.header_info.get("total_time_ns") or request.header_info.get("time_window_ns") or 0.0)
    if total_time_ns > 0.0:
        step_s = total_time_ns * 1.0e-9 / samples
        params.setdefault("time_step_s", step_s)
        params.setdefault("time_window_ns", total_time_ns)
        if request.method_id == "frequency_filter_1d":
            params.setdefault("sample_rate_hz", 1.0 / step_s)
    return params


def clone_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    return {
        str(key): np.array(item, copy=True) if isinstance(item, np.ndarray) else item
        for key, item in (value or {}).items()
    }


def clone_trace_metadata(value: dict[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    return {str(key): np.array(item, copy=True) for key, item in (value or {}).items()}


__all__ = [
    "NativeProcessingCatalog",
    "NativeProcessingExecutor",
]
