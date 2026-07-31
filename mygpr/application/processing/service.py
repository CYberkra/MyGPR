#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI-independent processing use cases."""
from __future__ import annotations

from typing import Any

import numpy as np

from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.common.errors import ProcessingMethodError
from mygpr.application.processing.validation import validate_parameters
from mygpr.application.processing.ports import (
    BlockPipelineConsumer,
    BlockPipelineExecutorPort,
    MatrixBlockSourcePort,
    ProcessingCatalogPort,
    ProcessingExecutorPort,
    ProcessingResourcePolicyPort,
)
from mygpr.domain.processing.models import (
    PipelineDefinition,
    PipelineExecutionResult,
    ProcessingMethodDescriptor,
    ProcessingRequest,
    ProcessingResult,
    ResourceEstimate,
)


class ProcessingApplicationError(ProcessingMethodError):
    """Stable processing application failure exposed through Backend API v1."""


class ProcessingService:
    """Stable façade for single-method and pipeline execution."""

    def __init__(
        self,
        catalog: ProcessingCatalogPort,
        executor: ProcessingExecutorPort,
        block_executor: BlockPipelineExecutorPort | None = None,
        resource_policy: ProcessingResourcePolicyPort | None = None,
    ) -> None:
        self._catalog = catalog
        self._executor = executor
        self._block_executor = block_executor
        self._resource_policy = resource_policy


    def get_method(self, method_id: str) -> ProcessingMethodDescriptor:
        descriptor = self._catalog.get(method_id)
        if descriptor is None:
            raise ProcessingApplicationError(f"unknown processing method: {method_id}")
        return descriptor

    def list_methods(self, *, public_only: bool = True) -> tuple[ProcessingMethodDescriptor, ...]:
        return tuple(self._catalog.list(public_only=public_only))

    def validate_request(self, request: ProcessingRequest) -> ProcessingRequest:
        descriptor = self.get_method(request.method_id)
        request.params = validate_parameters(descriptor, request.params)
        return request

    def estimate(self, request: ProcessingRequest) -> ResourceEstimate:
        self.validate_request(request)
        return self._executor.estimate(request)

    def execute_method(
        self,
        request: ProcessingRequest,
        context: ExecutionContext | None = None,
    ) -> ProcessingResult:
        descriptor = self.get_method(request.method_id)
        request.params = validate_parameters(descriptor, request.params)
        execution_context = context or ExecutionContext.null()
        execution_context.raise_if_cancelled()
        estimate = self._executor.estimate(request)
        self.validate_estimate(estimate, context=execution_context, operation=descriptor.name)
        execution_context.report_progress(0, 1, f"开始处理: {descriptor.name}")
        result = self._executor.execute(request, execution_context)
        execution_context.raise_if_cancelled()
        execution_context.report_progress(1, 1, f"处理完成: {descriptor.name}")
        return result

    def validate_estimate(
        self,
        estimate: ResourceEstimate,
        *,
        context: ExecutionContext | None = None,
        operation: str = "processing",
    ) -> None:
        if self._resource_policy is not None:
            self._resource_policy.validate(estimate, context=context, operation=operation)

    def estimate_pipeline_shape(
        self,
        shape: tuple[int, int],
        dtype: str,
        pipeline: PipelineDefinition,
    ) -> ResourceEstimate:
        if self.supports_block_pipeline(pipeline):
            return self.estimate_block_pipeline(shape, dtype, pipeline)
        base = int(np.prod(shape, dtype=np.int64)) * max(1, np.dtype(dtype).itemsize)
        multiplier = 3
        notes: list[str] = ["pipeline requires loaded or global execution"]
        for step in pipeline.steps:
            if not step.enabled:
                continue
            descriptor = self.get_method(step.method_id)
            if "global_transform" in descriptor.capabilities:
                multiplier = max(multiplier, 8)
                notes.append(f"global transform: {step.method_id}")
            elif step.method_id in {"svd_bg", "svd_subspace", "wavelet_svd", "hankel_svd", "rpca_background"}:
                multiplier = max(multiplier, 6)
                notes.append(f"high-memory method: {step.method_id}")
        return ResourceEstimate(
            memory_bytes=base * multiplier,
            temporary_disk_bytes=base * 2,
            relative_cost="high" if multiplier >= 6 else "medium",
            supports_cancellation=True,
            supports_chunking=False,
            notes=tuple(dict.fromkeys(notes)),
        )

    def supports_block_pipeline(self, pipeline: PipelineDefinition) -> bool:
        return self._block_executor is not None and self._block_executor.supports(pipeline)

    def estimate_block_pipeline(
        self,
        shape: tuple[int, int],
        dtype: str,
        pipeline: PipelineDefinition,
    ) -> ResourceEstimate:
        if self._block_executor is None:
            return ResourceEstimate(
                relative_cost="unsupported",
                supports_chunking=False,
                notes=("block executor is not configured",),
            )
        return self._block_executor.estimate(shape, dtype, pipeline)

    def execute_pipeline_blocks(
        self,
        source: MatrixBlockSourcePort,
        pipeline: PipelineDefinition,
        *,
        header_info: dict[str, Any] | None = None,
        trace_metadata: dict[str, np.ndarray] | None = None,
        consumer: BlockPipelineConsumer[Any],
        context: ExecutionContext | None = None,
    ) -> Any:
        if self._block_executor is None or not self._block_executor.supports(pipeline):
            raise ProcessingApplicationError("pipeline does not support block execution")
        execution_context = context or ExecutionContext.null()
        estimate = self._block_executor.estimate(source.shape, source.dtype, pipeline)
        self.validate_estimate(estimate, context=execution_context, operation=pipeline.name)
        return self._block_executor.execute(
            source,
            pipeline,
            header_info=dict(header_info or {}),
            trace_metadata={
                key: np.array(value, copy=True)
                for key, value in (trace_metadata or {}).items()
            },
            consumer=consumer,
            context=execution_context,
        )

    def execute_pipeline(
        self,
        data: np.ndarray,
        pipeline: PipelineDefinition,
        *,
        header_info: dict[str, Any] | None = None,
        trace_metadata: dict[str, np.ndarray] | None = None,
        context: ExecutionContext | None = None,
    ) -> PipelineExecutionResult:
        execution_context = context or ExecutionContext.null()
        current_data = np.asarray(data)
        current_header = dict(header_info or {})
        current_trace_metadata = {
            key: np.array(value, copy=True)
            for key, value in (trace_metadata or {}).items()
        }
        step_results: list[ProcessingResult] = []
        enabled_steps = [step for step in pipeline.steps if step.enabled]
        if not enabled_steps:
            raise ProcessingApplicationError("pipeline contains no enabled step")

        total = len(enabled_steps)
        for index, step in enumerate(enabled_steps, start=1):
            execution_context.raise_if_cancelled()
            execution_context.report_progress(
                index - 1,
                total,
                f"流水线 {index}/{total}: {step.label}",
            )
            request = ProcessingRequest(
                data=current_data,
                method_id=step.method_id,
                params=dict(step.params),
                header_info=current_header,
                trace_metadata=current_trace_metadata,
            )
            request.params = validate_parameters(self.get_method(step.method_id), request.params)
            step_context = execution_context.child(index - 1, total)
            self.validate_estimate(
                self._executor.estimate(request),
                context=step_context,
                operation=step.label,
            )
            result = self._executor.execute(request, step_context)
            step_results.append(result)
            current_data = np.asarray(result.data)
            current_header = dict(result.header_info)
            current_trace_metadata = {
                key: np.array(value, copy=True)
                for key, value in result.trace_metadata.items()
            }

        execution_context.report_progress(total, total, "流水线处理完成")
        return PipelineExecutionResult(
            data=current_data,
            header_info=current_header,
            trace_metadata=current_trace_metadata,
            step_results=step_results,
        )


__all__ = ["ProcessingApplicationError", "ProcessingService"]
