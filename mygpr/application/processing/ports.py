#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ports consumed by processing and AutoTune application services."""
from __future__ import annotations

from typing import Any, Iterator, Protocol, Sequence, TypeVar

from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.processing.models import (
    ProcessingMethodDescriptor,
    ProcessingRequest,
    ProcessingResult,
    ResourceEstimate,
    BlockPipelineSummary,
    PipelineDefinition,
)


class ProcessingCatalogPort(Protocol):
    """Read-only processing-method catalogue."""

    def get(self, method_id: str) -> ProcessingMethodDescriptor | None: ...

    def list(self, *, public_only: bool = False) -> Sequence[ProcessingMethodDescriptor]: ...

    def auto_tune_stage(self, method_id: str) -> str: ...

    def raw_metadata(self, method_id: str) -> dict[str, Any]: ...


class ProcessingExecutorPort(Protocol):
    """Concrete algorithm execution boundary."""

    def execute(
        self,
        request: ProcessingRequest,
        context: ExecutionContext | None = None,
    ) -> ProcessingResult: ...

    def estimate(self, request: ProcessingRequest) -> ResourceEstimate: ...


class ProcessingResourcePolicyPort(Protocol):
    """Validate a resource estimate before allocation-heavy execution."""

    def validate(
        self,
        estimate: ResourceEstimate,
        *,
        context: ExecutionContext | None = None,
        operation: str = "processing",
    ) -> None: ...


class MatrixBlockSourcePort(Protocol):
    """Read-only matrix source that yields bounded row blocks."""

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def dtype(self) -> str: ...

    def iter_blocks(self, *, block_rows: int) -> Iterator[tuple[int, int, Any]]: ...


TConsumerResult = TypeVar("TConsumerResult")


class BlockPipelineConsumer(Protocol[TConsumerResult]):
    def __call__(
        self,
        matrix: Any,
        summary: BlockPipelineSummary,
    ) -> TConsumerResult: ...


class BlockPipelineExecutorPort(Protocol):
    """File-backed pipeline boundary for project-sized matrices."""

    def supports(self, pipeline: PipelineDefinition) -> bool: ...

    def estimate(
        self,
        shape: tuple[int, int],
        dtype: str,
        pipeline: PipelineDefinition,
    ) -> ResourceEstimate: ...

    def execute(
        self,
        source: MatrixBlockSourcePort,
        pipeline: PipelineDefinition,
        *,
        header_info: dict[str, Any],
        trace_metadata: dict[str, Any],
        consumer: BlockPipelineConsumer[Any],
        context: ExecutionContext | None = None,
    ) -> Any: ...


__all__ = [
    "BlockPipelineConsumer",
    "BlockPipelineExecutorPort",
    "MatrixBlockSourcePort",
    "ProcessingCatalogPort",
    "ProcessingExecutorPort",
    "ProcessingResourcePolicyPort",
]
