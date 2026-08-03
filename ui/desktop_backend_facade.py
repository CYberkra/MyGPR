#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DesktopBackendFacade — ui/ 唯一允许接触的 core/ 与跨层导入通道。

所有 ``ui/`` 模块必须通过本文件获取核心数据模型、GUI 渲染辅助、
方法注册表元数据、文件格式过滤器及 job 状态类型；禁止在 ui/ 内直接
``import core.*`` 或 ``import mygpr.domain.*`` / ``import mygpr.application.*``。

本文件是 **deepened** facade：对外只暴露 UI-facing DTO 和 wrapped functions，
raw core / domain 类型只在内部翻译后通过 DTO 返回。
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from ui.desktop_backend_dtos import (
    UiJobSnapshot,
    UiMethodEntry,
    UiPipelineDefinition,
    UiPipelineStep,
)
from mygpr.application.jobs.models import JobEventType, JobResultSummary  # noqa: E402 — re-exported for ui/controllers

# ---------------------------------------------------------------------------
# Backend imports (the only place in ui/ that touches core/mygpr directly)
# ---------------------------------------------------------------------------
from core.gpr_data_model import GPRDataSet
from core.gui_rendering import bundle_from_dataset, compute_levels
from core.gpr_format_registry import supported_file_dialog_filter
from core.method_registry_metadata import (
    METHOD_CATEGORY_LABELS,
    METHOD_METADATA,
    METHOD_TAGS,
    PREFERRED_METHOD_ORDER,
)
from core.methods_registry import PROCESSING_METHODS
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public wrapped functions
# ---------------------------------------------------------------------------

def compute_display_levels(
    matrix: Any,
    *,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> tuple[float, float]:
    """Compute nan/inf-safe percentile display levels for a matrix."""
    return compute_levels(matrix, p_low=p_low, p_high=p_high)


def build_preview_bundle(
    line_id: str,
    matrix: Any,
    *,
    distance_axis_m: Any = None,
    time_axis_ns: Any = None,
    title: str = "",
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> Any:
    """Build a :class:`PreviewBundle` from raw array data.

    Constructs a minimal ``GPRDataSet``-like object internally and forwards
    to ``core.gui_rendering.bundle_from_dataset``.  The UI never sees
    ``GPRDataSet``.
    """
    sample_count = int(getattr(matrix, "shape", (0,))[0])
    trace_count = int(getattr(matrix, "shape", (0, 1))[1])
    time_window_ns = 250.0
    dataset = GPRDataSet(
        line_id=line_id,
        matrix=matrix,
        distance_axis_m=distance_axis_m if distance_axis_m is not None else [],
        time_axis_ns=time_axis_ns if time_axis_ns is not None else [],
        depth_axis_m=[],
        sample_count=sample_count,
        trace_count=trace_count,
        time_window_ns=time_window_ns,
    )
    return bundle_from_dataset(dataset, title=title, p_low=p_low, p_high=p_high)


def file_dialog_filter() -> str:
    """Return the ``;;``-joined file filter string for GPR import dialogs."""
    return supported_file_dialog_filter()


def method_catalog() -> tuple[UiMethodEntry, ...]:
    """Return the processing method catalog as typed, immutable DTOs.

    Each entry is built by merging the raw registry descriptor with
    display metadata from ``core.method_registry_metadata``.
    """
    order = {method_id: index for index, method_id in enumerate(PREFERRED_METHOD_ORDER)}
    items: list[UiMethodEntry] = []
    for method_id, raw in PROCESSING_METHODS.items():
        meta = METHOD_METADATA.get(method_id, {})
        category = str(meta.get("category") or raw.get("category") or "experimental")
        tag = METHOD_TAGS.get(method_id)
        entry = UiMethodEntry(
            method_id=str(method_id),
            name=str(raw.get("name", method_id)),
            display_name=str(meta.get("display_name") or raw.get("name", method_id)),
            category=category,
            category_label=str(METHOD_CATEGORY_LABELS.get(category, category)),
            tags=(str(tag),) if tag else (),
            auto_tune_enabled=bool(raw.get("auto_tune_enabled", False)),
            parameter_schema=tuple(
                dict(item) if isinstance(item, Mapping) else {"name": str(key)}
                for key, item in (raw.get("parameter_schema") or {}).items()
            ),
            description=str(raw.get("description") or ""),
        )
        items.append(entry)
    items.sort(key=lambda entry: (order.get(entry.method_id, len(order)), entry.method_id))
    return tuple(items)


def pipeline_from_dicts(
    steps: list[dict[str, Any]],
    name: str = "",
) -> UiPipelineDefinition:
    """Convert a page-provided list of step dicts into an immutable pipeline DTO."""
    ui_steps = tuple(
        UiPipelineStep(
            method_id=str(step.get("method_id", "")),
            params=dict(step.get("params") or {}),
            enabled=bool(step.get("enabled", True)),
            label=str(step.get("label", "") or step.get("method_id", "")),
        )
        for step in steps
    )
    return UiPipelineDefinition(steps=ui_steps, name=str(name or "Processing pipeline"))


def pipeline_to_raw(ui_pipeline: UiPipelineDefinition) -> PipelineDefinition:
    """Convert a UI pipeline DTO back to the domain ``PipelineDefinition`` the backend expects."""
    raw_steps = tuple(
        PipelineStep(
            method_id=step.method_id,
            params=dict(step.params),
            enabled=step.enabled,
            label=step.label,
        )
        for step in ui_pipeline.steps
    )
    return PipelineDefinition(steps=raw_steps, name=ui_pipeline.name)


def job_snapshot_from_raw(raw: Any) -> UiJobSnapshot:
    """Adapt a raw backend ``JobSnapshot`` into a UI-safe DTO."""
    return UiJobSnapshot.from_raw(raw)


# ---------------------------------------------------------------------------
# Re-exports required by ui/controllers/* (no raw core/mygpr types in __all__)
# ---------------------------------------------------------------------------

__all__ = [
    # wrapped functions
    "compute_display_levels",
    "build_preview_bundle",
    "file_dialog_filter",
    "method_catalog",
    "pipeline_from_dicts",
    "pipeline_to_raw",
    "job_snapshot_from_raw",
    # DTOs (for isinstance checks and signal payloads)
    "UiMethodEntry",
    "UiPipelineStep",
    "UiPipelineDefinition",
    "UiJobSnapshot",
    # re-exported enums / lightweight types
    "JobEventType",
    "JobResultSummary",
]
