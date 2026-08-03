#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI-facing DTOs returned by ``ui.desktop_backend_facade``.

These types are the *only* backend/core types the UI layer is allowed to
see.  Every raw core or domain type is translated into one of these DTOs
before crossing the facade boundary.

The ``from_raw`` class methods are the translation seam: when the backend
evolves, only these adapters need to change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Method catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UiMethodEntry:
    """One row in the processing method catalog shown in the UI."""

    method_id: str
    name: str
    display_name: str
    category: str
    category_label: str
    tags: tuple[str, ...]
    auto_tune_enabled: bool
    parameter_schema: tuple[dict[str, Any], ...]
    description: str

    @classmethod
    def from_raw(
        cls,
        *,
        method_id: str,
        name: str,
        display_name: str,
        category: str,
        category_label: str,
        tag: str | None = None,
        auto_tune_enabled: bool = False,
        parameter_schema: Mapping[str, Any] | None = None,
        description: str = "",
    ) -> UiMethodEntry:
        schema_list = [
            dict(item) if isinstance(item, Mapping) else {"name": str(key)}
            for key, item in (parameter_schema or {}).items()
        ]
        tags: tuple[str, ...] = (str(tag),) if tag else ()
        return cls(
            method_id=str(method_id),
            name=str(name),
            display_name=str(display_name),
            category=str(category),
            category_label=str(category_label),
            tags=tags,
            auto_tune_enabled=bool(auto_tune_enabled),
            parameter_schema=tuple(schema_list),
            description=str(description),
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UiPipelineStep:
    """One immutable step in a processing pipeline."""

    method_id: str
    params: Mapping[str, Any]
    enabled: bool
    label: str

    @classmethod
    def from_raw(
        cls,
        *,
        method_id: str,
        params: Mapping[str, Any] | None = None,
        enabled: bool = True,
        label: str = "",
    ) -> UiPipelineStep:
        return cls(
            method_id=str(method_id),
            params=dict(params or {}),
            enabled=bool(enabled),
            label=str(label or method_id),
        )


@dataclass(frozen=True)
class UiPipelineDefinition:
    """Versioned sequence of processing steps as seen by the UI."""

    steps: tuple[UiPipelineStep, ...]
    name: str

    @classmethod
    def from_raw(
        cls,
        *,
        steps: Sequence[UiPipelineStep],
        name: str = "",
    ) -> UiPipelineDefinition:
        if not steps:
            raise ValueError("pipeline must contain at least one step")
        return cls(
            steps=tuple(steps),
            name=str(name or "Processing pipeline"),
        )


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UiJobSnapshot:
    """Lightweight job state consumed by the UI via JobBridge."""

    job_id: str
    title: str
    status: str
    completed: int
    total: int
    message: str
    progress: float | None
    is_terminal: bool
    error_code: str
    error_message: str

    @classmethod
    def from_raw(cls, raw: Any) -> UiJobSnapshot:
        status_value = str(getattr(raw, "status", None).value) if hasattr(getattr(raw, "status", None), "value") else str(getattr(raw, "status", ""))
        total = int(getattr(raw, "total", 0) or 0)
        completed = int(getattr(raw, "completed", 0) or 0)
        progress: float | None = None
        if total > 0:
            progress = min(1.0, max(0.0, completed / total))
        return cls(
            job_id=str(getattr(raw, "job_id", "")),
            title=str(getattr(raw, "title", "")),
            status=status_value,
            completed=completed,
            total=total,
            message=str(getattr(raw, "message", "")),
            progress=progress,
            is_terminal=bool(getattr(raw, "is_terminal", False)),
            error_code=str(getattr(raw, "error_code", "")),
            error_message=str(getattr(raw, "error_message", "")),
        )


__all__ = [
    "UiMethodEntry",
    "UiPipelineStep",
    "UiPipelineDefinition",
    "UiJobSnapshot",
]
