#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime request/result contracts for Workflow Studio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkflowRunRequest:
    """Canonical request passed from Workflow Studio into the app runtime."""

    methods: tuple[Any, ...]
    realtime: bool = False
    run_mode: str = ""

    @classmethod
    def from_signal_args(
        cls,
        methods: object,
        realtime: bool = False,
        run_mode: str = "",
    ) -> "WorkflowRunRequest":
        return cls(
            methods=tuple(list(methods or [])),
            realtime=bool(realtime),
            run_mode=str(run_mode or ""),
        )

    def as_signal_args(self) -> tuple[tuple[Any, ...], bool, str]:
        """Return legacy signal-compatible arguments."""
        return self.methods, self.realtime, self.run_mode


@dataclass(frozen=True)
class WorkflowNodeOutput:
    """One workflow node output cached for preview/evidence nodes."""

    node_id: str
    method_key: str
    method_name: str
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    elapsed_ms: float = 0.0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowRunResult:
    """Canonical result summary for Workflow Studio runtime updates."""

    outputs: tuple[WorkflowNodeOutput, ...] = ()
    realtime: bool = False
    run_mode: str = ""
    final_data: Any = None
    final_header_info: dict[str, Any] | None = None
    final_trace_metadata: Any = None

    @classmethod
    def from_worker_payload(
        cls,
        payload: dict[str, Any],
        *,
        realtime: bool = False,
        run_mode: str = "",
    ) -> "WorkflowRunResult":
        outputs: list[WorkflowNodeOutput] = []
        for item in payload.get("outputs", []) or []:
            if not isinstance(item, dict):
                continue
            warnings = item.get("warnings") or item.get("runtime_warnings") or ()
            if isinstance(warnings, str):
                warnings = (warnings,)
            outputs.append(
                WorkflowNodeOutput(
                    node_id=str(item.get("node_id") or ""),
                    method_key=str(item.get("method_key") or ""),
                    method_name=str(item.get("method_name") or item.get("method_key") or ""),
                    data=item.get("data"),
                    metadata=dict(item.get("metadata") or {}),
                    input_shape=tuple(item["input_shape"]) if item.get("input_shape") else None,
                    output_shape=tuple(item["output_shape"]) if item.get("output_shape") else None,
                    elapsed_ms=float(item.get("elapsed_ms") or 0.0),
                    warnings=tuple(str(warning) for warning in warnings),
                )
            )
        return cls(
            outputs=tuple(outputs),
            realtime=bool(realtime),
            run_mode=str(run_mode or ""),
            final_data=payload.get("final_data"),
            final_header_info=payload.get("final_header_info"),
            final_trace_metadata=payload.get("final_trace_metadata"),
        )
