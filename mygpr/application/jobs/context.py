#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution context shared by processing, AutoTune and future backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from mygpr.application.jobs.cancellation import CancellationToken

ProgressCallback = Callable[[int, int, str], None]
WarningCallback = Callable[[dict[str, Any]], None]
ArtifactCallback = Callable[[dict[str, Any]], None]


def _noop_progress(completed: int, total: int, message: str) -> None:
    del completed, total, message


def _noop_payload(payload: dict[str, Any]) -> None:
    del payload


@dataclass(slots=True)
class ExecutionContext:
    """Cooperative runtime services passed across backend boundaries."""

    cancellation_token: CancellationToken = field(default_factory=CancellationToken)
    progress_callback: ProgressCallback = _noop_progress
    warning_callback: WarningCallback = _noop_payload
    artifact_callback: ArtifactCallback = _noop_payload
    metadata: dict[str, Any] = field(default_factory=dict)
    _progress_offset: float = 0.0
    _progress_span: float = 1.0

    @classmethod
    def null(cls) -> "ExecutionContext":
        return cls()

    def raise_if_cancelled(self) -> None:
        self.cancellation_token.raise_if_cancelled()

    def is_cancelled(self) -> bool:
        return self.cancellation_token.is_cancelled

    def report_progress(self, completed: int, total: int, message: str = "") -> None:
        total_value = max(1, int(total))
        local_fraction = min(1.0, max(0.0, float(completed) / total_value))
        global_fraction = self._progress_offset + local_fraction * self._progress_span
        scaled_total = 10_000
        scaled_completed = int(round(min(1.0, max(0.0, global_fraction)) * scaled_total))
        self.progress_callback(scaled_completed, scaled_total, str(message))

    def emit_warning(self, warning: dict[str, Any]) -> None:
        self.warning_callback(dict(warning))

    def emit_artifact(self, artifact: dict[str, Any]) -> None:
        self.artifact_callback(dict(artifact))

    def child(self, zero_based_index: int, parent_total: int) -> "ExecutionContext":
        parent_total_value = max(1, int(parent_total))
        index = min(parent_total_value - 1, max(0, int(zero_based_index)))
        child_span = self._progress_span / parent_total_value
        return ExecutionContext(
            cancellation_token=self.cancellation_token,
            progress_callback=self.progress_callback,
            warning_callback=self.warning_callback,
            artifact_callback=self.artifact_callback,
            metadata=dict(self.metadata),
            _progress_offset=self._progress_offset + index * child_span,
            _progress_span=child_span,
        )
