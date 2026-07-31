#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data contracts for pipeline-level automatic tuning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.app_errors import MyGPRError

class AutoTunePipelineError(MyGPRError):
    """Raised when a pipeline-level auto-tune run cannot be executed."""

@dataclass
class PipelineCandidate:
    """One final branch produced by a pipeline-level auto-tune run."""

    name: str
    source: str
    pipeline: list[str]
    params_by_method: dict[str, dict[str, Any]]
    result: np.ndarray
    metadata: dict[str, Any]
    metrics: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    auto_tune_results: dict[str, dict[str, Any]] = field(default_factory=dict)

@dataclass
class PipelineStepRecord:
    """Per-step evidence for reports and research comparison."""

    index: int
    method_key: str
    method_name: str
    manual_params: dict[str, Any]
    auto_params: dict[str, Any]
    manual_before: np.ndarray
    manual_after: np.ndarray
    auto_before: np.ndarray
    auto_after: np.ndarray
    manual_metrics: dict[str, float]
    auto_metrics: dict[str, float]
    metric_delta: dict[str, float]
    auto_tune_result: dict[str, Any] | None
    manual_roi_before: dict[str, int]
    manual_roi_after: dict[str, int]
    auto_roi_before: dict[str, int]
    auto_roi_after: dict[str, int]
    warnings: dict[str, list[str]]
    risk_flags: list[str]
    recommendation: str
    reason: str
    rolled_back_to_manual: bool = False

@dataclass
class AutoTunePipelineRun:
    """Full pipeline-level auto-tune result."""

    input_shape: tuple[int, int]
    pipeline: list[str]
    baseline_profile_key: str | None
    manual_source: str
    roi_info: dict[str, Any]
    ground_truth_info: dict[str, Any]
    steps: list[PipelineStepRecord]
    manual: PipelineCandidate
    automatic: PipelineCandidate
    metric_delta: dict[str, float]
    overall_recommendation: str
    risk_flags: list[str]

@dataclass
class _BranchState:
    current: np.ndarray
    header_info: dict[str, Any]
    trace_metadata: dict[str, np.ndarray]
    params_by_method: dict[str, dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    auto_tune_results: dict[str, dict[str, Any]] = field(default_factory=dict)

__all__ = ["AutoTunePipelineError", "AutoTunePipelineRun", "PipelineCandidate", "PipelineStepRecord"]
