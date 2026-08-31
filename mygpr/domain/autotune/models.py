#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure domain models for automatic GPR parameter tuning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrialScore:
    score: float
    metrics: dict[str, float]
    penalties: dict[str, float]
    reason: str


@dataclass
class AutoTuneContext:
    full_data: np.ndarray
    header_info: dict[str, Any]
    trace_metadata: dict[str, np.ndarray]
    roi_source: str
    roi_label: str
    roi_bounds: dict[str, int] | None
    roi_data: np.ndarray | None
    context_bounds: dict[str, int] | None
    context_data: np.ndarray
    features: dict[str, Any]
    search_mode: str


@dataclass
class OuterSelectionScore:
    score: float
    metrics: dict[str, float]
    reason: str


PROFILE_LABELS = {
    "conservative": "保守档",
    "balanced": "平衡档",
    "aggressive": "增强档",
}

INVALID_TRIAL_SCORE = -1.0e9
FAILURE_PENALTY = 999.0


__all__ = [
    "AutoTuneContext",
    "FAILURE_PENALTY",
    "INVALID_TRIAL_SCORE",
    "OuterSelectionScore",
    "PROFILE_LABELS",
    "TrialScore",
]
