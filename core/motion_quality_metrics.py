#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility exports for motion benchmark metrics.

The canonical implementations live in ``core.quality_metrics`` so existing
Pyright/LSP import resolution stays stable while motion-specific callers can
still use this focused module path.
"""

from __future__ import annotations

from core.quality_metrics import compute_motion_quality_metrics
from core.quality_metrics import detect_ridge_indices
from core.quality_metrics import footprint_rmse
from core.quality_metrics import path_rmse
from core.quality_metrics import periodic_banding_ratio
from core.quality_metrics import ridge_error_metrics
from core.quality_metrics import target_preservation_ratio
from core.quality_metrics import trace_spacing_std

__all__ = [
    "compute_motion_quality_metrics",
    "detect_ridge_indices",
    "footprint_rmse",
    "path_rmse",
    "periodic_banding_ratio",
    "ridge_error_metrics",
    "target_preservation_ratio",
    "trace_spacing_std",
]
