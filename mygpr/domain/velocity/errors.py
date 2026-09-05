#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain errors for velocity analysis (hyperbola fitting)."""
from __future__ import annotations

from mygpr.domain.common.errors import MyGPRError


class VelocityAnalysisError(MyGPRError):
    """Raised when hyperbola velocity fitting cannot produce a physical result."""

    error_code = "MYGPR_VELOCITY_ANALYSIS_ERROR"
    category = "velocity_analysis"
    default_hint = "确认拾取点数量与坐标有效，且拾取走时确为双曲线形态。"


__all__ = ["VelocityAnalysisError"]
