#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain errors for grid analysis (grouping / gridding)."""
from __future__ import annotations

from mygpr.domain.common.errors import MyGPRError


class GridAnalysisError(MyGPRError):
    """Raised when track grouping or attribute gridding cannot proceed."""

    error_code = "MYGPR_GRID_ANALYSIS_ERROR"
    category = "grid_analysis"
    default_hint = "确认测线轨迹已投影且坐标有效，网格参数（cell_size）为正数。"


__all__ = ["GridAnalysisError"]
