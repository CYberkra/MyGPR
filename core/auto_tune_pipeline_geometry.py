#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROI geometry helpers shared by pipeline orchestration and evaluation."""
from __future__ import annotations

from typing import Any

from core.scalar_utils import to_int

def _clamp_bounds(shape: tuple[int, int], bounds: dict[str, Any]) -> dict[str, int]:
    samples, traces = int(shape[0]), int(shape[1])
    t0 = max(
        0,
        min(to_int(bounds.get("time_start_idx"), default=0), max(samples - 1, 0)),
    )
    t1 = max(
        t0 + 1,
        min(to_int(bounds.get("time_end_idx"), default=samples), samples),
    )
    d0 = max(
        0,
        min(to_int(bounds.get("dist_start_idx"), default=0), max(traces - 1, 0)),
    )
    d1 = max(
        d0 + 1,
        min(to_int(bounds.get("dist_end_idx"), default=traces), traces),
    )
    return {
        "time_start_idx": int(t0),
        "time_end_idx": int(t1),
        "dist_start_idx": int(d0),
        "dist_end_idx": int(d1),
    }

__all__ = ["_clamp_bounds"]
