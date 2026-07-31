#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small deterministic normalization helpers for AutoTune scoring v2.

The helpers deliberately avoid model training and external state.  They keep
AutoTune scores bounded, auditable, and stable across GUI / CLI execution.
"""

from __future__ import annotations

from math import exp, log
from typing import Mapping


def clamp01(value: float) -> float:
    """Return *value* clipped to the closed interval [0, 1]."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, value))


def norm01(value: float, lo: float, hi: float) -> float:
    """Linearly normalize *value* from [lo, hi] to [0, 1]."""
    if hi <= lo:
        return 0.0
    return clamp01((float(value) - float(lo)) / (float(hi) - float(lo)))


def logistic01(value: float, *, slope: float = 1.0, center: float = 0.0) -> float:
    """Map an unbounded score to [0, 1] with a logistic curve."""
    x = max(-60.0, min(60.0, float(slope) * (float(value) - float(center))))
    return float(1.0 / (1.0 + exp(-x)))


def ratio_similarity(ratio: float, *, tolerance_ratio: float = 3.0) -> float:
    """Return 1 when ratio≈1 and decline symmetrically for attenuation/amplification."""
    ratio = max(1e-12, float(ratio))
    tolerance_ratio = max(1.01, float(tolerance_ratio))
    distance = abs(log(ratio)) / log(tolerance_ratio)
    return clamp01(1.0 - distance)


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Return positive weights normalized to sum to 1.

    Empty or all-zero inputs return an empty dictionary.  Negative weights are
    clipped to zero because AutoTune v2 uses penalties as explicit 0..1 terms
    rather than negative coefficients.
    """
    clean = {str(k): max(0.0, float(v)) for k, v in dict(weights or {}).items()}
    total = sum(clean.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in clean.items()}


def weighted_sum(terms: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Return a bounded weighted sum over normalized 0..1 terms."""
    normalized = normalize_weights(weights)
    if not normalized:
        return 0.0
    return clamp01(sum(normalized[key] * clamp01(terms.get(key, 0.0)) for key in normalized))
