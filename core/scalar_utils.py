#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for accepting Python and NumPy scalar runtime parameters."""

from __future__ import annotations

from typing import Any

import numpy as np


def first_scalar(value: Any) -> Any:
    """Return the first scalar-like value from Python or NumPy inputs."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return value
    if arr.size == 0:
        return None
    return arr.reshape(-1)[0]


def to_float(value: Any, *, default: float) -> float:
    """Convert a runtime parameter to float, accepting NumPy scalar arrays."""
    scalar = first_scalar(value)
    if scalar is None:
        return float(default)
    if isinstance(scalar, str) and scalar.strip() == "":
        return float(default)
    try:
        return float(scalar)
    except (TypeError, ValueError):
        return float(default)


def to_optional_float(value: Any) -> float | None:
    """Convert to float while preserving explicit None as None."""
    if value is None:
        return None
    return to_float(value, default=0.0)


def to_float_or_none(value: Any) -> float | None:
    """Convert to float, returning None for empty or invalid values."""
    scalar = first_scalar(value)
    if scalar is None:
        return None
    if isinstance(scalar, str) and scalar.strip() == "":
        return None
    try:
        return float(scalar)
    except (TypeError, ValueError):
        return None


def to_int(value: Any, *, default: int) -> int:
    """Convert a runtime parameter to int via float for GUI numeric strings."""
    scalar = first_scalar(value)
    if scalar is None:
        return int(default)
    if isinstance(scalar, str) and scalar.strip() == "":
        return int(default)
    try:
        return int(float(scalar))
    except (TypeError, ValueError):
        return int(default)


def to_int_or_none(value: Any) -> int | None:
    """Convert to int via float, returning None for empty or invalid values."""
    parsed = to_float_or_none(value)
    if parsed is None:
        return None
    return int(round(parsed))


def first_two_floats(value: Any) -> tuple[float, float] | None:
    """Return the first two finite floats from a sequence-like value."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 2:
        return None
    low = float(arr[0])
    high = float(arr[1])
    if not (np.isfinite(low) and np.isfinite(high)):
        return None
    return low, high
