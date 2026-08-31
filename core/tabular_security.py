#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neutralise spreadsheet formula injection in exported text fields."""
from __future__ import annotations

import math
import re
from typing import Any

_SIGNED_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def safe_tabular_value(value: Any) -> Any:
    """Return a value safe for CSV/XLSX while preserving genuine numerics."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if not isinstance(value, str):
        return value
    stripped = value.lstrip(" \t\r\n")
    if not stripped:
        return value
    if stripped[0] in "=+-@" and not _SIGNED_NUMBER_RE.fullmatch(stripped):
        return "'" + value
    return value


def safe_tabular_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: safe_tabular_value(value) for key, value in row.items()}


__all__ = ["safe_tabular_row", "safe_tabular_value"]
