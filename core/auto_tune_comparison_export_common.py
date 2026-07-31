#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serialization and naming helpers for auto-tune comparison evidence."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

def _safe_bundle_name(bundle_name: str | None) -> str:
    raw = str(bundle_name or datetime.now().strftime("auto_tune_comparison_%Y%m%d_%H%M%S"))
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", raw).strip("._-")
    return safe or "auto_tune_comparison"

def _first_value(values: Any) -> Any:
    if isinstance(values, (list, tuple)) and values:
        return values[0]
    return None

def _unique_values(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        safe = _json_safe(value)
        key = json.dumps(safe, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(safe)
    return out

def _same_params(lhs: Any, rhs: Any) -> bool:
    return _json_safe(lhs or {}) == _json_safe(rhs or {})

def _artifact_name(path: Any) -> str:
    if not path:
        return ""
    return Path(str(path)).name

def _csv_value(value: Any) -> str:
    safe = _json_safe(value)
    if isinstance(safe, (dict, list)):
        return json.dumps(safe, ensure_ascii=False, sort_keys=True)
    if safe is None:
        return ""
    return str(safe)

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, int):
        return int(value)
    return str(value)

__all__ = ['_safe_bundle_name', '_first_value', '_unique_values', '_same_params', '_artifact_name', '_csv_value', '_json_safe']
