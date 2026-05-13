#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DC offset removal for trace-domain UAV-GPR preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.runtime_warnings import build_runtime_warning


def method_dc_shift(
    data: np.ndarray,
    estimator: str = "mean",
    scope: str = "per_trace",
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove DC bias from a B-scan.

    The default subtracts each trace's time-axis mean, which is the usual
    trace-domain DC correction before dewow and band control.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("输入数据为空")

    warnings: list[dict[str, Any]] = []
    if not np.isfinite(arr).all():
        finite = np.isfinite(arr)
        fill_value = float(np.nanmean(arr[finite])) if finite.any() else 0.0
        arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
        warnings.append(
            build_runtime_warning(
                "data_sanitized",
                "DC 去偏输入包含 NaN/Inf，已使用均值填充后处理。",
                method_id="dc_shift",
                fill_value=fill_value,
                stage="input",
            )
        )

    estimator_key = str(estimator or "mean").strip().lower()
    if estimator_key not in {"mean", "median"}:
        raise ValueError("dc_shift estimator 必须是 mean 或 median")

    scope_key = str(scope or "per_trace").strip().lower()
    if scope_key not in {"per_trace", "global"}:
        raise ValueError("dc_shift scope 必须是 per_trace 或 global")

    if scope_key == "global":
        offset = (
            float(np.median(arr))
            if estimator_key == "median"
            else float(np.mean(arr))
        )
    else:
        offset = (
            np.median(arr, axis=0)
            if estimator_key == "median"
            else np.mean(arr, axis=0)
        )

    result = arr - offset
    return result.astype(np.float32, copy=False), {
        "method": "dc_shift",
        "estimator": estimator_key,
        "scope": scope_key,
        "offset": np.asarray(offset, dtype=np.float32),
        "runtime_warnings": warnings,
    }
