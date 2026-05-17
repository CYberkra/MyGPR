#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual constant velocity model for migration and depth conversion."""

from __future__ import annotations

from typing import Any

import numpy as np


C0_M_PER_NS = 0.299792458


def method_manual_velocity_model(
    data: np.ndarray,
    mode: str = "velocity",
    velocity_m_per_ns: float = 0.10,
    epsilon_r: float = 9.0,
    uncertainty_fraction: float = 0.10,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Attach a constant velocity model without altering amplitudes."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("输入数据为空")

    mode_key = str(mode or "velocity").strip().lower()
    if mode_key not in {"velocity", "dielectric"}:
        raise ValueError("速度模型 mode 必须是 velocity 或 dielectric")

    if mode_key == "dielectric":
        eps = _finite_float(epsilon_r)
        if eps is None or eps <= 1.0:
            raise ValueError("介电常数 epsilon_r 必须是有限值且大于 1")
        velocity = C0_M_PER_NS / np.sqrt(eps)
    else:
        velocity = _finite_float(velocity_m_per_ns)
        if velocity is None:
            raise ValueError("速度 velocity_m_per_ns 必须是有限值")
        if not (0.01 <= velocity <= C0_M_PER_NS):
            raise ValueError("速度 velocity_m_per_ns 必须在 0.01~0.299792458 m/ns 内")
        eps = float((C0_M_PER_NS / velocity) ** 2)

    uncertainty = _finite_float(uncertainty_fraction)
    if uncertainty is None or uncertainty < 0.0:
        raise ValueError("uncertainty_fraction 必须是非负有限值")

    model = {
        "type": "constant_velocity",
        "mode": mode_key,
        "velocity_m_per_ns": float(velocity),
        "epsilon_r": float(eps),
        "uncertainty_fraction": float(uncertainty),
        "source": "manual_velocity_model",
    }

    return np.array(arr, copy=True), {
        "method": "manual_velocity_model",
        "velocity_model": model,
        "header_info_updates": {
            "velocity_model": model,
            "velocity_m_per_ns": float(velocity),
            "epsilon_r": float(eps),
        },
    }


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None
