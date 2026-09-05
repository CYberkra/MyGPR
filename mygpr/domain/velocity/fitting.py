#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双曲线速度拟合（纯 numpy，无 Qt/core 依赖）。

模型：t(x) = (2/v)·sqrt((x - x0)² + z0²)。
平方线性化：t² = A·x² + B·x + C，一次最小二乘解出 A/B/C，无需迭代初值：
    v  = 2/sqrt(A)
    x0 = -B / (2A)
    z0² = C - B²/(4A)   （A > 0 保证 v² > 0；z0² < 0 视为非物理）
"""
from __future__ import annotations

import numpy as np

from mygpr.domain.velocity.errors import VelocityAnalysisError
from mygpr.domain.velocity.models import HyperbolaFit, VelocityPick

_MIN_PICKS = 3


def fit_hyperbola(picks: list[VelocityPick]) -> HyperbolaFit:
    """对 ≥3 个拾取点做双曲线拟合，返回物理参数化结果。

    Raises:
        VelocityAnalysisError: 点数不足、坐标非有限、矩阵秩亏、
            拟合出非物理参数（A ≤ 0 或 z0² < 0）。
    """
    if len(picks) < _MIN_PICKS:
        raise VelocityAnalysisError(
            f"双曲线拟合至少需要 {_MIN_PICKS} 个拾取点，实际 {len(picks)} 个。",
            hint="在 B-scan 上沿双曲线形态至少拾取 3 个点。",
        )

    x = np.array([float(p.x_m) for p in picks], dtype=np.float64)
    t = np.array([float(p.t_ns) for p in picks], dtype=np.float64)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(t))):
        raise VelocityAnalysisError(
            "拾取点坐标包含 NaN/Inf。", hint="重新拾取，确认走时轴与里程轴有效。")

    # t² = A·x² + B·x + C：设计矩阵一次 lstsq
    design = np.column_stack([x * x, x, np.ones_like(x)])
    if np.linalg.matrix_rank(design) < 3:
        raise VelocityAnalysisError(
            "拾取点分布退化（里程共线或过少），双曲线参数不可辨识。",
            hint="沿双曲线拉开拾取间距，覆盖顶点与两翼。",
        )
    coef, *_ = np.linalg.lstsq(design, t * t, rcond=None)
    a_coef, b_coef, c_coef = (float(v) for v in coef)
    if a_coef <= 0.0:
        raise VelocityAnalysisError(
            "拟合曲率为负或为零（A ≤ 0），不存在物理双曲线解。",
            hint="确认拾取点确为绕射双曲线（两端走时大于顶点），"
                 "且未混入平界面同相轴。",
        )

    v_m_ns = 2.0 / np.sqrt(a_coef)
    x0_m = -b_coef / (2.0 * a_coef)
    # C = A·(x0² + z0²)  ⇒  z0² = C/A - x0² = C/A - B²/(4A²)
    z0_sq = c_coef / a_coef - x0_m * x0_m
    if z0_sq < 0.0:
        raise VelocityAnalysisError(
            f"拟合顶点深度平方为负（z0²={z0_sq:.3e}），几何上不可能。",
            hint="拾取点应覆盖双曲线顶点附近；远离顶点的拾取会低估 z0。",
        )
    z0_m = float(np.sqrt(max(z0_sq, 0.0)))

    # 拟合优度：对原始 t（非线性模型）计算，RMSE/R² 语义直观
    t_model = (2.0 / v_m_ns) * np.sqrt((x - x0_m) ** 2 + z0_m ** 2)
    residual = t - t_model
    rmse = float(np.sqrt(np.mean(residual * residual)))
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    return HyperbolaFit(
        v_m_ns=float(v_m_ns),
        x0_m=float(x0_m),
        z0_m=z0_m,
        rmse_ns=rmse,
        r_squared=float(r_squared),
        pick_count=len(picks),
    )


__all__ = ["fit_hyperbola"]
