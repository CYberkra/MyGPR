#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RPCA background suppression using inexact augmented Lagrange multiplier."""

from __future__ import annotations

import numpy as np
from scipy.linalg import norm, svd


def _soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def method_rpca_background(
    data: np.ndarray,
    lam: float | None = None,
    mu: float | None = None,
    max_iter: int = 120,
    tol: float = 1e-6,
    **kwargs,
):
    """Separate low-rank background and sparse reflections/clutter.

    Returns sparse foreground (`data - low_rank`) plus metadata for diagnostics.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")

    rows, cols = arr.shape
    lam_value = float(lam) if lam is not None else 1.0 / np.sqrt(max(rows, cols))
    max_iter = max(1, int(max_iter))
    tol = max(float(tol), 1e-9)
    mu_report = float(mu) if (mu is not None and float(mu) > 0.0) else 0.0

    fro_norm = norm(arr, ord="fro")
    if fro_norm == 0.0:
        zero = np.zeros_like(arr, dtype=np.float32)
        return zero, {
            "method": "rpca_background",
            "iterations": 0,
            "converged": True,
            "lambda": lam_value,
            "mu": mu_report,
            "rank": 0,
            "sparse_ratio": 0.0,
            "residual": 0.0,
        }

    dual_norm = max(norm(arr, 2), norm(arr.ravel(), np.inf) / max(lam_value, 1e-9))
    Y = arr / max(dual_norm, 1e-9)

    if mu is not None and float(mu) > 0.0:
        mu_initial = float(mu)
    else:
        mu_initial = 1.25 / max(norm(arr, 2), 1e-9)
    mu_value = max(mu_initial, 1e-9)
    mu_bar = mu_value * 1e5
    rho = 1.5

    low_rank = np.zeros_like(arr)
    sparse = np.zeros_like(arr)
    residual = float("inf")
    rank = 0
    converged = False

    iteration = 0
    for iteration in range(1, max_iter + 1):
        U, s, Vt = svd(arr - sparse + Y / mu_value, full_matrices=False)
        s_thresh = np.maximum(s - 1.0 / mu_value, 0.0)
        rank = int(np.sum(s_thresh > 0.0))
        if rank > 0:
            low_rank = (U[:, :rank] * s_thresh[:rank]) @ Vt[:rank, :]
        else:
            low_rank.fill(0.0)

        sparse = _soft_threshold(arr - low_rank + Y / mu_value, lam_value / mu_value)
        residual_matrix = arr - low_rank - sparse
        residual = norm(residual_matrix, ord="fro") / fro_norm
        if residual < tol:
            converged = True
            break

        Y = Y + mu_value * residual_matrix
        mu_value = min(mu_value * rho, mu_bar)

    result = sparse.astype(np.float32)
    return result, {
        "method": "rpca_background",
        "iterations": iteration,
        "converged": converged,
        "lambda": lam_value,
        "mu": mu_value,
        "rank": rank,
        "sparse_ratio": float(np.count_nonzero(np.abs(result) > 1e-6) / result.size),
        "residual": float(residual),
    }
