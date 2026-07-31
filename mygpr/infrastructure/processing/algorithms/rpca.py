"""Native robust-PCA background suppression with cancellation and hard caps."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import norm

from mygpr.infrastructure.processing.algorithms.common import (
    as_float,
    as_int,
    ensure_matrix,
    normalize_output,
    warning,
)
from mygpr.infrastructure.processing.algorithms.global_spectral import (
    GlobalProcessingCancelled,
    _cancel_check,
    _poll,
    leading_svd,
)


def _soft_threshold(value: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


def method_rpca_background_native(
    data: Any,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    rows, cols = arr.shape
    maximum_elements = max(1, as_int(params.get("max_elements"), 8_000_000))
    if arr.size > maximum_elements and not bool(params.get("allow_large", False)):
        raise MemoryError(
            f"RPCA input has {arr.size} elements, above managed cap {maximum_elements}; "
            "crop the line or explicitly allow a larger execution budget"
        )
    lam_param = params.get("lam")
    lam = as_float(lam_param, 0.0) if lam_param not in (None, "") else 0.0
    lam = lam if lam > 0.0 else 1.0 / np.sqrt(max(rows, cols))
    mu_param = as_float(params.get("mu"), 0.0)
    max_iter = max(1, as_int(params.get("max_iter"), 120))
    tolerance = max(1.0e-9, as_float(params.get("tol"), 1.0e-6))
    checker = _cancel_check(params)
    frobenius = float(norm(arr, ord="fro"))
    if frobenius == 0.0:
        return normalize_output(
            "rpca_background",
            np.zeros_like(arr),
            {"method": "rpca_background", "iterations": 0, "converged": True, "rank": 0},
            warnings,
        )
    spectral = float(norm(arr, 2))
    dual_norm = max(spectral, float(norm(arr.ravel(), np.inf)) / max(lam, 1.0e-9))
    dual = arr / max(dual_norm, 1.0e-9)
    mu = mu_param if mu_param > 0.0 else 1.25 / max(spectral, 1.0e-9)
    mu = max(mu, 1.0e-9)
    mu_bar = mu * 1.0e5
    low_rank = np.zeros_like(arr)
    sparse = np.zeros_like(arr)
    residual = float("inf")
    rank = 0
    converged = False
    iteration = 0
    # RPCA needs adaptive rank.  The initial bound is intentionally modest and
    # grows when all computed singular values survive shrinkage.
    rank_bound = min(min(arr.shape), max(8, as_int(params.get("rank_cap"), 64)))
    solver_counts: dict[str, int] = {}
    for iteration in range(1, max_iter + 1):
        _poll(checker, f"RPCA iteration {iteration}")
        working = arr - sparse + dual / mu
        factors = leading_svd(
            working,
            rank_bound,
            {
                **params,
                "solver": params.get("svd_solver", "auto"),
                "cancel_checker": checker,
            },
        )
        solver_counts[factors.solver] = solver_counts.get(factors.solver, 0) + 1
        shrunk = np.maximum(factors.singular_values - 1.0 / mu, 0.0)
        rank = int(np.count_nonzero(shrunk > 0.0))
        if rank:
            low_rank = (factors.left[:, :rank] * shrunk[:rank]) @ factors.right[:rank]
        else:
            low_rank.fill(0.0)
        if rank == factors.effective_rank and rank_bound < min(arr.shape):
            rank_bound = min(min(arr.shape), max(rank_bound + 8, rank_bound * 2))
        sparse = _soft_threshold(arr - low_rank + dual / mu, lam / mu)
        residual_matrix = arr - low_rank - sparse
        residual = float(norm(residual_matrix, ord="fro") / frobenius)
        if residual < tolerance:
            converged = True
            break
        dual = dual + mu * residual_matrix
        mu = min(mu * 1.5, mu_bar)
    if not converged:
        warnings.append(
            warning(
                "rpca_not_converged",
                "RPCA 在最大迭代次数内未达到收敛阈值。",
                "rpca_background",
                iterations=iteration,
                residual=residual,
            )
        )
    metadata = {
        "method": "rpca_background",
        "iterations": iteration,
        "converged": converged,
        "lambda": lam,
        "mu": mu,
        "rank": rank,
        "rank_bound": rank_bound,
        "sparse_ratio": float(np.count_nonzero(np.abs(sparse) > 1.0e-6) / sparse.size),
        "residual": residual,
        "svd_solvers": solver_counts,
    }
    return normalize_output("rpca_background", sparse, metadata, warnings)


__all__ = ["GlobalProcessingCancelled", "method_rpca_background_native"]
