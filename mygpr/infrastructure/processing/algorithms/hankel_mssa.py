#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D MSSA (Multivariate Singular Spectrum Analysis) 去噪。

实现要点：
1. 垂直 MSSA (V-MSSA)：将所有道的 Hankel 矩阵垂直堆叠后统一做 SVD。
2. 利用跨道相关性分离信号与噪声，解决单道 Hankel SVD 在真实 GPR 数据上的失败问题。
3. 支持固定 window_size / rank，也支持 SVHT 自动选择 rank。
4. 保留原有 hankel_svd 接口兼容性。

参考：
- pymssa (https://github.com/kieferk/pymssa) - V-MSSA 实现
- Golyandina et al. 2015, J. Stat. Software - MSSA 理论
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import svd as scipy_svd


DEFAULT_SVHT_BASE = 2.858
DEFAULT_WINDOW_SIZE = 0  # 0 表示自动 (N//2)


class ProcessingCancelled(Exception):
    """用户主动取消处理。"""


def _append_warning(metadata: dict[str, object], message: str) -> None:
    """向 metadata.warning 列表追加去重告警。"""
    warnings = metadata.setdefault("warnings", [])
    if not isinstance(warnings, list):
        warnings = list(warnings) if isinstance(warnings, (list, tuple, set)) else []
        metadata["warnings"] = warnings
    text = str(message)
    if text not in warnings:
        warnings.append(text)


def _poll_cancel(cancel_checker) -> None:
    """Raise standard cancellation exception when user aborts."""
    if cancel_checker and bool(cancel_checker()):
        raise ProcessingCancelled("用户已取消（Hankel-SVD / MSSA）")


def _resolve_window_size(n_samples: int, window_length: int | None) -> int:
    """解析 window_size，默认 N//2。"""
    if window_length is not None and int(window_length) > 0:
        return min(int(window_length), n_samples - 1)
    return n_samples // 2


def _build_vmssa_trajectory(
    data: np.ndarray,
    window_size: int,
    dtype: np.dtype | type[np.floating] | None = None,
) -> np.ndarray:
    """构建垂直 MSSA 轨迹矩阵。

    对每道（列）构建 Hankel 矩阵 (L, K)，然后垂直堆叠成 (P*L, K)。
    """
    n_samples, n_traces = data.shape
    L = window_size
    K = n_samples - L + 1

    work_dtype = np.dtype(dtype) if dtype is not None else data.dtype

    if K <= 0:
        # 太短，无法构建轨迹矩阵
        return np.zeros((0, 0), dtype=work_dtype)

    blocks = []
    for p in range(n_traces):
        trace = np.asarray(data[:, p], dtype=work_dtype)
        # sliding_window_view 返回 (K, L)
        windows = sliding_window_view(trace, L)
        blocks.append(windows.T)  # (L, K)

    return np.concatenate(blocks, axis=0)  # (P*L, K)


def _svd_trajectory(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, str]:
    """对轨迹矩阵做 SVD 分解。

    返回：(right_singular_vectors, singular_values, rank, backend)
    """
    if trajectory.size == 0:
        return np.zeros((0, 0)), np.zeros(0), 0, "none"

    rows, cols = trajectory.shape
    if rows >= cols:
        # V-MSSA 轨迹矩阵通常是 tall-skinny。用 X.T @ X 求右奇异向量，
        # 避免保存 rows x cols 的左奇异向量，显著降低峰值内存。
        gram = trajectory.T @ trajectory
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        right_singular_vectors = eigenvectors[:, order].T
        singular_values = np.sqrt(eigenvalues)
        backend = "gram_eigh"
    else:
        # 宽矩阵较少见，保留直接 SVD 路径以覆盖极小窗口/单道输入。
        _U, singular_values, right_singular_vectors = scipy_svd(
            trajectory,
            full_matrices=False,
            check_finite=False,
        )
        backend = "full"

    rank = int(np.sum(singular_values > 1e-12))

    return right_singular_vectors, singular_values, rank, backend


def _svht_rank_selection(singular_values: np.ndarray, aggressiveness: float = 1.0) -> int:
    """Singular Value Hard Thresholding (SVHT) 自动选择 rank。

    参考：Donoho & Gavish 2013, https://arxiv.org/abs/1305.5870

    aggressiveness: 去噪激进程度
        - 0.1: 非常保守，保留大量成分
        - 1.0: 标准 SVHT
        - 2.0: 激进，去掉更多成分
    """
    if len(singular_values) == 0:
        return 1

    # 有效特征值
    s_valid = singular_values[singular_values > 1e-12]
    if len(s_valid) == 0:
        return 1

    median_sv = float(np.median(s_valid))
    if median_sv <= 1e-12:
        return 1

    # aggressiveness 越高，threshold 越低（更激进，去掉更多）
    threshold = DEFAULT_SVHT_BASE / max(float(aggressiveness), 0.01)
    sv_threshold = threshold * median_sv

    rank = int(np.sum(singular_values >= sv_threshold))
    return max(1, min(rank, len(singular_values)))


def _diagonal_average(hankel_matrix: np.ndarray) -> np.ndarray:
    """对角平均重建时间序列。

    输入: Hankel 矩阵 (L, K)
    输出: 时间序列，长度 N = L + K - 1
    """
    L, K = hankel_matrix.shape
    N = L + K - 1
    output = np.zeros(N, dtype=np.float64)
    counts = np.convolve(np.ones(L, dtype=np.float64), np.ones(K, dtype=np.float64))

    for i, row in enumerate(hankel_matrix):
        output[i : i + K] += row
    return output / counts


def _reconstruct_vmssa(
    trajectory: np.ndarray,
    right_singular_vectors: np.ndarray,
    rank: int,
    n_samples: int,
    n_traces: int,
    window_size: int,
) -> np.ndarray:
    """V-MSSA 重建：用前 r 个成分重建轨迹矩阵，然后对角平均恢复每道。

    返回: 重建后的数据 (n_samples, n_traces)
    """
    L = window_size

    if trajectory.size == 0 or rank <= 0:
        return np.zeros((n_samples, n_traces), dtype=np.float64)

    V_r = right_singular_vectors[:rank, :]
    projected = trajectory @ V_r.T

    # 将重建后的轨迹矩阵分割回每道，然后对角平均
    result = np.zeros((n_samples, n_traces), dtype=np.float64)
    for p in range(n_traces):
        sidx = p * L
        eidx = sidx + L
        block = projected[sidx:eidx, :] @ V_r  # (L, K)
        result[:, p] = _diagonal_average(block)

    return result


def _initial_metadata(
    n_samples: int,
    rank: int | None,
    aggressiveness: float,
    preview: bool,
) -> dict[str, object]:
    fixed_rank = rank is not None and int(rank) > 0
    return {
        "method": "hankel_svd",
        "window_length": None,
        "rank_requested": int(rank) if fixed_rank else None,
        "rank_mode": "fixed" if fixed_rank else "auto",
        "effective_rank_min": None,
        "effective_rank_max": None,
        "svd_backend": "full",
        "fallback_columns": 0,
        "window_selection_mode": "fixed",
        "rank_selection_mode": "fixed" if fixed_rank else "svht_auto",
        "rho": aggressiveness,
        "recovery_mode": "mssa",
        "candidate_windows": [],
        "calibration_traces": [],
        "preview": preview,
        "caps": {
            "max_candidate_windows": 6,
            "max_calibration_traces": 8,
            "preview_window_cap": max(1, n_samples - 1),
            "auto_rank_fallback_cap": 3,
            "column_cancel_interval": 8,
        },
        "post_recovery_gain_min": 1.0,
        "post_recovery_gain_max": 1.0,
        "saliency_preservation_fraction_min": 0.0,
        "saliency_preservation_fraction_max": 0.0,
        "warnings": [],
    }


def _short_input_metadata(aggressiveness: float, preview: bool) -> dict[str, object]:
    metadata = _initial_metadata(1, 1, aggressiveness, preview)
    metadata.update(
        window_length=1,
        rank_requested=1,
        rank_mode="fixed",
        effective_rank_min=1,
        effective_rank_max=1,
        svd_backend="none",
        rank_selection_mode="fixed",
        candidate_windows=[1],
        caps={},
    )
    _append_warning(metadata, "input has fewer than 2 samples; returning original data")
    return metadata


def _select_window(
    n_samples: int,
    requested: int | None,
    preview: bool,
    metadata: dict[str, object],
) -> int:
    automatic = requested is None or int(requested) <= 0
    if automatic:
        window = n_samples // 2
        metadata["window_selection_mode"] = "auto_n_div_2"
    else:
        window = _resolve_window_size(n_samples, requested)
        metadata["window_selection_mode"] = "fixed"
        if int(requested) != window:
            _append_warning(
                metadata,
                f"requested window_length {int(requested)} was adjusted to feasible value {window}",
            )
    if preview and not automatic:
        cap = max(1, n_samples // 4)
        if window > cap:
            _append_warning(metadata, f"preview capped requested window_length {window} to {cap}")
            window = cap
    metadata["window_length"] = window
    metadata["candidate_windows"] = [window]
    return window


def _select_rank(
    singular_values: np.ndarray,
    full_rank: int,
    requested: int | None,
    max_rank: int,
    window_size: int,
    n_samples: int,
    preview: bool,
    aggressiveness: float,
    metadata: dict[str, object],
) -> int:
    automatic = requested is None or int(requested) <= 0
    if automatic:
        selected = _svht_rank_selection(singular_values, aggressiveness=aggressiveness)
        cap = max(1, min(int(max_rank), full_rank))
        if selected > cap:
            _append_warning(metadata, f"auto-selected rank {selected} was capped to max_rank {cap}")
            selected = cap
        if preview and selected > max(1, min(5, full_rank)):
            preview_cap = max(1, min(5, full_rank))
            _append_warning(metadata, f"preview capped auto-selected rank {selected} to {preview_cap}")
            selected = preview_cap
        metadata["rank_selection_mode"] = "svht_auto"
    else:
        columns = n_samples - window_size + 1
        feasible = max(1, min(full_rank, max_rank, window_size - 1, columns - 1))
        selected = max(1, min(int(requested), feasible))
        metadata["rank_selection_mode"] = "fixed"
        if int(requested) != selected:
            _append_warning(metadata, f"requested rank {int(requested)} was adjusted to feasible value {selected}")
    metadata["effective_rank_min"] = selected
    metadata["effective_rank_max"] = selected
    return selected


def method_hankel_svd(data, window_length=None, rank=None, max_rank=100, preview=False, **kwargs):
    """Denoise a B-scan through vertical multivariate SSA."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    preview = bool(kwargs.get("preview", preview))
    checker = kwargs.get("cancel_checker")
    aggressiveness = max(0.01, min(float(kwargs.get("aggressiveness", 1.0)), 10.0))
    n_samples, n_traces = arr.shape
    if n_samples < 2:
        return arr.copy(), _short_input_metadata(aggressiveness, preview)
    metadata = _initial_metadata(n_samples, rank, aggressiveness, preview)
    if n_samples == 0 or n_traces == 0:
        _append_warning(metadata, "empty input matrix received")
        return np.zeros_like(arr), metadata
    window_size = _select_window(n_samples, window_length, preview, metadata)
    _poll_cancel(checker)
    trajectory_bytes = n_traces * window_size * (n_samples - window_size + 1) * 8
    trajectory_dtype = np.float32 if trajectory_bytes >= 32 * 1024 * 1024 else np.float64
    trajectory = _build_vmssa_trajectory(arr.astype(trajectory_dtype, copy=False), window_size)
    if trajectory.size == 0:
        _append_warning(metadata, "trajectory matrix is empty; returning zero output")
        return np.zeros_like(arr), metadata
    _poll_cancel(checker)
    vectors, singular_values, full_rank, solver = _svd_trajectory(trajectory)
    metadata.update(
        svd_backend="mixed" if solver == "gram_eigh" else solver,
        svd_solver=solver,
        trajectory_dtype=str(trajectory.dtype),
    )
    selected_rank = _select_rank(
        singular_values, full_rank, rank, max_rank, window_size,
        n_samples, preview, aggressiveness, metadata,
    )
    _poll_cancel(checker)
    result = _reconstruct_vmssa(
        trajectory, vectors, selected_rank, n_samples, n_traces, window_size,
    )
    if preview:
        _append_warning(metadata, "preview mode: MSSA with bounded parameters")
    return result, metadata


# 保留旧的辅助函数以兼容可能的直接调用
_build_hankel = _build_vmssa_trajectory


def method_hankel_svd_native(data, params):
    """Application-adapter signature for the native MSSA kernel."""
    from mygpr.infrastructure.processing.algorithms.common import normalize_output, warning

    runtime = dict(params or {})
    context = runtime.pop("_execution_context", None)
    if context is not None:
        runtime.setdefault("cancel_checker", context.is_cancelled)
    result, metadata = method_hankel_svd(data, **runtime)
    runtime_warnings = []
    for message in metadata.pop("warnings", []) or []:
        runtime_warnings.append(
            warning("hankel_mssa_warning", str(message), "hankel_svd")
        )
    return normalize_output("hankel_svd", result, metadata, runtime_warnings)


__all__ = ["method_hankel_svd_native"]
