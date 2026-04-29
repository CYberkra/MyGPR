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

import math

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


def _build_vmssa_trajectory(data: np.ndarray, window_size: int) -> np.ndarray:
    """构建垂直 MSSA 轨迹矩阵。

    对每道（列）构建 Hankel 矩阵 (L, K)，然后垂直堆叠成 (P*L, K)。
    """
    n_samples, n_traces = data.shape
    L = window_size
    K = n_samples - L + 1

    if K <= 0:
        # 太短，无法构建轨迹矩阵
        return np.zeros((0, 0), dtype=np.float64)

    blocks = []
    for p in range(n_traces):
        trace = data[:, p]
        # sliding_window_view 返回 (K, L)
        windows = sliding_window_view(trace, L)
        blocks.append(windows.T)  # (L, K)

    return np.concatenate(blocks, axis=0)  # (P*L, K)


def _svd_trajectory(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """对轨迹矩阵做 SVD 分解。

    返回：(left_singular_vectors, singular_values, rank)
    """
    if trajectory.size == 0:
        return np.zeros((0, 0)), np.zeros(0), 0

    # 直接 SVD（当 K < P*L 时，比 covariance 方法更快）
    U, s, Vh = scipy_svd(trajectory, full_matrices=False, check_finite=False)

    # 排序（按降序，scipy_svd 默认已排序）
    rank = int(np.sum(s > 1e-12))

    return U, s, rank


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
    counts = np.zeros(N, dtype=np.float64)

    for i in range(L):
        for j in range(K):
            idx = i + j
            output[idx] += hankel_matrix[i, j]
            counts[idx] += 1.0

    counts[counts == 0.0] = 1.0
    return output / counts


def _reconstruct_vmssa(
    trajectory: np.ndarray,
    left_singular_vectors: np.ndarray,
    rank: int,
    n_samples: int,
    n_traces: int,
    window_size: int,
) -> np.ndarray:
    """V-MSSA 重建：用前 r 个成分重建轨迹矩阵，然后对角平均恢复每道。

    返回: 重建后的数据 (n_samples, n_traces)
    """
    L = window_size
    K = n_samples - L + 1

    if trajectory.size == 0 or rank <= 0:
        return np.zeros((n_samples, n_traces), dtype=np.float64)

    # 重建轨迹矩阵 = U_r @ U_r.T @ X
    U_r = left_singular_vectors[:, :rank]
    reconstructed_trajectory = U_r @ (U_r.T @ trajectory)

    # 将重建后的轨迹矩阵分割回每道，然后对角平均
    result = np.zeros((n_samples, n_traces), dtype=np.float64)
    for p in range(n_traces):
        sidx = p * L
        eidx = sidx + L
        block = reconstructed_trajectory[sidx:eidx, :]  # (L, K)
        result[:, p] = _diagonal_average(block)

    return result


def method_hankel_svd(data, window_length=None, rank=None, max_rank=100, preview=False, **kwargs):
    """2D MSSA (V-MSSA) 去噪。

    实现要点：
    1. 垂直 MSSA：所有道的 Hankel 矩阵垂直堆叠后统一 SVD
    2. 利用跨道相关性分离信号与噪声
    3. 支持 SVHT 自动选择 rank，也支持固定 rank
    4. 保留原有接口兼容性

    参数：
        data: 2D 数组 (n_samples, n_traces)
        window_length: 窗口大小，0/None 表示自动 (N//2)
        rank: 保留的成分数，0/None 表示自动 (SVHT)
        max_rank: 最大有效 rank
        preview: 预览模式
        aggressiveness: 去噪激进程度 (0.1~2.0)
        cancel_checker: 取消检查函数

    返回：
        (result_array, metadata_dict)
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")

    # 预处理：替换 NaN/Inf 为 0（MSSA 对缺失值敏感）
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    _ = kwargs.get("batch_size")
    if "preview" in kwargs:
        preview = kwargs.get("preview", preview)
    preview = bool(preview)
    cancel_checker = kwargs.get("cancel_checker")
    aggressiveness = float(kwargs.get("aggressiveness", 1.0))
    aggressiveness = max(0.01, min(aggressiveness, 10.0))

    n_samples, n_traces = arr.shape
    result = np.zeros_like(arr, dtype=np.float64)

    # 边界情况：如果样本数太少，无法构建有效的轨迹矩阵
    if n_samples < 2:
        metadata: dict[str, object] = {
            "method": "hankel_svd",
            "window_length": 1,
            "rank_requested": 1,
            "rank_mode": "fixed",
            "effective_rank_min": 1,
            "effective_rank_max": 1,
            "svd_backend": "none",
            "fallback_columns": 0,
            "window_selection_mode": "fixed",
            "rank_selection_mode": "fixed",
            "rho": aggressiveness,
            "recovery_mode": "mssa",
            "candidate_windows": [1],
            "calibration_traces": [],
            "preview": preview,
            "caps": {},
            "post_recovery_gain_min": 1.0,
            "post_recovery_gain_max": 1.0,
            "saliency_preservation_fraction_min": 0.0,
            "saliency_preservation_fraction_max": 0.0,
            "warnings": ["input has fewer than 2 samples; returning original data"],
        }
        return arr.copy(), metadata

    metadata: dict[str, object] = {
        "method": "hankel_svd",
        "window_length": None,
        "rank_requested": int(rank) if rank is not None and int(rank) > 0 else None,
        "rank_mode": "fixed" if rank is not None and int(rank) > 0 else "auto",
        "effective_rank_min": None,
        "effective_rank_max": None,
        "svd_backend": "full",
        "fallback_columns": 0,
        "window_selection_mode": "fixed",
        "rank_selection_mode": "fixed" if rank is not None and int(rank) > 0 else "svht_auto",
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

    if n_samples == 0 or n_traces == 0:
        metadata["window_length"] = 1 if n_samples <= 1 else _resolve_window_size(n_samples, window_length)
        _append_warning(metadata, "empty input matrix received")
        return result, metadata

    # 确定 window_size
    auto_window = window_length is None or int(window_length) <= 0
    if auto_window:
        window_size = n_samples // 2
        metadata["window_selection_mode"] = "auto_n_div_2"
    else:
        window_size = _resolve_window_size(n_samples, window_length)
        metadata["window_selection_mode"] = "fixed"
        if window_length is not None and int(window_length) != window_size:
            _append_warning(
                metadata,
                f"requested window_length {int(window_length)} was adjusted to feasible value {window_size}",
            )

    # 预览模式：限制窗口大小
    if preview and not auto_window:
        preview_window_cap = max(1, n_samples // 4)
        if window_size > preview_window_cap:
            _append_warning(
                metadata,
                f"preview capped requested window_length {window_size} to {preview_window_cap}",
            )
            window_size = preview_window_cap

    metadata["window_length"] = window_size
    metadata["candidate_windows"] = [window_size]

    # 构建轨迹矩阵
    _poll_cancel(cancel_checker)
    trajectory = _build_vmssa_trajectory(arr, window_size)

    if trajectory.size == 0:
        _append_warning(metadata, "trajectory matrix is empty; returning zero output")
        return result, metadata

    # SVD 分解
    _poll_cancel(cancel_checker)
    left_singular_vectors, singular_values, full_rank = _svd_trajectory(trajectory)
    metadata["svd_backend"] = "full"

    # 确定 rank
    auto_rank = rank is None or int(rank) <= 0
    if auto_rank:
        selected_rank = _svht_rank_selection(singular_values, aggressiveness=aggressiveness)
        metadata["rank_selection_mode"] = "svht_auto"
        if preview:
            preview_rank_cap = max(1, min(5, full_rank))
            if selected_rank > preview_rank_cap:
                _append_warning(
                    metadata,
                    f"preview capped auto-selected rank {selected_rank} to {preview_rank_cap}",
                )
                selected_rank = preview_rank_cap
    else:
        # 对于2D MSSA，rank上限是min(P*L, K)，但为了兼容性，也限制为min(window_size, K)-1
        K = n_samples - window_size + 1
        theoretical_max = min(full_rank, max_rank, window_size - 1, K - 1)
        theoretical_max = max(1, theoretical_max)
        selected_rank = min(int(rank), theoretical_max)
        selected_rank = max(1, selected_rank)
        metadata["rank_selection_mode"] = "fixed"
        if rank is not None and int(rank) != selected_rank:
            _append_warning(
                metadata,
                f"requested rank {int(rank)} was adjusted to feasible value {selected_rank}",
            )

    metadata["effective_rank_min"] = selected_rank
    metadata["effective_rank_max"] = selected_rank

    # 重建
    _poll_cancel(cancel_checker)
    reconstructed = _reconstruct_vmssa(
        trajectory,
        left_singular_vectors,
        selected_rank,
        n_samples,
        n_traces,
        window_size,
    )

    result = reconstructed

    # 记录一些统计信息
    metadata["post_recovery_gain_min"] = 1.0
    metadata["post_recovery_gain_max"] = 1.0
    metadata["saliency_preservation_fraction_min"] = 0.0
    metadata["saliency_preservation_fraction_max"] = 0.0

    if preview:
        _append_warning(metadata, "preview mode: MSSA with bounded parameters")

    return result, metadata


# 保留旧的辅助函数以兼容可能的直接调用
_build_hankel = _build_vmssa_trajectory
