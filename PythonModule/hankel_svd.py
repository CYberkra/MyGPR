#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hankel 矩阵 SVD 去噪（截断 SVD 主路径）。"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.linalg import svd as full_svd
from scipy.sparse.linalg import svds as truncated_svds


class ProcessingCancelled(Exception):
    """用户主动取消处理。"""


def _resolve_window_length(ny: int, window_length: int | None) -> int:
    """解析 Hankel 嵌入窗口长度。"""
    if window_length is None or int(window_length) <= 0:
        return max(1, ny // 4)
    return min(int(window_length), ny - 1)


def _resolve_target_rank(
    m: int,
    window_length: int,
    rank: int | None,
    max_rank: int,
) -> int:
    """解析截断 SVD 目标秩。"""
    k_target = min(int(max_rank), min(m, window_length) - 1)
    if rank is not None and int(rank) > 0:
        k_target = min(int(rank), k_target)
    return max(k_target, 1)


def _detect_rank_from_spectrum(S: np.ndarray) -> int | None:
    """根据奇异值差分的稳定段估计秩。"""
    if len(S) <= 2:
        return None

    diff_spec = np.diff(S)
    threshold = np.mean(np.abs(diff_spec))
    below = (np.abs(diff_spec[:-1]) < threshold) & (np.abs(diff_spec[1:]) < threshold)
    hits = np.where(below)[0]
    if len(hits) == 0:
        return None
    return max(int(hits[0]) + 1, 1)


def _build_hankel(trace: np.ndarray, m: int, window_length: int) -> np.ndarray:
    """构建单道 Hankel 矩阵。"""
    return as_strided(
        trace,
        shape=(m, window_length),
        strides=(trace.strides[0], trace.strides[0]),
    ).copy()


def _build_diagonal_average_counts(ny: int, m: int, window_length: int) -> np.ndarray:
    """计算反对角线平均时每个位置的累加次数。"""
    counts = np.zeros(ny)
    for j in range(window_length):
        counts[j : j + m] += 1
    return counts


def _build_hankel_indices(window_length: int, m: int) -> np.ndarray:
    """预计算反对角线回写索引。"""
    return (np.arange(window_length)[:, None] + np.arange(m)[None, :]).ravel()


def method_hankel_svd(data, window_length=None, rank=None, max_rank=10, **kwargs):
    """Hankel 矩阵 SVD 去噪。

    当前实现按列逐道处理，每道数据走：
    1. Hankel 嵌入
    2. 截断 SVD（失败时回退全量 SVD）
    3. 低秩重建
    4. 反对角线平均回写

    说明：
    - `batch_size` 等旧实验参数会被静默忽略；当前主路径未使用 batch SVD。
    - 返回轻量 metadata，不包含大数组。
    """
    ny, nx = data.shape
    window_length = _resolve_window_length(ny, window_length)

    result = np.zeros_like(data, dtype=float)
    m = ny - window_length + 1
    if m <= 0:
        return result, {
            "method": "hankel_svd",
            "window_length": int(window_length),
            "rank_requested": int(rank) if rank is not None and rank > 0 else None,
            "rank_mode": "fixed" if rank is not None and rank > 0 else "auto",
            "effective_rank_min": None,
            "effective_rank_max": None,
            "svd_backend": "none",
            "fallback_columns": 0,
        }

    counts = _build_diagonal_average_counts(ny, m, window_length)
    hankel_indices = _build_hankel_indices(window_length, m)
    k_target = _resolve_target_rank(m, window_length, rank, max_rank)
    cancel_checker = kwargs.get("cancel_checker")
    fallback_columns = 0
    effective_rank_min = None
    effective_rank_max = None

    for col in range(nx):
        if cancel_checker and col % 8 == 0 and bool(cancel_checker()):
            raise ProcessingCancelled("用户已取消（Hankel-SVD）")

        trace = data[:, col].astype(float)
        hankel = _build_hankel(trace, m, window_length)

        try:
            U, S, Vt = truncated_svds(hankel, k=k_target, which="LM")
        except Exception:
            fallback_columns += 1
            U, S, Vt = full_svd(hankel, full_matrices=False)
            U, S, Vt = U[:, :k_target], S[:k_target], Vt[:k_target, :]

        if rank is None or rank <= 0:
            rank_val = _detect_rank_from_spectrum(S)
            if rank_val is not None:
                U, S, Vt = U[:, :rank_val], S[:rank_val], Vt[:rank_val, :]

        current_rank = int(len(S))
        effective_rank_min = (
            current_rank
            if effective_rank_min is None
            else min(effective_rank_min, current_rank)
        )
        effective_rank_max = (
            current_rank
            if effective_rank_max is None
            else max(effective_rank_max, current_rank)
        )

        hankel_filtered = (U * S) @ Vt

        trace_filtered = np.zeros(ny)
        np.add.at(trace_filtered, hankel_indices, hankel_filtered.ravel())
        result[:, col] = trace_filtered / counts

    if fallback_columns == 0:
        svd_backend = "truncated"
    elif fallback_columns >= nx:
        svd_backend = "full"
    else:
        svd_backend = "mixed"

    return result, {
        "method": "hankel_svd",
        "window_length": int(window_length),
        "rank_requested": int(rank) if rank is not None and rank > 0 else None,
        "rank_mode": "fixed" if rank is not None and rank > 0 else "auto",
        "effective_rank_min": effective_rank_min,
        "effective_rank_max": effective_rank_max,
        "svd_backend": svd_backend,
        "fallback_columns": int(fallback_columns),
    }
