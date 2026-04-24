#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVD subspace reconstruction denoising."""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd


def method_svd_subspace(
    data: np.ndarray,
    rank_start: int = 2,
    rank_end: int = 40,
    **kwargs,
):
    """Reconstruct data using a selected SVD singular-value subspace.

    Args:
        data: Input B-scan array with shape (samples, traces).
        rank_start: 1-based inclusive singular-value index to keep.
        rank_end: 1-based inclusive singular-value index to keep.

    Returns:
        tuple: (reconstructed_array, metadata_dict)
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")

    min_dim = min(arr.shape)
    if min_dim < 2:
        return arr.astype(np.float32), {
            "method": "svd_subspace",
            "rank_start": 1,
            "rank_end": 1,
        }

    rank_start = max(1, int(rank_start))
    rank_end = max(rank_start, int(rank_end))
    rank_end = min(rank_end, min_dim)

    U, S, Vt = svd(arr, full_matrices=False)
    start_idx = rank_start - 1
    end_idx = rank_end
    reconstructed = (U[:, start_idx:end_idx] * S[start_idx:end_idx]) @ Vt[
        start_idx:end_idx, :
    ]

    return reconstructed.astype(np.float32), {
        "method": "svd_subspace",
        "rank_start": rank_start,
        "rank_end": rank_end,
        "rank_count": rank_end - rank_start + 1,
    }
