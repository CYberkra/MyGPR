#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVD背景去除（低秩近似）"""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd


def method_svd_background(
    data: np.ndarray,
    rank: int = 1,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    """SVD背景去除（低秩近似）"""
    U, S, Vt = svd(data, full_matrices=False, check_finite=False)
    S_bg = np.zeros_like(S)
    S_bg[:rank] = S[:rank]
    background = (U * S_bg) @ Vt
    return data - background, background
