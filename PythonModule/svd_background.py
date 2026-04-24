#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVD背景去除（低秩近似）"""

import numpy as np
from scipy.linalg import svd


def method_svd_background(data, rank=1, **kwargs):
    """SVD背景去除（低秩近似）"""
    U, S, Vt = svd(data, full_matrices=False)
    S_bg = np.zeros_like(S)
    S_bg[:rank] = S[:rank]
    background = (U * S_bg) @ Vt
    return data - background, background
