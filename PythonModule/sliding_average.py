#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""滑动平均背景去除"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def method_sliding_average(data, window_size=10, axis=1, **kwargs):
    """滑动平均背景去除"""
    background = uniform_filter1d(data, size=window_size, axis=axis, mode="nearest")
    return data - background, background
