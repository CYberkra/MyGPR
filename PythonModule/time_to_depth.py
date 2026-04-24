#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深度转换与标定"""

import numpy as np
from scipy.interpolate import interp1d


def method_time_to_depth(data, dt=0.1, v=0.10, dz=0.02, **kwargs):
    """深度转换与标定"""
    ny, nx = data.shape
    t = np.arange(ny) * dt
    z_old = t * v / 2.0

    z_max = z_old[-1] if ny > 0 else 0.0
    num_z = int(z_max / dz) + 1 if dz > 0 else ny
    z_new = np.linspace(0, z_max, max(num_z, 1))

    f = interp1d(z_old, data, axis=0, bounds_error=False, fill_value=0.0)
    depth_data = f(z_new)
    return depth_data, {"is_depth": True, "z_max": z_max}
