#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F-K锥形滤波器"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift


def method_fk_filter(data, angle_low=10, angle_high=65, taper_width=5, **kwargs):
    """F-K锥形滤波器"""
    F = fftshift(fft2(data))
    ny, nx = F.shape

    ky = fftshift(np.fft.fftfreq(ny))
    kx = fftshift(np.fft.fftfreq(nx))

    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    angle = np.degrees(np.arctan2(np.abs(KY), np.abs(KX)))

    band_mask = (angle >= angle_low) & (angle <= angle_high)
    mask = np.ones_like(angle, dtype=float)

    if taper_width > 0:
        sigma = taper_width
        dist_to_low = np.abs(angle - angle_low)
        dist_to_high = np.abs(angle - angle_high)
        dist = np.minimum(dist_to_low, dist_to_high)

        mask[band_mask] = 0.05
        taper_region = band_mask & (dist < taper_width)
        if np.any(taper_region):
            mask[taper_region] = 1 - np.exp(-(dist[taper_region] ** 2) / (2 * sigma**2))
    else:
        mask[band_mask] = 0.0

    F_filtered = F * mask
    result = np.real(ifft2(ifftshift(F_filtered)))
    return result, mask
