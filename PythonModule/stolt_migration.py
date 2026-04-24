#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stolt / ω-k 迁移（增强版）"""

import numpy as np
from scipy.ndimage import uniform_filter1d
import scipy.fft as sp_fft

_STOLT_AXIS_CACHE = {}
_STOLT_GAIN_CACHE = {}


def _to_float32_2d(data: np.ndarray) -> np.ndarray:
    """转换为float32二维数组"""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array")
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return arr


def _get_stolt_axes(nt_pad: int, nx_pad: int, dt: float, dx: float):
    """获取Stolt迁移的频域坐标（带缓存）"""
    key = (int(nt_pad), int(nx_pad), float(dt), float(dx))
    if key in _STOLT_AXIS_CACHE:
        return _STOLT_AXIS_CACHE[key]
    omega = (2.0 * np.pi * sp_fft.rfftfreq(nt_pad, d=dt)).astype(np.float32, copy=False)
    kx = (2.0 * np.pi * sp_fft.fftshift(sp_fft.fftfreq(nx_pad, d=dx))).astype(
        np.float32, copy=False
    )
    if len(_STOLT_AXIS_CACHE) > 16:
        _STOLT_AXIS_CACHE.clear()
    _STOLT_AXIS_CACHE[key] = (omega, kx)
    return omega, kx


def _get_depth_gain(nt: int, depth_gain: float, depth_gain_power: float) -> np.ndarray:
    """获取深度增益曲线（带缓存）"""
    key = (int(nt), float(depth_gain), float(depth_gain_power))
    if key in _STOLT_GAIN_CACHE:
        return _STOLT_GAIN_CACHE[key]
    z = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    gain = (
        1.0 + depth_gain * np.power(z, max(depth_gain_power, 0.1), dtype=np.float32)
    ).astype(np.float32, copy=False)
    if len(_STOLT_GAIN_CACHE) > 32:
        _STOLT_GAIN_CACHE.clear()
    _STOLT_GAIN_CACHE[key] = gain
    return gain


def method_stolt_migration(data, dx=0.05, dt=0.1, v=0.10, pad_x=1, pad_t=1, **kwargs):
    """Stolt / ω-k 迁移（增强版）"""
    cancel_checker = kwargs.get("cancel_checker")
    arr = _to_float32_2d(data)

    nt, nx = arr.shape
    if nt < 2 or nx < 2:
        raise ValueError("Stolt migration requires at least 2x2 samples")

    dx = float(dx)
    dt = float(dt)
    v = float(v)
    if dx <= 0 or dt <= 0 or v <= 0:
        raise ValueError("dx, dt, v must be positive")

    pad_x = max(0, int(pad_x))
    pad_t = max(0, int(pad_t))

    jacobian_power = float(kwargs.get("stolt_jacobian_power", 0.05))
    obliquity_power = float(kwargs.get("stolt_obliquity_power", 0.05))
    mask_softness = float(kwargs.get("stolt_mask_softness", 0.03))
    kz_smooth = max(1, int(kwargs.get("stolt_kz_smooth", 3)))
    depth_gain = float(kwargs.get("stolt_depth_gain", 0.0))
    depth_gain_power = float(kwargs.get("stolt_depth_gain_power", 1.1))
    clip_percentile = float(kwargs.get("stolt_clip_percentile", 100.0))

    nt_pad = int(2 ** np.ceil(np.log2(nt * (1 + pad_t))))
    nx_pad = int(2 ** np.ceil(np.log2(nx * (1 + pad_x))))
    nt_pad = max(nt_pad, nt)
    nx_pad = max(nx_pad, nx)

    work = np.zeros((nt_pad, nx_pad), dtype=np.float32)
    work[:nt, :nx] = arr

    spec_tx = sp_fft.rfft(work, axis=0)
    spec_wkx = sp_fft.fftshift(sp_fft.fft(spec_tx, axis=1), axes=1)

    omega, kx = _get_stolt_axes(nt_pad, nx_pad, dt, dx)
    if omega.size < 2:
        raise ValueError("Insufficient frequency bins for Stolt mapping")

    omega_max = float(omega[-1])
    eps = 1e-6
    kz = np.linspace(0.0, 2.0 * omega_max / v, omega.size, dtype=np.float32)
    stolt_wkx = np.zeros_like(spec_wkx, dtype=np.complex64)
    mapped_mask = np.zeros_like(spec_wkx.real, dtype=np.float32)

    for ix in range(nx_pad):
        if cancel_checker and ix % 4 == 0 and bool(cancel_checker()):
            raise Exception("用户已取消（Stolt映射插值）")
        kx_val = float(kx[ix])
        kmag = np.sqrt(kx_val * kx_val + kz * kz, dtype=np.float32)
        omega_src = (0.5 * v * kmag).astype(np.float32, copy=False)
        valid = omega_src <= omega_max
        if not np.any(valid):
            continue

        omega_v = omega_src[valid]
        kz_v = kz[valid]

        col = spec_wkx[:, ix]
        re = np.interp(omega_v, omega, col.real, left=0.0, right=0.0)
        im = np.interp(omega_v, omega, col.imag, left=0.0, right=0.0)
        samp = (re + 1j * im).astype(np.complex64, copy=False)

        jac = np.clip((kz_v + eps) / (omega_v + eps), 0.0, 8.0).astype(
            np.float32, copy=False
        )
        amp = np.power(jac, max(jacobian_power, 0.0), dtype=np.float32)

        ob = np.clip(kz_v / (kmag[valid] + eps), 0.0, 1.0).astype(
            np.float32, copy=False
        )
        if obliquity_power > 0:
            amp *= np.power(ob, obliquity_power, dtype=np.float32)

        edge = (omega_max - omega_v) / (omega_max + eps)
        msoft = max(mask_softness, 1e-4)
        soft = np.where(edge >= msoft, 1.0, np.clip(edge / msoft, 0.0, 1.0)).astype(
            np.float32, copy=False
        )
        amp *= soft

        stolt_wkx[valid, ix] = samp * amp
        mapped_mask[valid, ix] = soft

    if kz_smooth > 1:
        stolt_wkx = (
            uniform_filter1d(stolt_wkx.real, size=kz_smooth, axis=0, mode="nearest")
            + 1j
            * uniform_filter1d(stolt_wkx.imag, size=kz_smooth, axis=0, mode="nearest")
        ).astype(np.complex64, copy=False)

    stolt_tx = sp_fft.ifft(sp_fft.ifftshift(stolt_wkx, axes=1), axis=1)
    migrated = sp_fft.irfft(stolt_tx, n=nt_pad, axis=0)
    migrated = np.asarray(migrated[:nt, :nx], dtype=np.float32)

    if depth_gain > 0:
        migrated = migrated * _get_depth_gain(nt, depth_gain, depth_gain_power)[:, None]

    clip_percentile = float(np.clip(clip_percentile, 95.0, 100.0))
    if clip_percentile < 99.999:
        clip_thr = float(np.percentile(np.abs(migrated), clip_percentile))
        if clip_thr > 0:
            migrated = (clip_thr * np.tanh(migrated / clip_thr)).astype(
                np.float32, copy=False
            )

    return migrated, {
        "mapped_params": {
            "dx": dx,
            "dt": dt,
            "v": v,
            "pad_x": int(pad_x),
            "pad_t": int(pad_t),
            "nt_pad": int(nt_pad),
            "nx_pad": int(nx_pad),
            "jacobian_power": jacobian_power,
            "obliquity_power": obliquity_power,
            "mask_softness": mask_softness,
            "kz_smooth": kz_smooth,
            "depth_gain": depth_gain,
            "depth_gain_power": depth_gain_power,
            "clip_percentile": clip_percentile,
            "mapped_coverage": float(np.mean(mapped_mask > 0)),
            "note": "enhanced_stolt_omega_k_with_amp_mapping_comp_fp32",
        }
    }
