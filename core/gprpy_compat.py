#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility helpers that mirror the baseline GPRPy windowed operators."""

from __future__ import annotations

import numpy as np


def apply_gprpy_dewow(data: np.ndarray, window: int) -> np.ndarray:
    """Mirror GPRPy toolbox dewow edge handling."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前: {arr.ndim}维")
    totsamps = arr.shape[0]
    if totsamps == 0:
        raise ValueError("输入数据为空")

    window = int(window)
    if window >= totsamps:
        result = arr - np.mean(arr, axis=0, keepdims=True)
        return result.astype(np.float32, copy=False)

    halfwid = int(np.ceil(window / 2.0))
    result = np.zeros(arr.shape, dtype=np.float64)

    avg = np.mean(arr[0 : halfwid + 1, :], axis=0)
    result[0 : halfwid + 1, :] = arr[0 : halfwid + 1, :] - avg

    for smp in range(halfwid, totsamps - halfwid + 1):
        winstart = int(smp - halfwid)
        winend = int(smp + halfwid)
        avg = np.mean(arr[winstart : winend + 1, :], axis=0)
        result[smp, :] = arr[smp, :] - avg

    avg = np.mean(arr[totsamps - halfwid : totsamps + 1, :], axis=0)
    result[totsamps - halfwid : totsamps + 1, :] = (
        arr[totsamps - halfwid : totsamps + 1, :] - avg
    )

    return result.astype(np.float32, copy=False)


def apply_gprpy_rem_mean_trace(data: np.ndarray, ntraces: int) -> np.ndarray:
    """Mirror GPRPy toolbox remMeanTrace edge handling."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前: {arr.ndim}维")
    tottraces = arr.shape[1]
    if tottraces == 0:
        raise ValueError("输入数据为空")

    ntraces = int(ntraces)
    if ntraces >= tottraces:
        result = arr - np.mean(arr, axis=1, keepdims=True)
        return result.astype(np.float32, copy=False)

    halfwid = int(np.ceil(ntraces / 2.0))
    result = np.zeros(arr.shape, dtype=np.float64)

    avg = np.mean(arr[:, 0 : halfwid + 1], axis=1, keepdims=True)
    result[:, 0 : halfwid + 1] = arr[:, 0 : halfwid + 1] - avg

    for tr in range(halfwid, tottraces - halfwid + 1):
        winstart = int(tr - halfwid)
        winend = int(tr + halfwid)
        avg = np.mean(arr[:, winstart : winend + 1], axis=1, keepdims=True)
        result[:, tr : tr + 1] = arr[:, tr : tr + 1] - avg

    avg = np.mean(arr[:, tottraces - halfwid : tottraces + 1], axis=1, keepdims=True)
    result[:, tottraces - halfwid : tottraces + 1] = (
        arr[:, tottraces - halfwid : tottraces + 1] - avg
    )

    return result.astype(np.float32, copy=False)


def apply_gprpy_agc_gain(data: np.ndarray, window: int) -> np.ndarray:
    """Mirror GPRPy toolbox AGC normalization."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前: {arr.ndim}维")
    totsamps = arr.shape[0]
    if totsamps == 0:
        raise ValueError("输入数据为空")

    eps = 1e-8
    window = int(window)
    if window > totsamps:
        energy = np.maximum(np.linalg.norm(arr, axis=0), eps)
        return np.divide(arr, energy).astype(np.float32, copy=False)

    halfwid = int(np.ceil(window / 2.0))
    result = np.zeros(arr.shape, dtype=np.float64)

    energy = np.maximum(np.linalg.norm(arr[0 : halfwid + 1, :], axis=0), eps)
    result[0 : halfwid + 1, :] = np.divide(arr[0 : halfwid + 1, :], energy)

    for smp in range(halfwid, totsamps - halfwid + 1):
        winstart = int(smp - halfwid)
        winend = int(smp + halfwid)
        energy = np.maximum(np.linalg.norm(arr[winstart : winend + 1, :], axis=0), eps)
        result[smp, :] = np.divide(arr[smp, :], energy)

    energy = np.maximum(np.linalg.norm(arr[totsamps - halfwid : totsamps + 1, :], axis=0), eps)
    result[totsamps - halfwid : totsamps + 1, :] = np.divide(
        arr[totsamps - halfwid : totsamps + 1, :],
        energy,
    )

    return result.astype(np.float32, copy=False)
