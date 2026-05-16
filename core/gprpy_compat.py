#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility helpers that mirror the baseline GPRPy windowed operators."""

from __future__ import annotations

import numpy as np


def _window_sums_axis0(arr: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    padded = np.vstack([np.zeros((1, arr.shape[1]), dtype=np.float64), np.cumsum(arr, axis=0)])
    return padded[ends, :] - padded[starts, :]


def _window_sums_axis1(arr: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    padded = np.hstack([np.zeros((arr.shape[0], 1), dtype=np.float64), np.cumsum(arr, axis=1)])
    return padded[:, ends] - padded[:, starts]


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

    sample_idx = np.arange(halfwid, totsamps - halfwid, dtype=np.int32)
    if sample_idx.size:
        starts = sample_idx - halfwid
        ends = sample_idx + halfwid + 1
        avg = _window_sums_axis0(arr, starts, ends) / (2 * halfwid + 1)
        result[sample_idx, :] = arr[sample_idx, :] - avg

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

    trace_idx = np.arange(halfwid, tottraces - halfwid, dtype=np.int32)
    if trace_idx.size:
        starts = trace_idx - halfwid
        ends = trace_idx + halfwid + 1
        avg = _window_sums_axis1(arr, starts, ends) / (2 * halfwid + 1)
        result[:, trace_idx] = arr[:, trace_idx] - avg

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

    sample_idx = np.arange(halfwid, totsamps - halfwid, dtype=np.int32)
    if sample_idx.size:
        starts = sample_idx - halfwid
        ends = sample_idx + halfwid + 1
        energy = np.maximum(np.sqrt(_window_sums_axis0(arr * arr, starts, ends)), eps)
        result[sample_idx, :] = np.divide(arr[sample_idx, :], energy)

    energy = np.maximum(np.linalg.norm(arr[totsamps - halfwid : totsamps + 1, :], axis=0), eps)
    result[totsamps - halfwid : totsamps + 1, :] = np.divide(
        arr[totsamps - halfwid : totsamps + 1, :],
        energy,
    )

    return result.astype(np.float32, copy=False)
