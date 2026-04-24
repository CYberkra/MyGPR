#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UAV-GPR 轨迹平滑模块

对经纬度轨迹进行平滑滤波，消除 GPS 高频噪声，不改变 B-scan 振幅数据。
支持 Savitzky-Golay 滤波和滑动平均滤波。
"""

import numpy as np


def _ensure_odd_window(window: int, max_window: int) -> int:
    """确保窗口为奇数且不超过上限。"""
    w = int(window)
    w = min(w, max_window)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w -= 1
    return w


def _moving_average_smooth_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """一维滑动平均平滑，边界保持原始值。"""
    n = arr.size
    if window < 3 or n <= window:
        return arr.copy()
    half = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(arr.astype(np.float64), kernel, mode="same")
    # 边界用原始值填充
    smoothed[:half] = arr[:half]
    smoothed[-half:] = arr[-half:]
    return smoothed


def _savgol_smooth_1d(arr: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay 平滑，边界保持原始值。"""
    from scipy.signal import savgol_filter

    n = arr.size
    if n <= window_length or window_length < polyorder + 2:
        return arr.copy()
    smoothed = savgol_filter(arr.astype(np.float64), window_length, polyorder)
    half = window_length // 2
    smoothed[:half] = arr[:half]
    smoothed[-half:] = arr[-half:]
    return smoothed


def method_trajectory_smoothing(
    data: np.ndarray,
    method: str = "savgol",
    window_length: int = 21,
    polyorder: int = 3,
    trace_metadata: dict | None = None,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """轨迹平滑处理。

    Args:
        data: 输入 B-scan 数据，形状 (samples, traces)
        method: 平滑方法，"savgol" 或 "moving_average"
        window_length: Savitzky-Golay 窗口长度（奇数）
        polyorder: Savitzky-Golay 多项式阶数
        trace_metadata: 每道元数据，必须包含 "longitude" 和 "latitude"
        **kwargs: 兼容其他参数

    Returns:
        (data, meta) — 振幅数据原样返回，meta 记录平滑参数
    """
    arr = np.asarray(data, dtype=np.float32)
    meta = {"method": "trajectory_smoothing"}

    if trace_metadata is None or "longitude" not in trace_metadata or "latitude" not in trace_metadata:
        meta["skipped"] = True
        meta["reason"] = "缺少 trace_metadata['longitude'] 或 trace_metadata['latitude']"
        return arr.copy(), meta

    longitude = np.asarray(trace_metadata["longitude"], dtype=np.float64)
    latitude = np.asarray(trace_metadata["latitude"], dtype=np.float64)
    n = min(longitude.size, latitude.size, arr.shape[1])

    if n == 0:
        meta["skipped"] = True
        meta["reason"] = "轨迹数据为空"
        return arr.copy(), meta

    longitude = longitude[:n]
    latitude = latitude[:n]

    # 备份原始值
    if "longitude_raw" not in trace_metadata:
        trace_metadata["longitude_raw"] = longitude.copy()
    if "latitude_raw" not in trace_metadata:
        trace_metadata["latitude_raw"] = latitude.copy()

    # 自动限制窗口大小（不超过总道数的 1/5，且至少为 3）
    max_window = max(3, int(n // 5) | 1)  # 确保奇数

    if method == "savgol":
        wl = _ensure_odd_window(window_length, max_window)
        po = max(1, min(int(polyorder), wl - 2))
        lon_smooth = _savgol_smooth_1d(longitude, wl, po)
        lat_smooth = _savgol_smooth_1d(latitude, wl, po)
        meta["window_length"] = int(wl)
        meta["polyorder"] = int(po)
    elif method == "moving_average":
        wl = _ensure_odd_window(window_length, max_window)
        lon_smooth = _moving_average_smooth_1d(longitude, wl)
        lat_smooth = _moving_average_smooth_1d(latitude, wl)
        meta["window_length"] = int(wl)
    else:
        meta["skipped"] = True
        meta["reason"] = f"不支持的平滑方法: {method}"
        return arr.copy(), meta

    # 更新 metadata
    trace_metadata["longitude"] = lon_smooth.astype(np.float64, copy=False)
    trace_metadata["latitude"] = lat_smooth.astype(np.float64, copy=False)

    # 计算平滑前后偏移量
    dx = lon_smooth - longitude
    dy = lat_smooth - latitude
    displacements_m = np.sqrt(dx ** 2 + dy ** 2) * 111320.0  # 粗略换算为米（纬度方向）
    meta["max_displacement_m"] = float(np.max(displacements_m))
    meta["mean_displacement_m"] = float(np.mean(displacements_m))
    meta["smoothed_traces"] = int(n)

    return arr.copy(), meta
