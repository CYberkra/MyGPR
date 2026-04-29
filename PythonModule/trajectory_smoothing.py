#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UAV-GPR 轨迹平滑模块（V1 横向几何修正）

对经纬度轨迹进行平滑滤波，消除 GPS 高频噪声，不改变 B-scan 振幅数据。
支持 Savitzky-Golay 滤波和滑动平均滤波。

V1 行为：
- 不原地修改传入的 trace_metadata
- 返回 trace_metadata_updates，包含平滑后的 local_x_m / local_y_m /
  trace_distance_m 以及位移指标
- 保留原始 lon/lat 到 longitude_raw / latitude_raw
"""

from __future__ import annotations

import numpy as np

from core.trace_metadata_utils import derive_local_xy_m  # type: ignore[import]


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
    smoothed = np.asarray(
        savgol_filter(arr.astype(np.float64), window_length, polyorder),
        dtype=np.float64,
    )
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
    """轨迹平滑处理（V1 横向几何修正）。

    Args:
        data: 输入 B-scan 数据，形状 (samples, traces)
        method: 平滑方法，"savgol" 或 "moving_average"
        window_length: Savitzky-Golay 窗口长度（奇数）
        polyorder: Savitzky-Golay 多项式阶数
        trace_metadata: 每道元数据，必须包含 "longitude" 和 "latitude"
        **kwargs: 兼容其他参数

    Returns:
        (data_copy, meta) — 振幅数据拷贝返回；meta 包含平滑参数、
        位移指标以及 trace_metadata_updates（非原地修改）。
    """
    arr = np.asarray(data, dtype=np.float32)
    meta: dict[str, object] = {"method": "trajectory_smoothing"}

    if trace_metadata is None or "longitude" not in trace_metadata or "latitude" not in trace_metadata:
        meta["skipped"] = True
        meta["reason"] = "缺少 trace_metadata['longitude'] 或 trace_metadata['latitude']"
        return arr.copy(), meta

    longitude = np.asarray(trace_metadata["longitude"], dtype=np.float64)
    latitude = np.asarray(trace_metadata["latitude"], dtype=np.float64)

    if longitude.ndim != 1 or latitude.ndim != 1:
        meta["skipped"] = True
        meta["reason"] = "longitude/latitude 必须为一维数组"
        return arr.copy(), meta

    trace_count = int(arr.shape[1])
    if longitude.size != latitude.size:
        meta["skipped"] = True
        meta["reason"] = (
            f"longitude/latitude 长度不一致：longitude={longitude.size}, latitude={latitude.size}"
        )
        meta["metadata_length_mismatch"] = True
        return arr.copy(), meta

    if longitude.size == 0:
        meta["skipped"] = True
        meta["reason"] = "轨迹数据为空"
        return arr.copy(), meta

    if longitude.size != trace_count:
        meta["skipped"] = True
        meta["reason"] = (
            f"轨迹元数据长度与道数不一致：metadata={longitude.size}, traces={trace_count}"
        )
        meta["metadata_length_mismatch"] = True
        meta["metadata_trace_count"] = int(longitude.size)
        meta["data_trace_count"] = trace_count
        return arr.copy(), meta

    n = trace_count

    if n < 3:
        meta["skipped"] = True
        meta["reason"] = "轨迹数据过短（少于3道）"
        return arr.copy(), meta

    # 保留原始 lon/lat（若输入已含 _raw 则继承，否则以当前输入为准）
    longitude_raw = (
        np.asarray(trace_metadata["longitude_raw"], dtype=np.float64).copy()
        if "longitude_raw" in trace_metadata
        else longitude.copy()
    )
    latitude_raw = (
        np.asarray(trace_metadata["latitude_raw"], dtype=np.float64).copy()
        if "latitude_raw" in trace_metadata
        else latitude.copy()
    )

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

    # 使用统一原点的局部切平面 XY（米）
    lon0 = float(longitude[0])
    lat0 = float(latitude[0])
    local_x_raw, local_y_raw = derive_local_xy_m(
        longitude, latitude, origin_longitude=lon0, origin_latitude=lat0
    )
    local_x_smooth, local_y_smooth = derive_local_xy_m(
        lon_smooth, lat_smooth, origin_longitude=lon0, origin_latitude=lat0
    )

    # 沿平滑路径的累积距离
    dx_dist = np.diff(local_x_smooth)
    dy_dist = np.diff(local_y_smooth)
    trace_distance_m = np.concatenate(
        ([0.0], np.cumsum(np.sqrt(dx_dist ** 2 + dy_dist ** 2)))
    ).astype(np.float64)

    # 计算平滑前后偏移量（米）
    dx = local_x_smooth - local_x_raw
    dy = local_y_smooth - local_y_raw
    displacements_m = np.sqrt(dx ** 2 + dy ** 2)
    meta["max_displacement_m"] = float(np.max(displacements_m))
    meta["mean_displacement_m"] = float(np.mean(displacements_m))
    meta["smoothed_traces"] = int(n)

    # 构造非原地修改的 updates
    trace_metadata_updates = {
        "longitude_raw": longitude_raw,
        "latitude_raw": latitude_raw,
        "longitude": lon_smooth.astype(np.float64, copy=False),
        "latitude": lat_smooth.astype(np.float64, copy=False),
        "local_x_m": local_x_smooth.astype(np.float64, copy=False),
        "local_y_m": local_y_smooth.astype(np.float64, copy=False),
        "trace_distance_m": trace_distance_m,
    }
    meta["trace_metadata_updates"] = trace_metadata_updates

    return arr.copy(), meta
