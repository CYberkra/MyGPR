#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UAV-GPR 飞行高度归一化模块

基于每道数据的飞行高度（flight_height_m），对 B-scan 做振幅校正和/或时移校正，
消除因天线离地高度变化引起的能量衰减和同相轴浮动。

输入要求：
- data: 二维 numpy 数组 (samples, traces)
- trace_metadata: 字典，必须包含 "flight_height_m" 键，值为每道的高度数组
"""

import numpy as np


def method_motion_compensation_height(
    data: np.ndarray,
    reference_height_mode: str = "mean",
    manual_height: float = 0.0,
    compensate_amplitude: bool = True,
    compensate_time_shift: bool = True,
    wave_speed_m_per_ns: float = 0.1,
    trace_metadata: dict | None = None,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """飞行高度归一化处理。

    Args:
        data: 输入 B-scan 数据，形状 (samples, traces)
        reference_height_mode: 参考高度选择，"mean"/"min"/"manual"
        manual_height: manual 模式下的参考高度（米）
        compensate_amplitude: 是否做振幅校正
        compensate_time_shift: 是否做时移校正
        wave_speed_m_per_ns: 电磁波速（米/纳秒），默认 0.1 m/ns
        trace_metadata: 每道元数据，必须包含 "flight_height_m"
        **kwargs: 兼容其他参数

    Returns:
        (corrected_data, meta)
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("飞行高度归一化需要二维 B-scan 数据")

    samples, traces = arr.shape
    meta = {"method": "motion_compensation_height"}

    if trace_metadata is None or "flight_height_m" not in trace_metadata:
        # 无高度信息时原样返回，避免阻断处理流程
        meta["skipped"] = True
        meta["reason"] = "缺少 trace_metadata['flight_height_m']"
        return arr.copy(), meta

    flight_height = np.asarray(trace_metadata["flight_height_m"], dtype=np.float32)
    if flight_height.size < traces:
        # 尝试用最后一个值填充或报错
        if flight_height.size == 0:
            meta["skipped"] = True
            meta["reason"] = "flight_height_m 为空"
            return arr.copy(), meta
        # 广播或补齐
        flight_height = np.resize(flight_height, traces)

    flight_height = flight_height[:traces]

    # 计算参考高度
    if reference_height_mode == "mean":
        h_ref = float(np.mean(flight_height))
    elif reference_height_mode == "min":
        h_ref = float(np.min(flight_height))
    elif reference_height_mode == "manual":
        h_ref = float(manual_height)
    else:
        h_ref = float(np.mean(flight_height))

    if h_ref <= 0:
        h_ref = 1.0  # 防止除零

    meta["reference_height_m"] = h_ref
    meta["input_height_min_m"] = float(np.min(flight_height))
    meta["input_height_max_m"] = float(np.max(flight_height))
    meta["input_height_mean_m"] = float(np.mean(flight_height))

    corrected = arr.copy()

    # 振幅校正：能量扩散与高度平方成反比
    if compensate_amplitude:
        amp_factors = (h_ref / flight_height) ** 2
        corrected = corrected * amp_factors[np.newaxis, :]
        meta["amplitude_correction"] = True

    # 时移校正：按双程时平移样点
    if compensate_time_shift and wave_speed_m_per_ns > 0:
        # 双程时差异（ns）
        delta_t_ns = 2.0 * (flight_height - h_ref) / wave_speed_m_per_ns
        meta["time_shift_correction"] = True
        meta["wave_speed_m_per_ns"] = wave_speed_m_per_ns

        # 计算每道采样间隔（ns）
        # 尝试从 kwargs 或 metadata 读取 time_window_ns 和 samples
        time_window_ns = kwargs.get("time_window_ns")
        if time_window_ns is None and trace_metadata is not None:
            time_window_ns = trace_metadata.get("time_window_ns")
        if time_window_ns is None:
            # 尝试从 header_info 推断，若无法获取则跳过时移
            time_window_ns = 100.0  # 给一个保守默认值并标记
            meta["time_shift_correction"] = False
            meta["time_shift_reason"] = "无法获取时窗信息，跳过时移校正"

        if meta.get("time_shift_correction"):
            dt_ns = float(time_window_ns) / max(samples - 1, 1)
            shifts_samples = delta_t_ns / dt_ns

            # 用线性插值做亚样点平移
            sample_indices = np.arange(samples, dtype=np.float32)
            for tr in range(traces):
                shift = float(shifts_samples[tr])
                if abs(shift) < 1e-3:
                    continue
                shifted_indices = sample_indices - shift
                # 边界外推
                shifted_indices = np.clip(shifted_indices, 0, samples - 1)
                corrected[:, tr] = np.interp(
                    sample_indices,
                    shifted_indices,
                    corrected[:, tr],
                )
            meta["max_shift_samples"] = float(np.max(np.abs(shifts_samples)))

    return corrected, meta
