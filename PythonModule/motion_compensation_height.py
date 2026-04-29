#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UAV-GPR 飞行高度归一化模块

基于每道数据的飞行高度（flight_height_m），对 B-scan 做振幅校正和/或时移校正，
减弱因天线离地高度变化引起的能量起伏和同相轴漂移。

输入要求：
- data: 二维 numpy 数组 (samples, traces)
- trace_metadata: 字典，必须包含 "flight_height_m" 键，值为每道的高度数组
- time_window_ns: 通过 runtime contract 注入（Task 1），也可从 trace_metadata 读取

V1 说明：
- `wave_speed_m_per_ns=0.1` 是当前实现与 benchmark 对齐使用的传播速度常数。
- 该常数用于保持既有数值行为稳定，不应在文档中自动等同为自由空间 / 空气段物理波速。
"""

from __future__ import annotations

import numpy as np


def method_motion_compensation_height(
    data: np.ndarray,
    reference_height_mode: str = "mean",
    manual_height: float = 0.0,
    compensate_amplitude: bool = True,
    compensate_time_shift: bool = True,
    wave_speed_m_per_ns: float = 0.1,
    max_shift_samples: float | None = None,
    interpolation_mode: str = "linear",
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
        wave_speed_m_per_ns: 当前实现使用的传播速度常数（米/纳秒），默认 0.1 m/ns
        max_shift_samples: 时移样点上限；None 表示不限制
        interpolation_mode: 插值模式；V1 仅支持 "linear"
        trace_metadata: 每道元数据，必须包含 "flight_height_m"
        **kwargs: 兼容其他参数，优先使用 kwargs["time_window_ns"]

    Returns:
        (corrected_data, meta)
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("飞行高度归一化需要二维 B-scan 数据")

    samples, traces = arr.shape
    meta: dict[str, object] = {
        "method": "motion_compensation_height",
        "compensate_amplitude": bool(compensate_amplitude),
        "compensate_time_shift": bool(compensate_time_shift),
        "wave_speed_m_per_ns": float(wave_speed_m_per_ns),
        "max_shift_samples": max_shift_samples,
        "interpolation_mode": str(interpolation_mode),
    }

    # 1. 缺失 metadata 时安全跳过
    if trace_metadata is None or "flight_height_m" not in trace_metadata:
        meta["skipped"] = True
        meta["reason"] = "缺少 trace_metadata['flight_height_m']"
        return arr.copy(), meta

    flight_height = np.asarray(trace_metadata["flight_height_m"], dtype=np.float32)
    if flight_height.ndim != 1:
        meta["skipped"] = True
        meta["reason"] = "flight_height_m 必须为一维数组"
        meta["input_height_valid"] = False
        return arr.copy(), meta
    if flight_height.size == 0:
        meta["skipped"] = True
        meta["reason"] = "flight_height_m 为空"
        meta["input_height_valid"] = False
        return arr.copy(), meta
    if flight_height.size != traces:
        meta["skipped"] = True
        meta["reason"] = (
            f"flight_height_m 长度与道数不一致：metadata={flight_height.size}, traces={traces}"
        )
        meta["input_height_valid"] = False
        meta["height_length_mismatch"] = True
        meta["metadata_trace_count"] = int(flight_height.size)
        meta["data_trace_count"] = int(traces)
        return arr.copy(), meta

    # 2. 显式校验非正或 NaN 高度
    if np.any(np.isnan(flight_height)):
        meta["skipped"] = True
        meta["reason"] = "flight_height_m 包含 NaN 值"
        meta["input_height_valid"] = False
        return arr.copy(), meta
    if np.any(flight_height <= 0):
        meta["skipped"] = True
        meta["reason"] = "flight_height_m 包含零或负值"
        meta["input_height_valid"] = False
        return arr.copy(), meta

    meta["input_height_valid"] = True

    # 3. 计算参考高度
    if reference_height_mode == "mean":
        h_ref = float(np.mean(flight_height))
    elif reference_height_mode == "min":
        h_ref = float(np.min(flight_height))
    elif reference_height_mode == "manual":
        h_ref = float(manual_height)
    else:
        h_ref = float(np.mean(flight_height))

    if h_ref <= 0:
        h_ref = 1.0

    meta["reference_height_m"] = h_ref
    meta["input_height_min_m"] = float(np.min(flight_height))
    meta["input_height_max_m"] = float(np.max(flight_height))
    meta["input_height_mean_m"] = float(np.mean(flight_height))
    meta["input_height_std_m"] = float(np.std(flight_height))

    corrected = arr.copy()

    # 4. 振幅校正：基于高度变化的近似能量归一化
    # 这里使用工程化的 h^2 比例做参考高度归一化，用于减弱高度起伏带来的
    # 整体能量变化；它不等同于完整雷达方程补偿。
    if compensate_amplitude:
        amp_factors = (flight_height / h_ref) ** 2
        corrected = corrected * amp_factors[np.newaxis, :]
        meta["amplitude_correction_applied"] = True
    else:
        meta["amplitude_correction_applied"] = False

    # 5. 时移校正
    if compensate_time_shift and wave_speed_m_per_ns > 0:
        if interpolation_mode != "linear":
            raise ValueError(
                f"interpolation_mode '{interpolation_mode}' 不受支持；V1 仅支持 'linear'"
            )

        # 这里沿用当前 V1/benchmark 的传播速度常数，保持数值行为与既有
        # preset / benchmark / evidence 一致。
        delta_t_ns = 2.0 * (flight_height - h_ref) / wave_speed_m_per_ns
        meta["time_shift_correction_applied"] = True

        # 按 Task 1 runtime contract 读取 time_window_ns
        time_window_ns = kwargs.get("time_window_ns")
        if time_window_ns is None and trace_metadata is not None:
            time_window_ns = trace_metadata.get("time_window_ns")
        if time_window_ns is None:
            meta["time_shift_correction_applied"] = False
            meta["time_shift_skip_reason"] = "无法获取时窗信息（time_window_ns），跳过时移校正"
        else:
            meta["time_window_ns"] = float(time_window_ns)
            dt_ns = float(time_window_ns) / max(samples - 1, 1)
            shifts_samples = delta_t_ns / dt_ns

            # 6. shift clamp
            if max_shift_samples is not None and max_shift_samples > 0:
                clamp = float(max_shift_samples)
                shifts_clamped = np.clip(shifts_samples, -clamp, clamp)
                meta["max_shift_samples_applied"] = float(np.max(np.abs(shifts_clamped)))
                meta["shift_clamped"] = not np.allclose(shifts_samples, shifts_clamped)
                shifts_samples = shifts_clamped
            else:
                meta["max_shift_samples_applied"] = float(np.max(np.abs(shifts_samples)))
                meta["shift_clamped"] = False

            sample_indices = np.arange(samples, dtype=np.float32)
            for tr in range(traces):
                shift = float(shifts_samples[tr])
                if abs(shift) < 1e-3:
                    continue
                shifted_indices = sample_indices - shift
                shifted_indices = np.clip(shifted_indices, 0, samples - 1)
                corrected[:, tr] = np.interp(
                    sample_indices,
                    shifted_indices,
                    corrected[:, tr],
                )
    else:
        meta["time_shift_correction_applied"] = False

    return corrected, meta
