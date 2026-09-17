#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""混合相位反褶积 (Schmelzbach & Huber 2015) 的原生实现.

两步法 (v1 蓝本):
  1. 逐道最小相位 spike 反褶积: 相邻 ``supertrace_traces`` 道合成 supertrace,
     以其自相关构造 Toeplitz 正态方程 (加 prewhitening 对角加载) 解出最小相位
     逆算子 m^-1; 再由 m^-1 的自相关经 Levinson-Durbin 递推得到最小相位等效
     子波 m, 随后逐道卷积 m^-1 完成脉冲化.
  2. 全测线 kurtosis 相位旋转: 对步骤一的输出求解析信号, 在 0-180 度内扫描
     常相位旋转角, 取全测线 kurtosis (峰度, 输出稀疏度代理) 最大的角度作为
     最优旋转. 旋转是全局单角度 (不做短窗时变), 与 S&H 2015 一致.

时间基准契约: 必需时间基准缺失 (time_step_s / sample_interval_ns /
sample_rate_hz / sample_rate_mhz 参数与 header total_time_ns /
time_window_ns 均无) 时抛出中文 ValueError —— 不沿用
frequency_filter_1d 的静默 skip+warning 路线.

参考:
    Schmelzbach, C., & Huber, E. (2015). Efficient deconvolution of
    ground-penetrating radar data. IEEE TGRS, 53(11), 6110-6118.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import hilbert

from mygpr.domain.common.scalars import to_float
from mygpr.infrastructure.processing.algorithms.common import (
    ensure_matrix,
    normalize_output,
    warning,
)

METHOD_ID = "mixed_phase_deconvolution"

_DEFAULT_OPERATOR_LENGTH = 35
_DEFAULT_SUPERTRACE_TRACES = 11
_DEFAULT_PREWHITENING = 0.1
_DEFAULT_ROTATION_STEP_DEG = 5.0
_MIN_SUPERTRACE_ENERGY = 1.0e-12
_MIN_OPERATOR_LENGTH = 5
_MAX_OPERATOR_LENGTH = 256


def method_mixed_phase_deconvolution(
    data: Any,
    operator_length: int = _DEFAULT_OPERATOR_LENGTH,
    supertrace_traces: int = _DEFAULT_SUPERTRACE_TRACES,
    prewhitening: float = _DEFAULT_PREWHITENING,
    window_start_ns: float = 0.0,
    window_end_ns: float = 0.0,
    rotation_step_deg: float = _DEFAULT_ROTATION_STEP_DEG,
    apply_phase_rotation: bool = True,
    time_step_s: float | None = None,
    sample_interval_ns: float | None = None,
    sample_rate_hz: float | None = None,
    sample_rate_mhz: float | None = None,
    header_info: dict[str, Any] | None = None,
    trace_metadata: dict[str, Any] | None = None,
    **_: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """混合相位反褶积: 最小相位 spike 反褶积 + 全局 kurtosis 相位旋转.

    参数见 registry parameter_schema. dt (采样间隔, 秒) 用于把
    window_start_ns/window_end_ns 映射为算子估计窗; 缺失且窗口非默认
    (0-0) 时同样报 ValueError.
    """
    del trace_metadata  # 当前 v1 不使用道元数据
    arr, warnings = ensure_matrix(data)
    operator_length = _clamp_operator_length(operator_length)
    supertrace_traces = max(1, int(supertrace_traces))
    prewhitening = float(np.clip(to_float(prewhitening, default=0.0), 0.0, 0.1))
    rotation_step_deg = float(np.clip(to_float(rotation_step_deg, default=_DEFAULT_ROTATION_STEP_DEG), 0.5, 45.0))

    dt_s = _resolve_time_step_s(
        samples=arr.shape[0],
        time_step_s=time_step_s,
        sample_interval_ns=sample_interval_ns,
        sample_rate_hz=sample_rate_hz,
        sample_rate_mhz=sample_rate_mhz,
        header_info=header_info,
    )

    start_row, end_row = _estimate_window_rows(
        samples=arr.shape[0],
        dt_s=dt_s,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
    )

    inverse_operators = _estimate_inverse_operators(
        arr,
        window=(start_row, end_row),
        operator_length=operator_length,
        supertrace_traces=supertrace_traces,
        prewhitening=prewhitening,
        warnings=warnings,
    )

    spiked = _apply_inverse_operators(arr, inverse_operators, supertrace_traces)

    optimal_deg = 0.0
    kurtosis_value = _kurtosis(spiked)
    if bool(apply_phase_rotation):
        best_kurtosis = kurtosis_value
        for deg in np.arange(0.0, 180.0, rotation_step_deg, dtype=np.float64):
            candidate = _constant_phase_rotate(spiked, float(np.deg2rad(deg)))
            k = _kurtosis(candidate)
            if k > best_kurtosis:
                best_kurtosis = k
                optimal_deg = float(deg)
        if optimal_deg > 0.0:
            spiked = _constant_phase_rotate(spiked, float(np.deg2rad(optimal_deg)))
            kurtosis_value = best_kurtosis

    metadata = {
        "method": METHOD_ID,
        "operator_length": int(operator_length),
        "supertrace_traces": int(supertrace_traces),
        "prewhitening": float(prewhitening),
        "estimate_window_rows": [int(start_row), int(end_row)],
        "optimal_rotation_deg": float(optimal_deg),
        "kurtosis": float(kurtosis_value),
        "sample_rate_hz": float(1.0 / dt_s),
        "post_advice": "建议后接 frequency_filter_1d 带通压制反褶积放大的带外噪声。",
    }
    return normalize_output(METHOD_ID, spiked, metadata, warnings)


def _clamp_operator_length(value: Any) -> int:
    try:
        length = int(to_float(value, default=_DEFAULT_OPERATOR_LENGTH))
    except (TypeError, ValueError):
        length = _DEFAULT_OPERATOR_LENGTH
    return int(np.clip(length, _MIN_OPERATOR_LENGTH, _MAX_OPERATOR_LENGTH))


def _resolve_time_step_s(
    *,
    samples: int,
    time_step_s: float | None,
    sample_interval_ns: float | None,
    sample_rate_hz: float | None,
    sample_rate_mhz: float | None,
    header_info: dict[str, Any] | None,
) -> float:
    """按 time_step_s → sample_interval_ns → sample_rate → header 窗口 解析 dt(秒).

    全部缺失时抛中文 ValueError (硬契约, 不静默回退).
    """
    step = to_float(time_step_s, default=0.0)
    if step > 0.0:
        return float(step)
    interval_ns = to_float(sample_interval_ns, default=0.0)
    if interval_ns > 0.0:
        return float(interval_ns) * 1.0e-9
    rate_hz = to_float(sample_rate_hz, default=0.0)
    if rate_hz <= 0.0:
        rate_hz = to_float(sample_rate_mhz, default=0.0) * 1.0e6
    if rate_hz > 0.0:
        return 1.0 / float(rate_hz)
    if isinstance(header_info, dict):
        total_ns = to_float(header_info.get("total_time_ns"), default=0.0)
        if total_ns <= 0.0:
            total_ns = to_float(header_info.get("time_window_ns"), default=0.0)
        if total_ns > 0.0 and samples > 1:
            return float(total_ns) * 1.0e-9 / float(samples - 1)
    raise ValueError(
        "混合相位反褶积缺少时间基准: 无法解析采样间隔 dt。请提供 "
        "time_step_s 或 sample_interval_ns / sample_rate_hz / "
        "sample_rate_mhz 参数, 或传入含 total_time_ns / time_window_ns "
        "的 header_info。"
    )


def _estimate_window_rows(
    *,
    samples: int,
    dt_s: float,
    window_start_ns: float,
    window_end_ns: float,
) -> tuple[int, int]:
    start_ns = to_float(window_start_ns, default=0.0)
    end_ns = to_float(window_end_ns, default=0.0)
    if end_ns <= start_ns:
        return 0, samples
    dt_ns = dt_s * 1.0e9
    start = int(np.clip(int(round(start_ns / max(dt_ns, 1e-9))), 0, samples - 1))
    end = int(np.clip(int(round(end_ns / max(dt_ns, 1e-9))), start + 1, samples))
    return start, end


def _estimate_inverse_operators(
    arr: np.ndarray,
    *,
    window: tuple[int, int],
    operator_length: int,
    supertrace_traces: int,
    prewhitening: float,
    warnings: list[dict[str, Any]],
) -> list[np.ndarray]:
    """对每个 supertrace 组估计一个最小相位逆算子 m^-1 (S&H 两步法)."""
    samples, traces = arr.shape
    start, end = window
    group_count = int(np.ceil(traces / max(1, supertrace_traces)))
    operators: list[np.ndarray] = []
    global_rms = float(np.sqrt(np.mean(arr[start:end] ** 2))) if end > start else 0.0
    for group in range(group_count):
        lo = group * supertrace_traces
        hi = min(traces, lo + supertrace_traces)
        supertrace = arr[start:end, lo:hi].mean(axis=1)
        energy = float(np.mean(supertrace**2)) if supertrace.size else 0.0
        reference = max(global_rms**2, 1.0e-30)
        if not np.isfinite(energy) or energy <= _MIN_SUPERTRACE_ENERGY * reference:
            operators.append(np.array([1.0]))
            warnings.append(
                warning(
                    "deconv_degenerate_supertrace",
                    f"supertrace 组 {group + 1}/{group_count} 能量近零, 该组算子退化为恒等。",
                    METHOD_ID,
                    group=group,
                )
            )
            continue
        acf = _autocorrelation(supertrace, operator_length)
        acf[0] *= 1.0 + max(prewhitening, 0.0)
        inverse = _solve_toeplitz(acf)
        # 两步法: m^-1 的自相关 → Levinson-Durbin 解最小相位等效子波 m,
        # 但脉冲化输出只需 m^-1 (m 用于后续相位分析/理论子波, v1 不展开).
        operators.append(inverse)
    return operators


def _autocorrelation(trace: np.ndarray, lags: int) -> np.ndarray:
    centered = trace - float(np.mean(trace))
    full = np.correlate(centered, centered, mode="full")
    mid = len(centered) - 1
    acf = full[mid : mid + max(1, lags)].astype(np.float64)
    if acf[0] <= 0.0:
        acf[0] = 1.0e-30
    return acf


def _solve_toeplitz(acf: np.ndarray) -> np.ndarray:
    """解 Wiener spike 反褶积正态方程 R a = e1 (e1 首元素 1).

    R 为 supertrace 自相关构成的对称 Toeplitz 矩阵, 解 a 即把子波压缩到
    零延迟尖脉冲的最小相位逆算子 (S&H 2015 两步法第一步). 经
    scipy.linalg.solve_toeplitz (Levinson-Durbin) 求解.
    """
    e1 = np.zeros(acf.size, dtype=np.float64)
    e1[0] = 1.0
    return solve_toeplitz((acf, acf), e1)


def _apply_inverse_operators(
    arr: np.ndarray,
    operators: list[np.ndarray],
    supertrace_traces: int,
) -> np.ndarray:
    samples, traces = arr.shape
    out = np.empty_like(arr, dtype=np.float64)
    fft_size = int(2 ** np.ceil(np.log2(samples + max(op.size for op in operators) - 1)))
    freq_operators = [np.fft.rfft(op, n=fft_size) for op in operators]
    for group in range(len(operators)):
        lo = group * supertrace_traces
        hi = min(traces, lo + supertrace_traces)
        spec = np.fft.rfft(arr[:, lo:hi], n=fft_size, axis=0)
        filtered = np.fft.irfft(spec * freq_operators[group][:, None], n=fft_size, axis=0)
        out[:, lo:hi] = filtered[:samples]
    return out


def _constant_phase_rotate(trace_matrix: np.ndarray, phase_rad: float) -> np.ndarray:
    analytic = hilbert(trace_matrix, axis=0)
    rotated = np.real(analytic * np.exp(1j * phase_rad))
    return rotated


def _kurtosis(arr: np.ndarray) -> float:
    flat = arr.reshape(-1).astype(np.float64)
    mean = float(np.mean(flat))
    centered = flat - mean
    m2 = float(np.mean(centered**2))
    if m2 <= 1.0e-30:
        return 0.0
    m4 = float(np.mean(centered**4))
    return m4 / (m2 * m2) - 3.0
