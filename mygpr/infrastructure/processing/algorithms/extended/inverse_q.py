#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inverse-Q 衰减补偿 (Wang 2002 稳定化逆 Q 滤波) 的原生实现.

模型: constant-Q 衰减 (GPR 带宽内单参数 Q* 成立, Turner 1994 / Bano 1996)
    振幅衰减 A(t, f) = exp(-pi * t * f / Q)
    速度色散相位 exp(-i * 2 * t * f/Q * ln(f / f_ref))  (Wang 2002 式)
补偿 (Wang 2002 稳定化逆 Q 滤波, (t,f) 域逐道):
    G(f, t) = exp(pi * t * f / Q) * exp(i * 2 * t * f/Q * ln(f / f_ref))
f_ref = f_max / 2. 硬增益钳制: 任何 (t, f) 频点的总补偿增益不超过
G_MAX = 10^(gain_limit_db / 20), 超限频点钳到 G_MAX 并统计钳制比例.

内存设计 (契约 memory_multiplier=4.0): 不构造 n_samples x n_freq 的全
复核矩阵, 而是按输出时间行分块 (32 行) 的共轭对合成 —— 每块只持有
O(chunk x K) 的 (t, f) 核 (K = rfft 正频点数, 约为采样数一半), 负频
贡献以共轭对 (负频取共轭相位) 折入正频累加, 输出恒为实数.

时间基准契约: 必需时间基准缺失 (time_step_s / sample_interval_ns /
sample_rate_hz / sample_rate_mhz 参数与 header total_time_ns /
time_window_ns 均无) 时抛出中文 ValueError —— 不沿用
frequency_filter_1d 的静默 skip+warning 路线.

参考:
    Wang, Y. (2002). A stable and efficient approach to inverse Q
    filtering. Geophysics, 67(6), 1845-1847.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mygpr.domain.common.scalars import to_float
from mygpr.infrastructure.processing.algorithms.common import (
    ensure_matrix,
    normalize_output,
    warning,
)

METHOD_ID = "inverse_q"

_DEFAULT_Q = 50.0
_DEFAULT_GAIN_LIMIT_DB = 40.0
_MIN_GAIN_LIMIT_DB = 10.0
_MAX_GAIN_LIMIT_DB = 100.0
_MIN_Q = 1.0


def method_inverse_q(
    data: Any,
    q_value: float = _DEFAULT_Q,
    gain_limit_db: float = _DEFAULT_GAIN_LIMIT_DB,
    time_step_s: float | None = None,
    sample_interval_ns: float | None = None,
    sample_rate_hz: float | None = None,
    sample_rate_mhz: float | None = None,
    header_info: dict[str, Any] | None = None,
    trace_metadata: dict[str, Any] | None = None,
    **_: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """稳定化逆 Q 滤波: constant-Q 振幅恢复 + 硬增益钳制.

    q_value <= 0 或非有限 → ValueError (Q 必填语义). gain_limit_db 钳制在
    [10, 100] dB, 超范围自动钳制并给 warning.
    """
    del trace_metadata  # 当前 v1 不使用道元数据
    arr, warnings = ensure_matrix(data)

    q = to_float(q_value, default=0.0)
    if not np.isfinite(q) or q < _MIN_Q:
        raise ValueError(
            f"inverse_q 需要有效的 Q 值 (q_value >= {_MIN_Q:.0f}), 收到: {q_value!r}。"
        )
    requested_limit = to_float(gain_limit_db, default=_DEFAULT_GAIN_LIMIT_DB)
    gain_limit_db_eff = float(
        np.clip(requested_limit, _MIN_GAIN_LIMIT_DB, _MAX_GAIN_LIMIT_DB)
    )
    if gain_limit_db_eff != requested_limit:
        warnings.append(
            warning(
                "inverse_q_gain_limit_adjusted",
                "增益上限超出稳定化范围, 已钳制到 [10, 100] dB。",
                METHOD_ID,
                requested_db=float(requested_limit),
                applied_db=float(gain_limit_db_eff),
            )
        )

    dt_s = _resolve_time_step_s(
        samples=arr.shape[0],
        time_step_s=time_step_s,
        sample_interval_ns=sample_interval_ns,
        sample_rate_hz=sample_rate_hz,
        sample_rate_mhz=sample_rate_mhz,
        header_info=header_info,
    )
    return _compensate_constant_q(
        arr,
        q=q,
        gain_limit_db=gain_limit_db_eff,
        dt_s=dt_s,
        warnings=warnings,
    )


def _compensate_constant_q(
    arr: np.ndarray,
    *,
    q: float,
    gain_limit_db: float,
    dt_s: float,
    warnings: list[Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """constant-Q 补偿主体: 静默输入守卫 + 分块共轭对合成 (Wang 2002 式 5-7)."""
    samples = arr.shape[0]
    input_rms = float(np.sqrt(np.mean(arr**2)))
    if input_rms <= 1.0e-30:
        warnings.append(
            warning(
                "inverse_q_silent_input",
                "输入数据能量近零, 逆 Q 滤波按恒等处理。",
                METHOD_ID,
            )
        )
        metadata = {
            "method": METHOD_ID,
            "q_value": float(q),
            "gain_limit_db": float(gain_limit_db),
            "max_applied_gain_db": 0.0,
            "clamped_fraction": 0.0,
            "sample_rate_hz": float(1.0 / dt_s),
            "f_ref_hz": float(np.fft.rfftfreq(samples, d=dt_s)[-1] / 2.0),
            "skipped": True,
        }
        return normalize_output(METHOD_ID, arr.copy(), metadata, warnings)

    freqs = np.fft.rfftfreq(samples, d=dt_s)  # Hz (K = samples//2 + 1 个正频点)
    t = np.arange(samples, dtype=np.float64) * dt_s  # 秒
    gain_max = 10.0 ** (gain_limit_db / 20.0)
    f_max = float(freqs[-1])
    f_ref = 0.5 * f_max

    # 钳制统计随块累加, 从不物化全 (samples, K) 振幅网格.
    ln_ratio = np.log(np.maximum(freqs / f_ref, 1.0e-30))

    spectrum = np.fft.rfft(arr, axis=0)  # (K, traces) complex128
    recovered = np.empty_like(arr, dtype=np.float64)
    block = 32
    k_idx = np.arange(spectrum.shape[0], dtype=np.float64)  # 正频点索引
    clamped_cells = 0
    grid_cells = 0
    max_gain_seen = 1.0
    for start_row in range(0, samples, block):
        stop_row = min(samples, start_row + block)
        rows = t[start_row:stop_row, None] * freqs[None, :]  # (b, K)
        amp = np.exp(np.pi * rows / q)
        clamped_cells += int(np.count_nonzero(amp > gain_max))
        grid_cells += amp.size
        max_gain_seen = max(max_gain_seen, float(np.max(amp)))
        phase = np.exp(1j * 2.0 * rows / q * ln_ratio[None, :])  # 色散相位
        amp = np.minimum(amp, gain_max)
        kernel = amp * phase  # (b, K): 正频补偿核
        # 共轭对合成 (Wang 2002 式 5-7): E(n) = (2/N) Re Σ_k kernel·X_k·e^{i2πkn/N},
        # 色散相位共轭约定: 负频贡献取 conj(kernel), 实部相同.
        n_block = np.arange(start_row, stop_row)[:, None]  # 输出样本索引 (b, 1)
        basis = np.exp(1j * 2.0 * np.pi * n_block * k_idx[None, :] / samples)  # (b, K)
        weights = kernel * basis  # (b, K), 最大复块 O(b*K)
        recovered[start_row:stop_row] = 2.0 * np.real(weights @ spectrum) / samples
        # DC 与 Nyquist 项不乘 2
        recovered[start_row:stop_row] -= np.real(
            kernel[:, 0][:, None] * spectrum[0][None, :]
        ) / samples
        if samples % 2 == 0:
            recovered[start_row:stop_row] -= np.real(
                kernel[:, -1][:, None] * basis[:, -1][:, None] * spectrum[-1][None, :]
            ) / samples

    clamped_fraction = clamped_cells / max(grid_cells, 1)
    max_applied_gain_db = float(20.0 * np.log10(min(max_gain_seen, gain_max)))
    if clamped_fraction > 0.0:
        warnings.append(
            warning(
                "inverse_q_gain_clamped",
                f"部分频点补偿增益超过 {gain_limit_db:.0f} dB 上限, 已钳制。",
                METHOD_ID,
                clamped_fraction=clamped_fraction,
                gain_limit_db=float(gain_limit_db),
            )
        )

    metadata = {
        "method": METHOD_ID,
        "q_value": float(q),
        "gain_limit_db": float(gain_limit_db),
        "max_applied_gain_db": max_applied_gain_db,
        "clamped_fraction": clamped_fraction,
        "f_ref_hz": f_ref,
        "sample_rate_hz": float(1.0 / dt_s),
        "nyquist_hz": float(freqs[-1]),
        "post_advice": "建议后接 frequency_filter_1d 带通压制高频补偿噪声。",
    }
    return normalize_output(METHOD_ID, recovered, metadata, warnings)


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
        "Inverse-Q 滤波缺少时间基准: 无法解析采样间隔 dt。请提供 "
        "time_step_s 或 sample_interval_ns / sample_rate_hz / "
        "sample_rate_mhz 参数, 或传入含 total_time_ns / time_window_ns "
        "的 header_info。"
    )
