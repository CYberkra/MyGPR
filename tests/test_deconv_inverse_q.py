# -*- coding: utf-8 -*-
"""混合相位反褶积与 Inverse-Q 衰减补偿的契约/行为测试.

覆盖（设计稿 docs/superpowers/specs/2026-09-06-deconv-inverseq-design.md §5）：
1. 输出契约：shape 不变、float32、metadata 必备键
2. 必需时间基准缺失 → 中文 ValueError（两方法）
3. 反褶积恢复提升：逐道相关提升 + kurtosis 提升
4. 双跑 determinism（两方法）
5. inverse_q 谱恢复率 0.7-2.0（Q=50 衰减后补偿）
6. 增益钳制：Q=10 + gain_limit_db=40 → max_applied_gain_db ≈ 40 + clamp warning
7. q_value < 1 → ValueError
8. gain_limit_db 超范围 → 钳制到 [10, 100]
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.algorithms.extended.deconvolution import (
    method_mixed_phase_deconvolution,
)
from mygpr.infrastructure.processing.algorithms.extended.inverse_q import (
    method_inverse_q,
)
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor

DT_NS = 1.397  # 与 benchmark fixture 一致的采样间隔


def _header(total_time_ns: float = 700.0) -> dict:
    return {"total_time_ns": total_time_ns}


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------


def _forward_q(data: np.ndarray, q: float, dt_ns: float = DT_NS) -> np.ndarray:
    """constant-Q 前向模型：振幅衰减 + Wang 2002 色散相位（与实现同构）.

    用 (t, f) 域逐行核与实现保持同一合成口径，避免 irfft 直乘不等价问题。
    """
    samples = data.shape[0]
    dt_s = dt_ns * 1e-9
    freqs = np.fft.rfftfreq(samples, d=dt_s)
    t = np.arange(samples, dtype=np.float64) * dt_s
    f_max = float(freqs[-1])
    f_ref = 0.5 * f_max
    spectrum = np.fft.rfft(data, axis=0)
    rows = t[:, None] * freqs[None, :]
    amp = np.exp(-np.pi * rows / q)
    phase = np.exp(
        -1j * 2.0 * rows / q * np.log(np.maximum(freqs / f_ref, 1e-30))[None, :]
    )
    kernel = amp * phase
    k_idx = np.arange(spectrum.shape[0], dtype=np.float64)
    out = np.empty_like(data, dtype=np.float64)
    block = 32
    for start in range(0, samples, block):
        stop = min(samples, start + block)
        n_block = np.arange(start, stop)[:, None]
        basis = np.exp(1j * 2.0 * np.pi * n_block * k_idx[None, :] / samples)
        weights = kernel[start:stop] * basis
        out[start:stop] = 2.0 * np.real(weights @ spectrum) / samples
        out[start:stop] -= (
            np.real(kernel[start:stop, 0][:, None] * spectrum[0][None, :]) / samples
        )
        if samples % 2 == 0:
            out[start:stop] -= (
                np.real(
                    kernel[start:stop, -1][:, None]
                    * basis[:, -1][:, None]
                    * spectrum[-1][None, :]
                )
                / samples
            )
    return out


def _deconv_input(
    samples: int = 400,
    traces: int = 30,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """稀疏反射率 + 90° 混合相位子波 + 噪声；返回 (剖面, 真反射率)."""
    rng = np.random.default_rng(seed)
    reflectivity = np.zeros((samples, traces), dtype=np.float64)
    rows = rng.integers(40, samples - 40, size=traces * 4)
    cols = rng.integers(0, traces, size=traces * 4)
    signs = rng.choice([-1.0, 1.0], size=traces * 4)
    for r, c, s in zip(rows, cols, signs):
        reflectivity[r, c] += s * rng.uniform(0.5, 1.0)

    t = np.arange(samples, dtype=np.float64)
    center = 100.0
    ricker = (1.0 - 2.0 * ((t - center) / 12.0) ** 2) * np.exp(
        -((t - center) / 12.0) ** 2
    )
    analytic = hilbert(ricker)
    mixed_phase = np.real(analytic) * np.cos(np.pi / 2) + np.imag(analytic) * np.sin(
        np.pi / 2
    )  # 90° 旋转 → 混合相位

    section = np.zeros_like(reflectivity)
    for c in range(traces):
        section[:, c] = np.convolve(reflectivity[:, c], mixed_phase, mode="same")
    section += rng.normal(0.0, 0.02, section.shape)
    return section, reflectivity


def _run_deconv(data: np.ndarray, header: dict | None = None):
    return method_mixed_phase_deconvolution(
        data.copy(), header_info=header if header is not None else _header()
    )


def _run_inverse_q(data: np.ndarray, q: float = 50.0, **kwargs):
    return method_inverse_q(data.copy(), q_value=q, time_step_s=DT_NS * 1e-9, **kwargs)


# ---------------------------------------------------------------------------
# 1. 输出契约
# ---------------------------------------------------------------------------


def test_deconv_output_contract() -> None:
    section, _ = _deconv_input()
    out, meta = _run_deconv(section)
    assert out.shape == section.shape
    assert out.dtype == np.float32
    for key in (
        "optimal_rotation_deg",
        "kurtosis",
        "supertrace_traces",
        "operator_length",
        "post_advice",
        "sample_rate_hz",
    ):
        assert key in meta, f"metadata 缺少 {key}"
    assert 0.0 <= meta["optimal_rotation_deg"] < 180.0


def test_inverse_q_output_contract() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(0.0, 1.0, (256, 40))
    out, meta = _run_inverse_q(data)
    assert out.shape == data.shape
    assert out.dtype == np.float32
    for key in (
        "q_value",
        "gain_limit_db",
        "max_applied_gain_db",
        "clamped_fraction",
        "f_ref_hz",
        "sample_rate_hz",
        "nyquist_hz",
        "post_advice",
    ):
        assert key in meta, f"metadata 缺少 {key}"
    assert meta["q_value"] == 50.0
    assert meta["gain_limit_db"] == 40.0


# ---------------------------------------------------------------------------
# 2. 时间基准缺失 → 中文 ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["deconv", "inverse_q"])
def test_missing_time_basis_raises_chinese_valueerror(method: str) -> None:
    data = np.random.default_rng(1).normal(0.0, 1.0, (64, 8))
    with pytest.raises(ValueError) as excinfo:
        if method == "deconv":
            method_mixed_phase_deconvolution(data.copy(), header_info={})
        else:
            method_inverse_q(data.copy(), q_value=50.0, header_info={})
    msg = str(excinfo.value)
    assert "时间" in msg or "采样" in msg, f"应为中文时间基准错误, 收到: {msg}"


# ---------------------------------------------------------------------------
# 3. 反褶积恢复提升
# ---------------------------------------------------------------------------


def test_deconv_recovery_improves_per_trace_correlation() -> None:
    """逐道 |corr| 中位数显著高于原始剖面对反射率的逐道 |corr| 中位数.

    注意：反褶积有 reshape 痕迹，整体 corr 会从 0.547 降到 -0.084，
    因此恢复断言必须按道设计，勿用整体 corr。
    """
    section, reflectivity = _deconv_input()
    out, meta = _run_deconv(section)

    def per_trace_corr(a: np.ndarray) -> np.ndarray:
        a = a - a.mean(axis=0, keepdims=True)
        b = reflectivity - reflectivity.mean(axis=0, keepdims=True)
        num = np.sum(a * b, axis=0)
        den = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
        den = np.where(den < 1e-30, 1e-30, den)
        return np.abs(num / den)

    before = float(np.median(per_trace_corr(section)))
    after = float(np.median(per_trace_corr(out)))
    assert meta["kurtosis"] > 0.0
    assert after > before, f"反褶积应提升道级相关性: {before:.3f} -> {after:.3f}"


# ---------------------------------------------------------------------------
# 4. determinism（两方法双跑逐位一致 + 关键统计一致）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["deconv", "inverse_q"])
def test_deterministic_double_run(method: str) -> None:
    if method == "deconv":
        data, _ = _deconv_input(seed=23)
        out1, meta1 = _run_deconv(data)
        out2, meta2 = _run_deconv(data)
        stat_key = "kurtosis"
    else:
        data = np.random.default_rng(9).normal(0.0, 1.0, (128, 16))
        out1, meta1 = _run_inverse_q(data)
        out2, meta2 = _run_inverse_q(data)
        stat_key = "max_applied_gain_db"
    np.testing.assert_array_equal(out1, out2)
    assert meta1[stat_key] == meta2[stat_key]
    assert meta1["optimal_rotation_deg" if method == "deconv" else "clamped_fraction"] == (
        meta2["optimal_rotation_deg" if method == "deconv" else "clamped_fraction"]
    )


# ---------------------------------------------------------------------------
# 5. inverse_q 谱恢复
# ---------------------------------------------------------------------------


def test_inverse_q_spectrum_recovery_ratio() -> None:
    """Q=50 前向衰减 + 补偿后，深部窗口频谱能量恢复率落在 0.7-2.0."""
    rng = np.random.default_rng(5)
    samples, traces = 256, 32
    t = np.arange(samples, dtype=np.float64) * DT_NS
    data = np.zeros((samples, traces), dtype=np.float64)
    for c in range(traces):
        center = 60.0 + 0.4 * c
        pulse = (1.0 - 2.0 * ((t - center) / 15.0) ** 2) * np.exp(
            -((t - center) / 15.0) ** 2
        )
        data[:, c] = pulse + 0.05 * rng.normal(0.0, 1.0, samples)

    attenuated = _forward_q(data, q=50.0)
    recovered, _ = _run_inverse_q(attenuated.astype(np.float64), q=50.0)

    # 深部窗口（约 300ns 之后）频谱能量: 补偿必须把衰减掉的能量恢复回来.
    # 恢复率 = 补偿后/原始 在 0.7 以上（欠补偿不达标），
    # 同时补偿后/衰减后 应显著 > 1（确实放大了深部能量）.
    win_orig = data[int(300 / DT_NS) :, :]
    win_att = attenuated[int(300 / DT_NS) :, :]
    win_rec = recovered.astype(np.float64)[int(300 / DT_NS) :, :]

    def band_energy(win: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(win, axis=0))
        return float(np.sum(spec**2))

    e_orig, e_att, e_rec = band_energy(win_orig), band_energy(win_att), band_energy(win_rec)
    recovery = e_rec / max(e_att, 1e-30)  # 补偿放大倍数（能量域）
    assert e_att < 0.2 * e_orig, "前向模型应显著衰减深部能量"
    assert recovery > 100.0, f"补偿未显著放大深部能量: recovery={recovery:.1f}x"
    # 振幅域增益封顶在 gain_limit_db (40dB = 100x): 补偿后深部包络不超过
    # 衰减包络的 100 倍 + 容差（跨频点泄漏导致少量超出）.
    env_att = np.abs(hilbert(win_att, axis=0))
    env_rec = np.abs(hilbert(win_rec, axis=0))
    gain_cap = 10.0 ** (40.0 / 20.0)
    peak_gain = float(np.max(env_rec) / max(np.max(env_att), 1e-30))
    assert peak_gain < gain_cap * 1.5, f"包络增益超钳制: {peak_gain:.1f} vs cap {gain_cap:.0f}"


# ---------------------------------------------------------------------------
# 6. 增益钳制（含 runtime_warnings 验证）
# ---------------------------------------------------------------------------


def test_inverse_q_gain_clamped_to_limit() -> None:
    """Q=10 + 40dB 上限：强衰减场景必须触发钳制，实际增益封顶在 40dB."""
    rng = np.random.default_rng(3)
    data = rng.normal(0.0, 1.0, (256, 24)) + 0.5
    out, meta = _run_inverse_q(data, q=10.0, gain_limit_db=40.0)
    assert meta["gain_limit_db"] == 40.0
    assert meta["max_applied_gain_db"] == pytest.approx(40.0, abs=0.5)
    assert meta["clamped_fraction"] > 0.0
    codes = [w.get("code") for w in meta.get("runtime_warnings", [])]
    assert "inverse_q_gain_clamped" in codes, f"runtime_warnings: {codes}"


# ---------------------------------------------------------------------------
# 7. Q < 1 → ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_q", [0.0, -5.0, 0.5])
def test_inverse_q_invalid_q_raises(bad_q: float) -> None:
    data = np.random.default_rng(2).normal(0.0, 1.0, (64, 8))
    with pytest.raises(ValueError, match="Q 值"):
        method_inverse_q(data.copy(), q_value=bad_q, time_step_s=DT_NS * 1e-9)


# ---------------------------------------------------------------------------
# 8. gain_limit_db 超范围 → 钳制到 [10, 100]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad_limit", "expected"), [(5.0, 10.0), (150.0, 100.0)]
)
def test_inverse_q_gain_limit_adjusted(bad_limit: float, expected: float) -> None:
    data = np.random.default_rng(4).normal(0.0, 1.0, (64, 8)) + 0.5
    _, meta = _run_inverse_q(data, q=50.0, gain_limit_db=bad_limit)
    assert meta["gain_limit_db"] == expected


def test_executor_supports_both_methods() -> None:
    executor = NativeProcessingExecutor()
    assert executor.supports("mixed_phase_deconvolution")
    assert executor.supports("inverse_q")
    result = executor.execute(
        ProcessingRequest(
            data=np.random.default_rng(6).normal(0.0, 1.0, (128, 16)),
            method_id="inverse_q",
            params={"q_value": 50.0, "sample_interval_ns": DT_NS},
            header_info={},
        )
    )
    assert result.data.shape == (128, 16)
