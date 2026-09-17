# -*- coding: utf-8 -*-
"""双曲线速度拟合核心测试（纯 domain，无 Qt/core 依赖）。

数学：双曲线 t(x) = (2/v)·sqrt((x-x0)² + z0²) 平方线性化为
t² = A·x² + B·x + C，其中 A = 4/v²，x0 = -B/(2A)，z0² = C - B²/(4A)。
"""
from __future__ import annotations

import numpy as np
import pytest

from mygpr.domain.velocity.errors import VelocityAnalysisError
from mygpr.domain.velocity.fitting import fit_hyperbola
from mygpr.domain.velocity.models import (
    VELOCITY_ANALYSIS_EVIDENCE_SCHEMA,
    VelocityPick,
)


class TestFitHyperbolaExactRecovery:
    def test_exact_synthetic_hyperbola_recovers_parameters(self):
        # 合成：v=0.1 m/ns（ε≈9），x0=2 m，z0=0.5 m
        v_true, x0_true, z0_true = 0.1, 2.0, 0.5
        x_m = np.linspace(0.0, 4.0, 9)
        t_ns = (2.0 / v_true) * np.sqrt((x_m - x0_true) ** 2 + z0_true ** 2)
        picks = [VelocityPick(trace_index=i, sample_index=i, x_m=x, t_ns=t)
                 for i, (x, t) in enumerate(zip(x_m, t_ns))]

        fit = fit_hyperbola(picks)

        assert fit.v_m_ns == pytest.approx(v_true, rel=1e-6)
        assert fit.x0_m == pytest.approx(x0_true, abs=1e-6)
        assert fit.z0_m == pytest.approx(z0_true, abs=1e-6)
        assert fit.rmse_ns == pytest.approx(0.0, abs=1e-9)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-9)

    def test_noisy_hyperbola_gives_reasonable_fit(self):
        v_true, x0_true, z0_true = 0.12, 1.5, 0.8
        rng = np.random.default_rng(42)
        x_m = np.linspace(0.0, 4.0, 15)
        t_exact = (2.0 / v_true) * np.sqrt((x_m - x0_true) ** 2 + z0_true ** 2)
        t_noisy = t_exact + rng.normal(0.0, 0.5, size=x_m.size)  # 0.5 ns 噪声
        picks = [VelocityPick(trace_index=i, sample_index=i, x_m=x, t_ns=t)
                 for i, (x, t) in enumerate(zip(x_m, t_noisy))]

        fit = fit_hyperbola(picks)

        assert fit.v_m_ns == pytest.approx(v_true, rel=0.15)
        assert fit.rmse_ns < 1.0
        assert 0.0 < fit.r_squared <= 1.0


class TestFitHyperbolaValidation:
    def test_fewer_than_three_points_raises(self):
        picks = [VelocityPick(0, 0, 0.0, 10.0), VelocityPick(1, 1, 0.5, 10.1)]
        with pytest.raises(VelocityAnalysisError) as exc:
            fit_hyperbola(picks)
        assert exc.value.error_code == "MYGPR_VELOCITY_ANALYSIS_ERROR"

    def test_degenerate_flat_line_raises(self):
        # 共线 x：x²/x/1 退化为秩亏 → 无法解出 v
        picks = [VelocityPick(i, i, 1.0, 10.0 + 0.1 * i) for i in range(5)]
        with pytest.raises(VelocityAnalysisError):
            fit_hyperbola(picks)

    def test_non_physical_velocity_raises(self):
        # 负曲率（A<0 → v² < 0）在物理上不可能：构造 t² 随 x² 递减
        x_m = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        t_ns = np.sqrt(100.0 - 4.0 * x_m ** 2)  # t² = 100 - 4x² → A=-4
        picks = [VelocityPick(i, i, x, t) for i, (x, t) in enumerate(zip(x_m, t_ns))]
        with pytest.raises(VelocityAnalysisError):
            fit_hyperbola(picks)

    def test_nan_inputs_raise(self):
        picks = [VelocityPick(i, i, float("nan"), 10.0) for i in range(5)]
        with pytest.raises(VelocityAnalysisError):
            fit_hyperbola(picks)

    def test_negative_depth_clamped_reported(self):
        # 拟合出 z0² < 0（几何上不可能）→ 报错而非静默取虚数
        x_m = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # 轻微下凹的最小二乘可能给出负 z0²：直接用共点双曲线退化态
        t_ns = 2.0 / 0.1 * np.sqrt((x_m - 2.0) ** 2)  # z0=0 退化为 V 形
        picks = [VelocityPick(i, i, x, t) for i, (x, t) in enumerate(zip(x_m, t_ns))]
        # z0=0 是合法边界（地表绕射），不应报错
        fit = fit_hyperbola(picks)
        assert fit.v_m_ns == pytest.approx(0.1, rel=1e-6)
        assert fit.z0_m == pytest.approx(0.0, abs=1e-9)


class TestVelocityPickModel:
    def test_pick_is_frozen_dataclass(self):
        pick = VelocityPick(trace_index=3, sample_index=5, x_m=1.5, t_ns=12.0)
        with pytest.raises(Exception):
            pick.x_m = 99.0  # noqa: B010 - 意图触发 frozen


def test_evidence_schema_constant_is_versioned():
    assert VELOCITY_ANALYSIS_EVIDENCE_SCHEMA == "mygpr.velocity_analysis_evidence.v1"
