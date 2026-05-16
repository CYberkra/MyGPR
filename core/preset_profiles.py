#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI presets, workflow profiles, and lightweight quality helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.methods_registry import PROCESSING_METHODS


# ============ 预设配置 ============

STOLT_MIGRATION_PRESETS = {
    "balanced": {
        "label": "Stolt 平衡档",
        "params": {
            "dx": 0.05,
            "dt": 0.10,
            "v": 0.10,
            "pad_x": 1,
            "pad_t": 1,
            "stolt_jacobian_power": 0.05,
            "stolt_obliquity_power": 0.06,
            "stolt_mask_softness": 0.03,
            "stolt_kz_smooth": 3,
            "stolt_depth_gain": 0.15,
            "stolt_depth_gain_power": 1.1,
            "stolt_clip_percentile": 99.7,
        },
        "note": "速度与聚焦折中，适合大多数中等SNR场景。",
    },
    "focus_first": {
        "label": "Stolt 聚焦优先",
        "params": {
            "dx": 0.05,
            "dt": 0.10,
            "v": 0.10,
            "pad_x": 2,
            "pad_t": 2,
            "stolt_jacobian_power": 0.09,
            "stolt_obliquity_power": 0.12,
            "stolt_mask_softness": 0.05,
            "stolt_kz_smooth": 5,
            "stolt_depth_gain": 0.35,
            "stolt_depth_gain_power": 1.25,
            "stolt_clip_percentile": 99.5,
        },
        "note": "更强映射补偿与频域平滑，优先成像聚焦质量。",
    },
}


GUI_PRESETS_V1 = {
    "raw_fidelity": {
        "label": "原始保真（默认）",
        "ui": {
            "normalize": False,
            "demean": False,
            "percentile": False,
        },
        "method_params": {},
    },
    "denoise_first": {
        "label": "降噪优先（稳健）",
        "ui": {
            "normalize": True,
            "demean": True,
            "percentile": True,
            "p_low": 1.0,
            "p_high": 99.0,
        },
        "method_params": {
            "set_zero_time": {"new_zero_time": 5.0},
            "dc_shift": {"estimator": "mean", "scope": "per_trace"},
            "dewow": {"window": 61},
            "fk_filter": {"angle_low": 12, "angle_high": 55, "taper_width": 4},
            "sec_gain": {"gain_min": 1.0, "gain_max": 4.2, "power": 1.2},
            "svd_subspace": {"rank_start": 1, "rank_end": 20},
        },
    },
    "detail_first": {
        "label": "保细节（细节优先）",
        "ui": {
            "normalize": False,
            "demean": True,
            "percentile": True,
            "p_low": 0.5,
            "p_high": 99.5,
        },
        "method_params": {
            "set_zero_time": {"new_zero_time": 4.5},
            "dewow": {"window": 31},
            "fk_filter": {"angle_low": 8, "angle_high": 62, "taper_width": 2},
            "sec_gain": {"gain_min": 1.0, "gain_max": 5.0, "power": 1.05},
            "hankel_svd": {"window_length": 0, "rank": 0},
        },
    },
    "stolt_balanced": {
        "label": "Stolt 平衡档",
        "ui": {
            "normalize": True,
            "demean": True,
            "percentile": True,
            "p_low": 0.8,
            "p_high": 99.2,
        },
        "method_params": {
            "stolt_migration": dict(STOLT_MIGRATION_PRESETS["balanced"]["params"]),
        },
    },
    "stolt_focus_first": {
        "label": "Stolt 聚焦优先",
        "ui": {
            "normalize": True,
            "demean": True,
            "percentile": True,
            "p_low": 0.5,
            "p_high": 99.5,
        },
        "method_params": {
            "wavelet_svd": {
                "wavelet": "db4",
                "levels": 2,
                "threshold": 0.05,
                "rank_start": 1,
                "rank_end": 20,
            },
            "stolt_migration": dict(STOLT_MIGRATION_PRESETS["focus_first"]["params"]),
        },
    },
}

DEFAULT_STARTUP_PRESET_KEY = "raw_fidelity"
BASIC_PARAM_LIMIT = 4

RECOMMENDED_RUN_PROFILES = {
    "robust_imaging": {
        "label": "稳健成像",
        "preset_key": "denoise_first",
        "order": [
            "set_zero_time",
            "dewow",
            "subtracting_average_2D",
            "fk_filter",
            "sec_gain",
            "svd_subspace",
        ],
    },
    "mygpr_standard": {
        "label": "MyGPR 标准流程",
        "preset_key": "denoise_first",
        "method_params": {
            "set_zero_time": {"new_zero_time": 5.0},
            "dewow": {"window": 61},
            "subtracting_average_2D": {"ntraces": 51},
            "sec_gain": {"gain_min": 1.0, "gain_max": 4.2, "power": 1.2},
            "svd_subspace": {"rank_start": 1, "rank_end": 20},
        },
        "order": [
            "set_zero_time",
            "dewow",
            "subtracting_average_2D",
            "sec_gain",
            "svd_subspace",
        ],
    },
    "high_quality_uav_gpr": {
        "label": "高质量 UAV-GPR",
        "preset_key": "raw_fidelity",
        "method_params": {
            "set_zero_time": {"new_zero_time": 5.0},
            "dewow": {"window": 61},
            "frequency_filter_1d": {
                "filter_type": "bandpass",
                "low_freq_mhz": 20.0,
                "high_freq_mhz": 170.0,
                "taper_ratio": 0.08,
            },
            "motion_compensation_v2": {
                "height_reference_mode": "mean",
                "height_source": "auto",
                "compensate_time_shift": True,
                "compensate_amplitude": True,
                "max_shift_samples": 0.0,
                "max_shift_ns": 20.0,
                "max_amplitude_scale": 2.0,
                "resample_spacing_m": 0.0,
                "apc_offset_x_m": 0.0,
                "apc_offset_y_m": 0.0,
                "apc_offset_z_m": 0.0,
                "max_abs_tilt_deg": 20.0,
            },
            "subtracting_average_2D": {"ntraces": 51},
            "wavelet_svd": {
                "wavelet": "db4",
                "levels": 2,
                "threshold": 0.05,
                "rank_start": 1,
                "rank_end": 20,
            },
            "sec_gain": {"gain_min": 1.0, "gain_max": 4.5, "power": 1.1},
            "manual_velocity_model": {
                "mode": "velocity",
                "velocity_m_per_ns": 0.10,
                "epsilon_r": 9.0,
                "uncertainty_fraction": 0.10,
            },
            "geometry_depth_context": {
                "require_velocity_model": True,
                "require_trace_spacing": True,
                "require_time_window": True,
                "require_agl": False,
            },
        },
        "order": [
            "set_zero_time",
            "dc_shift",
            "dewow",
            "frequency_filter_1d",
            "motion_compensation_v2",
            "subtracting_average_2D",
            "wavelet_svd",
            "manual_velocity_model",
            "geometry_depth_context",
            "sec_gain",
        ],
    },
    "gprmax_impulse_validation": {
        "label": "gprMax impulse 仿真验证",
        "preset_key": "raw_fidelity",
        "method_params": {
            "set_zero_time": {"new_zero_time": 5.0},
            "dewow": {"window": 61},
            "subtracting_average_2D": {"ntraces": 31},
            "agcGain": {"window": 61},
            "wavelet_svd": {
                "wavelet": "db4",
                "levels": 2,
                "threshold": 0.05,
                "rank_start": 1,
                "rank_end": 20,
            },
        },
        "order": [
            "set_zero_time",
            "dewow",
            "subtracting_average_2D",
            "agcGain",
            "wavelet_svd",
        ],
    },
    "uav_gpr_experience_baseline_v1": {
        "label": "UAV-GPR 经验参数 baseline",
        "preset_key": "denoise_first",
        "method_params": {
            "set_zero_time": {"new_zero_time": 5.0},
            "dewow": {"window": 61},
            "subtracting_average_2D": {"ntraces": 51},
            "sec_gain": {"gain_min": 1.0, "gain_max": 4.2, "power": 1.2},
            "svd_subspace": {"rank_start": 1, "rank_end": 20},
        },
        "order": [
            "set_zero_time",
            "dewow",
            "subtracting_average_2D",
            "sec_gain",
            "svd_subspace",
        ],
    },
    "hankel_denoise": {
        "label": "Hankel-SVD 去噪",
        "preset_key": "denoise_first",
        "method_params": {
            "hankel_svd": {"window_length": 0, "rank": 0},
        },
        "order": [
            "set_zero_time",
            "dewow",
            "subtracting_average_2D",
            "fk_filter",
            "sec_gain",
            "hankel_svd",
        ],
    },
    "wavelet_2d_denoise": {
        "label": "Wavelet 2D 去噪",
        "preset_key": "denoise_first",
        "method_params": {
            "wavelet_2d": {"wavelet": "db4", "levels": 2, "threshold": 0.1},
        },
        "order": [
            "set_zero_time",
            "dewow",
            "subtracting_average_2D",
            "fk_filter",
            "sec_gain",
            "wavelet_2d",
        ],
    },
    "wavelet_svd_denoise": {
        "label": "Wavelet-SVD 去噪",
        "preset_key": "denoise_first",
        "method_params": {
            "wavelet_svd": {
                "wavelet": "db4",
                "levels": 2,
                "threshold": 0.05,
                "rank_start": 1,
                "rank_end": 20,
            },
        },
        "order": [
            "set_zero_time",
            "dewow",
            "subtracting_average_2D",
            "fk_filter",
            "sec_gain",
            "wavelet_svd",
        ],
    },
    "motion_compensation_v1": {
        "label": "运动补偿 V1",
        "preset_key": "raw_fidelity",
        "method_params": {
            "trajectory_smoothing": {
                "method": "savgol",
                "window_length": 21,
                "polyorder": 3,
            },
            "motion_compensation_speed": {"spacing_m": 0.0},
            "motion_compensation_attitude": {
                "apc_offset_x_m": 0.0,
                "apc_offset_y_m": 0.0,
                "apc_offset_z_m": 0.0,
                "max_abs_tilt_deg": 20.0,
            },
            "motion_compensation_height": {
                "reference_height_mode": "mean",
                "compensate_amplitude": True,
                "compensate_time_shift": True,
                "wave_speed_m_per_ns": 0.1,
            },
            "motion_compensation_vibration": {
                "smooth_window": 9,
                "preserve_row_percentile": 94.0,
                "preserve_mix": 0.35,
                "background_mix": 0.02,
                "max_restore_gain": 1.25,
            },
        },
        "order": [
            "trajectory_smoothing",
            "motion_compensation_speed",
            "motion_compensation_attitude",
            "motion_compensation_height",
            "motion_compensation_vibration",
        ],
    },
    "motion_compensation_v2": {
        "label": "UAV 运动补偿 V2",
        "preset_key": "raw_fidelity",
        "method_params": {
            "motion_compensation_v2": {
                "height_reference_mode": "mean",
                "height_source": "auto",
                "compensate_time_shift": True,
                "compensate_amplitude": True,
                "max_shift_samples": 0.0,
                "max_shift_ns": 20.0,
                "max_amplitude_scale": 2.0,
                "resample_spacing_m": 0.0,
                "apc_offset_x_m": 0.0,
                "apc_offset_y_m": 0.0,
                "apc_offset_z_m": 0.0,
                "max_abs_tilt_deg": 20.0,
            },
        },
        "order": ["motion_compensation_v2"],
    },
}


# ============ 质量指标相关 ============

DEFAULT_QUALITY_DASHBOARD_THRESHOLDS = {
    "focus_ratio": {"min": 0.25},
    "hot_pixels": {"max": 0},
    "spikiness": {"max": 2.0},
    "time_ms": {"max": 45.0},
}


def compute_quality_metrics(data: np.ndarray, elapsed_ms: float | None = None) -> dict:
    """计算轻量级处理后质量指标"""
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return {
            "focus_ratio": 0.0,
            "hot_pixels": 0,
            "spikiness": 0.0,
            "time_ms": float(elapsed_ms) if elapsed_ms is not None else 0.0,
        }

    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_clean = np.abs(clean)
    total_energy = float(np.sum(abs_clean**2))
    if total_energy <= 1e-12:
        focus_ratio = 0.0
    else:
        threshold = float(np.percentile(abs_clean, 95.0))
        focus_ratio = float(
            np.sum((abs_clean[abs_clean >= threshold]) ** 2) / total_energy
        )

    baseline = float(np.percentile(abs_clean, 99.5))
    sigma = float(np.std(clean))
    sigma_gate = max(6.0 * sigma, 1e-9)
    hot_gate = max(baseline, sigma_gate)
    hot_pixels = int(np.sum(abs_clean >= hot_gate))

    centered = clean - float(np.mean(clean))
    std = float(np.std(centered))

    if std <= 1e-12:
        spikiness = 0.0
    else:
        z = centered / std
        spikiness = float(max(0.0, np.mean(z**4) - 3.0))

    return {
        "focus_ratio": focus_ratio,
        "hot_pixels": hot_pixels,
        "spikiness": spikiness,
        "time_ms": float(max(0.0, elapsed_ms or 0.0)),
    }


def compute_stolt_data_stats(data: np.ndarray) -> dict:
    """计算用于自适应Stolt预设选择的数据统计"""
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.size < 16:
        return {
            "valid": False,
            "snr_db": 0.0,
            "spikiness": 0.0,
            "eff_bw_ratio": 0.0,
            "dynamic_range_db": 0.0,
        }

    finite = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    centered = finite - np.median(finite)
    abs_centered = np.abs(centered)

    q50 = float(np.percentile(abs_centered, 50))
    q95 = float(np.percentile(abs_centered, 95))
    q99 = float(np.percentile(abs_centered, 99))
    mad_sigma = max(q50 / 0.6745, 1e-9)
    snr_db = float(20.0 * np.log10(max(q95, 1e-9) / mad_sigma))
    dynamic_range_db = float(20.0 * np.log10(max(q99, 1e-9) / max(q50, 1e-9)))

    std = float(np.std(centered))
    if std <= 1e-12:
        spikiness = 0.0
    else:
        z = centered / std
        kurt = float(np.mean(z**4))
        spikiness = max(0.0, kurt - 3.0)

    spec = np.abs(np.fft.rfft(centered, axis=0))
    if spec.size == 0:
        eff_bw_ratio = 0.0
    else:
        mean_spec = np.mean(spec, axis=1)
        total = float(np.sum(mean_spec))
        if total <= 1e-12:
            eff_bw_ratio = 0.0
        else:
            cdf = np.cumsum(mean_spec) / total
            lo_idx = int(np.searchsorted(cdf, 0.1))
            hi_idx = int(np.searchsorted(cdf, 0.9))
            denom = max(1, len(mean_spec) - 1)
            eff_bw_ratio = max(0.0, min(1.0, (hi_idx - lo_idx) / denom))

    return {
        "valid": True,
        "snr_db": snr_db,
        "spikiness": spikiness,
        "eff_bw_ratio": float(eff_bw_ratio),
        "dynamic_range_db": dynamic_range_db,
    }


def choose_adaptive_stolt_preset(data: np.ndarray):
    """基于数据特征选择自适应Stolt预设"""
    stats = compute_stolt_data_stats(data)
    if not stats.get("valid", False):
        return "balanced", "数据维度/规模不足，回退平衡档。", stats

    snr = stats["snr_db"]
    spike = stats["spikiness"]
    bw = stats["eff_bw_ratio"]
    dr = stats["dynamic_range_db"]

    focus_score = 0
    reason_tokens = []

    if snr < 11.0:
        focus_score += 2
        reason_tokens.append(f"SNR偏低({snr:.1f}dB)")
    elif snr > 17.0:
        reason_tokens.append(f"SNR较高({snr:.1f}dB)")
    else:
        reason_tokens.append(f"SNR中等({snr:.1f}dB)")

    if spike > 8.0:
        focus_score += 2
        reason_tokens.append(f"尖峰度高({spike:.2f})")
    elif spike < 3.0:
        reason_tokens.append(f"尖峰度低({spike:.2f})")
    else:
        reason_tokens.append(f"尖峰度中等({spike:.2f})")

    if bw > 0.62:
        focus_score += 1
        reason_tokens.append(f"有效带宽较宽({bw:.2f})")
    elif bw < 0.38:
        reason_tokens.append(f"有效带宽较窄({bw:.2f})")
    else:
        reason_tokens.append(f"有效带宽中等({bw:.2f})")

    if dr > 26.0:
        focus_score += 1
        reason_tokens.append(f"动态范围较大({dr:.1f}dB)")
    elif dr < 16.0:
        reason_tokens.append(f"动态范围较小({dr:.1f}dB)")
    else:
        reason_tokens.append(f"动态范围中等({dr:.1f}dB)")

    if focus_score >= 2:
        chosen = "focus_first"
    else:
        chosen = "balanced"

    reason = "；".join(reason_tokens)
    return chosen, reason, stats


# ============================================================================
# 四阶段工作流配置 (UAV-GPR Standard Workflow)
# ============================================================================

WORKFLOW_STAGES = {
    "import": {
        "id": "import",
        "label": "数据导入",
        "icon": "📁",
        "subtitle": "加载GPR数据文件",
        "methods": {},
    },
    "stage1": {
        "id": "stage1",
        "label": "阶段一：一维单道校正",
        "icon": "📍",
        "subtitle": "基线修复 - 消除仪器系统误差",
        "description": "这是最基础的'大扫除'环节，首要任务是消除仪器系统误差，保证每一条独立的雷达波形在时间轴和振幅基准上是正常的。",
        "methods": {
            "spatial_resampling": {
                "name": "空间轨迹重采样",
                "available": False,
                "default_enabled": False,
                "tooltip": "由于无人机飞行速度不匀，需将RTK-GNSS坐标与雷达数据对齐并插值，强制映射到等间距网格上。",
                "params": [],
            },
            "set_zero_time": {
                "name": "时间零点校正",
                "available": True,
                "default_enabled": True,
                "tooltip": "统一地表起跑线，将初至波对齐到0ns。",
                "params": PROCESSING_METHODS["set_zero_time"]["params"],
            },
            "dewow": {
                "name": "去直流/去漂移",
                "available": True,
                "default_enabled": True,
                "tooltip": "消除低频漂移和基线扭曲。",
                "params": PROCESSING_METHODS["dewow"]["params"],
            },
            "dc_shift": {
                "name": "DC 去偏",
                "available": True,
                "default_enabled": True,
                "tooltip": "移除每道或全局直流偏置，是 dewow 前的基础迹线域校正。",
                "params": PROCESSING_METHODS["dc_shift"]["params"],
            },
        },
    },
    "stage2": {
        "id": "stage2",
        "label": "阶段二：二维宏观背景压制",
        "icon": "📍",
        "subtitle": "消除强干扰 - 去除直达波和地表反射",
        "description": "处理由于天线直达波、地表强反射以及无人机电机带来的'霸屏'干扰。注意：此阶段必须在去噪完成后执行！",
        "methods": {
            "frequency_filter_1d": {
                "name": "一维频域滤波",
                "available": True,
                "default_enabled": True,
                "tooltip": "沿时间轴做零相位带通/高通/低通/陷波滤波，用于压制有效频带外噪声。",
                "params": PROCESSING_METHODS["frequency_filter_1d"]["params"],
            },
            "motion_compensation_v2": {
                "name": "UAV运动补偿V2",
                "available": True,
                "default_enabled": True,
                "tooltip": "融合高度、RTK/IMU与等距重采样元数据，补偿无人机高度和姿态导致的道间畸变。",
                "params": PROCESSING_METHODS["motion_compensation_v2"]["params"],
            },
            "subtracting_average_2D": {
                "name": "平均道减法",
                "available": True,
                "default_enabled": True,
                "tooltip": "基础背景去除方法，减去水平方向上极其强烈的'横条纹'。",
                "params": PROCESSING_METHODS["subtracting_average_2D"]["params"],
            },
            "fk_filter": {
                "name": "F-K锥形滤波",
                "available": True,
                "default_enabled": False,
                "tooltip": "在频率-波数域进行锥形滤波，去除特定角度的干扰。",
                "params": PROCESSING_METHODS["fk_filter"]["params"],
            },
            "median_background_2D": {
                "name": "中位数背景抑制",
                "available": True,
                "default_enabled": False,
                "tooltip": "对异常强反射更稳健，适合存在局部离群道或尖峰杂波时的背景抑制。",
                "params": PROCESSING_METHODS["median_background_2D"]["params"],
            },
            "svd_bg": {
                "name": "SVD背景去除",
                "available": True,
                "default_enabled": False,
                "tooltip": "基于奇异值分解的背景去除，处理地表剧烈起伏时效果更好。",
                "params": PROCESSING_METHODS["svd_bg"]["params"],
            },
            "ccbs": {
                "name": "CCBS互相关背景减除",
                "available": True,
                "default_enabled": False,
                "tooltip": "前沿方法，动态权重加权背景减除，在复杂杂波场景下性能优异。",
                "params": PROCESSING_METHODS["ccbs"]["params"],
            },
        },
    },
    "stage3": {
        "id": "stage3",
        "label": "阶段三：能量恢复与精细提纯",
        "icon": "📍",
        "subtitle": "突出弱信号 - 增益补偿与结构化去噪",
        "description": "深部有效信号因电磁波衰减变得非常微弱，需要进行能量补偿与二次提纯。核心法则：必须在去噪与背景去除之后做增益！",
        "methods": {
            "sec_gain": {
                "name": "SEC球面指数增益",
                "available": True,
                "default_enabled": True,
                "tooltip": "推荐：保真度好，能完美保留真实物理反射强度的比例。",
                "params": PROCESSING_METHODS["sec_gain"]["params"],
            },
            "agcGain": {
                "name": "AGC自动增益控制",
                "available": True,
                "default_enabled": False,
                "tooltip": "备选：瞬间提升视觉效果，但破坏信号相对振幅关系。",
                "params": PROCESSING_METHODS["agcGain"]["params"],
            },
            "hankel_svd": {
                "name": "Hankel-SVD去噪",
                "available": True,
                "default_enabled": False,
                "tooltip": "增益放大后进行结构化去噪，平滑深部的细碎纹理噪声。",
                "params": PROCESSING_METHODS["hankel_svd"]["params"],
            },
            "svd_subspace": {
                "name": "SVD子空间去噪",
                "available": True,
                "default_enabled": False,
                "tooltip": "按奇异值子空间重构数据，适合实验性子空间去噪与直达波/噪声分离。",
                "params": PROCESSING_METHODS["svd_subspace"]["params"],
            },
            "wavelet_svd": {
                "name": "Wavelet-SVD复合去噪",
                "available": True,
                "default_enabled": False,
                "tooltip": "先做二维小波分解，再对低频近似系数做SVD子空间重构的复合去噪。",
                "params": PROCESSING_METHODS["wavelet_svd"]["params"],
            },
            "wavelet_2d": {
                "name": "Wavelet 2D去噪",
                "available": True,
                "default_enabled": False,
                "tooltip": "二维小波阈值去噪，适合先压制细碎高频纹理噪声。",
                "params": PROCESSING_METHODS["wavelet_2d"]["params"],
            },
            "running_average_2D": {
                "name": "尖锐杂波抑制",
                "available": True,
                "default_enabled": False,
                "tooltip": "CaGPR 中的 running_average_2D。对增益后的局部尖锐杂波做横向运行平均抑制。",
                "params": PROCESSING_METHODS["running_average_2D"]["params"],
            },
        },
    },
    "stage4": {
        "id": "stage4",
        "label": "阶段四：几何与物理属性还原",
        "icon": "📍",
        "subtitle": "聚焦与标定 - 消除无人机动态畸变",
        "description": "最后一步是消除无人机动态畸变，让图像回归真实的地下物理形态。这是工程解释的最后一步，直接决定目标体埋深准不准。",
        "methods": {
            "topographic_correction": {
                "name": "无人机高度/地形校正",
                "available": False,
                "default_enabled": False,
                "tooltip": "UAV-GPR专属：结合LiDAR或RTK高程，在时间轴上垂直平移数据道。",
                "params": [],
            },
            "stolt_migration": {
                "name": "Stolt/ω-k偏移迁移",
                "available": True,
                "default_enabled": False,
                "tooltip": "利用波场倒推，将发散的双曲线能量精确'收敛'压缩成真实的物理位置。",
                "params": PROCESSING_METHODS["stolt_migration"]["params"],
            },
            "kirchhoff_migration": {
                "name": "Kirchhoff偏移迁移",
                "available": True,
                "default_enabled": False,
                "tooltip": "移植自 CaGPR 的 Kirchhoff 成像主链，适合做层状/均匀速度模型下的初始成像。",
                "params": PROCESSING_METHODS["kirchhoff_migration"]["params"],
            },
            "time_to_depth": {
                "name": "深度转换与标定",
                "available": True,
                "default_enabled": False,
                "tooltip": "结合介电常数，把纵坐标从时间(ns)转换为绝对物理深度(m)。",
                "params": PROCESSING_METHODS["time_to_depth"]["params"],
            },
            "manual_velocity_model": {
                "name": "手动速度模型",
                "available": True,
                "default_enabled": True,
                "tooltip": "第一版速度模型入口：手动常速度或介电常数换算。",
                "params": PROCESSING_METHODS["manual_velocity_model"]["params"],
            },
            "geometry_depth_context": {
                "name": "几何-深度上下文检查",
                "available": True,
                "default_enabled": True,
                "tooltip": "检查速度、道距、时间窗和 AGL 元数据，并为迁移/时深转换传递上下文。",
                "params": PROCESSING_METHODS["geometry_depth_context"]["params"],
            },
        },
    },
}


WORKFLOW_PRESETS = {
    "robust_imaging": {
        "label": "稳健成像",
        "description": "标准四阶段流程，平衡速度与质量",
        "stages": {
            "stage1": {"set_zero_time": True, "dewow": True},
            "stage2": {"fk_filter": True, "subtracting_average_2D": True},
            "stage3": {"sec_gain": True},
            "stage4": {},
        },
    },
    "mygpr_standard": {
        "label": "MyGPR 标准流程",
        "description": "原 MyGPR 经典五步处理链：零时、低频漂移、背景、增益、去噪。",
        "stages": {
            "stage1": {"set_zero_time": True, "dewow": True},
            "stage2": {"subtracting_average_2D": True},
            "stage3": {"sec_gain": True, "svd_subspace": True},
            "stage4": {},
        },
    },
    "high_quality_uav_gpr": {
        "label": "高质量 UAV-GPR",
        "description": "面向无人机实测数据的最高质量默认链路，优先保留完整数据、传感器补偿和结构保真。",
        "stages": {
            "stage1": {"set_zero_time": True, "dc_shift": True, "dewow": True},
            "stage2": {
                "frequency_filter_1d": True,
                "motion_compensation_v2": True,
                "subtracting_average_2D": True,
            },
            "stage3": {"wavelet_svd": True},
            "stage4": {
                "manual_velocity_model": True,
                "geometry_depth_context": True,
                "sec_gain": True,
            },
        },
    },
    "gprmax_impulse_validation": {
        "label": "gprMax impulse 仿真验证",
        "description": "面向外部 gprMax impulse B-scan 的默认链路，不套用实测 SFCW 20-170MHz 固定频带。",
        "stages": {
            "stage1": {"set_zero_time": True, "dewow": True},
            "stage2": {"subtracting_average_2D": True},
            "stage3": {"agcGain": True, "wavelet_svd": True},
            "stage4": {},
        },
    },
    "custom": {
        "label": "自定义",
        "description": "完全自定义每个阶段的方法",
        "stages": {},
    },
}


WORKFLOW_STAGE_ORDER = ["import", "stage1", "stage2", "stage3", "stage4"]


_PROFILE_SENSOR_DEPENDENCY_WARNINGS: dict[str, dict[str, Any]] = {
    "motion_compensation_v2": {
        "code": "motion_compensation_v2_sensor_dependency",
        "method_key": "motion_compensation_v2",
        "message": (
            "motion_compensation_v2 依赖 trace_metadata；缺少 AGL 高度时会跳过"
            "高度/时移/振幅补偿并记录运行告警。"
        ),
        "required_any": ["height_agl_m", "flight_height_m"],
        "optional": [
            "local_x_m",
            "local_y_m",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "trace_distance_m",
        ],
    },
}


def build_profile_workflow_summary(profile_key: str) -> dict[str, Any]:
    """Build report-friendly stage and dependency metadata for a run profile."""
    profile = RECOMMENDED_RUN_PROFILES.get(profile_key)
    if profile is None:
        raise KeyError(f"未知推荐流程: {profile_key}")

    workflow = WORKFLOW_PRESETS.get(profile_key, {})
    stage_config = workflow.get("stages", {}) if isinstance(workflow, dict) else {}
    ordered_methods = [str(method_key) for method_key in profile.get("order", [])]

    method_to_stage: dict[str, str] = {}
    for stage_key in WORKFLOW_STAGE_ORDER:
        methods = stage_config.get(stage_key, {})
        if not isinstance(methods, dict):
            continue
        for method_key, enabled in methods.items():
            if enabled:
                method_to_stage[str(method_key)] = stage_key

    stages: list[dict[str, Any]] = []
    for stage_key in WORKFLOW_STAGE_ORDER:
        if stage_key == "import":
            continue
        stage_methods = [
            method_key
            for method_key in ordered_methods
            if method_to_stage.get(method_key) == stage_key
        ]
        if not stage_methods:
            continue
        stage_meta = WORKFLOW_STAGES.get(stage_key, {})
        stages.append(
            {
                "stage_key": stage_key,
                "stage_label": stage_meta.get("label", stage_key),
                "method_keys": stage_methods,
                "methods": [
                    {
                        "method_key": method_key,
                        "method_name": PROCESSING_METHODS.get(method_key, {}).get(
                            "name", method_key
                        ),
                    }
                    for method_key in stage_methods
                ],
            }
        )

    sensor_dependency_warnings = [
        dict(_PROFILE_SENSOR_DEPENDENCY_WARNINGS[method_key])
        for method_key in ordered_methods
        if method_key in _PROFILE_SENSOR_DEPENDENCY_WARNINGS
    ]

    return {
        "profile_key": profile_key,
        "label": profile.get("label", profile_key),
        "preset_key": profile.get("preset_key"),
        "ordered_methods": ordered_methods,
        "stages": stages,
        "unassigned_methods": [
            method_key
            for method_key in ordered_methods
            if method_key not in method_to_stage
        ],
        "sensor_dependency_warnings": sensor_dependency_warnings,
    }
