#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI presets, workflow profiles, and lightweight quality helpers."""

from __future__ import annotations

import numpy as np

from core.methods_registry import PROCESSING_METHODS


# ============ 预设配置 ============

STOLT_MIGRATION_PRESETS = {
    "speed_first": {
        "label": "Stolt 速度优先",
        "params": {
            "dx": 0.05,
            "dt": 0.10,
            "v": 0.10,
            "pad_x": 0,
            "pad_t": 1,
            "stolt_jacobian_power": 0.02,
            "stolt_obliquity_power": 0.02,
            "stolt_mask_softness": 0.015,
            "stolt_kz_smooth": 1,
            "stolt_depth_gain": 0.0,
            "stolt_depth_gain_power": 1.0,
            "stolt_clip_percentile": 99.8,
        },
        "note": "更少补零与平滑，优先吞吐与交互速度。",
    },
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
    "stolt_speed_first": {
        "label": "Stolt 速度优先",
        "ui": {
            "normalize": False,
            "demean": False,
            "percentile": False,
        },
        "method_params": {
            "stolt_migration": dict(STOLT_MIGRATION_PRESETS["speed_first"]["params"]),
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
    "high_focus": {
        "label": "高聚焦",
        "preset_key": "stolt_focus_first",
        "order": [
            "set_zero_time",
            "dewow",
            "subtracting_average_2D",
            "fk_filter",
            "sec_gain",
            "hankel_svd",
            "stolt_migration",
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

    speed_score = 0
    focus_score = 0
    reason_tokens = []

    if snr < 11.0:
        focus_score += 2
        reason_tokens.append(f"SNR偏低({snr:.1f}dB)")
    elif snr > 17.0:
        speed_score += 1
        reason_tokens.append(f"SNR较高({snr:.1f}dB)")
    else:
        reason_tokens.append(f"SNR中等({snr:.1f}dB)")

    if spike > 8.0:
        focus_score += 2
        reason_tokens.append(f"尖峰度高({spike:.2f})")
    elif spike < 3.0:
        speed_score += 1
        reason_tokens.append(f"尖峰度低({spike:.2f})")
    else:
        reason_tokens.append(f"尖峰度中等({spike:.2f})")

    if bw > 0.62:
        focus_score += 1
        reason_tokens.append(f"有效带宽较宽({bw:.2f})")
    elif bw < 0.38:
        speed_score += 1
        reason_tokens.append(f"有效带宽较窄({bw:.2f})")
    else:
        reason_tokens.append(f"有效带宽中等({bw:.2f})")

    if dr > 26.0:
        focus_score += 1
        reason_tokens.append(f"动态范围较大({dr:.1f}dB)")
    elif dr < 16.0:
        speed_score += 1
        reason_tokens.append(f"动态范围较小({dr:.1f}dB)")
    else:
        reason_tokens.append(f"动态范围中等({dr:.1f}dB)")

    diff = focus_score - speed_score
    if diff >= 2:
        chosen = "focus_first"
    elif diff <= -2:
        chosen = "speed_first"
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
        },
    },
    "stage2": {
        "id": "stage2",
        "label": "阶段二：二维宏观背景压制",
        "icon": "📍",
        "subtitle": "消除强干扰 - 去除直达波和地表反射",
        "description": "处理由于天线直达波、地表强反射以及无人机电机带来的'霸屏'干扰。注意：此阶段必须在去噪完成后执行！",
        "methods": {
            "bandpass_filter": {
                "name": "频带滤波",
                "available": False,
                "default_enabled": False,
                "tooltip": "应用带通滤波精准切割掉超出有效带宽的无人机电机高频电磁干扰噪声。",
                "params": [],
            },
            "fk_filter": {
                "name": "F-K锥形滤波",
                "available": True,
                "default_enabled": False,
                "tooltip": "在频率-波数域进行锥形滤波，去除特定角度的干扰。",
                "params": PROCESSING_METHODS["fk_filter"]["params"],
            },
            "subtracting_average_2D": {
                "name": "平均道减法",
                "available": True,
                "default_enabled": True,
                "tooltip": "基础背景去除方法，减去水平方向上极其强烈的'横条纹'。",
                "params": PROCESSING_METHODS["subtracting_average_2D"]["params"],
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
    "high_focus": {
        "label": "高聚焦",
        "description": "完整流程包含迁移，质量最优",
        "stages": {
            "stage1": {"set_zero_time": True, "dewow": True},
            "stage2": {"fk_filter": True, "svd_bg": True},
            "stage3": {"sec_gain": True, "hankel_svd": True},
            "stage4": {"stolt_migration": True, "time_to_depth": True},
        },
    },
    "custom": {
        "label": "自定义",
        "description": "完全自定义每个阶段的方法",
        "stages": {},
    },
}


WORKFLOW_STAGE_ORDER = ["import", "stage1", "stage2", "stage3", "stage4"]
