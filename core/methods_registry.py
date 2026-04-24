#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方法注册表模块 - 统一处理方法定义

此模块集中管理所有GPR处理方法的信息，供GUI和CLI共享使用。
- core 类型：PythonModule/ 下的旧式CSV I/O函数（compensatingGain, dewow 等）
- local 类型：PythonModule/ 下的numpy数组函数（svd_background, fk_filter 等）
"""

from typing import Any

import numpy as np

from PythonModule.svd_background import method_svd_background
from PythonModule.fk_filter import method_fk_filter
from PythonModule.hankel_svd import method_hankel_svd
from PythonModule.kirchhoff_migration import method_kirchhoff_migration
from PythonModule.stolt_migration import method_stolt_migration
from PythonModule.time_to_depth import method_time_to_depth
from PythonModule.sec_gain import method_sec_gain
from PythonModule.sliding_average import method_sliding_average
from PythonModule.rpca_placeholder import method_rpca_placeholder
from PythonModule.rpca_background import method_rpca_background
from PythonModule.wnnm_placeholder import method_wnnm_placeholder
from PythonModule.ccbs_filter import method_ccbs
from PythonModule.median_background_2D import method_median_background_2d
from PythonModule.svd_subspace import method_svd_subspace

_method_wavelet_2d: Any
_method_wavelet_svd: Any

try:
    from PythonModule.wavelet_2d import method_wavelet_2d as _imported_method_wavelet_2d
    from PythonModule.wavelet_svd import method_wavelet_svd as _imported_method_wavelet_svd

    HAS_PYWAVELETS = True
    _method_wavelet_2d = _imported_method_wavelet_2d
    _method_wavelet_svd = _imported_method_wavelet_svd
except ModuleNotFoundError as e:
    if e.name != "pywt":
        raise

    HAS_PYWAVELETS = False

    def _missing_wavelet_2d(*args, **kwargs):
        raise ImportError(
            "Wavelet 2D 去噪需要安装 PyWavelets。请执行: pip install PyWavelets"
        )

    def _missing_wavelet_svd(*args, **kwargs):
        raise ImportError(
            "Wavelet-SVD 需要安装 PyWavelets。请执行: pip install PyWavelets"
        )

    _method_wavelet_2d = _missing_wavelet_2d
    _method_wavelet_svd = _missing_wavelet_svd


from PythonModule.dewow import method_dewow
from PythonModule.set_zero_time import method_set_zero_time
from PythonModule.motion_compensation_height import method_motion_compensation_height
from PythonModule.trajectory_smoothing import method_trajectory_smoothing


# ============ 方法注册表 ============

PROCESSING_METHODS = {
    "compensatingGain": {
        "name": "0 compensatingGain (manual gain compensation)",
        "type": "core",
        "module": "compensatingGain",
        "func": "compensatingGain",
        "params": [
            {
                "name": "gain_min",
                "label": "Gain min",
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 20.0,
            },
            {
                "name": "gain_max",
                "label": "Gain max",
                "type": "float",
                "default": 6.0,
                "min": 0.1,
                "max": 50.0,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "gain",
        "auto_tune_candidates": {
            "gain_min": [0.8, 1.0, 1.2],
            "gain_max": [2.5, 3.5, 4.5, 5.5, 7.0, 9.0, 12.0],
        },
    },
    "dewow": {
        "name": "1 dewow (low-frequency drift correction)",
        "type": "local",
        "module": "dewow",
        "func": method_dewow,
        "params": [
            {
                "name": "window",
                "label": "Window (samples)",
                "type": "int",
                "default": 23,
                "min": 1,
                "max": 1000,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "drift",
        "auto_tune_candidates": {"window": [16, 32, 64, 128, 256]},
    },
    "set_zero_time": {
        "name": "2 set_zero_time (zero-time correction)",
        "type": "local",
        "module": "set_zero_time",
        "func": method_set_zero_time,
        "params": [
            {
                "name": "new_zero_time",
                "label": "Zero-time (ns)",
                "type": "float",
                "default": 5.0,
                "min": 0.0,
                "max": 1000.0,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "zero_time",
        "auto_tune_candidates": {
            "detectors": ["threshold", "peak", "first_break"],
            "thresholds": [0.02, 0.05, 0.08, 0.12],
            "backup_samples": [2, 4, 6, 8],
            "search_ratio": 0.35,
        },
    },
    "agcGain": {
        "name": "3 agcGain (AGC correction)",
        "type": "core",
        "module": "agcGain",
        "func": "agcGain",
        "params": [
            {
                "name": "window",
                "label": "Window (samples)",
                "type": "int",
                "default": 11,
                "min": 1,
                "max": 1000,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "gain",
        "auto_tune_candidates": {"window": [7, 11, 21, 31, 41, 61, 81, 121]},
    },
    "subtracting_average_2D": {
        "name": "4 subtracting_average_2D (background removal)",
        "type": "core",
        "module": "subtracting_average_2D",
        "func": "subtracting_average_2D",
        "params": [
            {
                "name": "ntraces",
                "label": "Window traces",
                "type": "int",
                "default": 501,
                "min": 1,
                "max": 2001,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "background",
        "auto_tune_candidates": {
            "ntraces": [7, 11, 21, 31, 51, 81, 101, 151, 201, 301, 401, 501]
        },
    },
    "median_background_2D": {
        "name": "4.1 median_background_2D (median background removal)",
        "type": "local",
        "func": method_median_background_2d,
        "params": [
            {
                "name": "ntraces",
                "label": "Window traces",
                "type": "int",
                "default": 51,
                "min": 1,
                "max": 2001,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "background",
        "auto_tune_candidates": {
            "ntraces": [7, 11, 21, 31, 51, 81, 101, 151, 201, 301]
        },
    },
    "running_average_2D": {
        "name": "5 sharp clutter suppression",
        "type": "core",
        "module": "running_average_2D",
        "func": "running_average_2D",
        "params": [
            {
                "name": "ntraces",
                "label": "Window traces",
                "type": "int",
                "default": 9,
                "min": 1,
                "max": 2001,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "impulse",
        "auto_tune_candidates": {"ntraces": [3, 5, 7, 9, 11]},
    },
    "svd_bg": {
        "name": "SVD background removal (low-rank)",
        "type": "local",
        "func": method_svd_background,
        "params": [
            {
                "name": "rank",
                "label": "Rank (remove top r)",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 20,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "background",
        "auto_tune_candidates": {"rank": [1, 2, 3, 4, 5, 6, 8]},
    },
    "fk_filter": {
        "name": "F-K cone filter",
        "type": "local",
        "func": method_fk_filter,
        "params": [
            {
                "name": "angle_low",
                "label": "Stopband start angle (°)",
                "type": "int",
                "default": 12,
                "min": 0,
                "max": 90,
            },
            {
                "name": "angle_high",
                "label": "Stopband end angle (°)",
                "type": "int",
                "default": 55,
                "min": 0,
                "max": 90,
            },
            {
                "name": "taper_width",
                "label": "Taper width (°)",
                "type": "int",
                "default": 4,
                "min": 0,
                "max": 20,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "fk",
        "auto_tune_candidates": {
            "angle_low": [6, 8, 10, 12, 15, 18],
            "angle_high": [40, 48, 55, 62, 70, 78],
            "taper_width": [0, 2, 4, 6, 8],
        },
    },
    "hankel_svd": {
        "name": "Hankel SVD denoising",
        "type": "local",
        "func": method_hankel_svd,
        "params": [
            {
                "name": "window_length",
                "label": "Window length (0=auto)",
                "type": "int",
                "default": 80,
                "min": 0,
                "max": 2000,
            },
            {
                "name": "rank",
                "label": "Rank kept (0=auto)",
                "type": "int",
                "default": 5,
                "min": 0,
                "max": 100,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "denoise",
        "auto_tune_candidates": {
            "window_length": [0, 32, 48, 64, 80],
            "rank": [0, 2, 4, 6, 8],
        },
    },
    "svd_subspace": {
        "name": "SVD子空间重构去噪",
        "type": "local",
        "func": method_svd_subspace,
        "params": [
            {
                "name": "rank_start",
                "label": "Start rank (1-based)",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 500,
            },
            {
                "name": "rank_end",
                "label": "End rank (1-based)",
                "type": "int",
                "default": 20,
                "min": 1,
                "max": 2000,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "denoise",
        "auto_tune_candidates": {
            "rank_start": [1, 2],
            "rank_end": [8, 12, 16, 20, 24, 32, 40],
        },
    },
    "wavelet_2d": {
        "name": "Wavelet 2D 去噪"
        + ("（需PyWavelets）" if not HAS_PYWAVELETS else ""),
        "type": "local",
        "func": _method_wavelet_2d,
        "params": [
            {
                "name": "wavelet",
                "label": "Wavelet",
                "type": "str",
                "default": "db4",
            },
            {
                "name": "levels",
                "label": "Levels",
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 8,
            },
            {
                "name": "threshold",
                "label": "Threshold (0-1)",
                "type": "float",
                "default": 0.1,
                "min": 0.0,
                "max": 1.0,
            },
            {
                "name": "threshold_strategy",
                "label": "Threshold strategy",
                "type": "str",
                "default": "mad_universal",
            },
        ],
        "auto_tune_enabled": HAS_PYWAVELETS,
        "auto_tune_family": "denoise",
        "auto_tune_candidates": {
            "wavelet": ["db4"],
            "levels": [1, 2, 3],
            "threshold": [0.05, 0.08, 0.1, 0.12],
            "threshold_strategy": ["mad_universal"],
        },
    },
    "wavelet_svd": {
        "name": "Wavelet-SVD 复合去噪"
        + ("（需PyWavelets）" if not HAS_PYWAVELETS else ""),
        "type": "local",
        "func": _method_wavelet_svd,
        "params": [
            {
                "name": "wavelet",
                "label": "Wavelet",
                "type": "str",
                "default": "db4",
            },
            {
                "name": "levels",
                "label": "Levels",
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 8,
            },
            {
                "name": "threshold",
                "label": "Threshold (0-1)",
                "type": "float",
                "default": 0.05,
                "min": 0.0,
                "max": 1.0,
            },
            {
                "name": "threshold_strategy",
                "label": "Threshold strategy",
                "type": "str",
                "default": "mad_universal",
            },
            {
                "name": "rank_start",
                "label": "Start rank (1-based)",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 500,
            },
            {
                "name": "rank_end",
                "label": "End rank (1-based)",
                "type": "int",
                "default": 20,
                "min": 1,
                "max": 2000,
            },
        ],
        "auto_tune_enabled": HAS_PYWAVELETS,
        "auto_tune_family": "denoise",
        "auto_tune_candidates": {
            "wavelet": ["db4"],
            "levels": [1, 2, 3],
            "threshold": [0.03, 0.05, 0.08],
            "threshold_strategy": ["mad_universal"],
            "rank_start": [1, 2],
            "rank_end": [8, 12, 16, 20, 24],
        },
    },
    "rpca_background": {
        "name": "RPCA背景抑制",
        "type": "local",
        "func": method_rpca_background,
        "params": [
            {
                "name": "lam",
                "label": "稀疏权重",
                "type": "float",
                "default": 0.08,
                "min": 0.001,
                "max": 1.0,
            },
            {
                "name": "mu",
                "label": "初始罚参数",
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 10.0,
            },
            {
                "name": "max_iter",
                "label": "最大迭代次数",
                "type": "int",
                "default": 120,
                "min": 10,
                "max": 1000,
            },
            {
                "name": "tol",
                "label": "收敛阈值",
                "type": "float",
                "default": 1e-6,
                "min": 1e-8,
                "max": 1e-2,
            },
        ],
        "auto_tune_enabled": False,
    },
    "wnnm_placeholder": {
        "name": "WNNM背景抑制（延期）",
        "type": "local",
        "func": method_wnnm_placeholder,
        "params": [
            {
                "name": "weight",
                "label": "权重",
                "type": "float",
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
            },
        ],
        "auto_tune_enabled": False,
    },
    "ccbs": {
        "name": "CCBS cross-correlation background subtraction",
        "type": "local",
        "func": method_ccbs,
        "params": [
            {
                "name": "use_custom_ref",
                "label": "Use custom reference wave",
                "type": "bool",
                "default": False,
            },
        ],
    },
    "motion_compensation_height": {
        "name": "飞行高度归一化",
        "type": "local",
        "func": method_motion_compensation_height,
        "params": [
            {
                "name": "reference_height_mode",
                "label": "参考高度",
                "type": "choice",
                "choices": ["mean", "min", "manual"],
                "default": "mean",
            },
            {
                "name": "manual_height",
                "label": "手动参考高度 (m)",
                "type": "float",
                "default": 10.0,
                "min": 0.0,
                "max": 500.0,
            },
            {
                "name": "compensate_amplitude",
                "label": "振幅校正",
                "type": "bool",
                "default": True,
            },
            {
                "name": "compensate_time_shift",
                "label": "时移校正",
                "type": "bool",
                "default": True,
            },
            {
                "name": "wave_speed_m_per_ns",
                "label": "波速 (m/ns)",
                "type": "float",
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "motion_comp",
        "auto_tune_candidates": {
            "reference_height_mode": ["mean", "min"],
            "compensate_amplitude": [True, False],
            "compensate_time_shift": [True, False],
            "wave_speed_m_per_ns": [0.05, 0.1, 0.15],
        },
    },
    "trajectory_smoothing": {
        "name": "轨迹平滑",
        "type": "local",
        "func": method_trajectory_smoothing,
        "params": [
            {
                "name": "method",
                "label": "平滑方法",
                "type": "choice",
                "choices": ["savgol", "moving_average"],
                "default": "savgol",
            },
            {
                "name": "window_length",
                "label": "窗口长度",
                "type": "int",
                "default": 21,
                "min": 3,
                "max": 501,
            },
            {
                "name": "polyorder",
                "label": "多项式阶数",
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 7,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "motion_comp",
        "auto_tune_candidates": {
            "method": ["savgol"],
            "window_length": [11, 21, 31, 51],
            "polyorder": [2, 3],
        },
    },
    "stolt_migration": {
        "name": "Stolt/ω-k 迁移（增强版）",
        "type": "local",
        "func": method_stolt_migration,
        "params": [
            {
                "name": "dx",
                "label": "道间距 dx (m)",
                "type": "float",
                "default": 0.05,
                "min": 1e-4,
                "max": 10.0,
            },
            {
                "name": "dt",
                "label": "时间步长 dt (ns)",
                "type": "float",
                "default": 0.1,
                "min": 1e-5,
                "max": 10.0,
            },
            {
                "name": "v",
                "label": "传播速度 v (m/ns)",
                "type": "float",
                "default": 0.10,
                "min": 1e-4,
                "max": 1.0,
            },
            {
                "name": "pad_x",
                "label": "横向补零倍数",
                "type": "int",
                "default": 1,
                "min": 0,
                "max": 8,
            },
            {
                "name": "pad_t",
                "label": "时间补零倍数",
                "type": "int",
                "default": 1,
                "min": 0,
                "max": 8,
            },
            {
                "name": "stolt_jacobian_power",
                "label": "Jacobian补偿幂次",
                "type": "float",
                "default": 0.05,
                "min": 0.0,
                "max": 1.0,
            },
            {
                "name": "stolt_obliquity_power",
                "label": "Obliquity补偿幂次",
                "type": "float",
                "default": 0.05,
                "min": 0.0,
                "max": 1.0,
            },
            {
                "name": "stolt_mask_softness",
                "label": "边缘软掩膜",
                "type": "float",
                "default": 0.03,
                "min": 0.0,
                "max": 0.3,
            },
            {
                "name": "stolt_kz_smooth",
                "label": "kz平滑窗口",
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 15,
            },
            {
                "name": "stolt_depth_gain",
                "label": "深度增益系数",
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 2.0,
            },
            {
                "name": "stolt_depth_gain_power",
                "label": "深度增益幂次",
                "type": "float",
                "default": 1.1,
                "min": 0.5,
                "max": 3.0,
            },
            {
                "name": "stolt_clip_percentile",
                "label": "幅值截断百分位",
                "type": "float",
                "default": 100.0,
                "min": 95.0,
                "max": 100.0,
            },
        ],
    },
    "kirchhoff_migration": {
        "name": "Kirchhoff 迁移",
        "type": "local",
        "func": method_kirchhoff_migration,
        "params": [
            {
                "name": "freq",
                "label": "中心频率 freq (Hz)",
                "type": "float",
                "default": 5.0e7,
                "min": 1.0e6,
                "max": 1.0e9,
            },
            {
                "name": "depth",
                "label": "成像深度 depth (m)",
                "type": "float",
                "default": 40.0,
                "min": 0.1,
                "max": 500.0,
            },
            {
                "name": "v",
                "label": "波速 v (m/ns)",
                "type": "float",
                "default": 0.10,
                "min": 0.01,
                "max": 1.0,
            },
            {
                "name": "weight",
                "label": "TV去噪权重",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
            },
            {
                "name": "alpha",
                "label": "幂增益 alpha",
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 3.0,
            },
            {
                "name": "num_cal",
                "label": "并行分块 num_cal",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 128,
            },
            {
                "name": "hei_cor",
                "label": "高度校正 hei_cor",
                "type": "int",
                "default": 1,
                "min": 0,
                "max": 2,
            },
            {
                "name": "topo_cor",
                "label": "地形校正 topo_cor",
                "type": "int",
                "default": 1,
                "min": 0,
                "max": 2,
            },
        ],
    },
    "time_to_depth": {
        "name": "深度转换与标定",
        "type": "local",
        "func": method_time_to_depth,
        "params": [
            {
                "name": "dt",
                "label": "时间步长 (ns)",
                "type": "float",
                "default": 0.1,
                "min": 0.01,
                "max": 10.0,
            },
            {
                "name": "v",
                "label": "波速 (m/ns)",
                "type": "float",
                "default": 0.10,
                "min": 0.01,
                "max": 0.3,
            },
            {
                "name": "dz",
                "label": "深度网格步长 (m)",
                "type": "float",
                "default": 0.02,
                "min": 0.001,
                "max": 1.0,
            },
        ],
    },
    "sec_gain": {
        "name": "SEC增益（深度补偿）",
        "type": "local",
        "func": method_sec_gain,
        "params": [
            {
                "name": "gain_min",
                "label": "增益下限",
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 10.0,
            },
            {
                "name": "gain_max",
                "label": "增益上限",
                "type": "float",
                "default": 4.5,
                "min": 0.1,
                "max": 20.0,
            },
            {
                "name": "power",
                "label": "曲线幂次",
                "type": "float",
                "default": 1.1,
                "min": 0.2,
                "max": 3.0,
            },
        ],
        "auto_tune_enabled": True,
        "auto_tune_family": "gain",
        "auto_tune_candidates": {
            "gain_min": [1.0],
            "gain_max": [2.5, 3.5, 4.5, 5.5, 7.0, 9.0, 12.0],
            "power": [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2],
        },
    },
    "sliding_avg": {
        "name": "Sliding-average background removal",
        "type": "local",
        "func": method_sliding_average,
        "params": [
            {
                "name": "window_size",
                "label": "Window size",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 200,
            },
            {
                "name": "axis",
                "label": "Axis (0/1)",
                "type": "int",
                "default": 1,
                "min": 0,
                "max": 1,
            },
        ],
    },
}


METHOD_METADATA = {
    "compensatingGain": {
        "category": "gain",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "增益补偿",
    },
    "dewow": {
        "category": "drift_correction",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "低频漂移矫正（Dewow）",
    },
    "set_zero_time": {
        "category": "time_correction",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "零时矫正",
    },
    "agcGain": {
        "category": "gain",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "自动增益控制（AGC）",
    },
    "subtracting_average_2D": {
        "category": "background_suppression",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "平均背景抑制",
    },
    "median_background_2D": {
        "category": "background_suppression",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "中值背景抑制",
    },
    "running_average_2D": {
        "category": "clutter_suppression",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "尖锐杂波抑制",
    },
    "svd_bg": {
        "category": "background_suppression",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "SVD 背景抑制",
    },
    "fk_filter": {
        "category": "filtering",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "F-K 锥形滤波",
    },
    "hankel_svd": {
        "category": "denoising",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "Hankel-SVD 去噪",
    },
    "svd_subspace": {
        "category": "denoising",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "SVD 子空间去噪",
    },
    "wavelet_2d": {
        "category": "denoising",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "Wavelet 2D 去噪",
    },
    "wavelet_svd": {
        "category": "denoising",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "Wavelet-SVD 复合去噪",
    },
    "rpca_background": {
        "category": "background_suppression",
        "maturity": "experimental",
        "visibility": "hidden",
        "display_name": "RPCA 背景抑制",
    },
    "wnnm_placeholder": {
        "category": "background_suppression",
        "maturity": "deferred",
        "visibility": "hidden",
        "display_name": "WNNM 背景抑制（延期）",
    },
    "ccbs": {
        "category": "background_suppression",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "互相关背景抑制（CCBS）",
    },
    "motion_compensation_height": {
        "category": "preprocessing",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "飞行高度归一化",
    },
    "trajectory_smoothing": {
        "category": "preprocessing",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "轨迹平滑",
    },
    "stolt_migration": {
        "category": "migration",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "Stolt 迁移",
    },
    "kirchhoff_migration": {
        "category": "migration",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "Kirchhoff 迁移",
    },
    "time_to_depth": {
        "category": "depth_conversion",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "时间-深度转换",
    },
    "sec_gain": {
        "category": "gain",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "SEC 增益",
    },
    "sliding_avg": {
        "category": "background_suppression",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "滑动平均背景抑制",
    },
}

for _method_key, _meta in METHOD_METADATA.items():
    if _method_key in PROCESSING_METHODS:
        PROCESSING_METHODS[_method_key].update(_meta)
        PROCESSING_METHODS[_method_key]["name"] = _meta["display_name"]


# ============ 显示名称与排序 ============

METHOD_DISPLAY_NAMES = {
    key: value["display_name"] for key, value in METHOD_METADATA.items()
}

PREFERRED_METHOD_ORDER = [
    "set_zero_time",
    "dewow",
    "subtracting_average_2D",
    "median_background_2D",
    "svd_bg",
    "ccbs",
    "sliding_avg",
    "fk_filter",
    "sec_gain",
    "compensatingGain",
    "agcGain",
    "hankel_svd",
    "svd_subspace",
    "wavelet_2d",
    "wavelet_svd",
    "running_average_2D",
    "stolt_migration",
    "kirchhoff_migration",
    "time_to_depth",
]

METHOD_TAGS = {
    "subtracting_average_2D": "推荐",
    "median_background_2D": "备选",
    "agcGain": "备选",
    "dewow": "推荐",
    "set_zero_time": "推荐",
    "svd_bg": "备选",
    "fk_filter": "实验",
    "hankel_svd": "实验",
    "svd_subspace": "实验",
    "wavelet_2d": "实验",
    "wavelet_svd": "实验",
    "stolt_migration": "实验",
    "kirchhoff_migration": "实验",
    "sliding_avg": "实验",
    "running_average_2D": "备选",
}

METHOD_CATEGORY_LABELS = {
    "time_correction": "时间校正",
    "drift_correction": "低频漂移矫正",
    "background_suppression": "背景抑制",
    "clutter_suppression": "尖锐杂波抑制",
    "filtering": "频域滤波",
    "denoising": "去噪",
    "gain": "增益",
    "migration": "迁移成像",
    "depth_conversion": "时间深度转换",
    "experimental": "实验功能",
}

AUTO_TUNE_STAGE_BY_METHOD = {
    "set_zero_time": "zero_time",
    "dewow": "drift",
    "subtracting_average_2D": "background",
    "median_background_2D": "background",
    "svd_bg": "background",
    "fk_filter": "background",
    "ccbs": "background",
    "sec_gain": "gain",
    "compensatingGain": "gain",
    "agcGain": "gain",
    "running_average_2D": "impulse",
    "hankel_svd": "denoise",
    "svd_subspace": "denoise",
    "wavelet_2d": "denoise",
    "wavelet_svd": "denoise",
    "motion_compensation_height": "motion_comp",
    "trajectory_smoothing": "motion_comp",
}

for _method_key, _stage in AUTO_TUNE_STAGE_BY_METHOD.items():
    if _method_key in PROCESSING_METHODS:
        PROCESSING_METHODS[_method_key]["auto_tune_stage"] = _stage


def is_public_method(method_key: str) -> bool:
    """Whether a method should appear in the public GUI lists."""
    method = PROCESSING_METHODS.get(method_key, {})
    if not method or str(method_key).startswith("_"):
        return False
    return str(method.get("visibility", "public")) == "public"


def get_method_display_name(method_key: str) -> str:
    """Return unified user-facing method name."""
    method = PROCESSING_METHODS.get(method_key, {})
    return str(
        METHOD_DISPLAY_NAMES.get(method_key)
        or method.get("display_name")
        or method.get("name")
        or method_key
    )


def get_method_category(method_key: str) -> str:
    """Return internal category key for a method."""
    method = PROCESSING_METHODS.get(method_key, {})
    return str(method.get("category", "experimental"))


def get_auto_tune_stage(method_key: str) -> str:
    """Return stage-level auto-tune grouping for a method."""
    method = PROCESSING_METHODS.get(method_key, {})
    return str(
        method.get("auto_tune_stage")
        or AUTO_TUNE_STAGE_BY_METHOD.get(method_key)
        or method.get("auto_tune_family")
        or ""
    )


def get_method_category_label(method_key: str) -> str:
    """Return user-facing category label for a method."""
    category = get_method_category(method_key)
    return str(METHOD_CATEGORY_LABELS.get(category, category))


def get_public_method_keys() -> list[str]:
    """Return public method keys in preferred display order."""
    ordered = [key for key in PREFERRED_METHOD_ORDER if is_public_method(key)]
    tail = [
        key
        for key in PROCESSING_METHODS.keys()
        if key not in ordered and is_public_method(key)
    ]
    return ordered + tail


def get_public_methods_grouped_by_category() -> list[tuple[str, list[str]]]:
    """Return public methods grouped by category while preserving preferred order."""
    grouped: dict[str, list[str]] = {}
    for key in get_public_method_keys():
        category = get_method_category(key)
        grouped.setdefault(category, []).append(key)

    ordered_categories = []
    for key in get_public_method_keys():
        category = get_method_category(key)
        if category not in ordered_categories:
            ordered_categories.append(category)

    return [(category, grouped.get(category, [])) for category in ordered_categories]


# GUI presets, quality helpers, and workflow presets were moved to
# `core.preset_profiles.py` to keep this module focused on method registration.
