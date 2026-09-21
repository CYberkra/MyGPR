"""Native algorithm registry and execution characteristics.

v0.9.38 启动性能：NATIVE_ALGORITHMS 是注册表元数据（参数 schema、auto-tune
家族、资源系数等）的事实源，启动期 UI 目录构建只需要元数据，不需要实现函数。
算法实现模块（basic/extended/frequency/...）顶层拉起 scipy 栈（importtime
实测 ~1.1s），因此所有实现引用改为 ``_lazy(模块, 属性)`` 惰性解析——首次
调用时才导入实现模块，元数据构建保持零 scipy。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class _LazyCallable:
    """(module, attr) 形式的惰性算法实现引用。

    首次调用时导入目标模块并缓存真实函数；``__module__`` 返回目标模块路径，
    供 methods_registry 由 ``func.__module__`` 推导展示模块名。
    """

    __slots__ = ("_module", "_attr", "_cached")

    def __init__(self, module: str, attr: str) -> None:
        self._module = module
        self._attr = attr
        self._cached: Any = None

    def _resolve(self) -> Any:
        if self._cached is None:
            import importlib

            self._cached = getattr(importlib.import_module(self._module), self._attr)
        return self._cached

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    @property
    def __module__(self) -> str:
        return self._module

    def __repr__(self) -> str:
        return f"_LazyCallable({self._module}:{self._attr})"


_ALG = "mygpr.infrastructure.processing.algorithms"


def _lazy(module: str, attr: str) -> Any:
    """惰性引用 ``module.attr`` 形式的算法实现（见模块 docstring）。"""
    return _LazyCallable(module, attr)


AlgorithmFunction = Callable[[Any, dict[str, Any]], tuple[np.ndarray, dict[str, Any]]]
GlobalFileFunction = Callable[[Any, Any, dict[str, Any], Any, int], dict[str, Any]]
ResourceEstimator = Callable[[tuple[int, int], np.dtype | str, dict[str, Any], dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class NativeAlgorithm:
    method_id: str
    name: str
    category: str
    function: AlgorithmFunction
    block_axis: str
    implementation_version: str = "native-1.0"
    auto_tune_family: str = ""
    auto_tune_stage: str = ""
    parameter_schema: dict[str, Any] | None = None
    memory_multiplier: float = 3.0
    temporary_multiplier: float = 1.0
    relative_cost: str = "low"
    file_function: GlobalFileFunction | None = None
    resource_estimator: ResourceEstimator | None = None

    @property
    def supports_chunking(self) -> bool:
        return self.block_axis in {"rows", "columns"}

    @property
    def supports_file_backed(self) -> bool:
        return self.block_axis in {"rows", "columns", "global"}

    def supports_block_params(self, params: dict[str, Any]) -> bool:
        if self.method_id in {"subtracting_average_2D", "median_background_2D"}:
            ranged = any(
                params.get(key) not in (None, "", 0, 0.0)
                for key in ("time_start_idx", "time_end_idx", "time_start_ns", "time_end_ns", "edge_taper_samples")
            )
            return not ranged
        if self.method_id == "sliding_avg":
            return int(params.get("axis", 1)) == 1
        if self.method_id == "agcGain" and bool(params.get("_low_energy_guard", False)):
            return False
        if self.method_id == "trace_median_filter" and bool(params.get("preserve_mean", False)):
            return False
        return True


def _schema(**items: Any) -> dict[str, Any]:
    return {key: value for key, value in items.items()}


NATIVE_ALGORITHMS: dict[str, NativeAlgorithm] = {
    "time_cut": NativeAlgorithm(
        "time_cut", "Time-window crop", "preprocess", _lazy(f"{_ALG}.extended", "native_time_cut"), "loaded_global",
        implementation_version="native-extended-1.0",
        parameter_schema=_schema(mode={"type": "str", "default": "remove_below"}, time_start_ns={"type": "float", "default": 0.0}, time_end_ns={"type": "float", "default": 0.0}),
        memory_multiplier=2.0, temporary_multiplier=1.0, relative_cost="low",
    ),
    "trace_qc": NativeAlgorithm(
        "trace_qc", "Trace quality control", "preprocess", _lazy(f"{_ALG}.extended", "native_trace_qc"), "loaded_global",
        implementation_version="native-extended-1.0",
        parameter_schema=_schema(mode={"type": "str", "default": "mark"}, empty_rms_threshold={"type": "float", "default": 0.0}, spike_zscore={"type": "float", "default": 6.0, "min": 0.0}, manual_trace_indices={"type": "str", "default": ""}),
        memory_multiplier=3.0, temporary_multiplier=1.0, relative_cost="low",
    ),
    "energy_decay_gain": NativeAlgorithm(
        "energy_decay_gain", "Robust energy-decay gain", "gain", _lazy(f"{_ALG}.extended", "native_energy_decay_gain"), "loaded_global",
        implementation_version="native-extended-1.0", auto_tune_family="gain", auto_tune_stage="gain",
        parameter_schema=_schema(strength={"type": "float", "default": 1.0}, smoothing_samples={"type": "int", "default": 31, "min": 1}, min_gain={"type": "float", "default": 0.5}, max_gain={"type": "float", "default": 8.0}, floor_ratio={"type": "float", "default": 0.05}),
        memory_multiplier=3.0, temporary_multiplier=1.0, relative_cost="medium",
    ),
    "amplitude_scale": NativeAlgorithm(
        "amplitude_scale", "Amplitude scaling", "gain", _lazy(f"{_ALG}.extended", "native_amplitude_scale"), "loaded_global",
        implementation_version="native-extended-1.0",
        parameter_schema=_schema(mode={"type": "str", "default": "constant"}, scale={"type": "float", "default": 1.0}, target={"type": "float", "default": 1.0}),
        memory_multiplier=2.0, temporary_multiplier=1.0, relative_cost="low",
    ),
    "median_background_2D": NativeAlgorithm(
        "median_background_2D", "Median background suppression", "background", _lazy(f"{_ALG}.extended", "native_median_background"), "rows",
        implementation_version="native-extended-1.0", auto_tune_family="background", auto_tune_stage="background",
        parameter_schema=_schema(ntraces={"type": "int", "default": 51, "min": 1}, time_start_ns={"type": "float", "default": 0.0}, time_end_ns={"type": "float", "default": 0.0}),
        memory_multiplier=4.0, temporary_multiplier=1.0, relative_cost="medium",
    ),
    "wavelet_2d": NativeAlgorithm(
        "wavelet_2d", "2D wavelet denoising", "denoise", _lazy(f"{_ALG}.extended", "native_wavelet_2d"), "global",
        implementation_version="native-extended-1.0", auto_tune_family="denoise", auto_tune_stage="denoise",
        parameter_schema=_schema(wavelet={"type": "str", "default": "db4"}, levels={"type": "int", "default": 2, "min": 1}, threshold={"type": "float", "default": 1.0, "min": 0.0, "max": 1.0}, threshold_strategy={"type": "str", "default": "mad_universal"}),
        memory_multiplier=8.0, temporary_multiplier=3.0, relative_cost="high",
    ),
    "hilbert_envelope": NativeAlgorithm(
        "hilbert_envelope", "Hilbert envelope", "attribute", _lazy(f"{_ALG}.extended", "native_hilbert_envelope"), "columns",
        implementation_version="native-extended-1.0",
        parameter_schema=_schema(normalize={"type": "bool", "default": False}, log_compress={"type": "bool", "default": False}),
        memory_multiplier=4.0, temporary_multiplier=1.0, relative_cost="medium",
    ),
    "ccbs": NativeAlgorithm(
        "ccbs", "Cross-correlation background subtraction", "background", _lazy(f"{_ALG}.extended", "native_ccbs"), "loaded_global",
        implementation_version="native-extended-1.0",
        parameter_schema=_schema(
            use_custom_ref={"type": "bool", "default": False},
            reference_trace_index={"type": "int", "default": -1, "min": -1},
        ),
        memory_multiplier=5.0, temporary_multiplier=1.0, relative_cost="medium",
    ),
    "time_to_depth": NativeAlgorithm(
        "time_to_depth", "Time-to-depth conversion", "migration", _lazy(f"{_ALG}.extended", "native_time_to_depth"), "loaded_global",
        implementation_version="native-extended-1.0",
        parameter_schema=_schema(dt={"type": "float", "default": 0.1}, v={"type": "float", "default": 0.1}, dz={"type": "float", "default": 0.02}),
        memory_multiplier=4.0, temporary_multiplier=2.0, relative_cost="medium",
    ),
    "compensatingGain": NativeAlgorithm(
        "compensatingGain", "Manual gain compensation", "gain", _lazy(f"{_ALG}.basic", "method_compensating_gain"), "columns",
        auto_tune_family="gain", parameter_schema=_schema(gain_min={"type": "float", "default": 1.0}, gain_max={"type": "float", "default": 6.0}),
    ),
    "dewow": NativeAlgorithm(
        "dewow", "Low-frequency drift correction", "baseline", _lazy(f"{_ALG}.basic", "method_dewow_native"), "columns",
        auto_tune_family="drift", parameter_schema=_schema(window={"type": "int", "default": 23, "min": 1}),
    ),
    "set_zero_time": NativeAlgorithm(
        "set_zero_time", "Zero-time correction", "baseline", _lazy(f"{_ALG}.basic", "method_zero_time"), "columns",
        auto_tune_family="zero_time", parameter_schema=_schema(new_zero_time={"type": "float", "default": 5.0, "min": 0.0}),
    ),
    "agcGain": NativeAlgorithm(
        "agcGain", "Automatic gain control", "gain", _lazy(f"{_ALG}.basic", "method_agc"), "columns",
        auto_tune_family="gain", parameter_schema=_schema(window={"type": "int", "default": 11, "min": 1}),
    ),
    "sec_gain": NativeAlgorithm(
        "sec_gain", "SEC gain", "gain", _lazy(f"{_ALG}.basic", "method_sec_gain_native"), "columns",
        auto_tune_family="gain", parameter_schema=_schema(gain_min={"type": "float", "default": 1.0}, gain_max={"type": "float", "default": 6.0}, power={"type": "float", "default": 1.0}),
    ),
    "subtracting_average_2D": NativeAlgorithm(
        "subtracting_average_2D", "Mean-trace background suppression", "background", _lazy(f"{_ALG}.basic", "method_remove_background"), "rows",
        auto_tune_family="background", parameter_schema=_schema(ntraces={"type": "int", "default": 501, "min": 1}),
    ),
    "running_average_2D": NativeAlgorithm(
        "running_average_2D", "Trace running average", "denoise", _lazy(f"{_ALG}.basic", "method_running_average"), "rows",
        auto_tune_family="impulse", parameter_schema=_schema(ntraces={"type": "int", "default": 9, "min": 1}),
    ),
    "sliding_avg": NativeAlgorithm(
        "sliding_avg", "Sliding-average background suppression", "background", _lazy(f"{_ALG}.basic", "method_sliding_background"), "rows",
        auto_tune_family="background", parameter_schema=_schema(window_size={"type": "int", "default": 10, "min": 1}, axis={"type": "int", "default": 1}),
    ),
    "frequency_filter_1d": NativeAlgorithm(
        "frequency_filter_1d", "Frequency filter", "filter", _lazy(f"{_ALG}.frequency", "method_frequency_filter"), "columns",
        auto_tune_family="frequency", parameter_schema=_schema(filter_type={"type": "str", "default": "bandpass"}, low_freq_mhz={"type": "float", "default": 10.0}, high_freq_mhz={"type": "float", "default": 800.0}, taper_ratio={"type": "float", "default": 0.0, "min": 0.0, "max": 0.5}),
    ),
    "trace_median_filter": NativeAlgorithm(
        "trace_median_filter", "Trace median filter", "denoise", _lazy(f"{_ALG}.basic", "method_trace_median"), "rows",
        auto_tune_family="denoise", parameter_schema=_schema(window_traces={"type": "int", "default": 5, "min": 1}),
    ),
    "trace_savgol_filter": NativeAlgorithm(
        "trace_savgol_filter", "Trace Savitzky-Golay filter", "denoise", _lazy(f"{_ALG}.basic", "method_trace_savgol"), "rows",
        auto_tune_family="denoise", parameter_schema=_schema(window_traces={"type": "int", "default": 7, "min": 1}, polyorder={"type": "int", "default": 2}),
    ),
    "svd_bg": NativeAlgorithm(
        "svd_bg", "SVD background removal", "background", _lazy(f"{_ALG}.global_spectral", "method_svd_background_native"), "global",
        implementation_version="native-global-1.0", auto_tune_family="background",
        parameter_schema=_schema(rank={"type": "int", "default": 1, "min": 1}, solver={"type": "str", "default": "auto"}),
        memory_multiplier=3.5, temporary_multiplier=2.0, relative_cost="high",
        file_function=_lazy(f"{_ALG}.global_spectral", "file_svd_background_native"),
    ),
    "svd_subspace": NativeAlgorithm(
        "svd_subspace", "SVD subspace reconstruction", "denoise", _lazy(f"{_ALG}.global_spectral", "method_svd_subspace_native"), "global",
        implementation_version="native-global-1.0", auto_tune_family="denoise",
        parameter_schema=_schema(rank_start={"type": "int", "default": 1, "min": 1}, rank_end={"type": "int", "default": 2, "min": 1}, solver={"type": "str", "default": "auto"}),
        memory_multiplier=3.5, temporary_multiplier=2.0, relative_cost="high",
        file_function=_lazy(f"{_ALG}.global_spectral", "file_svd_subspace_native"),
    ),
    "fk_filter": NativeAlgorithm(
        "fk_filter", "F-K cone filter", "filter", _lazy(f"{_ALG}.global_spectral", "method_fk_filter_native"), "global",
        implementation_version="native-global-1.0", auto_tune_family="fk", auto_tune_stage="frequency",
        parameter_schema=_schema(angle_low={"type": "float", "default": 10.0}, angle_high={"type": "float", "default": 65.0}, taper_width={"type": "float", "default": 5.0}),
        memory_multiplier=7.0, temporary_multiplier=2.0, relative_cost="high",
    ),
    "stolt_migration": NativeAlgorithm(
        "stolt_migration", "Stolt migration", "migration", _lazy(f"{_ALG}.global_spectral", "method_stolt_migration_native"), "global",
        implementation_version="native-global-1.0",
        parameter_schema=_schema(dx={"type": "float", "default": 0.05}, dt={"type": "float", "default": 0.1}, v={"type": "float", "default": 0.10}, pad_x={"type": "int", "default": 1}, pad_t={"type": "int", "default": 1},
                                 stolt_jacobian_power={"type": "float", "default": 0.0, "min": 0.0, "max": 1.0},
                                 stolt_obliquity_power={"type": "float", "default": 1.0, "min": 0.0, "max": 1.0}),
        memory_multiplier=12.0, temporary_multiplier=3.0, relative_cost="very_high",
    ),
    "motion_compensation_height": NativeAlgorithm(
        "motion_compensation_height", "Height motion compensation", "motion_compensation",
        _lazy(f"{_ALG}.motion", "native_motion_height"), "loaded_global", implementation_version="native-motion-2.0",
        auto_tune_family="motion_comp", auto_tune_stage="motion_comp",
        parameter_schema=_schema(
            reference_height_mode={"type": "str", "default": "mean"},
            manual_height={"type": "float", "default": 10.0, "min": 0.0},
            height_source={"type": "str", "default": "auto", "choices": ("auto", "height_agl_m", "flight_height_m")},
            compensate_amplitude={"type": "bool", "default": True},
            compensate_time_shift={"type": "bool", "default": True},
            wave_speed_m_per_ns={"type": "float", "default": 0.299792458, "min": 0.01},
        ),
        memory_multiplier=4.0, temporary_multiplier=1.0, relative_cost="medium",
    ),
    "motion_compensation_speed": NativeAlgorithm(
        "motion_compensation_speed", "Equal-distance trace resampling", "motion_compensation",
        _lazy(f"{_ALG}.motion", "native_motion_speed"), "loaded_global", implementation_version="native-motion-2.0",
        auto_tune_family="motion_comp", auto_tune_stage="motion_comp",
        parameter_schema=_schema(spacing_m={"type": "float", "default": 0.0, "min": 0.0}),
        memory_multiplier=4.0, temporary_multiplier=2.0, relative_cost="medium",
    ),
    "trajectory_smoothing": NativeAlgorithm(
        "trajectory_smoothing", "Trajectory smoothing", "motion_compensation",
        _lazy(f"{_ALG}.motion", "native_trajectory_smoothing"), "loaded_global", implementation_version="native-motion-2.0",
        auto_tune_family="motion_comp", auto_tune_stage="motion_comp",
        parameter_schema=_schema(method={"type": "str", "default": "savgol"}, window_length={"type": "int", "default": 21, "min": 3}, polyorder={"type": "int", "default": 3, "min": 1}),
        memory_multiplier=2.0, temporary_multiplier=1.0, relative_cost="low",
    ),
    "motion_compensation_attitude": NativeAlgorithm(
        "motion_compensation_attitude", "Attitude/APC correction", "motion_compensation",
        _lazy(f"{_ALG}.motion", "native_motion_attitude"), "loaded_global", implementation_version="native-motion-2.0",
        auto_tune_family="motion_comp", auto_tune_stage="motion_comp",
        parameter_schema=_schema(
            apc_offset_x_m={"type": "float", "default": 0.0},
            apc_offset_y_m={"type": "float", "default": 0.0},
            apc_offset_z_m={"type": "float", "default": 0.0},
            max_abs_tilt_deg={"type": "float", "default": 20.0, "min": 1.0},
        ),
        memory_multiplier=2.5, temporary_multiplier=1.0, relative_cost="low",
    ),
    "motion_compensation_vibration": NativeAlgorithm(
        "motion_compensation_vibration", "Vibration suppression", "motion_compensation",
        _lazy(f"{_ALG}.motion", "native_motion_vibration"), "loaded_global", implementation_version="native-motion-2.0",
        auto_tune_family="denoise", auto_tune_stage="artifact",
        parameter_schema=_schema(
            smooth_window={"type": "int", "default": 9, "min": 1},
            preserve_row_percentile={"type": "float", "default": 94.0},
            preserve_mix={"type": "float", "default": 0.35},
            background_mix={"type": "float", "default": 0.02},
            max_restore_gain={"type": "float", "default": 1.25},
        ),
        memory_multiplier=6.0, temporary_multiplier=2.0, relative_cost="high",
    ),
    "motion_compensation_v2": NativeAlgorithm(
        "motion_compensation_v2", "Unified motion compensation V2", "motion_compensation",
        _lazy(f"{_ALG}.motion", "native_motion_v2"), "loaded_global", implementation_version="native-motion-2.0",
        auto_tune_family="motion_comp", auto_tune_stage="motion_comp",
        parameter_schema=_schema(
            height_reference_mode={"type": "str", "default": "mean"},
            manual_height_m={"type": "float", "default": 1.5, "min": 0.01},
            height_source={"type": "str", "default": "auto"},
            compensate_time_shift={"type": "bool", "default": True},
            compensate_amplitude={"type": "bool", "default": True},
            max_shift_samples={"type": "float", "default": 0.0},
            max_shift_ns={"type": "float", "default": 20.0},
            max_amplitude_scale={"type": "float", "default": 2.0},
            resample_spacing_m={"type": "float", "default": 0.0, "min": 0.0},
            interpolation_mode={"type": "str", "default": "linear"},
            apc_offset_x_m={"type": "float", "default": 0.0},
            apc_offset_y_m={"type": "float", "default": 0.0},
            apc_offset_z_m={"type": "float", "default": 0.0},
            max_abs_tilt_deg={"type": "float", "default": 20.0, "min": 1.0},
            air_wave_speed_m_per_ns={"type": "float", "default": 0.299792458, "min": 0.01},
        ),
        memory_multiplier=6.0, temporary_multiplier=2.0, relative_cost="high",
    ),
    "kirchhoff_migration": NativeAlgorithm(
        "kirchhoff_migration", "Kirchhoff migration", "migration",
        _lazy(f"{_ALG}.kirchhoff", "method_kirchhoff_migration_native"), "loaded_global",
        implementation_version="native-kirchhoff-2.0",
        parameter_schema=_schema(
            freq={"type": "float", "default": 5.0e7, "min": 1.0},
            depth={"type": "float", "default": 40.0, "min": 0.01},
            v={"type": "float", "default": 0.10, "min": 0.001},
            alpha={"type": "float", "default": 1.0},
            weight={"type": "float", "default": 0.05, "min": 0.0},
            num_cal={"type": "int", "default": 1, "min": 1},
            topo_cor={"type": "int", "default": 0},
            hei_cor={"type": "int", "default": 0},
            backend={"type": "str", "default": "auto"},
            length_m={"type": "float", "default": 0.0, "min": 0.0},
            time_window_ns={"type": "float", "default": 0.0, "min": 0.0},
            allow_gpu_runtime_fallback={"type": "bool", "default": True},
        ),
        memory_multiplier=12.0, temporary_multiplier=2.0, relative_cost="very_high",
        resource_estimator=_lazy(f"{_ALG}.kirchhoff", "estimate_kirchhoff_resources"),
    ),
    "rtm_migration": NativeAlgorithm(
        "rtm_migration", "Experimental RTM migration", "migration",
        _lazy(f"{_ALG}.rtm", "method_rtm_migration_native"), "loaded_global",
        implementation_version="experimental-rtm-1.0",
        parameter_schema=_schema(
            time_window_ns={"type": "float", "default": 0.0, "min": 0.0},
            dt_ns={"type": "float", "default": 0.1, "min": 1.0e-9},
            length_m={"type": "float", "default": 0.0, "min": 0.0},
            dx_m={"type": "float", "default": 0.0, "min": 0.0},
            dx={"type": "float", "default": 0.0, "min": 0.0},
            v={"type": "float", "default": 0.10, "min": 0.001},
            velocity_m_per_ns={"type": "float", "default": 0.10, "min": 0.001},
            depth_m={"type": "float", "default": 20.0, "min": 0.01},
            depth={"type": "float", "default": 20.0, "min": 0.01},
            dz_m={"type": "float", "default": 0.0, "min": 0.0},
            dz={"type": "float", "default": 0.0, "min": 0.0},
            cfl_limit={"type": "float", "default": 0.65, "min": 0.20, "max": 0.70},
            max_grid_elements={"type": "int", "default": 40_000_000, "min": 1},
            max_cell_updates={"type": "int", "default": 1_500_000_000, "min": 1},
            source_depth_index={"type": "int", "default": 1, "min": 0},
            boundary_width={"type": "int", "default": 24, "min": 0},
            boundary_strength={"type": "float", "default": 3.5, "min": 0.0, "max": 12.0},
            injection_scale={"type": "float", "default": 1.0},
            remove_surface_row={"type": "bool", "default": True},
            normalize={"type": "bool", "default": False},
        ),
        memory_multiplier=8.0, temporary_multiplier=1.0, relative_cost="very_high",
        resource_estimator=_lazy(f"{_ALG}.rtm", "estimate_rtm_resources"),
    ),
    "hankel_svd": NativeAlgorithm(
        "hankel_svd", "Hankel SVD / V-MSSA", "denoise", _lazy(f"{_ALG}.hankel_mssa", "method_hankel_svd_native"), "global",
        implementation_version="native-global-1.0", auto_tune_family="denoise",
        parameter_schema=_schema(window_length={"type": "int", "default": 0, "min": 0}, rank={"type": "int", "default": 0, "min": 0}, aggressiveness={"type": "float", "default": 0.5}),
        memory_multiplier=10.0, temporary_multiplier=2.0, relative_cost="very_high",
    ),
    "rpca_background": NativeAlgorithm(
        "rpca_background", "Robust PCA background suppression", "background", _lazy(f"{_ALG}.rpca", "method_rpca_background_native"), "global",
        implementation_version="native-global-1.0",
        parameter_schema=_schema(lam={"type": "float", "default": 0.08}, mu={"type": "float", "default": 0.0}, max_iter={"type": "int", "default": 120}, tol={"type": "float", "default": 1e-6}),
        memory_multiplier=9.0, temporary_multiplier=3.0, relative_cost="very_high",
    ),
    "mixed_phase_deconvolution": NativeAlgorithm(
        "mixed_phase_deconvolution", "混合相位反褶积 (Schmelzbach 2015)", "denoise", _lazy(f"{_ALG}.extended", "native_mixed_phase_deconvolution"), "loaded_global",
        implementation_version="native-extended-1.0", auto_tune_family="denoise", auto_tune_stage="denoise",
        parameter_schema=_schema(
            operator_length={"type": "int", "default": 35, "min": 5},
            supertrace_traces={"type": "int", "default": 11, "min": 3},
            prewhitening={"type": "float", "default": 0.1, "min": 0.0, "max": 0.1},
            window_start_ns={"type": "float", "default": 0.0},
            window_end_ns={"type": "float", "default": 0.0},
            rotation_step_deg={"type": "float", "default": 5.0, "min": 1.0, "max": 45.0},
            apply_phase_rotation={"type": "bool", "default": True},
        ),
        memory_multiplier=4.0, temporary_multiplier=1.0, relative_cost="medium",
    ),
    "inverse_q": NativeAlgorithm(
        "inverse_q", "Inverse-Q 衰减补偿 (Wang 2002)", "filter", _lazy(f"{_ALG}.extended", "native_inverse_q"), "loaded_global",
        implementation_version="native-extended-1.0", auto_tune_family="filter", auto_tune_stage="frequency",
        parameter_schema=_schema(
            q_value={"type": "float", "default": 50.0, "min": 1.0},
            gain_limit_db={"type": "float", "default": 40.0, "min": 10.0, "max": 100.0},
        ),
        memory_multiplier=4.0, temporary_multiplier=1.0, relative_cost="medium",
    ),

}
