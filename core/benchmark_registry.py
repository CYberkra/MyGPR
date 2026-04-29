#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark sample registry and scenario taxonomy for Stage A foundations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from core.methods_registry import PROCESSING_METHODS
from core.quality_metrics import compute_motion_quality_metrics


DEFAULT_BENCHMARK_SEED = 42


@dataclass(frozen=True)
class BenchmarkSampleSpec:
    """Declarative benchmark sample specification."""

    sample_id: str
    scenario: str
    title: str
    description: str
    default_methods: tuple[str, ...]
    focus_metrics: tuple[str, ...]
    tags: tuple[str, ...]
    builder: Callable[[int], tuple[np.ndarray, dict[str, Any]]]


BENCHMARK_SCENARIOS: dict[str, dict[str, Any]] = {
    "zero_time": {
        "label": "零时与首波定位",
        "goal": "验证零时矫正是否压低前零时能量并稳定首波。",
    },
    "drift_background": {
        "label": "低频漂移与背景抑制",
        "goal": "验证低频漂移、水平背景和主反射保真之间的平衡。",
    },
    "clutter_gain": {
        "label": "增益与尖锐杂波抑制",
        "goal": "验证深部可见性增强与尖锐杂波控制之间的平衡。",
    },
    "motion_compensation": {
        "label": "五项运动误差基准",
        "goal": "验证高度、横向、速度、姿态与周期性振动背景误差可被客观度量。",
    },
}


def _base_header_info(
    samples: int,
    traces: int,
    trace_interval_m: float = 0.09,
    total_time_ns: float = 700.0,
) -> dict[str, Any]:
    return {
        "a_scan_length": int(samples),
        "num_traces": int(traces),
        "total_time_ns": float(total_time_ns),
        "trace_interval_m": float(trace_interval_m),
    }


def _shift_traces_linear(data: np.ndarray, shifts_samples: np.ndarray) -> np.ndarray:
    """Shift each trace by a fractional number of samples with zero fill."""
    arr = np.asarray(data, dtype=np.float64)
    shifts = np.asarray(shifts_samples, dtype=np.float64).reshape(-1)
    samples, traces = arr.shape
    sample_index = np.arange(samples, dtype=np.float64)
    shifted = np.zeros_like(arr)
    for trace_idx in range(min(traces, shifts.size)):
        shifted[:, trace_idx] = np.interp(
            sample_index - float(shifts[trace_idx]),
            sample_index,
            arr[:, trace_idx],
            left=0.0,
            right=0.0,
        )
    return shifted


def _local_xy_to_lon_lat(
    local_x_m: np.ndarray,
    local_y_m: np.ndarray,
    lon0: float = 116.3913,
    lat0: float = 39.9075,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert local metric coordinates to deterministic lon/lat arrays."""
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(lat0))
    longitude = lon0 + np.asarray(local_x_m, dtype=np.float64) / meters_per_deg_lon
    latitude = lat0 + np.asarray(local_y_m, dtype=np.float64) / meters_per_deg_lat
    return longitude, latitude


def _compute_footprint_xy(
    local_x_m: np.ndarray,
    local_y_m: np.ndarray,
    roll_deg: np.ndarray,
    pitch_deg: np.ndarray,
    yaw_deg: np.ndarray,
    target_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate nadir footprint displacement from attitude angles."""
    yaw_rad = np.deg2rad(np.asarray(yaw_deg, dtype=np.float64))
    pitch_offset = float(target_depth_m) * np.tan(np.deg2rad(pitch_deg))
    roll_offset = float(target_depth_m) * np.tan(np.deg2rad(roll_deg))
    local_x = np.asarray(local_x_m, dtype=np.float64)
    local_y = np.asarray(local_y_m, dtype=np.float64)
    footprint_x = local_x + pitch_offset * np.cos(yaw_rad) - roll_offset * np.sin(yaw_rad)
    footprint_y = local_y + pitch_offset * np.sin(yaw_rad) + roll_offset * np.cos(yaw_rad)
    return footprint_x, footprint_y


def _build_zero_time_fixture(seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples, traces = 192, 72
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, traces, dtype=np.float64)[None, :]
    data = 0.03 * rng.normal(size=(samples, traces))
    data += 0.08 * np.sin(2.0 * np.pi * 2.2 * t)
    data += 0.04 * (t - 0.3)

    first_break = 18 + np.round(
        3.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, traces))
    ).astype(int)
    for col, idx in enumerate(first_break):
        pulse = np.array([0.7, 1.8, 1.1, 0.45], dtype=np.float64)
        data[idx : idx + len(pulse), col] += pulse

    data[95:100, :] += 0.18
    data[122:126, 16:54] += 0.32
    return data.astype(np.float32), {
        "header_info": _base_header_info(samples, traces),
        "reference_zero_idx": 18,
        "notes": "Synthetic sample emphasizing first-break stability and pre-zero suppression.",
    }


def _build_drift_background_fixture(seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples, traces = 224, 84
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, traces, dtype=np.float64)[None, :]
    data = 0.025 * rng.normal(size=(samples, traces))
    data += 0.22 * np.sin(2.0 * np.pi * 1.4 * t)
    data += 0.14 * np.cos(2.0 * np.pi * 0.5 * x)
    data += 0.18 * (t - 0.5)

    data[64:68, :] += 0.34
    data[108:113, :] += 0.22
    data[150:154, 10:72] += 0.45
    data[178:183, 26:64] += 0.29
    return data.astype(np.float32), {
        "header_info": _base_header_info(samples, traces),
        "reference_zero_idx": 16,
        "notes": "Synthetic sample emphasizing drift removal and horizontal background suppression.",
    }


def _build_clutter_gain_fixture(seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples, traces = 256, 96
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    data = 0.02 * rng.normal(size=(samples, traces))
    attenuation = np.exp(-2.5 * t)
    data += 0.12 * attenuation * np.sin(2.0 * np.pi * 3.5 * t)

    for center in (72, 118, 186, 220):
        width = 2 + (center % 3)
        amp = 0.55 if center < 150 else 0.82
        data[center : center + width, 12:84] += amp

    spike_rows = rng.integers(12, samples - 12, size=18)
    spike_cols = rng.integers(0, traces, size=18)
    data[spike_rows, spike_cols] += rng.uniform(1.5, 2.4, size=18)
    data[198:206, 18:74] += 0.22
    return data.astype(np.float32), {
        "header_info": _base_header_info(samples, traces),
        "reference_zero_idx": 14,
        "notes": "Synthetic sample emphasizing deep visibility, clipping risk, and sharp-clutter control.",
    }


def _build_motion_compensation_fixture(seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a deterministic UAV-GPR line with five injected motion errors."""
    rng = np.random.default_rng(seed)
    samples, traces = 240, 96
    total_time_ns = 180.0
    trace_interval_m = 0.45
    # Keep the benchmark aligned with the current V1 implementation contract.
    # This constant is an implementation assumption for deterministic evidence,
    # not a claim that free-space / air-path propagation should always be 0.1 m/ns.
    wave_speed_m_per_ns = 0.1
    target_depth_m = 1.8
    ridge_row_range = (68, 92)
    target_row_range = (136, 176)
    banding_trace_band = (0.05, 0.18)
    banding_row_range = (0, samples)

    trace_phase = np.linspace(0.0, 1.0, traces, dtype=np.float64)
    sample_phase = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    trace_axis = trace_phase[None, :]
    sample_axis = sample_phase[:, None]

    clean = 0.010 * rng.normal(size=(samples, traces))
    clean += 0.022 * np.sin(2.0 * np.pi * 1.4 * sample_axis)
    clean += 0.015 * np.cos(2.0 * np.pi * 1.2 * trace_axis) * np.exp(-2.6 * sample_axis)
    clean += 0.010 * (sample_axis - 0.45)

    reflector_ridge_idx = np.full(traces, 78, dtype=np.int32)
    ridge_pulse = np.array([0.22, 1.15, 2.25, 1.05, 0.32], dtype=np.float64)
    target_pulse = np.array([0.16, 0.52, 0.88, 0.56, 0.18], dtype=np.float64)
    for trace_idx, ridge_idx in enumerate(reflector_ridge_idx):
        clean[ridge_idx - 2 : ridge_idx + 3, trace_idx] += ridge_pulse
        target_center = 148 + int(np.round(2.0 * np.sin(2.0 * np.pi * trace_phase[trace_idx])))
        target_gain = 0.95 + 0.18 * np.cos(2.0 * np.pi * 2.0 * trace_phase[trace_idx])
        clean[target_center - 2 : target_center + 3, trace_idx] += target_gain * target_pulse
        hyperbola_row = 170 + int(np.round(10.0 * ((trace_idx - (traces - 1) / 2.0) / traces) ** 2))
        clean[hyperbola_row - 1 : hyperbola_row + 2, trace_idx] += np.array(
            [0.18, 0.62, 0.24],
            dtype=np.float64,
        )

    gt_spacing = np.full(traces - 1, trace_interval_m, dtype=np.float64)
    gt_trace_distance = np.concatenate(([0.0], np.cumsum(gt_spacing)))
    gt_local_x = gt_trace_distance.copy()
    gt_local_y = 0.22 * np.sin(2.0 * np.pi * trace_phase) + 0.05 * np.sin(
        2.0 * np.pi * 3.0 * trace_phase
    )
    gt_height = 1.60 + 0.04 * np.cos(2.0 * np.pi * trace_phase)
    gt_roll = 0.7 * np.sin(2.0 * np.pi * trace_phase)
    gt_pitch = 0.5 * np.cos(2.0 * np.pi * 1.5 * trace_phase)
    gt_yaw = 0.8 * np.sin(2.0 * np.pi * 0.5 * trace_phase)
    gt_longitude, gt_latitude = _local_xy_to_lon_lat(gt_local_x, gt_local_y)
    gt_footprint_x, gt_footprint_y = _compute_footprint_xy(
        gt_local_x,
        gt_local_y,
        gt_roll,
        gt_pitch,
        gt_yaw,
        target_depth_m=target_depth_m,
    )

    height_error = 0.13 * np.sin(2.0 * np.pi * 2.3 * trace_phase + 0.2) + 0.045 * np.cos(
        2.0 * np.pi * 8.0 * trace_phase
    )
    lateral_error = 0.38 * np.sin(2.0 * np.pi * 1.1 * trace_phase + 0.3) + 0.09 * np.cos(
        2.0 * np.pi * 6.0 * trace_phase
    )
    spacing_scale = 1.0 + 0.28 * np.sin(2.0 * np.pi * 1.2 * trace_phase[1:] - 0.4) + 0.10 * np.cos(
        2.0 * np.pi * 4.1 * trace_phase[1:]
    )
    obs_spacing = trace_interval_m * np.clip(spacing_scale, 0.58, None)
    obs_trace_distance = np.concatenate(([0.0], np.cumsum(obs_spacing)))
    obs_local_x = obs_trace_distance.copy()
    obs_local_y = gt_local_y + lateral_error
    obs_height = gt_height + height_error
    obs_roll = gt_roll + 4.5 * np.sin(2.0 * np.pi * 1.4 * trace_phase + 0.5) + 0.8 * np.cos(
        2.0 * np.pi * 5.0 * trace_phase
    )
    obs_pitch = gt_pitch + 3.6 * np.cos(2.0 * np.pi * 1.7 * trace_phase - 0.1)
    obs_yaw = gt_yaw + 5.2 * np.sin(2.0 * np.pi * 0.9 * trace_phase + 0.15)
    obs_longitude, obs_latitude = _local_xy_to_lon_lat(obs_local_x, obs_local_y)
    obs_footprint_x, obs_footprint_y = _compute_footprint_xy(
        obs_local_x,
        obs_local_y,
        obs_roll,
        obs_pitch,
        obs_yaw,
        target_depth_m=target_depth_m,
    )

    dt_ns = total_time_ns / max(samples - 1, 1)
    shift_samples = 2.0 * (obs_height - gt_height) / wave_speed_m_per_ns / dt_ns
    raw = _shift_traces_linear(clean, shift_samples)
    amplitude_scale = (gt_height / np.maximum(obs_height, 0.25)) ** 2
    raw *= amplitude_scale[np.newaxis, :]

    depth_envelope = 0.28 + 0.72 * np.exp(-2.2 * sample_axis)
    banding = 0.18 * depth_envelope * np.sin(
        2.0 * np.pi * (7.0 * trace_axis + 0.35 * np.sin(2.0 * np.pi * 4.0 * sample_axis))
    )
    background = 0.055 * np.cos(2.0 * np.pi * 0.85 * sample_axis) * (
        1.0 + 0.35 * np.sin(2.0 * np.pi * 0.6 * trace_axis)
    )
    raw += banding + background

    trace_metadata = {
        "trace_index": np.arange(traces, dtype=np.int32),
        "trace_distance_m": obs_trace_distance.astype(np.float64),
        "local_x_m": obs_local_x.astype(np.float64),
        "local_y_m": obs_local_y.astype(np.float64),
        "longitude": obs_longitude.astype(np.float64),
        "latitude": obs_latitude.astype(np.float64),
        "flight_height_m": obs_height.astype(np.float64),
        "roll_deg": obs_roll.astype(np.float64),
        "pitch_deg": obs_pitch.astype(np.float64),
        "yaw_deg": obs_yaw.astype(np.float64),
        "footprint_x_m": obs_footprint_x.astype(np.float64),
        "footprint_y_m": obs_footprint_y.astype(np.float64),
        "time_window_ns": float(total_time_ns),
    }
    ground_truth_trace_metadata = {
        "trace_index": np.arange(traces, dtype=np.int32),
        "trace_distance_m": gt_trace_distance.astype(np.float64),
        "local_x_m": gt_local_x.astype(np.float64),
        "local_y_m": gt_local_y.astype(np.float64),
        "longitude": gt_longitude.astype(np.float64),
        "latitude": gt_latitude.astype(np.float64),
        "flight_height_m": gt_height.astype(np.float64),
        "roll_deg": gt_roll.astype(np.float64),
        "pitch_deg": gt_pitch.astype(np.float64),
        "yaw_deg": gt_yaw.astype(np.float64),
        "footprint_x_m": gt_footprint_x.astype(np.float64),
        "footprint_y_m": gt_footprint_y.astype(np.float64),
        "reflector_ridge_idx": reflector_ridge_idx.astype(np.int32),
        "time_window_ns": float(total_time_ns),
    }
    metric_config = {
        "ridge_row_range": list(ridge_row_range),
        "target_row_range": list(target_row_range),
        "banding_trace_band": list(banding_trace_band),
        "banding_row_range": list(banding_row_range),
    }
    raw_metrics = compute_motion_quality_metrics(
        raw,
        trace_metadata,
        ground_truth_trace_metadata,
        ground_truth_data=clean,
        ridge_row_range=ridge_row_range,
        target_row_range=target_row_range,
        banding_trace_band=banding_trace_band,
        banding_row_range=banding_row_range,
    )

    expected_metrics = {
        "raw": raw_metrics,
        "minimum_uncompensated": {
            "raw_ridge_rmse_samples": 1.0,
            "trace_spacing_std_m": 0.05,
            "path_rmse_m": 0.25,
            "footprint_rmse_m": 0.30,
            "periodic_banding_ratio": 0.04,
            "target_preservation_ratio": 0.80,
        },
        "metric_config": metric_config,
    }

    return raw.astype(np.float32), {
        "header_info": _base_header_info(
            samples,
            traces,
            trace_interval_m=trace_interval_m,
            total_time_ns=total_time_ns,
        ),
        "trace_metadata": trace_metadata,
        "ground_truth_trace_metadata": ground_truth_trace_metadata,
        "ground_truth_data": clean.astype(np.float32),
        "expected_metrics": expected_metrics,
        "injected_errors": {
            "height_error_m": height_error.astype(np.float64),
            "lateral_error_m": lateral_error.astype(np.float64),
            "trace_spacing_error_m": np.diff(obs_trace_distance - gt_trace_distance).astype(np.float64),
            "roll_error_deg": (obs_roll - gt_roll).astype(np.float64),
            "pitch_error_deg": (obs_pitch - gt_pitch).astype(np.float64),
            "yaw_error_deg": (obs_yaw - gt_yaw).astype(np.float64),
            "footprint_error_m": np.sqrt(
                (obs_footprint_x - gt_footprint_x) ** 2
                + (obs_footprint_y - gt_footprint_y) ** 2
            ).astype(np.float64),
            "height_shift_samples": shift_samples.astype(np.float64),
        },
        "notes": "Synthetic UAV-GPR line with deterministic height, lateral, speed, attitude, and vibration/background errors.",
        "wave_speed_m_per_ns": wave_speed_m_per_ns,
        "target_depth_m": target_depth_m,
    }


BENCHMARK_SAMPLES: dict[str, BenchmarkSampleSpec] = {
    "zero_time_reference": BenchmarkSampleSpec(
        sample_id="zero_time_reference",
        scenario="zero_time",
        title="零时首波基准样本",
        description="用于比较零时矫正与首波清晰度的合成样本。",
        default_methods=("set_zero_time", "dewow"),
        focus_metrics=(
            "pre_zero_energy_ratio_after",
            "first_break_sharpness_after",
            "baseline_bias_reduction",
        ),
        tags=("synthetic", "zero_time", "first_break"),
        builder=_build_zero_time_fixture,
    ),
    "drift_background_reference": BenchmarkSampleSpec(
        sample_id="drift_background_reference",
        scenario="drift_background",
        title="漂移背景抑制基准样本",
        description="用于比较低频漂移、水平背景和主反射保真的合成样本。",
        default_methods=(
            "dewow",
            "subtracting_average_2D",
            "median_background_2D",
            "rpca_background",
        ),
        focus_metrics=(
            "low_freq_energy_reduction",
            "horizontal_coherence_reduction",
            "target_band_energy_ratio",
            "local_saliency_preservation",
        ),
        tags=("synthetic", "drift", "background"),
        builder=_build_drift_background_fixture,
    ),
    "clutter_gain_reference": BenchmarkSampleSpec(
        sample_id="clutter_gain_reference",
        scenario="clutter_gain",
        title="增益与尖锐杂波基准样本",
        description="用于比较深部可见性增强与尖锐杂波抑制代价的合成样本。",
        default_methods=("agcGain", "running_average_2D", "sec_gain"),
        focus_metrics=(
            "deep_zone_contrast_gain",
            "edge_preservation",
            "clipping_ratio_after",
            "hot_pixel_ratio_after",
        ),
        tags=("synthetic", "gain", "clutter"),
        builder=_build_clutter_gain_fixture,
    ),
    "motion_compensation_v1": BenchmarkSampleSpec(
        sample_id="motion_compensation_v1",
        scenario="motion_compensation",
        title="五项运动补偿合成航线基准",
        description="用于度量高度、横向、速度、姿态和周期性振动背景误差的合成 UAV-GPR 航线样本。",
        default_methods=("trajectory_smoothing", "motion_compensation_height"),
        focus_metrics=(
            "raw_ridge_rmse_samples",
            "trace_spacing_std_m",
            "path_rmse_m",
            "footprint_rmse_m",
            "periodic_banding_ratio",
            "target_preservation_ratio",
        ),
        tags=("synthetic", "motion", "uav", "benchmark"),
        builder=_build_motion_compensation_fixture,
    ),
}


for _sample_id, _spec in BENCHMARK_SAMPLES.items():
    missing = [
        method_key
        for method_key in _spec.default_methods
        if method_key not in PROCESSING_METHODS
    ]
    if missing:
        raise KeyError(
            f"Benchmark sample {_sample_id} references unknown methods: {missing}"
        )
    if _spec.scenario not in BENCHMARK_SCENARIOS:
        raise KeyError(
            f"Benchmark sample {_sample_id} references unknown scenario: {_spec.scenario}"
        )


def list_benchmark_sample_ids() -> list[str]:
    """Return benchmark sample ids in declaration order."""
    return list(BENCHMARK_SAMPLES.keys())


def get_benchmark_sample_spec(sample_id: str) -> BenchmarkSampleSpec:
    """Return a benchmark sample specification by id."""
    try:
        return BENCHMARK_SAMPLES[sample_id]
    except KeyError as exc:
        raise KeyError(f"未知 benchmark sample: {sample_id}") from exc


def generate_benchmark_sample(
    sample_id: str,
    seed: int = DEFAULT_BENCHMARK_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate benchmark input data and metadata deterministically."""
    spec = get_benchmark_sample_spec(sample_id)
    data, meta = spec.builder(int(seed))
    payload = dict(meta)
    payload.setdefault("sample_id", sample_id)
    payload.setdefault("scenario", spec.scenario)
    payload.setdefault("seed", int(seed))
    payload.setdefault("title", spec.title)
    payload.setdefault("focus_metrics", list(spec.focus_metrics))
    payload.setdefault("default_methods", list(spec.default_methods))
    payload.setdefault("tags", list(spec.tags))
    return np.asarray(data, dtype=np.float32), payload
