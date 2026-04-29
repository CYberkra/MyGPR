#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export isolated demo cases for each V1 motion-compensation stage."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.benchmark_registry import (
    _local_xy_to_lon_lat,
    _shift_traces_linear,
    generate_benchmark_sample,
)
from core.processing_engine import (
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.quality_metrics import (
    compute_motion_quality_metrics,
    footprint_rmse,
    path_rmse,
    periodic_banding_ratio,
    ridge_error_metrics,
    target_preservation_ratio,
    trace_spacing_std,
)
from core.trace_metadata_utils import derive_local_xy_m


OUTPUT_ROOT = Path("output/single_motion_demo_cases")
SEED = 123


def _clone_trace_metadata(trace_metadata: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in trace_metadata.items():
        if isinstance(value, np.ndarray):
            cloned[key] = np.array(value, copy=True)
        else:
            cloned[key] = value
    return cloned


def _clone_header_info(header_info: dict[str, Any]) -> dict[str, Any]:
    return dict(header_info)


def _compute_trace_distance(local_x_m: np.ndarray, local_y_m: np.ndarray) -> np.ndarray:
    if local_x_m.size == 0:
        return np.array([], dtype=np.float64)
    step = np.sqrt(np.diff(local_x_m) ** 2 + np.diff(local_y_m) ** 2)
    return np.concatenate(([0.0], np.cumsum(step))).astype(np.float64)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _save_case_bundle(path: Path, data: np.ndarray, header_info: dict[str, Any], trace_metadata: dict[str, Any]) -> None:
    payload = {"raw_data": np.asarray(data, dtype=np.float32)}
    for key, value in trace_metadata.items():
        payload[f"trace_metadata__{key}"] = np.asarray(value)
    np.savez_compressed(path, **payload)
    _write_json(path.with_suffix(".header.json"), header_info)


def _plot_bscan(ax: plt.Axes, data: np.ndarray, title: str, cmap: str = "gray") -> None:
    ax.imshow(np.asarray(data), cmap=cmap, aspect="auto", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Sample")


def _sample_bscan_on_distance(ideal_data: np.ndarray, ideal_distance: np.ndarray, observed_distance: np.ndarray) -> np.ndarray:
    """Sample an ideal B-scan on a nonuniform distance axis."""
    samples = ideal_data.shape[0]
    out = np.empty((samples, observed_distance.size), dtype=np.float32)
    for row in range(samples):
        out[row, :] = np.interp(observed_distance, ideal_distance, ideal_data[row, :]).astype(np.float32)
    return out


def _add_realistic_shallow_wavefield(data: np.ndarray, *, include_secondary: bool = True) -> np.ndarray:
    """Inject a visible shallow direct wave and optional weaker shallow response."""
    result = np.asarray(data, dtype=np.float32).copy()
    samples, traces = result.shape
    phase = np.linspace(0.0, 1.0, traces, dtype=np.float64)
    direct_row = 11.0 + 1.0 * np.sin(2.0 * np.pi * 0.45 * phase) + 0.6 * np.cos(2.0 * np.pi * 1.2 * phase)
    direct_pulse = np.array([0.14, 0.45, 1.10, 2.05, 2.80, 2.05, 1.10, 0.45, 0.14], dtype=np.float32)
    tail_pulse = np.array([-0.05, -0.16, -0.28, -0.16, -0.05], dtype=np.float32)
    shallow_reflection = np.array([0.05, 0.16, 0.34, 0.52, 0.34, 0.16, 0.05], dtype=np.float32)

    for col in range(traces):
        center = int(round(direct_row[col]))
        start = center - direct_pulse.size // 2
        if 0 <= start and start + direct_pulse.size <= samples:
            result[start : start + direct_pulse.size, col] += direct_pulse

        tail_center = center + 8
        tail_start = tail_center - tail_pulse.size // 2
        if 0 <= tail_start and tail_start + tail_pulse.size <= samples:
            result[tail_start : tail_start + tail_pulse.size, col] += tail_pulse

        if include_secondary:
            shallow_center = center + 18 + int(round(1.8 * np.sin(2.0 * np.pi * 0.9 * phase[col])))
            shallow_start = shallow_center - shallow_reflection.size // 2
            if 0 <= shallow_start and shallow_start + shallow_reflection.size <= samples:
                result[shallow_start : shallow_start + shallow_reflection.size, col] += shallow_reflection

    return result


def _add_realistic_near_surface_structure(data: np.ndarray) -> np.ndarray:
    """Build a simpler scene: shallow direct wave plus one obvious hyperbola."""
    result = _add_realistic_shallow_wavefield(np.zeros_like(data, dtype=np.float32), include_secondary=False)
    samples, traces = result.shape
    phase = np.linspace(0.0, 1.0, traces, dtype=np.float64)
    trace_idx = np.arange(traces, dtype=np.float64)

    center_trace = 0.53 * (traces - 1)
    hyperbola_rows = 72.0 + 20.0 * (np.sqrt(1.0 + ((trace_idx - center_trace) / 7.0) ** 2) - 1.0)
    hyperbola_pulse = np.array([0.10, 0.28, 0.62, 1.05, 1.55, 2.10, 1.55, 1.05, 0.62, 0.28, 0.10], dtype=np.float32)
    for col in range(traces):
        row_center = int(round(hyperbola_rows[col]))
        row_start = row_center - hyperbola_pulse.size // 2
        if 0 <= row_start and row_start + hyperbola_pulse.size <= samples:
            result[row_start : row_start + hyperbola_pulse.size, col] += hyperbola_pulse

    sample_axis = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    background = 0.006 * np.exp(-3.0 * sample_axis) * np.sin(2.0 * np.pi * (0.55 * sample_axis + 0.10 * phase[None, :]))
    result += background.astype(np.float32)
    return result


def _run_motion_method(
    method_key: str,
    data: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime_params = prepare_runtime_params(
        method_key,
        params or {},
        _clone_header_info(header_info),
        _clone_trace_metadata(trace_metadata),
        data.shape,
    )
    corrected, runtime_meta = run_processing_method(np.asarray(data, dtype=np.float32), method_key, runtime_params)
    corrected_header = merge_result_header_info(header_info, runtime_meta, corrected.shape)
    corrected_trace_metadata = merge_result_trace_metadata(trace_metadata, runtime_meta)
    return corrected, corrected_header, corrected_trace_metadata, runtime_meta


def _expected_attitude_updates(
    trace_metadata: dict[str, Any],
    *,
    apc_offset_x_m: float = 0.0,
    apc_offset_y_m: float = 0.0,
    apc_offset_z_m: float = 0.0,
    max_abs_tilt_deg: float = 20.0,
) -> dict[str, np.ndarray]:
    local_x_m = np.asarray(trace_metadata["local_x_m"], dtype=np.float64)
    local_y_m = np.asarray(trace_metadata["local_y_m"], dtype=np.float64)
    yaw_rad = np.deg2rad(np.asarray(trace_metadata["yaw_deg"], dtype=np.float64))
    roll_deg = np.asarray(trace_metadata["roll_deg"], dtype=np.float64)
    pitch_deg = np.asarray(trace_metadata["pitch_deg"], dtype=np.float64)
    roll_used_rad = np.deg2rad(np.clip(roll_deg, -max_abs_tilt_deg, max_abs_tilt_deg))
    pitch_used_rad = np.deg2rad(np.clip(pitch_deg, -max_abs_tilt_deg, max_abs_tilt_deg))

    apc_x = apc_offset_x_m * np.cos(yaw_rad) - apc_offset_y_m * np.sin(yaw_rad)
    apc_y = apc_offset_x_m * np.sin(yaw_rad) + apc_offset_y_m * np.cos(yaw_rad)
    projection_height_m = np.asarray(trace_metadata["flight_height_m"], dtype=np.float64) + apc_offset_z_m
    pitch_body_m = projection_height_m * np.tan(pitch_used_rad)
    roll_body_m = projection_height_m * np.tan(roll_used_rad)
    footprint_dx = pitch_body_m * np.cos(yaw_rad) - roll_body_m * np.sin(yaw_rad)
    footprint_dy = pitch_body_m * np.sin(yaw_rad) + roll_body_m * np.cos(yaw_rad)

    corrected_x = local_x_m + apc_x + footprint_dx
    corrected_y = local_y_m + apc_y + footprint_dy
    distance = _compute_trace_distance(corrected_x, corrected_y)
    return {
        "local_x_m": corrected_x,
        "local_y_m": corrected_y,
        "trace_distance_m": distance,
        "footprint_x_m": corrected_x.copy(),
        "footprint_y_m": corrected_y.copy(),
    }


def _height_case(
    out_dir: Path,
    clean: np.ndarray,
    header_info: dict[str, Any],
    raw_trace_metadata: dict[str, Any],
    ground_truth_trace_metadata: dict[str, Any],
    metric_config: dict[str, Any],
) -> dict[str, Any]:
    clean_with_shallow = _add_realistic_near_surface_structure(clean)
    gt_height = np.asarray(ground_truth_trace_metadata["flight_height_m"], dtype=np.float64)
    obs_height = np.asarray(raw_trace_metadata["flight_height_m"], dtype=np.float64)
    total_time_ns = float(header_info["total_time_ns"])
    dt_ns = total_time_ns / max(clean.shape[0] - 1, 1)
    shift_samples = 2.0 * (obs_height - gt_height) / 0.1 / dt_ns
    raw_data = _shift_traces_linear(clean_with_shallow, shift_samples).astype(np.float32)
    raw_data *= ((gt_height / np.maximum(obs_height, 0.25)) ** 2)[np.newaxis, :]

    raw_meta = _clone_trace_metadata(ground_truth_trace_metadata)
    raw_meta["flight_height_m"] = obs_height.copy()
    raw_meta["time_window_ns"] = float(total_time_ns)

    corrected, corrected_header, corrected_meta, _ = _run_motion_method(
        "motion_compensation_height",
        raw_data,
        header_info,
        raw_meta,
    )

    raw_metrics = compute_motion_quality_metrics(
        raw_data,
        raw_meta,
        ground_truth_trace_metadata,
        ground_truth_data=clean_with_shallow,
        ridge_row_range=tuple(metric_config["ridge_row_range"]),
        target_row_range=tuple(metric_config["target_row_range"]),
        banding_trace_band=tuple(metric_config["banding_trace_band"]),
        banding_row_range=tuple(metric_config["banding_row_range"]),
    )
    final_metrics = compute_motion_quality_metrics(
        corrected,
        corrected_meta,
        ground_truth_trace_metadata,
        ground_truth_data=clean_with_shallow,
        ridge_row_range=tuple(metric_config["ridge_row_range"]),
        target_row_range=tuple(metric_config["target_row_range"]),
        banding_trace_band=tuple(metric_config["banding_trace_band"]),
        banding_row_range=tuple(metric_config["banding_row_range"]),
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _plot_bscan(axes[0], raw_data, "Height error / before")
    _plot_bscan(axes[1], corrected, "Height compensation / after")
    _plot_bscan(axes[2], corrected - raw_data, "Difference", cmap="seismic")
    fig.tight_layout()
    fig.savefig(out_dir / "height_demo.png", dpi=150)
    plt.close(fig)

    _save_case_bundle(out_dir / "height_input_bundle.npz", raw_data, header_info, raw_meta)
    _write_json(
        out_dir / "height_metrics.json",
        {
            "case": "height",
            "raw_metrics": raw_metrics,
            "final_metrics": final_metrics,
            "summary": {
                "raw_ridge_rmse": raw_metrics["raw_ridge_rmse_samples"],
                "final_ridge_rmse": final_metrics["raw_ridge_rmse_samples"],
            },
            "artifacts": {"main_figure": str(out_dir / "height_demo.png")},
        },
    )
    return {
        "name": "height",
        "figure": str(out_dir / "height_demo.png"),
        "raw_metrics": raw_metrics,
        "final_metrics": final_metrics,
        "input_shape": list(raw_data.shape),
        "output_shape": list(corrected.shape),
        "header_info": corrected_header,
    }


def _trajectory_case(
    out_dir: Path,
    clean: np.ndarray,
    header_info: dict[str, Any],
    raw_trace_metadata: dict[str, Any],
    ground_truth_trace_metadata: dict[str, Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED + 100)
    n = 200
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    gt_local_x = 50.0 * t
    gt_local_y = 2.0 * np.sin(2.0 * np.pi * 0.5 * t)
    noise_freq = 20.0
    noise_amp = 0.8
    noise_x = noise_amp * np.sin(2.0 * np.pi * noise_freq * t + rng.uniform(0.0, 2.0 * np.pi))
    noise_y = noise_amp * np.cos(2.0 * np.pi * noise_freq * t + rng.uniform(0.0, 2.0 * np.pi))
    noise_x += 0.02 * rng.normal(size=n)
    noise_y += 0.02 * rng.normal(size=n)
    obs_local_x = gt_local_x + noise_x
    obs_local_y = gt_local_y + noise_y
    obs_longitude, obs_latitude = _local_xy_to_lon_lat(obs_local_x, obs_local_y)
    gt_longitude, gt_latitude = _local_xy_to_lon_lat(gt_local_x, gt_local_y)
    obs_x, obs_y = derive_local_xy_m(obs_longitude, obs_latitude)
    gt_x, gt_y = derive_local_xy_m(
        gt_longitude,
        gt_latitude,
        origin_longitude=float(obs_longitude[0]),
        origin_latitude=float(obs_latitude[0]),
    )

    local_header = _clone_header_info(header_info)
    local_header["num_traces"] = int(n)

    raw_meta = {
        "longitude": obs_longitude.astype(np.float64),
        "latitude": obs_latitude.astype(np.float64),
        "local_x_m": obs_x.astype(np.float64),
        "local_y_m": obs_y.astype(np.float64),
    }
    gt_trace_metadata = {
        "local_x_m": gt_x.astype(np.float64),
        "local_y_m": gt_y.astype(np.float64),
    }
    trajectory_data = np.zeros((64, n), dtype=np.float32)

    corrected, corrected_header, corrected_meta, _ = _run_motion_method(
        "trajectory_smoothing",
        trajectory_data,
        local_header,
        raw_meta,
    )

    before_rmse = path_rmse(raw_meta, gt_trace_metadata)
    after_rmse = path_rmse(corrected_meta, gt_trace_metadata)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(raw_meta["local_x_m"], raw_meta["local_y_m"], label="raw", linewidth=1.5)
    axes[0].plot(corrected_meta["local_x_m"], corrected_meta["local_y_m"], label="corrected", linewidth=1.5)
    axes[0].plot(
        gt_trace_metadata["local_x_m"],
        gt_trace_metadata["local_y_m"],
        label="ground truth",
        linewidth=1.5,
    )
    axes[0].set_title("Trajectory path")
    axes[0].set_xlabel("local_x_m")
    axes[0].set_ylabel("local_y_m")
    axes[0].legend()

    displacement = np.sqrt(
        (np.asarray(corrected_meta["local_x_m"]) - np.asarray(raw_meta["local_x_m"])) ** 2
        + (np.asarray(corrected_meta["local_y_m"]) - np.asarray(raw_meta["local_y_m"])) ** 2
    )
    axes[1].plot(displacement)
    axes[1].set_title("Per-trace displacement")
    axes[1].set_xlabel("trace")
    axes[1].set_ylabel("meters")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_demo.png", dpi=150)
    plt.close(fig)

    _save_case_bundle(out_dir / "trajectory_input_bundle.npz", trajectory_data, local_header, raw_meta)
    _write_json(
        out_dir / "trajectory_metrics.json",
        {
            "case": "trajectory",
            "before_path_rmse_m": before_rmse,
            "after_path_rmse_m": after_rmse,
            "artifacts": {"main_figure": str(out_dir / "trajectory_demo.png")},
        },
    )
    return {
        "name": "trajectory",
        "figure": str(out_dir / "trajectory_demo.png"),
        "before_path_rmse_m": before_rmse,
        "after_path_rmse_m": after_rmse,
        "input_shape": list(trajectory_data.shape),
        "output_shape": list(corrected.shape),
        "header_info": corrected_header,
    }


def _speed_case(
    out_dir: Path,
    clean: np.ndarray,
    header_info: dict[str, Any],
    raw_trace_metadata: dict[str, Any],
    ground_truth_trace_metadata: dict[str, Any],
) -> dict[str, Any]:
    samples, traces = 240, 96
    ideal_distance = np.linspace(0.0, 42.0, traces, dtype=np.float64)
    phase = np.linspace(0.0, 1.0, traces, dtype=np.float64)
    nominal_spacing = ideal_distance[-1] / max(traces - 1, 1)
    spacing_scale = 1.0 + 0.95 * np.sin(2.0 * np.pi * 1.15 * phase[1:] - 0.85)
    spacing_scale += 0.42 * np.cos(2.0 * np.pi * 3.85 * phase[1:] + 0.25)
    raw_spacing = nominal_spacing * np.clip(spacing_scale, 0.18, 2.10)
    raw_distance = np.concatenate(([0.0], np.cumsum(raw_spacing)))
    raw_distance *= ideal_distance[-1] / max(raw_distance[-1], 1.0e-6)

    ideal_data = np.zeros((samples, traces), dtype=np.float32)
    sample_axis = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    trace_axis = ideal_distance[None, :]
    ideal_data += (0.006 * np.sin(2.0 * np.pi * 1.6 * sample_axis)).astype(np.float32)
    ideal_data += (0.004 * np.cos(2.0 * np.pi * 0.6 * trace_axis / max(ideal_distance[-1], 1.0))).astype(np.float32)
    ideal_data[28:30, :] += 0.05
    ideal_data[52:54, :] += 0.08

    x0 = ideal_distance[traces // 2]
    direct_wave_row = 11.5 + 1.2 * np.sin(2.0 * np.pi * 0.55 * phase) + 0.8 * np.cos(2.0 * np.pi * 1.35 * phase)
    hyperbola_row = 74.0 + 28.0 * (np.sqrt(1.0 + ((ideal_distance - x0) / 1.85) ** 2) - 1.0)
    secondary_row = 130.0 + 17.0 * (np.sqrt(1.0 + ((ideal_distance - (x0 + 7.5)) / 2.8) ** 2) - 1.0)
    direct_pulse = np.array([0.18, 0.62, 1.55, 2.85, 3.55, 2.85, 1.55, 0.62, 0.18], dtype=np.float32)
    direct_tail = np.array([-0.06, -0.18, -0.30, -0.18, -0.06], dtype=np.float32)
    main_pulse = np.array([0.10, 0.28, 0.70, 1.35, 2.10, 2.85, 3.45, 2.85, 2.10, 1.35, 0.70, 0.28, 0.10], dtype=np.float32)
    secondary_pulse = np.array([0.08, 0.20, 0.42, 0.75, 1.10, 1.35, 1.10, 0.75, 0.42, 0.20, 0.08], dtype=np.float32)
    for col in range(traces):
        direct_center = int(round(direct_wave_row[col]))
        direct_start = direct_center - direct_pulse.size // 2
        if 0 <= direct_start and direct_start + direct_pulse.size <= samples:
            ideal_data[direct_start : direct_start + direct_pulse.size, col] += direct_pulse
        tail_center = direct_center + 8
        tail_start = tail_center - direct_tail.size // 2
        if 0 <= tail_start and tail_start + direct_tail.size <= samples:
            ideal_data[tail_start : tail_start + direct_tail.size, col] += direct_tail

        center = int(round(hyperbola_row[col]))
        start = center - main_pulse.size // 2
        if 0 <= start and start + main_pulse.size <= samples:
            ideal_data[start : start + main_pulse.size, col] += main_pulse
        center2 = int(round(secondary_row[col]))
        start2 = center2 - secondary_pulse.size // 2
        if 0 <= start2 and start2 + secondary_pulse.size <= samples:
            ideal_data[start2 : start2 + secondary_pulse.size, col] += secondary_pulse

    raw_data = _sample_bscan_on_distance(ideal_data, ideal_distance, raw_distance)

    gt_local_y = np.zeros(traces, dtype=np.float64)
    raw_longitude, raw_latitude = _local_xy_to_lon_lat(raw_distance, gt_local_y)

    raw_meta = {
        "trace_index": np.arange(traces, dtype=np.int32),
        "trace_distance_m": raw_distance.copy(),
        "local_x_m": raw_distance.copy(),
        "local_y_m": gt_local_y.copy(),
        "longitude": raw_longitude,
        "latitude": raw_latitude,
    }

    local_header = _clone_header_info(header_info)
    local_header["a_scan_length"] = int(samples)
    local_header["num_traces"] = int(traces)
    local_header["trace_interval_m"] = float(ideal_distance[-1] / (traces - 1))

    corrected, corrected_header, corrected_meta, _ = _run_motion_method(
        "motion_compensation_speed",
        raw_data,
        local_header,
        raw_meta,
    )

    before_spacing = np.diff(np.asarray(raw_meta["trace_distance_m"], dtype=np.float64))
    after_spacing = np.diff(np.asarray(corrected_meta["trace_distance_m"], dtype=np.float64))
    before_std = trace_spacing_std(raw_meta)
    after_std = trace_spacing_std(corrected_meta)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _plot_bscan(axes[0], raw_data, "Speed error input")
    _plot_bscan(axes[1], corrected, "Speed compensation / after")
    axes[2].plot(before_spacing, label="raw spacing")
    axes[2].plot(after_spacing, label="corrected spacing")
    axes[2].set_title("Trace spacing")
    axes[2].set_xlabel("segment")
    axes[2].set_ylabel("meters")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "speed_demo.png", dpi=150)
    plt.close(fig)

    _save_case_bundle(out_dir / "speed_input_bundle.npz", raw_data, local_header, raw_meta)
    _write_json(
        out_dir / "speed_metrics.json",
        {
            "case": "speed",
            "before_spacing_std_m": before_std,
            "after_spacing_std_m": after_std,
            "input_traces": int(raw_data.shape[1]),
            "output_traces": int(corrected.shape[1]),
            "artifacts": {"main_figure": str(out_dir / "speed_demo.png")},
        },
    )
    return {
        "name": "speed",
        "figure": str(out_dir / "speed_demo.png"),
        "before_spacing_std_m": before_std,
        "after_spacing_std_m": after_std,
        "input_shape": list(raw_data.shape),
        "output_shape": list(corrected.shape),
        "header_info": corrected_header,
    }


def _attitude_case(
    out_dir: Path,
    clean: np.ndarray,
    header_info: dict[str, Any],
    raw_trace_metadata: dict[str, Any],
    ground_truth_trace_metadata: dict[str, Any],
) -> dict[str, Any]:
    n = 64
    phase = np.linspace(0.0, 1.0, n, dtype=np.float64)
    attitude_data = np.zeros((32, n), dtype=np.float32)
    raw_meta = {
        "local_x_m": np.linspace(0.0, 30.0, n, dtype=np.float64),
        "local_y_m": 0.4 * np.sin(2.0 * np.pi * phase),
        "roll_deg": 7.0 * np.sin(2.0 * np.pi * 0.8 * phase),
        "pitch_deg": 6.0 * np.cos(2.0 * np.pi * 1.1 * phase),
        "yaw_deg": 14.0 * np.sin(2.0 * np.pi * 0.4 * phase),
        "flight_height_m": 3.0 + 0.15 * np.cos(2.0 * np.pi * phase),
    }
    raw_meta["trace_distance_m"] = _compute_trace_distance(raw_meta["local_x_m"], raw_meta["local_y_m"])
    raw_meta["footprint_x_m"] = np.asarray(raw_meta["local_x_m"], dtype=np.float64).copy()
    raw_meta["footprint_y_m"] = np.asarray(raw_meta["local_y_m"], dtype=np.float64).copy()
    target_updates = _expected_attitude_updates(raw_meta)
    gt_trace_metadata = {
        "footprint_x_m": target_updates["footprint_x_m"],
        "footprint_y_m": target_updates["footprint_y_m"],
    }

    local_header = _clone_header_info(header_info)
    local_header["num_traces"] = int(n)

    corrected, corrected_header, corrected_meta, _ = _run_motion_method(
        "motion_compensation_attitude",
        attitude_data,
        local_header,
        raw_meta,
    )

    before_rmse = footprint_rmse(raw_meta, gt_trace_metadata)
    after_rmse = footprint_rmse(corrected_meta, gt_trace_metadata)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(raw_meta["footprint_x_m"], raw_meta["footprint_y_m"], label="raw footprint")
    axes[0].plot(corrected_meta["footprint_x_m"], corrected_meta["footprint_y_m"], label="corrected footprint")
    axes[0].plot(
        gt_trace_metadata["footprint_x_m"],
        gt_trace_metadata["footprint_y_m"],
        label="ground truth",
    )
    axes[0].set_title("Footprint path")
    axes[0].set_xlabel("footprint_x_m")
    axes[0].set_ylabel("footprint_y_m")
    axes[0].legend()

    footprint_error = np.sqrt(
        (np.asarray(corrected_meta["footprint_x_m"]) - np.asarray(gt_trace_metadata["footprint_x_m"])) ** 2
        + (np.asarray(corrected_meta["footprint_y_m"]) - np.asarray(gt_trace_metadata["footprint_y_m"])) ** 2
    )
    axes[1].plot(footprint_error)
    axes[1].set_title("Corrected footprint error")
    axes[1].set_xlabel("trace")
    axes[1].set_ylabel("meters")
    fig.tight_layout()
    fig.savefig(out_dir / "attitude_demo.png", dpi=150)
    plt.close(fig)

    _save_case_bundle(out_dir / "attitude_input_bundle.npz", attitude_data, local_header, raw_meta)
    _write_json(
        out_dir / "attitude_metrics.json",
        {
            "case": "attitude",
            "before_footprint_rmse_m": before_rmse,
            "after_footprint_rmse_m": after_rmse,
            "artifacts": {"main_figure": str(out_dir / "attitude_demo.png")},
        },
    )
    return {
        "name": "attitude",
        "figure": str(out_dir / "attitude_demo.png"),
        "before_footprint_rmse_m": before_rmse,
        "after_footprint_rmse_m": after_rmse,
        "input_shape": list(attitude_data.shape),
        "output_shape": list(corrected.shape),
        "header_info": corrected_header,
    }


def _vibration_case(
    out_dir: Path,
    clean: np.ndarray,
    header_info: dict[str, Any],
    ground_truth_trace_metadata: dict[str, Any],
    metric_config: dict[str, Any],
) -> dict[str, Any]:
    clean_with_shallow = _add_realistic_near_surface_structure(clean)
    samples, traces = clean_with_shallow.shape
    trace_phase = np.linspace(0.0, 1.0, traces, dtype=np.float64)[None, :]
    sample_phase = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    depth_envelope = 0.28 + 0.72 * np.exp(-2.2 * sample_phase)
    banding = 0.18 * depth_envelope * np.sin(
        2.0 * np.pi * (7.0 * trace_phase + 0.35 * np.sin(2.0 * np.pi * 4.0 * sample_phase))
    )
    background = 0.055 * np.cos(2.0 * np.pi * 0.85 * sample_phase) * (
        1.0 + 0.35 * np.sin(2.0 * np.pi * 0.6 * trace_phase)
    )
    raw_data = np.asarray(clean_with_shallow + banding + background, dtype=np.float32)
    raw_meta = _clone_trace_metadata(ground_truth_trace_metadata)

    corrected, corrected_header, corrected_meta, _ = _run_motion_method(
        "motion_compensation_vibration",
        raw_data,
        header_info,
        raw_meta,
    )

    before_banding = periodic_banding_ratio(
        raw_data,
        trace_band=tuple(metric_config["banding_trace_band"]),
        row_range=tuple(metric_config["banding_row_range"]),
    )
    after_banding = periodic_banding_ratio(
        corrected,
        trace_band=tuple(metric_config["banding_trace_band"]),
        row_range=tuple(metric_config["banding_row_range"]),
    )
    before_target = target_preservation_ratio(raw_data, clean_with_shallow, tuple(metric_config["target_row_range"]))
    after_target = target_preservation_ratio(corrected, clean_with_shallow, tuple(metric_config["target_row_range"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _plot_bscan(axes[0], raw_data, "Vibration / before")
    _plot_bscan(axes[1], corrected, "Vibration compensation / after")
    _plot_bscan(axes[2], corrected - raw_data, "Difference", cmap="seismic")
    fig.tight_layout()
    fig.savefig(out_dir / "vibration_demo.png", dpi=150)
    plt.close(fig)

    _save_case_bundle(out_dir / "vibration_input_bundle.npz", raw_data, header_info, raw_meta)
    _write_json(
        out_dir / "vibration_metrics.json",
        {
            "case": "vibration",
            "before_periodic_banding_ratio": before_banding,
            "after_periodic_banding_ratio": after_banding,
            "before_target_preservation_ratio": before_target,
            "after_target_preservation_ratio": after_target,
            "artifacts": {"main_figure": str(out_dir / "vibration_demo.png")},
        },
    )
    return {
        "name": "vibration",
        "figure": str(out_dir / "vibration_demo.png"),
        "before_periodic_banding_ratio": before_banding,
        "after_periodic_banding_ratio": after_banding,
        "before_target_preservation_ratio": before_target,
        "after_target_preservation_ratio": after_target,
        "input_shape": list(raw_data.shape),
        "output_shape": list(corrected.shape),
        "header_info": corrected_header,
        "trace_metadata_keys": sorted(corrected_meta.keys()),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _, meta = generate_benchmark_sample("motion_compensation_v1", seed=SEED)
    clean = np.asarray(meta["ground_truth_data"], dtype=np.float32)
    header_info = _clone_header_info(meta["header_info"])
    raw_trace_metadata = _clone_trace_metadata(meta["trace_metadata"])
    ground_truth_trace_metadata = _clone_trace_metadata(meta["ground_truth_trace_metadata"])
    metric_config = dict(meta["expected_metrics"]["metric_config"])

    summary: dict[str, Any] = {
        "seed": SEED,
        "output_root": str(OUTPUT_ROOT),
        "cases": {},
    }

    height_dir = OUTPUT_ROOT / "height"
    height_dir.mkdir(parents=True, exist_ok=True)
    summary["cases"]["height"] = _height_case(
        height_dir,
        clean,
        header_info,
        raw_trace_metadata,
        ground_truth_trace_metadata,
        metric_config,
    )

    trajectory_dir = OUTPUT_ROOT / "trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    summary["cases"]["trajectory"] = _trajectory_case(
        trajectory_dir,
        clean,
        header_info,
        raw_trace_metadata,
        ground_truth_trace_metadata,
    )

    speed_dir = OUTPUT_ROOT / "speed"
    speed_dir.mkdir(parents=True, exist_ok=True)
    summary["cases"]["speed"] = _speed_case(
        speed_dir,
        clean,
        header_info,
        raw_trace_metadata,
        ground_truth_trace_metadata,
    )

    attitude_dir = OUTPUT_ROOT / "attitude"
    attitude_dir.mkdir(parents=True, exist_ok=True)
    summary["cases"]["attitude"] = _attitude_case(
        attitude_dir,
        clean,
        header_info,
        raw_trace_metadata,
        ground_truth_trace_metadata,
    )

    vibration_dir = OUTPUT_ROOT / "vibration"
    vibration_dir.mkdir(parents=True, exist_ok=True)
    summary["cases"]["vibration"] = _vibration_case(
        vibration_dir,
        clean,
        header_info,
        ground_truth_trace_metadata,
        metric_config,
    )

    _write_json(OUTPUT_ROOT / "single_motion_demo_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
