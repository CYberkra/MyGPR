#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate and process a visible UAV-GPR motion compensation effect demo."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpr_io import extract_airborne_csv_payload
from core.processing_engine import (  # noqa: E402
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.uav_georeference_3d import (  # noqa: E402
    build_airborne_georeference_3d_payload,
    save_airborne_georeference_3d_preview_png,
)
from read_file_data import readcsv  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "output" / "mygpr_uav_motion_effect_demo_v1"
SAMPLES = 240
TRACES = 210
TOTAL_TIME_NS = 120.0
START_TIMESTAMP_S = 3000.0
TRACE_PERIOD_S = 0.075
AIR_WAVE_SPEED_M_PER_NS = 0.299792458
ORIGIN_LONGITUDE = 104.123456
ORIGIN_LATITUDE = 30.654321

PIPELINE: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "trajectory_smoothing",
        {
            "method": "savgol",
            "window_length": 31,
            "polyorder": 3,
        },
    ),
    (
        "motion_compensation_speed",
        {
            "spacing_m": 0.42,
            "interpolation_mode": "linear",
        },
    ),
    (
        "motion_compensation_attitude",
        {
            "apc_offset_x_m": 0.04,
            "apc_offset_y_m": -0.02,
            "apc_offset_z_m": 0.0,
            "max_abs_tilt_deg": 18.0,
        },
    ),
    (
        "motion_compensation_height",
        {
            "reference_height_mode": "mean",
            "manual_height": 0.0,
            "compensate_amplitude": True,
            "compensate_time_shift": True,
            "wave_speed_m_per_ns": AIR_WAVE_SPEED_M_PER_NS,
            "max_shift_samples": 12.0,
            "interpolation_mode": "linear",
        },
    ),
)


@dataclass(frozen=True)
class MotionEffectDemoResult:
    """Paths and metrics produced by the motion effect demo."""

    output_dir: Path
    main_csv: Path
    summary_json: Path
    final_data_shape: tuple[int, int]
    bscan_rms_delta: float
    top_interface_std_before: float
    top_interface_std_after: float


def _lon_lat_from_xy(x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(ORIGIN_LATITUDE))
    longitude = ORIGIN_LONGITUDE + np.asarray(x_m, dtype=np.float64) / meters_per_deg_lon
    latitude = ORIGIN_LATITUDE + np.asarray(y_m, dtype=np.float64) / meters_per_deg_lat
    return longitude, latitude


def _add_ricker(data: np.ndarray, row: np.ndarray, amplitude: float, width: float) -> None:
    sample_axis = np.arange(data.shape[0], dtype=np.float64)[:, None]
    row = np.asarray(row, dtype=np.float64)[None, :]
    tau = (sample_axis - row) / float(width)
    data += float(amplitude) * (1.0 - 2.0 * tau**2) * np.exp(-tau**2)


def _shift_trace(trace: np.ndarray, shift_samples: float) -> np.ndarray:
    sample_index = np.arange(trace.size, dtype=np.float64)
    return np.interp(
        sample_index - float(shift_samples),
        sample_index,
        np.asarray(trace, dtype=np.float64),
        left=0.0,
        right=0.0,
    ).astype(np.float32)


def _interp_columns(source: np.ndarray, source_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    out = np.empty((source.shape[0], target_x.size), dtype=np.float32)
    for row in range(source.shape[0]):
        out[row, :] = np.interp(target_x, source_x, source[row, :], left=0.0, right=0.0)
    return out


def _build_ideal_bscan(uniform_x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    traces = uniform_x.size
    phase = np.linspace(0.0, 1.0, traces, dtype=np.float64)
    sample_axis = np.arange(SAMPLES, dtype=np.float64)[:, None]
    data = 0.010 * rng.normal(size=(SAMPLES, traces))
    data += 0.030 * np.sin(2.0 * np.pi * (sample_axis / 9.0 + 0.12 * phase[None, :]))
    data += 0.018 * np.sin(2.0 * np.pi * 5.0 * phase)[None, :]

    direct_air = 18.0 + 1.2 * np.sin(2.0 * np.pi * 0.45 * phase)
    shallow_layer = 42.0 + 3.0 * np.sin(2.0 * np.pi * 0.65 * phase + 0.2)
    middle_layer = 91.0 + 5.5 * np.sin(2.0 * np.pi * 0.45 * phase - 0.35)
    deep_layer = 158.0 + 6.5 * np.sin(2.0 * np.pi * 0.35 * phase + 0.7)
    _add_ricker(data, direct_air, 0.45, 1.6)
    _add_ricker(data, shallow_layer, 0.70, 2.4)
    _add_ricker(data, middle_layer, -0.50, 3.0)
    _add_ricker(data, deep_layer, 0.32, 4.0)

    center_x = 44.0
    apex_sample = 108.0
    hyperbola = apex_sample + 0.125 * (uniform_x - center_x) ** 2
    _add_ricker(data, hyperbola, 1.35, 2.3)
    _add_ricker(data, hyperbola + 6.0, -0.82, 2.6)
    _add_ricker(data, hyperbola + 13.0, 0.42, 3.2)

    # A weaker secondary response helps reveal over/under correction in reports.
    weak_center_x = 66.0
    weak_hyperbola = 138.0 + 0.080 * (uniform_x - weak_center_x) ** 2
    _add_ricker(data, weak_hyperbola, 0.38, 2.6)

    data /= max(float(np.nanmax(np.abs(data))), 1e-6)
    truth = {
        "primary_target": {
            "type": "pipe_like_hyperbola",
            "center_x_m": center_x,
            "apex_sample": apex_sample,
            "sample_window": [92, 138],
            "trace_window": [70, 135],
        },
        "direct_air_wave_sample": float(np.mean(direct_air)),
        "shallow_layer_sample": float(np.mean(shallow_layer)),
    }
    return data.astype(np.float32), truth


def _build_motion_payload() -> dict[str, Any]:
    rng = np.random.default_rng(20260518)
    uniform_x = np.linspace(0.0, 88.0, TRACES, dtype=np.float64)
    phase = np.linspace(0.0, 1.0, TRACES, dtype=np.float64)
    ideal, truth = _build_ideal_bscan(uniform_x, rng)

    nominal_spacing = float(np.mean(np.diff(uniform_x)))
    raw_spacing = nominal_spacing * (
        1.0
        + 0.34 * np.sin(2.0 * np.pi * 2.2 * phase + 0.25)
        + 0.16 * np.cos(2.0 * np.pi * 4.1 * phase)
    )
    raw_spacing += rng.normal(0.0, nominal_spacing * 0.035, size=TRACES)
    raw_spacing = np.clip(raw_spacing, nominal_spacing * 0.55, nominal_spacing * 1.55)
    local_x = np.cumsum(raw_spacing)
    local_x -= local_x[0]
    local_x *= uniform_x[-1] / max(float(local_x[-1]), 1e-6)
    smooth_y = 1.15 * np.sin(2.0 * np.pi * 0.70 * phase - 0.25)
    jitter_y = 0.26 * np.sin(2.0 * np.pi * 13.0 * phase) + rng.normal(0.0, 0.055, size=TRACES)
    local_y = smooth_y + jitter_y

    ground_elevation = 116.0 + 0.45 * np.sin(2.0 * np.pi * 0.34 * phase)
    height_agl = 0.62 + 0.20 * np.sin(2.0 * np.pi * 1.45 * phase + 0.45)
    height_agl += 0.04 * np.sin(2.0 * np.pi * 8.0 * phase)
    height_agl = np.clip(height_agl, 0.34, 0.92)
    flight_height = height_agl.copy()
    local_z = ground_elevation + flight_height
    longitude, latitude = _lon_lat_from_xy(local_x, local_y)

    roll = 5.5 * np.sin(2.0 * np.pi * 2.0 * phase + 0.4) + 1.0 * np.sin(2.0 * np.pi * 9.0 * phase)
    pitch = 4.8 * np.cos(2.0 * np.pi * 1.7 * phase - 0.1)
    yaw = 6.0 + 5.5 * np.sin(2.0 * np.pi * 0.55 * phase)
    timestamps = START_TIMESTAMP_S + TRACE_PERIOD_S * np.arange(TRACES, dtype=np.float64)

    raw = _interp_columns(ideal, uniform_x, local_x)
    reference_height = float(np.mean(height_agl))
    dt_ns = TOTAL_TIME_NS / max(SAMPLES - 1, 1)
    shift_samples = 2.0 * (height_agl - reference_height) / AIR_WAVE_SPEED_M_PER_NS / dt_ns
    observed = np.empty_like(raw, dtype=np.float32)
    for col in range(TRACES):
        observed[:, col] = _shift_trace(raw[:, col], shift_samples[col])
        observed[:, col] *= np.float32(np.clip((reference_height / height_agl[col]) ** 2, 0.45, 2.30))
    observed += 0.018 * rng.normal(size=observed.shape).astype(np.float32)
    observed += (0.025 * np.sin(2.0 * np.pi * 7.0 * phase))[None, :].astype(np.float32)
    observed /= max(float(np.nanmax(np.abs(observed))), 1e-6)

    trace_distance = np.empty(TRACES, dtype=np.float64)
    trace_distance[0] = 0.0
    trace_distance[1:] = np.cumsum(np.hypot(np.diff(local_x), np.diff(local_y)), dtype=np.float64)
    metadata = {
        "trace_index": np.arange(TRACES, dtype=np.int32),
        "trace_timestamp_s": timestamps,
        "longitude": longitude,
        "latitude": latitude,
        "ground_elevation_m": ground_elevation,
        "flight_height_m": flight_height,
        "height_agl_m": height_agl,
        "local_x_m": local_x,
        "local_y_m": local_y,
        "local_z_m": local_z,
        "trace_distance_m": trace_distance,
        "roll_deg": roll,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
    }
    header = {
        "a_scan_length": SAMPLES,
        "num_traces": TRACES,
        "total_time_ns": TOTAL_TIME_NS,
        "trace_interval_m": nominal_spacing,
        "data_context": "uav_gpr_motion_effect_demo",
    }
    return {
        "data": observed.astype(np.float32),
        "ideal_data": ideal,
        "header_info": header,
        "trace_metadata": metadata,
        "truth": truth,
        "uniform_x_m": uniform_x,
        "reference_height_m": reference_height,
        "height_shift_samples": shift_samples,
    }


def _write_main_csv(path: Path, payload: dict[str, Any]) -> None:
    data = np.asarray(payload["data"], dtype=np.float32)
    meta = payload["trace_metadata"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"Number of Samples = {SAMPLES}\n")
        handle.write(f"Time windows (ns) = {TOTAL_TIME_NS:.6f}\n")
        handle.write(f"Number of Traces = {TRACES}\n")
        handle.write(f"Trace interval (m) = {float(payload['header_info']['trace_interval_m']):.6f}\n")
        writer = csv.writer(handle)
        for trace_idx in range(TRACES):
            for sample_idx in range(SAMPLES):
                writer.writerow(
                    [
                        f"{meta['longitude'][trace_idx]:.10f}",
                        f"{meta['latitude'][trace_idx]:.10f}",
                        f"{meta['ground_elevation_m'][trace_idx]:.6f}",
                        f"{data[sample_idx, trace_idx]:.8f}",
                        f"{meta['flight_height_m'][trace_idx]:.6f}",
                        f"{meta['trace_timestamp_s'][trace_idx]:.6f}",
                    ]
                )


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_sidecars(output_dir: Path, payload: dict[str, Any]) -> None:
    meta = payload["trace_metadata"]
    timestamp_rows: list[dict[str, Any]] = []
    rtk_rows: list[dict[str, Any]] = []
    imu_rows: list[dict[str, Any]] = []
    altimeter_rows: list[dict[str, Any]] = []
    for idx in range(TRACES):
        ts = float(meta["trace_timestamp_s"][idx])
        timestamp_rows.append({"trace_index": idx, "timestamp_s": f"{ts:.6f}"})
        rtk_rows.append(
            {
                "timestamp_s": f"{ts:.6f}",
                "longitude": f"{meta['longitude'][idx]:.10f}",
                "latitude": f"{meta['latitude'][idx]:.10f}",
                "ground_elevation_m": f"{meta['ground_elevation_m'][idx]:.6f}",
                "flight_height_m": f"{meta['flight_height_m'][idx]:.6f}",
                "local_x_m": f"{meta['local_x_m'][idx]:.6f}",
                "local_y_m": f"{meta['local_y_m'][idx]:.6f}",
                "local_z_m": f"{meta['local_z_m'][idx]:.6f}",
                "rtk_fix_type": 5,
                "satellites": 17 + int(idx % 6),
                "hdop": f"{0.52 + 0.05 * ((idx + 1) % 5):.3f}",
            }
        )
        imu_rows.append(
            {
                "timestamp_s": f"{ts:.6f}",
                "roll_deg": f"{meta['roll_deg'][idx]:.6f}",
                "pitch_deg": f"{meta['pitch_deg'][idx]:.6f}",
                "yaw_deg": f"{meta['yaw_deg'][idx]:.6f}",
                "angular_rate_x": f"{0.18 * np.cos(idx * 0.17):.6f}",
                "angular_rate_y": f"{0.14 * np.sin(idx * 0.15):.6f}",
                "angular_rate_z": f"{0.06 * np.cos(idx * 0.10):.6f}",
            }
        )
        confidence = 0.84 + 0.10 * np.sin(idx / 19.0)
        altimeter_rows.append(
            {
                "timestamp_s": f"{ts:.6f}",
                "height_agl_m": f"{meta['height_agl_m'][idx]:.6f}",
                "height_source": "synthetic_motion_effect_demo",
                "snr": f"{17.0 + 4.0 * confidence:.3f}",
                "target_count": 1,
                "valid": 1,
                "height_confidence": f"{np.clip(confidence, 0.72, 0.98):.3f}",
            }
        )

    _write_dict_rows(output_dir / "trace_timestamps.csv", timestamp_rows)
    _write_dict_rows(output_dir / "rtk.csv", rtk_rows)
    _write_dict_rows(output_dir / "imu.csv", imu_rows)
    _write_dict_rows(output_dir / "altimeter.csv", altimeter_rows)


def _run_method(
    data: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    method_id: str,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    runtime_params = prepare_runtime_params(
        method_id,
        params,
        header_info,
        trace_metadata,
        tuple(data.shape),
    )
    result, meta = run_processing_method(data, method_id, runtime_params)
    next_header = merge_result_header_info(header_info, meta, tuple(result.shape))
    next_trace_metadata = merge_result_trace_metadata(trace_metadata, meta)
    return result, next_header, next_trace_metadata, meta


def _run_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    data = np.asarray(payload["data"], dtype=np.float32)
    header = dict(payload["header_info"])
    metadata = {key: np.asarray(value).copy() for key, value in payload["trace_metadata"].items()}
    stages: list[dict[str, Any]] = [
        {
            "method": "raw",
            "label": "Raw",
            "data": data.copy(),
            "header_info": dict(header),
            "trace_metadata": {key: value.copy() for key, value in metadata.items()},
            "meta": {},
        }
    ]
    for method_id, params in PIPELINE:
        data, header, metadata, method_meta = _run_method(data, header, metadata, method_id, params)
        stages.append(
            {
                "method": method_id,
                "label": method_id,
                "data": np.asarray(data, dtype=np.float32).copy(),
                "header_info": dict(header),
                "trace_metadata": {key: np.asarray(value).copy() for key, value in metadata.items()},
                "meta": method_meta,
                "params": dict(params),
            }
        )
    return {
        "final_data": data,
        "final_header_info": header,
        "final_trace_metadata": metadata,
        "stages": stages,
    }


def _clip_for_display(data: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(data, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    scale = float(np.nanpercentile(np.abs(finite), 98.5))
    if scale <= 0:
        scale = 1.0
    return -scale, scale


def _save_bscan(path: Path, data: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _clip_for_display(data)
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=150)
    ax.imshow(data, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Sample")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_bscan_comparison(path: Path, stages: list[dict[str, Any]]) -> None:
    selected = [
        stages[0],
        next(stage for stage in stages if stage["method"] == "motion_compensation_speed"),
        next(stage for stage in stages if stage["method"] == "motion_compensation_height"),
    ]
    titles = ["Raw", "After speed resample", "After four motion steps"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), dpi=150)
    for ax, stage, title in zip(axes, selected, titles):
        data = np.asarray(stage["data"], dtype=np.float32)
        vmin, vmax = _clip_for_display(data)
        ax.imshow(data, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _track_top_interface_std(data: np.ndarray) -> float:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 70:
        return 0.0
    band = np.abs(arr[28:62, :])
    peak_rows = np.argmax(band, axis=0) + 28
    if peak_rows.size > 9:
        kernel = np.ones(9, dtype=np.float64) / 9.0
        trend = np.convolve(peak_rows.astype(np.float64), kernel, mode="same")
        peak_rows = peak_rows[4:-4] - trend[4:-4]
    return float(np.std(peak_rows))


def _resample_like_reference(data: np.ndarray, target_traces: int) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.shape[1] == target_traces:
        return arr
    source_axis = np.linspace(0.0, 1.0, arr.shape[1], dtype=np.float64)
    target_axis = np.linspace(0.0, 1.0, target_traces, dtype=np.float64)
    return _interp_columns(arr, source_axis, target_axis)


def _save_3d_previews(
    output_dir: Path,
    raw_stage: dict[str, Any],
    final_stage: dict[str, Any],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    raw_payload = build_airborne_georeference_3d_payload(
        raw_stage["data"],
        raw_stage["header_info"],
        raw_stage["trace_metadata"],
        max_preview_traces=260,
        max_preview_samples=180,
    )
    final_payload = build_airborne_georeference_3d_payload(
        final_stage["data"],
        final_stage["header_info"],
        final_stage["trace_metadata"],
        max_preview_traces=260,
        max_preview_samples=180,
    )
    if raw_payload is not None:
        raw_path = output_dir / "raw_3d_preview.png"
        save_airborne_georeference_3d_preview_png(raw_payload, raw_path, title="Raw 3D motion preview")
        paths["raw_3d_preview_png"] = str(raw_path)
    if final_payload is not None:
        final_path = output_dir / "final_3d_preview.png"
        save_airborne_georeference_3d_preview_png(final_payload, final_path, title="Final 3D motion preview")
        paths["final_3d_preview_png"] = str(final_path)
    return paths


def _write_readme(output_dir: Path) -> None:
    text = """# MyGPR UAV-GPR Motion Effect Demo v1

This package is a synthetic validation/demo dataset for MyGPR motion compensation.
It is intentionally more visible than field data: non-equidistant spacing,
trajectory jitter, roll/pitch/yaw, and AGL height variation are all injected so
the four core motion compensation steps can be inspected in B-scan and 3D views.

It is not real field evidence and should not be used as a geological conclusion.

## How to Try It in MyGPR

1. Import `main.csv`.
2. The sixth CSV column carries `trace_timestamp_s`, so RTK/IMU/altimeter
   sidecars in this folder can be aligned without the previous missing timestamp
   warning.
3. Run the four-step motion workflow:
   `trajectory_smoothing -> motion_compensation_speed -> motion_compensation_attitude -> motion_compensation_height`.
4. Open the quality/export 3D preview. Raw and current 3D trajectories should
   differ, and the B-scan curtain should show a clearer target hyperbola after
   height compensation.

Recommended height parameter for this demo:
`wave_speed_m_per_ns = 0.299792458`.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _write_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    manifest = {
        "schema": "mygpr_uav_motion_effect_demo_v1",
        "description": "Synthetic UAV-GPR motion compensation effect validation dataset.",
        "data_file": "main.csv",
        "trace_timestamps_file": "trace_timestamps.csv",
        "rtk_file": "rtk.csv",
        "imu_file": "imu.csv",
        "altimeter_file": "altimeter.csv",
        "metadata_file": "metadata.json",
        "recommended_workflow": [method for method, _ in PIPELINE],
        "recommended_params": {method: params for method, params in PIPELINE},
        "expected_target": payload["truth"]["primary_target"],
        "notes": [
            "Synthetic only.",
            "Designed to make motion compensation effects visible in B-scan and 3D preview.",
            "Use wave_speed_m_per_ns=0.299792458 for the height step in this air-path demo.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate_motion_effect_demo(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> MotionEffectDemoResult:
    """Generate input sidecars, run four motion steps, and export demo evidence."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    payload = _build_motion_payload()
    _write_main_csv(out / "main.csv", payload)
    _write_sidecars(out, payload)
    _write_manifest(out, payload)
    _write_readme(out)
    metadata = {
        "schema": "mygpr_uav_motion_effect_demo_v1_metadata",
        "samples": SAMPLES,
        "traces": TRACES,
        "total_time_ns": TOTAL_TIME_NS,
        "air_wave_speed_m_per_ns": AIR_WAVE_SPEED_M_PER_NS,
        "reference_height_m": float(payload["reference_height_m"]),
        "height_min_m": float(np.min(payload["trace_metadata"]["height_agl_m"])),
        "height_max_m": float(np.max(payload["trace_metadata"]["height_agl_m"])),
        "max_abs_injected_shift_samples": float(np.max(np.abs(payload["height_shift_samples"]))),
        "truth": payload["truth"],
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    raw_csv = readcsv(str(out / "main.csv"))
    data, trace_metadata, header_info = extract_airborne_csv_payload(
        raw_csv,
        {
            "a_scan_length": SAMPLES,
            "num_traces": TRACES,
            "total_time_ns": TOTAL_TIME_NS,
            "trace_interval_m": payload["header_info"]["trace_interval_m"],
        },
        rtk_path=out / "rtk.csv",
        imu_path=out / "imu.csv",
        altimeter_path=out / "altimeter.csv",
    )
    if trace_metadata is None or header_info is None:
        raise RuntimeError("failed to parse generated motion sidecars")

    result = _run_pipeline(
        {
            "data": data,
            "header_info": header_info,
            "trace_metadata": trace_metadata,
        }
    )
    stages = result["stages"]
    final_stage = stages[-1]
    final_data = np.asarray(final_stage["data"], dtype=np.float32)
    raw_resampled = _resample_like_reference(np.asarray(stages[0]["data"], dtype=np.float32), final_data.shape[1])
    bscan_rms_delta = float(np.sqrt(np.mean((final_data - raw_resampled) ** 2)))
    top_before = _track_top_interface_std(np.asarray(stages[0]["data"], dtype=np.float32))
    top_after = _track_top_interface_std(final_data)

    _save_bscan(out / "raw_bscan.png", np.asarray(stages[0]["data"], dtype=np.float32), "Raw synthetic UAV-GPR B-scan")
    _save_bscan(out / "final_motion_bscan.png", final_data, "After four motion compensation steps")
    _save_bscan_comparison(out / "bscan_motion_comparison.png", stages)
    preview_paths = _save_3d_previews(out, stages[0], final_stage)

    stage_summaries = []
    for stage in stages:
        meta = stage.get("meta") or {}
        stage_summaries.append(
            {
                "method": stage["method"],
                "shape": list(np.asarray(stage["data"]).shape),
                "params": stage.get("params") or {},
                "skipped": bool(meta.get("skipped", False)),
                "reason": meta.get("reason"),
                "warnings": meta.get("warnings") or [],
                "key_metrics": {
                    key: value
                    for key, value in meta.items()
                    if key
                    in {
                        "max_displacement_m",
                        "mean_displacement_m",
                        "spacing_m",
                        "source_traces",
                        "target_traces",
                        "max_shift_samples_applied",
                        "input_height_std_m",
                        "shift_clamped",
                    }
                },
            }
        )
    summary = {
        "schema": "mygpr_uav_motion_effect_demo_v1_summary",
        "output_dir": str(out),
        "main_csv": str(out / "main.csv"),
        "pipeline": [method for method, _ in PIPELINE],
        "raw_shape": list(np.asarray(stages[0]["data"]).shape),
        "final_shape": list(final_data.shape),
        "bscan_rms_delta": bscan_rms_delta,
        "top_interface_std_before": top_before,
        "top_interface_std_after": top_after,
        "before_spacing_std_m": float(np.std(np.diff(np.asarray(stages[0]["trace_metadata"]["trace_distance_m"], dtype=np.float64)))),
        "after_speed_spacing_std_m": float(np.std(np.diff(np.asarray(stages[2]["trace_metadata"]["trace_distance_m"], dtype=np.float64)))),
        "after_spacing_std_m": float(np.std(np.diff(np.asarray(final_stage["trace_metadata"]["trace_distance_m"], dtype=np.float64)))),
        "preview_paths": preview_paths,
        "stages": stage_summaries,
    }
    summary_path = out / "processing_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return MotionEffectDemoResult(
        output_dir=out,
        main_csv=out / "main.csv",
        summary_json=summary_path,
        final_data_shape=tuple(final_data.shape),
        bscan_rms_delta=bscan_rms_delta,
        top_interface_std_before=top_before,
        top_interface_std_after=top_after,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a visible UAV-GPR motion compensation effect demo."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    result = generate_motion_effect_demo(args.output_dir)
    print(f"Generated motion effect demo: {result.output_dir}")
    print(f"Main CSV: {result.main_csv}")
    print(f"Summary: {result.summary_json}")
    print(f"Final shape: {result.final_data_shape}")
    print(f"B-scan RMS delta: {result.bscan_rms_delta:.6f}")
    print(
        "Top interface std before/after: "
        f"{result.top_interface_std_before:.3f} -> {result.top_interface_std_after:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
