#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a motion-compensation validation package from a gprMax dataset."""

from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.uav_georeference_3d import (  # noqa: E402
    build_airborne_georeference_3d_payload,
)
from core.trace_metadata_utils import resample_bscan_columns_linear  # noqa: E402
from PythonModule.motion_compensation_attitude import method_motion_compensation_attitude  # noqa: E402
from PythonModule.motion_compensation_height import method_motion_compensation_height  # noqa: E402
from PythonModule.motion_compensation_speed import method_motion_compensation_speed  # noqa: E402
from PythonModule.motion_compensation_v2 import method_motion_compensation_v2  # noqa: E402
from PythonModule.trajectory_smoothing import method_trajectory_smoothing  # noqa: E402
from read_file_data import readcsv  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "output" / "gprmax_motion_validation"
AIR_WAVE_SPEED_M_PER_NS = 0.299792458
START_TIMESTAMP_S = 4200.0
TRACE_PERIOD_S = 0.08
ORIGIN_LONGITUDE = 104.123456
ORIGIN_LATITUDE = 30.654321
MOTION_DEMO_PROFILE: dict[str, float] = {
    "spacing_sin_fraction": 0.40,
    "spacing_cos_fraction": 0.20,
    "spacing_noise_fraction": 0.055,
    "spacing_min_fraction": 0.42,
    "spacing_max_fraction": 1.72,
    "lateral_track_fraction": 0.14,
    "lateral_min_trace_intervals": 1.10,
    "lateral_max_trace_intervals": 3.80,
    "target_height_shift_samples": 40.0,
    "height_min_m": 0.22,
    "height_max_m": 1.10,
    "roll_amp_deg": 9.0,
    "pitch_amp_deg": 8.0,
    "yaw_base_deg": 4.0,
    "yaw_amp_deg": 9.0,
    "noise_std": 0.014,
    "striping_amp": 0.024,
}


@dataclass(frozen=True)
class GprMaxMotionValidationResult:
    """Paths and metrics for a generated gprMax motion validation package."""

    output_dir: Path
    main_csv: Path
    summary_json: Path
    report_md: Path
    raw_shape: tuple[int, int]
    atomic_shape: tuple[int, int]
    v2_shape: tuple[int, int]
    spacing_std_before_m: float
    spacing_std_atomic_m: float
    spacing_std_atomic_after_speed_m: float
    spacing_std_v2_m: float
    target_ratio_raw: float | None
    target_ratio_atomic: float | None
    target_ratio_v2: float | None


ATOMIC_PIPELINE: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "trajectory_smoothing",
        {
            "method": "savgol",
            "window_length": 31,
            "polyorder": 3,
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
        "motion_compensation_speed",
        {
            "spacing_m": 0.0,
            "interpolation_mode": "linear",
        },
    ),
    (
        "motion_compensation_height",
        {
            "reference_height_mode": "mean",
            "manual_height": 0.0,
            "height_source": "auto",
            "compensate_amplitude": True,
            "compensate_time_shift": True,
            "wave_speed_m_per_ns": AIR_WAVE_SPEED_M_PER_NS,
            "max_shift_samples": 64.0,
            "max_shift_ns": 20.0,
            "interpolation_mode": "linear",
        },
    ),
)

V2_PARAMS: dict[str, Any] = {
    "height_reference_mode": "mean",
    "height_source": "auto",
    "compensate_time_shift": True,
    "compensate_amplitude": True,
    "max_shift_samples": 64.0,
    "max_shift_ns": 20.0,
    "max_amplitude_scale": 2.0,
    "resample_spacing_m": 0.0,
    "apc_offset_x_m": 0.04,
    "apc_offset_y_m": -0.02,
    "apc_offset_z_m": 0.0,
    "max_abs_tilt_deg": 18.0,
}

METHOD_CALLS = {
    "trajectory_smoothing": method_trajectory_smoothing,
    "motion_compensation_speed": method_motion_compensation_speed,
    "motion_compensation_attitude": method_motion_compensation_attitude,
    "motion_compensation_height": method_motion_compensation_height,
    "motion_compensation_v2": method_motion_compensation_v2,
}


def generate_gprmax_motion_validation_package(
    dataset: str | Path,
    output_dir: str | Path | None = None,
    *,
    seed: int = 20260519,
    target_traces: int | None = None,
) -> GprMaxMotionValidationResult:
    """Generate a UAV-motion validation package from a gprMax manifest/folder."""
    from core.gpr_io import extract_airborne_csv_payload
    from core.gprmax_dataset_contract import load_gprmax_dataset_contract

    package = load_gprmax_dataset_contract(dataset)
    out = Path(output_dir or _default_output_dir(package.scenario_id)).resolve()
    out.mkdir(parents=True, exist_ok=True)

    source = _build_source_payload(
        package.data,
        package.header_info,
        seed=seed,
        target_traces=target_traces,
    )
    source["scenario_id"] = package.scenario_id
    source["source_manifest"] = str(package.manifest_path)
    source["source_primary_out_file"] = str(package.primary_out_file)
    source["source_ground_truth_file"] = str(package.ground_truth_file)
    source["ground_truth"] = _scale_ground_truth_for_trace_count(
        package.ground_truth,
        int(source["original_gprmax_shape"][1]),
        int(source["traces"]),
    )
    source["ground_truth_raw"] = package.ground_truth_raw

    _write_main_csv(out / "main.csv", source)
    _write_sidecars(out, source)
    source["copied_source_artifacts"] = _copy_source_artifacts(out, source)
    _write_manifest(out, source)
    _write_metadata(out, source)

    raw_csv = readcsv(str(out / "main.csv"))
    raw_data, trace_metadata, header_info = extract_airborne_csv_payload(
        raw_csv,
        _csv_header(source),
        rtk_path=out / "rtk.csv",
        imu_path=out / "imu.csv",
        altimeter_path=out / "altimeter.csv",
    )
    if trace_metadata is None or header_info is None:
        raise RuntimeError("failed to parse generated UAV motion sidecars")

    atomic = _run_atomic_pipeline(raw_data, header_info, trace_metadata, source["trace_interval_m"])
    v2 = _run_single_method(
        raw_data,
        header_info,
        trace_metadata,
        "motion_compensation_v2",
        {**V2_PARAMS, "resample_spacing_m": source["trace_interval_m"]},
    )

    summary = _build_summary(out, source, raw_data, trace_metadata, atomic, v2)
    _write_images(out, source, raw_data, trace_metadata, header_info, atomic, v2, summary)
    summary["artifacts"]["images"] = _artifact_images(out)
    summary_path = out / "processing_summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_md = out / "motion_validation_report.md"
    report_md.write_text(_render_report(summary), encoding="utf-8")
    report_html = out / "motion_validation_report.html"
    report_html.write_text(_render_html_report(summary), encoding="utf-8")
    _write_readme(out, source)

    return GprMaxMotionValidationResult(
        output_dir=out,
        main_csv=out / "main.csv",
        summary_json=summary_path,
        report_md=report_md,
        raw_shape=tuple(np.asarray(raw_data).shape),
        atomic_shape=tuple(np.asarray(atomic["data"]).shape),
        v2_shape=tuple(np.asarray(v2["data"]).shape),
        spacing_std_before_m=float(summary["metrics"]["spacing_std_before_m"]),
        spacing_std_atomic_m=float(summary["metrics"]["spacing_std_atomic_m"]),
        spacing_std_atomic_after_speed_m=float(summary["metrics"]["spacing_std_atomic_after_speed_m"]),
        spacing_std_v2_m=float(summary["metrics"]["spacing_std_v2_m"]),
        target_ratio_raw=summary["metrics"].get("target_ratio_raw"),
        target_ratio_atomic=summary["metrics"].get("target_ratio_atomic"),
        target_ratio_v2=summary["metrics"].get("target_ratio_v2"),
    )


def _default_output_dir(scenario_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in scenario_id)
    return DEFAULT_OUTPUT_ROOT / f"{safe}_{timestamp}"


def _build_source_payload(
    data: np.ndarray,
    header_info: dict[str, Any],
    *,
    seed: int,
    target_traces: int | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ideal = _normalize_bscan(data)
    original_samples, original_traces = ideal.shape
    if target_traces is not None and int(target_traces) > original_traces:
        ideal = _resample_like_reference(ideal, int(target_traces))
    samples, traces = ideal.shape
    if traces < 2:
        raise ValueError("gprMax motion validation requires at least two traces")
    total_time_ns = _resolve_total_time_ns(header_info, samples)
    trace_interval = _resolve_trace_interval_m(header_info)
    uniform_x = np.arange(traces, dtype=np.float64) * trace_interval
    track_length = max(float(uniform_x[-1] - uniform_x[0]), trace_interval * max(traces - 1, 1))
    phase = np.linspace(0.0, 1.0, traces, dtype=np.float64)
    profile = MOTION_DEMO_PROFILE

    spacing = trace_interval * (
        1.0
        + profile["spacing_sin_fraction"] * np.sin(2.0 * np.pi * 1.8 * phase + 0.25)
        + profile["spacing_cos_fraction"] * np.cos(2.0 * np.pi * 3.3 * phase)
    )
    spacing += rng.normal(0.0, trace_interval * profile["spacing_noise_fraction"], size=traces)
    spacing = np.clip(
        spacing,
        trace_interval * profile["spacing_min_fraction"],
        trace_interval * profile["spacing_max_fraction"],
    )
    local_x = np.cumsum(spacing)
    local_x -= local_x[0]
    local_x *= uniform_x[-1] / max(float(local_x[-1]), 1.0e-9)
    lateral_amp = min(
        max(track_length * profile["lateral_track_fraction"], trace_interval * profile["lateral_min_trace_intervals"]),
        trace_interval * profile["lateral_max_trace_intervals"],
    )
    local_y = lateral_amp * np.sin(2.0 * np.pi * 0.72 * phase - 0.2)
    local_y += lateral_amp * 0.35 * np.sin(2.0 * np.pi * 8.0 * phase)

    ground_elevation = 116.0 + 0.30 * np.sin(2.0 * np.pi * 0.35 * phase)
    dt_ns = total_time_ns / max(samples - 1, 1)
    height_amp = min(
        max(profile["target_height_shift_samples"] * dt_ns * AIR_WAVE_SPEED_M_PER_NS / 2.0, 0.025),
        0.32,
    )
    height_agl = 0.60 + height_amp * np.sin(2.0 * np.pi * 1.35 * phase + 0.45)
    height_agl += height_amp * 0.35 * np.sin(2.0 * np.pi * 6.0 * phase)
    height_agl = np.clip(height_agl, profile["height_min_m"], profile["height_max_m"])
    flight_height = height_agl.copy()
    local_z = ground_elevation + flight_height
    longitude, latitude = _lon_lat_from_xy(local_x, local_y)
    roll = profile["roll_amp_deg"] * np.sin(2.0 * np.pi * 1.9 * phase + 0.4)
    pitch = profile["pitch_amp_deg"] * np.cos(2.0 * np.pi * 1.5 * phase - 0.1)
    yaw = profile["yaw_base_deg"] + profile["yaw_amp_deg"] * np.sin(2.0 * np.pi * 0.55 * phase)
    timestamps = START_TIMESTAMP_S + TRACE_PERIOD_S * np.arange(traces, dtype=np.float64)

    observed = _interp_columns(ideal, uniform_x, local_x)
    reference_height = float(np.mean(height_agl))
    shift_samples = 2.0 * (height_agl - reference_height) / AIR_WAVE_SPEED_M_PER_NS / max(dt_ns, 1.0e-9)
    for trace_idx in range(traces):
        observed[:, trace_idx] = _shift_trace(observed[:, trace_idx], shift_samples[trace_idx])
        observed[:, trace_idx] *= np.float32(np.clip((reference_height / height_agl[trace_idx]) ** 2, 0.50, 2.0))
    observed += profile["noise_std"] * rng.normal(size=observed.shape).astype(np.float32)
    observed += (profile["striping_amp"] * np.sin(2.0 * np.pi * 5.5 * phase))[None, :].astype(np.float32)
    observed = _normalize_bscan(observed)

    trace_distance = np.empty(traces, dtype=np.float64)
    trace_distance[0] = 0.0
    trace_distance[1:] = np.cumsum(np.hypot(np.diff(local_x), np.diff(local_y)), dtype=np.float64)
    return {
        "ideal_data": ideal,
        "original_gprmax_shape": (int(original_samples), int(original_traces)),
        "derived_longline": bool(traces != original_traces),
        "data": observed,
        "samples": samples,
        "traces": traces,
        "total_time_ns": total_time_ns,
        "trace_interval_m": trace_interval,
        "uniform_x_m": uniform_x,
        "reference_height_m": reference_height,
        "height_shift_samples": shift_samples,
        "motion_profile": {
            **profile,
            "local_y_peak_to_peak_m": float(np.ptp(local_y)),
            "height_agl_peak_to_peak_m": float(np.ptp(height_agl)),
            "max_abs_height_shift_samples": float(np.max(np.abs(shift_samples))),
            "trace_spacing_peak_to_peak_m": float(np.ptp(np.diff(trace_distance))) if trace_distance.size > 2 else 0.0,
            "roll_peak_to_peak_deg": float(np.ptp(roll)),
            "pitch_peak_to_peak_deg": float(np.ptp(pitch)),
            "yaw_peak_to_peak_deg": float(np.ptp(yaw)),
        },
        "trace_metadata": {
            "trace_index": np.arange(traces, dtype=np.int32),
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
        },
    }


def _normalize_bscan(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - np.median(arr, axis=0, keepdims=True).astype(np.float32)
    scale = float(np.percentile(np.abs(arr), 99.0)) if arr.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (arr / scale).astype(np.float32)


def _resolve_total_time_ns(header_info: dict[str, Any], samples: int) -> float:
    for key in ("total_time_ns", "time_window_ns"):
        value = header_info.get(key)
        if value is not None:
            candidate = float(value)
            if np.isfinite(candidate) and candidate > 0.0:
                return candidate
    dt_s = header_info.get("gprmax_dt_s")
    if dt_s is not None:
        candidate = float(dt_s) * float(samples) * 1.0e9
        if np.isfinite(candidate) and candidate > 0.0:
            return candidate
    return 120.0


def _resolve_trace_interval_m(header_info: dict[str, Any]) -> float:
    value = header_info.get("trace_interval_m")
    if value is not None:
        candidate = float(value)
        if np.isfinite(candidate) and candidate > 0.0:
            return candidate
    return 0.25


def _interp_columns(source: np.ndarray, source_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    return resample_bscan_columns_linear(source, source_x, target_x)


def _shift_trace(trace: np.ndarray, shift_samples: float) -> np.ndarray:
    sample_index = np.arange(trace.size, dtype=np.float64)
    return np.interp(
        sample_index - float(shift_samples),
        sample_index,
        np.asarray(trace, dtype=np.float64),
        left=0.0,
        right=0.0,
    ).astype(np.float32)


def _lon_lat_from_xy(x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(ORIGIN_LATITUDE))
    longitude = ORIGIN_LONGITUDE + np.asarray(x_m, dtype=np.float64) / meters_per_deg_lon
    latitude = ORIGIN_LATITUDE + np.asarray(y_m, dtype=np.float64) / meters_per_deg_lat
    return longitude, latitude


def _csv_header(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "a_scan_length": int(source["samples"]),
        "num_traces": int(source["traces"]),
        "total_time_ns": float(source["total_time_ns"]),
        "trace_interval_m": float(source["trace_interval_m"]),
    }


def _write_main_csv(path: Path, source: dict[str, Any]) -> None:
    data = np.asarray(source["data"], dtype=np.float32)
    meta = source["trace_metadata"]
    samples = int(source["samples"])
    traces = int(source["traces"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"Number of Samples = {samples}\n")
        handle.write(f"Time windows (ns) = {float(source['total_time_ns']):.9f}\n")
        handle.write(f"Number of Traces = {traces}\n")
        handle.write(f"Trace interval (m) = {float(source['trace_interval_m']):.9f}\n")
        writer = csv.writer(handle)
        for trace_idx in range(traces):
            for sample_idx in range(samples):
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_sidecars(output_dir: Path, source: dict[str, Any]) -> None:
    meta = source["trace_metadata"]
    timestamps = []
    rtk_rows = []
    imu_rows = []
    altimeter_rows = []
    for idx in range(int(source["traces"])):
        ts = float(meta["trace_timestamp_s"][idx])
        timestamps.append({"trace_index": idx, "timestamp_s": f"{ts:.6f}"})
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
                "satellites": 16 + int(idx % 5),
                "hdop": f"{0.55 + 0.04 * (idx % 4):.3f}",
            }
        )
        imu_rows.append(
            {
                "timestamp_s": f"{ts:.6f}",
                "roll_deg": f"{meta['roll_deg'][idx]:.6f}",
                "pitch_deg": f"{meta['pitch_deg'][idx]:.6f}",
                "yaw_deg": f"{meta['yaw_deg'][idx]:.6f}",
            }
        )
        confidence = 0.86 + 0.08 * math.sin(idx / 17.0)
        altimeter_rows.append(
            {
                "timestamp_s": f"{ts:.6f}",
                "height_agl_m": f"{meta['height_agl_m'][idx]:.6f}",
                "height_source": "synthetic_gprmax_motion",
                "snr": f"{18.0 + 4.0 * confidence:.3f}",
                "target_count": 1,
                "valid": 1,
                "height_confidence": f"{np.clip(confidence, 0.72, 0.98):.3f}",
            }
        )
    _write_dict_rows(output_dir / "trace_timestamps.csv", timestamps)
    _write_dict_rows(output_dir / "rtk.csv", rtk_rows)
    _write_dict_rows(output_dir / "imu.csv", imu_rows)
    _write_dict_rows(output_dir / "altimeter.csv", altimeter_rows)


def _write_manifest(output_dir: Path, source: dict[str, Any]) -> None:
    manifest = {
        "schema": "mygpr_gprmax_motion_validation_package_v1",
        "scenario_id": source.get("scenario_id"),
        "description": "gprMax-derived B-scan with injected UAV trajectory/attitude/height motion for MyGPR motion compensation validation.",
        "data_file": "main.csv",
        "trace_timestamps_file": "trace_timestamps.csv",
        "rtk_file": "rtk.csv",
        "imu_file": "imu.csv",
        "altimeter_file": "altimeter.csv",
        "metadata_file": "metadata.json",
        "source_manifest": source.get("source_manifest"),
        "source_primary_out_file": source.get("source_primary_out_file"),
        "source_ground_truth_file": source.get("source_ground_truth_file"),
        "copied_source_artifacts": source.get("copied_source_artifacts", {}),
        "source_shape": list(source.get("original_gprmax_shape", [])),
        "derived_longline": bool(source.get("derived_longline", False)),
        "motion_profile": _jsonable(source.get("motion_profile", {})),
        "recommended_workflow": [method for method, _params in ATOMIC_PIPELINE],
        "recommended_v2_method": "motion_compensation_v2",
        "recommended_params": {
            **{method: params for method, params in ATOMIC_PIPELINE},
            "motion_compensation_v2": {
                **V2_PARAMS,
                "resample_spacing_m": float(source["trace_interval_m"]),
            },
        },
        "notes": [
            "The underground wavefield comes from gprMax.",
            "UAV motion metadata and motion artifacts are injected by MyGPR for controlled validation.",
            "This is validation evidence, not field geology evidence.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_metadata(output_dir: Path, source: dict[str, Any]) -> None:
    metadata = {
        "schema": "mygpr_gprmax_motion_validation_metadata_v1",
        "scenario_id": source.get("scenario_id"),
        "source_manifest": source.get("source_manifest"),
        "source_primary_out_file": source.get("source_primary_out_file"),
        "source_ground_truth_file": source.get("source_ground_truth_file"),
        "copied_source_artifacts": source.get("copied_source_artifacts", {}),
        "source_shape": list(source.get("original_gprmax_shape", [])),
        "derived_longline": bool(source.get("derived_longline", False)),
        "samples": int(source["samples"]),
        "traces": int(source["traces"]),
        "total_time_ns": float(source["total_time_ns"]),
        "trace_interval_m": float(source["trace_interval_m"]),
        "reference_height_m": float(source["reference_height_m"]),
        "height_min_m": float(np.min(source["trace_metadata"]["height_agl_m"])),
        "height_max_m": float(np.max(source["trace_metadata"]["height_agl_m"])),
        "max_abs_injected_shift_samples": float(np.max(np.abs(source["height_shift_samples"]))),
        "motion_profile": _jsonable(source.get("motion_profile", {})),
        "ground_truth": source.get("ground_truth"),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _copy_source_artifacts(output_dir: Path, source: dict[str, Any]) -> dict[str, str]:
    """Copy small source descriptors next to the derived validation package."""
    copied: dict[str, str] = {}
    candidates = {
        "source_manifest": source.get("source_manifest"),
        "source_ground_truth": source.get("source_ground_truth_file"),
    }
    primary = Path(str(source.get("source_primary_out_file") or ""))
    manifest = Path(str(source.get("source_manifest") or ""))
    search_dirs = [path.parent for path in (primary, manifest) if str(path) and path.parent.exists()]
    in_file: Path | None = None
    for folder in search_dirs:
        matches = sorted(folder.glob("*.in"))
        if matches:
            in_file = matches[0]
            break
    if in_file is not None:
        candidates["source_model_in"] = str(in_file)

    for key, value in candidates.items():
        if not value:
            continue
        source_path = Path(str(value))
        if not source_path.exists() or not source_path.is_file():
            continue
        suffix = source_path.suffix or ".txt"
        target = output_dir / f"{key}{suffix}"
        shutil.copy2(source_path, target)
        copied[key] = str(target)
    return copied


def _scale_ground_truth_for_trace_count(
    ground_truth: dict[str, Any] | None,
    source_traces: int,
    target_traces: int,
) -> dict[str, Any] | None:
    """Scale MyGPR half-open trace ROIs when a source B-scan is trace-resampled."""
    if not isinstance(ground_truth, dict):
        return ground_truth
    if source_traces <= 0 or target_traces <= 0 or source_traces == target_traces:
        return copy.deepcopy(ground_truth)
    scaled = copy.deepcopy(ground_truth)
    scale = float(target_traces) / float(source_traces)

    def scale_roi(roi: dict[str, Any] | None) -> None:
        if not isinstance(roi, dict):
            return
        d0 = int(roi.get("dist_start_idx", 0))
        d1 = int(roi.get("dist_end_idx", target_traces))
        roi["dist_start_idx"] = max(0, min(target_traces, int(math.floor(d0 * scale))))
        roi["dist_end_idx"] = max(roi["dist_start_idx"], min(target_traces, int(math.ceil(d1 * scale))))

    scale_roi(scaled.get("analysis_roi"))
    for target in scaled.get("targets") or []:
        if isinstance(target, dict):
            scale_roi(target.get("roi"))
    for background in scaled.get("background_rois") or []:
        scale_roi(background)
    scaled.setdefault("conversion_notes", []).append(
        {
            "code": "trace_roi_scaled_for_motion_validation_longline",
            "message": "Trace ROI was scaled after resampling the source gprMax B-scan for a longer validation line.",
            "source_traces": int(source_traces),
            "target_traces": int(target_traces),
        }
    )
    return scaled


def _scale_single_roi_for_trace_count(
    roi: dict[str, Any] | None,
    source_traces: int,
    target_traces: int,
) -> dict[str, int] | None:
    if not isinstance(roi, dict):
        return None
    scaled = dict(roi)
    if source_traces <= 0 or target_traces <= 0 or source_traces == target_traces:
        return {key: int(value) for key, value in scaled.items() if key.endswith("_idx")}
    scale = float(target_traces) / float(source_traces)
    d0 = int(scaled.get("dist_start_idx", 0))
    d1 = int(scaled.get("dist_end_idx", target_traces))
    scaled["dist_start_idx"] = max(0, min(target_traces, int(math.floor(d0 * scale))))
    scaled["dist_end_idx"] = max(scaled["dist_start_idx"], min(target_traces, int(math.ceil(d1 * scale))))
    return {key: int(value) for key, value in scaled.items() if key.endswith("_idx")}


def _run_atomic_pipeline(
    data: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    trace_interval_m: float,
) -> dict[str, Any]:
    stages: list[dict[str, Any]] = [
        {
            "method": "raw",
            "data": np.asarray(data, dtype=np.float32).copy(),
            "header_info": dict(header_info),
            "trace_metadata": {key: np.asarray(value).copy() for key, value in trace_metadata.items()},
            "meta": {},
            "params": {},
        }
    ]
    current = np.asarray(data, dtype=np.float32)
    header = dict(header_info)
    metadata = {key: np.asarray(value).copy() for key, value in trace_metadata.items()}
    for method_id, base_params in ATOMIC_PIPELINE:
        params = dict(base_params)
        if method_id == "trajectory_smoothing":
            params["window_length"] = _safe_savgol_window(current.shape[1], int(params["window_length"]))
        if method_id == "motion_compensation_speed":
            params["spacing_m"] = float(trace_interval_m)
        current, header, metadata, method_meta = _run_method(
            current,
            header,
            metadata,
            method_id,
            params,
        )
        stages.append(
            {
                "method": method_id,
                "data": np.asarray(current, dtype=np.float32).copy(),
                "header_info": dict(header),
                "trace_metadata": {key: np.asarray(value).copy() for key, value in metadata.items()},
                "meta": method_meta,
                "params": params,
            }
        )
    return {"data": current, "header_info": header, "trace_metadata": metadata, "stages": stages}


def _safe_savgol_window(trace_count: int, requested: int) -> int:
    if trace_count < 5:
        return 3 if trace_count >= 3 else max(1, trace_count)
    window = min(int(requested), trace_count if trace_count % 2 == 1 else trace_count - 1)
    return max(5, window if window % 2 == 1 else window - 1)


def _run_single_method(
    data: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    method_id: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    result, next_header, next_trace_metadata, meta = _run_method(
        data,
        header_info,
        trace_metadata,
        method_id,
        params,
    )
    return {
        "data": np.asarray(result, dtype=np.float32),
        "header_info": next_header,
        "trace_metadata": next_trace_metadata,
        "meta": meta,
        "params": params,
    }


def _run_method(
    data: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    method_id: str,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    method = METHOD_CALLS[method_id]
    runtime_params = {
        **dict(params),
        "trace_metadata": trace_metadata,
        "header_info": header_info,
        "time_window_ns": header_info.get("total_time_ns"),
    }
    result, meta = method(data, **runtime_params)
    next_header = _merge_header_info(header_info, meta, tuple(np.asarray(result).shape))
    next_trace_metadata = _merge_trace_metadata(trace_metadata, meta)
    return np.asarray(result, dtype=np.float32), next_header, next_trace_metadata, meta


def _merge_header_info(
    header_info: dict[str, Any],
    meta: dict[str, Any],
    shape: tuple[int, int],
) -> dict[str, Any]:
    merged = dict(header_info or {})
    updates = meta.get("header_info_updates")
    if isinstance(updates, dict):
        merged.update(updates)
    merged["a_scan_length"] = int(shape[0])
    merged["num_traces"] = int(shape[1])
    return merged


def _merge_trace_metadata(
    trace_metadata: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> dict[str, np.ndarray]:
    if isinstance(meta.get("trace_metadata_out"), dict):
        return {
            key: np.asarray(value).copy()
            for key, value in meta["trace_metadata_out"].items()
        }
    merged = {
        key: np.asarray(value).copy()
        for key, value in (trace_metadata or {}).items()
    }
    updates = meta.get("trace_metadata_updates")
    if isinstance(updates, dict):
        for key, value in updates.items():
            merged[key] = np.asarray(value).copy()
    return merged


def _write_images(
    output_dir: Path,
    source: dict[str, Any],
    raw_data: np.ndarray,
    raw_metadata: dict[str, np.ndarray],
    raw_header: dict[str, Any],
    atomic: dict[str, Any],
    v2: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    _save_bscan(output_dir / "source_gprmax_bscan.png", source["ideal_data"], "Source gprMax B-scan")
    _save_bscan(output_dir / "motion_injected_raw_bscan.png", raw_data, "Motion-injected raw B-scan")
    _save_bscan(output_dir / "atomic_motion_final_bscan.png", atomic["data"], "After four atomic motion steps")
    _save_bscan(output_dir / "motion_v2_final_bscan.png", v2["data"], "After motion_compensation_v2")
    _save_comparison(
        output_dir / "bscan_motion_validation_comparison.png",
        [
            ("gprMax source", source["ideal_data"]),
            ("motion-injected raw", raw_data),
            ("four atomic steps", atomic["data"]),
            ("motion_compensation_v2", v2["data"]),
        ],
    )
    _save_paper_comparison(
        output_dir / "paper_motion_validation_comparison.png",
        [
            ("source", source["ideal_data"]),
            ("raw", raw_data),
            ("atomic", atomic["data"]),
            ("motion v2", v2["data"]),
        ],
        summary,
    )
    for name, data, header, metadata, title in [
        ("raw_3d_preview.png", raw_data, raw_header, raw_metadata, "Motion-injected raw 3D preview"),
        ("atomic_3d_preview.png", atomic["data"], atomic["header_info"], atomic["trace_metadata"], "Atomic motion 3D preview"),
        ("motion_v2_3d_preview.png", v2["data"], v2["header_info"], v2["trace_metadata"], "Motion V2 3D preview"),
    ]:
        payload = build_airborne_georeference_3d_payload(
            data,
            header,
            metadata,
            max_preview_traces=260,
            max_preview_samples=180,
        )
        if payload is not None:
            _save_3d_preview_png_safe(payload, output_dir / name, title=title)


def _save_3d_preview_png_safe(payload: dict[str, Any], path: Path, *, title: str) -> None:
    """Save a 3D preview, falling back to a small diagnostic PNG on renderer pressure."""
    try:
        _save_lightweight_3d_preview(payload, path, title=title)
    except (MemoryError, RuntimeError, OSError) as exc:
        _save_preview_placeholder(path, title, f"3D preview unavailable: {exc}")


def _save_lightweight_3d_preview(payload: dict[str, Any], path: Path, *, title: str) -> None:
    preview = payload.get("preview") or {}
    curtain_z = np.asarray(preview.get("curtain_z_m", []), dtype=np.float64)
    amplitude = np.asarray(preview.get("amplitude", []), dtype=np.float64)
    if curtain_z.size == 0 or amplitude.shape != curtain_z.shape:
        raise ValueError("3D preview payload has no valid curtain mesh")
    row_step = max(1, int(math.ceil(amplitude.shape[0] / 96)))
    col_step = max(1, int(math.ceil(amplitude.shape[1] / 160)))
    amp = amplitude[::row_step, ::col_step]
    finite_amp = amp[np.isfinite(amp)]
    if finite_amp.size:
        vmin, vmax = np.percentile(finite_amp, [2.0, 98.0])
    else:
        vmin, vmax = 0.0, 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    x_m = np.asarray(payload.get("local_x_m", []), dtype=np.float64)
    z_m = np.asarray(payload.get("airborne_z_m", []), dtype=np.float64)
    if x_m.size:
        x_axis = np.linspace(float(np.nanmin(x_m)), float(np.nanmax(x_m)), amp.shape[1])
    else:
        x_axis = np.arange(amp.shape[1], dtype=np.float64)
    if curtain_z.size:
        z_min = float(np.nanmin(curtain_z))
        z_max = float(np.nanmax(curtain_z))
    else:
        z_min, z_max = float(amp.shape[0]), 0.0

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), dpi=120, height_ratios=[3, 1])
    try:
        ax_img, ax_line = axes
        ax_img.imshow(
            amp,
            cmap="gray",
            aspect="auto",
            vmin=float(vmin),
            vmax=float(vmax),
            extent=[float(x_axis[0]), float(x_axis[-1]), z_min, z_max],
        )
        ax_img.set_title(title)
        ax_img.set_xlabel("Along-track X (m)")
        ax_img.set_ylabel("Curtain Z (m)")
        if x_m.size and z_m.size:
            ax_line.plot(x_m, z_m, color="#0f766e", linewidth=1.3)
            ax_line.scatter([x_m[0]], [z_m[0]], color="#22c55e", s=18, label="start")
            ax_line.scatter([x_m[-1]], [z_m[-1]], color="#ef4444", s=18, label="end")
        ax_line.set_xlabel("Along-track X (m)")
        ax_line.set_ylabel("UAV Z (m)")
        ax_line.grid(True, alpha=0.25)
        if ax_line.get_legend_handles_labels()[0]:
            ax_line.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_preview_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 2.4), dpi=100)
    try:
        ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=11)
        ax.text(0.5, 0.38, message, ha="center", va="center", fontsize=8, wrap=True)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_bscan(path: Path, data: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=150)
    try:
        vmin, vmax = _clip_for_display(data)
        ax.imshow(np.asarray(data, dtype=np.float32), cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_comparison(path: Path, panels: list[tuple[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.0), dpi=150)
    try:
        axes_arr = np.asarray(axes).reshape(-1)
        vmin, vmax = _clip_for_display(*(data for _title, data in panels))
        for ax, (title, data) in zip(axes_arr, panels):
            ax.imshow(np.asarray(data, dtype=np.float32), cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample")
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def _save_paper_comparison(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
    summary: dict[str, Any],
) -> None:
    """Save a paper-friendly four-panel B-scan comparison with locked scale."""
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), dpi=170)
    try:
        axes_arr = np.asarray(axes).reshape(-1)
        vmin, vmax = _clip_for_display(*(data for _title, data in panels))
        metrics = summary.get("metrics", {})
        shapes = summary.get("shapes", {})
        metric_keys = {
            "source": None,
            "raw": "raw_vs_source_rms",
            "atomic": "atomic_vs_source_rms",
            "motion v2": "v2_vs_source_rms",
        }
        for ax, (title, data) in zip(axes_arr, panels):
            arr = np.asarray(data, dtype=np.float32)
            ax.imshow(arr, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
            rms_key = metric_keys.get(title)
            rms = metrics.get(rms_key) if rms_key else None
            ridge_key = f"ridge_rmse_samples_{title.replace('motion v2', 'v2').replace(' ', '_')}"
            ridge = metrics.get(ridge_key)
            parts = [f"{title}  {tuple(shapes.get(title.replace('motion v2', 'v2'), arr.shape))}"]
            if rms is not None:
                parts.append(f"RMS={float(rms):.4g}")
            if ridge is not None:
                parts.append(f"ridge={float(ridge):.3g} samp")
            ax.set_title("\n".join(parts), fontsize=10)
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample")
        fig.suptitle("gprMax Motion Validation: Source vs Injected Raw vs Compensation", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(path)
    finally:
        plt.close(fig)


def _clip_for_display(*arrays: np.ndarray) -> tuple[float, float]:
    chunks: list[np.ndarray] = []
    for data in arrays:
        values = np.asarray(data, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            chunks.append(np.abs(finite))
    if not chunks:
        return -1.0, 1.0
    scale = float(np.percentile(np.concatenate(chunks), 98.5))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return -scale, scale


def _build_summary(
    output_dir: Path,
    source: dict[str, Any],
    raw_data: np.ndarray,
    raw_metadata: dict[str, np.ndarray],
    atomic: dict[str, Any],
    v2: dict[str, Any],
) -> dict[str, Any]:
    raw_spacing_stats = _spacing_stats(raw_metadata)
    atomic_spacing_stats = _spacing_stats(atomic["trace_metadata"])
    atomic_speed_stage = _stage_by_method(atomic.get("stages") or [], "motion_compensation_speed")
    atomic_speed_spacing_stats = _spacing_stats(atomic_speed_stage["trace_metadata"]) if atomic_speed_stage else atomic_spacing_stats
    atomic_speed_meta = atomic_speed_stage.get("meta", {}) if atomic_speed_stage else {}
    atomic_speed_params = atomic_speed_stage.get("params", {}) if atomic_speed_stage else {}
    v2_spacing_stats = _spacing_stats(v2["trace_metadata"])
    raw_shape = tuple(int(v) for v in np.asarray(raw_data).shape)
    atomic_shape = tuple(int(v) for v in np.asarray(atomic["data"]).shape)
    v2_shape = tuple(int(v) for v in np.asarray(v2["data"]).shape)
    source_resampled_atomic = _resample_like_reference(source["ideal_data"], np.asarray(atomic["data"]).shape[1])
    source_resampled_v2 = _resample_like_reference(source["ideal_data"], np.asarray(v2["data"]).shape[1])
    source_resampled_raw = _resample_like_reference(source["ideal_data"], np.asarray(raw_data).shape[1])
    target_roi = _first_target_roi(source.get("ground_truth"))
    raw_target_roi = _scale_single_roi_for_trace_count(target_roi, int(source["traces"]), int(raw_shape[1]))
    atomic_target_roi = _scale_single_roi_for_trace_count(target_roi, int(source["traces"]), int(atomic_shape[1]))
    v2_target_roi = _scale_single_roi_for_trace_count(target_roi, int(source["traces"]), int(v2_shape[1]))
    source_energy = _target_roi_energy(source["ideal_data"], target_roi)
    metrics = {
        "spacing_std_before_m": raw_spacing_stats["std_m"],
        "spacing_std_atomic_m": atomic_spacing_stats["std_m"],
        "spacing_std_atomic_after_speed_m": atomic_speed_spacing_stats["std_m"],
        "spacing_std_v2_m": v2_spacing_stats["std_m"],
        "trace_spacing_cv_before": raw_spacing_stats["cv"],
        "trace_spacing_cv_atomic": atomic_spacing_stats["cv"],
        "trace_spacing_cv_v2": v2_spacing_stats["cv"],
        "max_gap_ratio_before": raw_spacing_stats["max_gap_ratio"],
        "max_gap_ratio_atomic": atomic_spacing_stats["max_gap_ratio"],
        "max_gap_ratio_v2": v2_spacing_stats["max_gap_ratio"],
        "raw_vs_source_rms": _rms_delta(raw_data, source_resampled_raw),
        "atomic_vs_source_rms": _rms_delta(atomic["data"], source_resampled_atomic),
        "v2_vs_source_rms": _rms_delta(v2["data"], source_resampled_v2),
        "atomic_rms_delta_from_raw": _rms_delta(atomic["data"], _resample_like_reference(raw_data, np.asarray(atomic["data"]).shape[1])),
        "v2_rms_delta_from_raw": _rms_delta(v2["data"], _resample_like_reference(raw_data, np.asarray(v2["data"]).shape[1])),
        "target_ratio_raw": _target_energy_ratio(raw_data, raw_target_roi),
        "target_ratio_atomic": _target_energy_ratio(atomic["data"], atomic_target_roi),
        "target_ratio_v2": _target_energy_ratio(v2["data"], v2_target_roi),
        "ridge_rmse_samples_raw": _ridge_rmse_samples(raw_data, source_resampled_raw, raw_target_roi),
        "ridge_rmse_samples_atomic": _ridge_rmse_samples(atomic["data"], source_resampled_atomic, atomic_target_roi),
        "ridge_rmse_samples_v2": _ridge_rmse_samples(v2["data"], source_resampled_v2, v2_target_roi),
        "reflector_flatness_metric_raw": _reflector_flatness_metric(raw_data, raw_target_roi),
        "reflector_flatness_metric_atomic": _reflector_flatness_metric(atomic["data"], atomic_target_roi),
        "reflector_flatness_metric_v2": _reflector_flatness_metric(v2["data"], v2_target_roi),
        "target_apex_error_samples_raw": _target_apex_error_samples(raw_data, source_resampled_raw, raw_target_roi),
        "target_apex_error_samples_atomic": _target_apex_error_samples(atomic["data"], source_resampled_atomic, atomic_target_roi),
        "target_apex_error_samples_v2": _target_apex_error_samples(v2["data"], source_resampled_v2, v2_target_roi),
        "target_roi_energy_preservation_raw": _energy_preservation(_target_roi_energy(raw_data, raw_target_roi), source_energy),
        "target_roi_energy_preservation_atomic": _energy_preservation(_target_roi_energy(atomic["data"], atomic_target_roi), source_energy),
        "target_roi_energy_preservation_v2": _energy_preservation(_target_roi_energy(v2["data"], v2_target_roi), source_energy),
        "resample_spacing_m": _resample_spacing_from_meta(v2.get("meta", {}), source["trace_interval_m"]),
        "target_traces": int(v2_shape[1]),
        "atomic_resample_spacing_m": _resample_spacing_from_meta(
            atomic_speed_meta,
            source["trace_interval_m"],
        ),
        "atomic_target_traces": int(atomic_shape[1]),
    }
    runtime_warnings = {
        "atomic": _collect_runtime_warnings(*(stage.get("meta", {}) for stage in atomic.get("stages", []))),
        "motion_v2": _collect_runtime_warnings(v2.get("meta", {})),
    }
    validation_notes = _build_validation_notes(metrics)
    return {
        "schema": "mygpr_gprmax_motion_validation_summary_v1",
        "output_dir": str(output_dir),
        "shapes": {
            "source": [int(source["samples"]), int(source["traces"])],
            "raw": list(raw_shape),
            "atomic": list(atomic_shape),
            "v2": list(v2_shape),
            "original_gprmax": list(source.get("original_gprmax_shape", [])),
        },
        "source": {
            "scenario_id": source.get("scenario_id"),
            "source_manifest": source.get("source_manifest"),
            "source_primary_out_file": source.get("source_primary_out_file"),
            "source_ground_truth_file": source.get("source_ground_truth_file"),
            "copied_source_artifacts": source.get("copied_source_artifacts", {}),
            "shape": [int(source["samples"]), int(source["traces"])],
            "original_gprmax_shape": list(source.get("original_gprmax_shape", [])),
            "derived_longline": bool(source.get("derived_longline", False)),
            "trace_interval_m": float(source["trace_interval_m"]),
            "total_time_ns": float(source["total_time_ns"]),
            "motion_profile": _jsonable(source.get("motion_profile", {})),
            "target_roi": target_roi,
            "target_geometry": _first_target_geometry(source.get("ground_truth")),
        },
        "pipeline": {
            "atomic": [method for method, _params in ATOMIC_PIPELINE],
            "motion_v2": "motion_compensation_v2",
        },
        "metrics": metrics,
        "resampling_explanation": {
            "atomic_resampled": bool(atomic_shape[1] != raw_shape[1]),
            "atomic_source_traces": int(raw_shape[1]),
            "atomic_target_traces": int(atomic_shape[1]),
            "atomic_resample_spacing_m": metrics["atomic_resample_spacing_m"],
            "atomic_resample_spacing_mode": "manual" if float(atomic_speed_params.get("spacing_m", 0.0) or 0.0) > 0.0 else "auto",
            "source_resampled_for_rms_to_atomic_trace_count": bool(atomic_shape[1] != source["traces"]),
            "motion_v2_resampled": bool(v2_shape[1] != raw_shape[1]),
            "source_traces": int(raw_shape[1]),
            "target_traces": int(v2_shape[1]),
            "resample_spacing_m": metrics["resample_spacing_m"],
            "source_resampled_for_rms_to_v2_trace_count": bool(v2_shape[1] != source["traces"]),
            "comparison_note": "RMS/ROI metrics compare each processed B-scan against the source gprMax B-scan resampled onto the processed trace axis when trace counts differ.",
        },
        "quality_flags": {
            "atomic": _collect_quality_flags(*(stage.get("meta", {}) for stage in atomic.get("stages", []))),
            "motion_v2": list(v2.get("meta", {}).get("quality_flags", []) or []),
        },
        "runtime_warnings": runtime_warnings,
        "validation_notes": validation_notes,
        "stage_meta": {
            "atomic_final": _compact_meta(atomic.get("stages", [])[-1].get("meta", {}) if atomic.get("stages") else {}),
            "motion_v2": _compact_meta(v2.get("meta", {})),
        },
        "artifacts": {
            "main_csv": str(output_dir / "main.csv"),
            "manifest": str(output_dir / "manifest.json"),
            "metadata": str(output_dir / "metadata.json"),
            "report_md": str(output_dir / "motion_validation_report.md"),
            "report_html": str(output_dir / "motion_validation_report.html"),
            "images": {
                name: str(output_dir / name)
                for name in _image_names()
                if (output_dir / name).exists()
            },
        },
    }


def _spacing_stats(metadata: dict[str, np.ndarray]) -> dict[str, float]:
    distance = np.asarray(metadata.get("trace_distance_m", []), dtype=np.float64)
    if distance.size < 3:
        return {"std_m": 0.0, "mean_m": 0.0, "cv": 0.0, "max_gap_ratio": 0.0}
    gaps = np.diff(distance)
    finite = gaps[np.isfinite(gaps) & (gaps > 0.0)]
    if finite.size == 0:
        return {"std_m": 0.0, "mean_m": 0.0, "cv": 0.0, "max_gap_ratio": 0.0}
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    return {
        "std_m": std,
        "mean_m": mean,
        "cv": float(std / mean) if mean > 0.0 else 0.0,
        "max_gap_ratio": float(np.max(finite) / mean) if mean > 0.0 else 0.0,
    }


def _stage_by_method(stages: list[dict[str, Any]], method_id: str) -> dict[str, Any] | None:
    for stage in stages:
        if stage.get("method") == method_id:
            return stage
    return None


def _resample_like_reference(data: np.ndarray, target_traces: int) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.shape[1] == target_traces:
        return arr
    source_axis = np.linspace(0.0, 1.0, arr.shape[1], dtype=np.float64)
    target_axis = np.linspace(0.0, 1.0, target_traces, dtype=np.float64)
    return _interp_columns(arr, source_axis, target_axis)


def _rms_delta(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    samples = min(aa.shape[0], bb.shape[0])
    traces = min(aa.shape[1], bb.shape[1])
    if samples <= 0 or traces <= 0:
        return 0.0
    diff = aa[:samples, :traces] - bb[:samples, :traces]
    return float(np.sqrt(np.mean(diff * diff)))


def _first_target_roi(ground_truth: dict[str, Any] | None) -> dict[str, int] | None:
    if not isinstance(ground_truth, dict):
        return None
    targets = ground_truth.get("targets") or []
    if not targets:
        return None
    roi = targets[0].get("roi") if isinstance(targets[0], dict) else None
    return roi if isinstance(roi, dict) else None


def _target_energy_ratio(data: np.ndarray, roi: dict[str, int] | None) -> float | None:
    if not roi:
        return None
    arr = np.asarray(data, dtype=np.float64)
    t0 = max(0, min(arr.shape[0], int(roi.get("time_start_idx", 0))))
    t1 = max(t0, min(arr.shape[0], int(roi.get("time_end_idx", arr.shape[0]))))
    d0 = max(0, min(arr.shape[1], int(roi.get("dist_start_idx", 0))))
    d1 = max(d0, min(arr.shape[1], int(roi.get("dist_end_idx", arr.shape[1]))))
    if t1 <= t0 or d1 <= d0:
        return None
    target = arr[t0:t1, d0:d1]
    total = arr[np.isfinite(arr)]
    if target.size == 0 or total.size == 0:
        return None
    return float(np.mean(target * target) / max(float(np.mean(total * total)), 1.0e-12))


def _target_roi_energy(data: np.ndarray, roi: dict[str, int] | None) -> float | None:
    window = _roi_window(data, roi)
    if window is None or window.size == 0:
        return None
    return float(np.mean(window * window))


def _energy_preservation(candidate_energy: float | None, source_energy: float | None) -> float | None:
    if candidate_energy is None or source_energy is None or source_energy <= 0.0:
        return None
    return float(candidate_energy / source_energy)


def _roi_window(data: np.ndarray, roi: dict[str, int] | None) -> np.ndarray | None:
    if not roi:
        return None
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        return None
    t0 = max(0, min(arr.shape[0], int(roi.get("time_start_idx", 0))))
    t1 = max(t0, min(arr.shape[0], int(roi.get("time_end_idx", arr.shape[0]))))
    d0 = max(0, min(arr.shape[1], int(roi.get("dist_start_idx", 0))))
    d1 = max(d0, min(arr.shape[1], int(roi.get("dist_end_idx", arr.shape[1]))))
    if t1 <= t0 or d1 <= d0:
        return None
    return arr[t0:t1, d0:d1]


def _ridge_rows(data: np.ndarray, roi: dict[str, int] | None) -> np.ndarray | None:
    window = _roi_window(data, roi)
    if window is None or window.size == 0:
        return None
    t0 = int(roi.get("time_start_idx", 0))
    return t0 + np.argmax(np.abs(window), axis=0).astype(np.float64)


def _ridge_rmse_samples(candidate: np.ndarray, source: np.ndarray, roi: dict[str, int] | None) -> float | None:
    a = _ridge_rows(candidate, roi)
    b = _ridge_rows(source, roi)
    if a is None or b is None or a.size == 0 or b.size == 0:
        return None
    count = min(a.size, b.size)
    return float(np.sqrt(np.mean((a[:count] - b[:count]) ** 2)))


def _reflector_flatness_metric(data: np.ndarray, roi: dict[str, int] | None) -> float | None:
    rows = _ridge_rows(data, roi)
    if rows is None or rows.size == 0:
        return None
    return float(np.std(rows))


def _target_apex_error_samples(candidate: np.ndarray, source: np.ndarray, roi: dict[str, int] | None) -> float | None:
    cand = _roi_window(candidate, roi)
    ref = _roi_window(source, roi)
    if cand is None or ref is None or cand.size == 0 or ref.size == 0:
        return None
    cand_row = int(np.unravel_index(np.argmax(np.abs(cand)), cand.shape)[0])
    ref_row = int(np.unravel_index(np.argmax(np.abs(ref)), ref.shape)[0])
    return float(abs(cand_row - ref_row))


def _resample_spacing_from_meta(meta: dict[str, Any], fallback: float) -> float:
    for key in ("spacing_m", "resample_spacing_m"):
        value = meta.get(key)
        if value is not None:
            candidate = float(value)
            if np.isfinite(candidate) and candidate > 0.0:
                return candidate
    return float(fallback)


def _collect_runtime_warnings(*metas: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for meta in metas:
        for key in ("runtime_warnings", "warnings"):
            value = meta.get(key)
            if isinstance(value, str):
                warnings.append(value)
            elif isinstance(value, (list, tuple)):
                warnings.extend(str(item) for item in value if item)
    return warnings


def _collect_quality_flags(*metas: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for meta in metas:
        value = meta.get("quality_flags")
        if isinstance(value, str):
            flags.append(value)
        elif isinstance(value, (list, tuple)):
            flags.extend(str(item) for item in value if item)
    return flags


def _build_validation_notes(metrics: dict[str, Any]) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    raw_apex = metrics.get("target_apex_error_samples_raw")
    atomic_apex = metrics.get("target_apex_error_samples_atomic")
    if raw_apex is not None and atomic_apex is not None and float(atomic_apex) > float(raw_apex):
        notes.append(
            {
                "code": "atomic_target_apex_error_worse_than_raw",
                "severity": "warning",
                "message": (
                    "The reordered atomic chain fixes the final trace spacing, but the target apex sample error "
                    "is still worse than raw in this run. Treat the four atomic steps as an ablation/debug view; "
                    "motion_compensation_v2 remains the preferred paper baseline when target localization matters."
                ),
            }
        )
    raw_cv = metrics.get("trace_spacing_cv_before")
    atomic_cv = metrics.get("trace_spacing_cv_atomic")
    if raw_cv is not None and atomic_cv is not None and float(atomic_cv) < float(raw_cv):
        notes.append(
            {
                "code": "atomic_spacing_improved_after_reorder",
                "severity": "info",
                "message": "The final atomic trace spacing CV is lower than raw after running attitude/APC before speed compensation.",
            }
        )
    return notes


def _first_target_geometry(ground_truth: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(ground_truth, dict):
        return None
    targets = ground_truth.get("targets") or []
    if not targets or not isinstance(targets[0], dict):
        return None
    target = targets[0]
    keys = ("id", "type", "material", "depth_m", "center_x_m", "center_y_m", "radius_m", "roi")
    return {key: _jsonable(target.get(key)) for key in keys if key in target}


def _image_names() -> list[str]:
    return [
        "source_gprmax_bscan.png",
        "motion_injected_raw_bscan.png",
        "atomic_motion_final_bscan.png",
        "motion_v2_final_bscan.png",
        "bscan_motion_validation_comparison.png",
        "paper_motion_validation_comparison.png",
        "raw_3d_preview.png",
        "atomic_3d_preview.png",
        "motion_v2_3d_preview.png",
    ]


def _artifact_images(output_dir: Path) -> dict[str, str]:
    return {name: str(output_dir / name) for name in _image_names() if (output_dir / name).exists()}


def _compact_meta(meta: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "skipped",
        "reason",
        "height_source_used",
        "reference_height_m",
        "max_shift_samples_applied",
        "max_shift_samples_effective",
        "shift_clamped",
        "source_traces",
        "target_traces",
        "spacing_m",
        "quality_flags",
        "warnings",
    }
    return {key: _jsonable(value) for key, value in meta.items() if key in keep}


def _render_report(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    shapes = summary.get("shapes", {})
    resampling = summary.get("resampling_explanation", {})
    lines = [
        "# gprMax Motion Compensation Validation",
        "",
        "## Source",
        "",
        f"- Scenario: `{summary['source'].get('scenario_id')}`",
        f"- gprMax manifest: `{summary['source'].get('source_manifest')}`",
        f"- Primary .out: `{summary['source'].get('source_primary_out_file')}`",
        f"- Source ground truth: `{summary['source'].get('source_ground_truth_file')}`",
        f"- Shape: `{summary['source'].get('shape')}`",
        f"- Original gprMax shape: `{summary['source'].get('original_gprmax_shape')}`",
        f"- Derived long-line scaffold: `{summary['source'].get('derived_longline')}`",
        f"- Trace interval: `{summary['source'].get('trace_interval_m')}` m",
        f"- Time window: `{summary['source'].get('total_time_ns')}` ns",
        "- This run uses an exaggerated demo/stress UAV motion profile for visibility and is not a field-flight baseline.",
        "",
        "## Shapes",
        "",
        "| data | shape |",
        "| --- | --- |",
        f"| gprMax source | `{shapes.get('source')}` |",
        f"| motion-injected raw | `{shapes.get('raw')}` |",
        f"| four atomic steps | `{shapes.get('atomic')}` |",
        f"| motion_compensation_v2 | `{shapes.get('v2')}` |",
        "",
        "## Target / ROI",
        "",
        f"- Target geometry: `{summary['source'].get('target_geometry')}`",
        f"- Target ROI: `{summary['source'].get('target_roi')}`",
        "",
        "## Workflow",
        "",
        "- Atomic: `" + " -> ".join(summary["pipeline"]["atomic"]) + "`",
        "- Unified: `motion_compensation_v2`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        if value is None:
            display = "--"
        elif isinstance(value, (int, float)):
            display = f"{float(value):.6g}"
        else:
            display = str(value)
        lines.append(f"| `{key}` | {display} |")
    lines.extend(
        [
            "",
            "## Atomic Resampling Explanation",
            "",
            f"- Atomic performed equal-distance resampling: `{resampling.get('atomic_resampled')}`",
            f"- Atomic source_traces: `{resampling.get('atomic_source_traces')}`",
            f"- Atomic target_traces: `{resampling.get('atomic_target_traces')}`",
            f"- Atomic resample_spacing_m: `{resampling.get('atomic_resample_spacing_m')}`",
            f"- Atomic resample_spacing_mode: `{resampling.get('atomic_resample_spacing_mode')}`",
            "- Atomic RMS and ROI metrics compare against the gprMax source B-scan resampled to the atomic processed trace axis when trace counts differ.",
            "- Therefore an atomic shape mismatch is expected after equal-distance resampling; it is not treated as a processing error.",
            "",
            "## V2 Resampling Explanation",
            "",
            f"- V2 performed equal-distance resampling: `{resampling.get('motion_v2_resampled')}`",
            f"- Source/raw trace count: `{resampling.get('source_traces')}`",
            f"- V2 target_traces: `{resampling.get('target_traces')}`",
            f"- V2 resample_spacing_m: `{resampling.get('resample_spacing_m')}`",
            "- RMS and ROI metrics are computed against the gprMax source B-scan resampled to the processed trace axis when trace counts differ.",
            "- Therefore a V2 shape mismatch is expected when equal-distance resampling changes the trace count; it is not treated as a processing error.",
            "",
            "## Quality Flags / Runtime Warnings",
            "",
            f"- Atomic quality_flags: `{summary.get('quality_flags', {}).get('atomic', [])}`",
            f"- V2 quality_flags: `{summary.get('quality_flags', {}).get('motion_v2', [])}`",
            f"- Atomic runtime_warnings: `{summary.get('runtime_warnings', {}).get('atomic', [])}`",
            f"- V2 runtime_warnings: `{summary.get('runtime_warnings', {}).get('motion_v2', [])}`",
            "",
            "## Validation Notes",
            "",
        ]
    )
    notes = summary.get("validation_notes") or []
    if notes:
        for note in notes:
            lines.append(
                f"- `{note.get('severity', 'info')}` `{note.get('code', 'note')}`: {note.get('message', '')}"
            )
    else:
        lines.append("- No additional validation notes.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Main CSV: `{summary['artifacts']['main_csv']}`",
            f"- Comparison image: `{summary['artifacts']['images'].get('bscan_motion_validation_comparison.png', '--')}`",
            f"- Paper comparison image: `{summary['artifacts']['images'].get('paper_motion_validation_comparison.png', '--')}`",
            f"- Raw 3D preview: `{summary['artifacts']['images'].get('raw_3d_preview.png', '--')}`",
            f"- Motion V2 3D preview: `{summary['artifacts']['images'].get('motion_v2_3d_preview.png', '--')}`",
            f"- Copied source artifacts: `{summary['source'].get('copied_source_artifacts', {})}`",
            "",
            "## Current Limitation",
            "",
            "- 这里的地下波场来自 gprMax，但 UAV 运动扰动是 MyGPR 侧可控注入；它用于验证运动补偿链路，不代表实测外业结论。",
            "- 短测线 gprMax 数据只能做 smoke；论文展示建议后续使用更长测线和更完整双曲线。",
        ]
    )
    return "\n".join(lines) + "\n"


def _html(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _fmt_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}g}"
    return str(value)


def _rel_img(summary: dict[str, Any], name: str) -> str | None:
    value = ((summary.get("artifacts") or {}).get("images") or {}).get(name)
    if not value:
        return None
    return Path(str(value)).name


def _metric_delta_badge(before: Any, after: Any, *, lower_is_better: bool = True) -> str:
    try:
        before_value = float(before)
        after_value = float(after)
    except (TypeError, ValueError):
        return '<span class="badge neutral">n/a</span>'
    improved = after_value < before_value if lower_is_better else after_value > before_value
    klass = "good" if improved else "warn"
    label = "改善" if improved else "未改善"
    return f'<span class="badge {klass}">{label}</span>'


def _render_html_report(summary: dict[str, Any]) -> str:
    """Render an evidence-first HTML report for group discussion."""
    source = summary.get("source") or {}
    metrics = summary.get("metrics") or {}
    shapes = summary.get("shapes") or {}
    resampling = summary.get("resampling_explanation") or {}
    notes = summary.get("validation_notes") or []
    comparison_img = _rel_img(summary, "paper_motion_validation_comparison.png")
    bscan_img = _rel_img(summary, "bscan_motion_validation_comparison.png")
    raw_3d = _rel_img(summary, "raw_3d_preview.png")
    v2_3d = _rel_img(summary, "motion_v2_3d_preview.png")
    atomic_route = " -> ".join((summary.get("pipeline") or {}).get("atomic") or [])
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metric_rows = [
        ("Raw vs Source RMS", metrics.get("raw_vs_source_rms"), "运动扰动注入后与原始 gprMax 的差异"),
        ("Atomic vs Source RMS", metrics.get("atomic_vs_source_rms"), "四原子补偿后残差"),
        ("Motion V2 vs Source RMS", metrics.get("v2_vs_source_rms"), "统一 V2 补偿后残差"),
        ("Trace Spacing CV Before", metrics.get("trace_spacing_cv_before"), "补偿前道距不均匀程度"),
        ("Trace Spacing CV Atomic", metrics.get("trace_spacing_cv_atomic"), "四原子补偿后道距不均匀程度"),
        ("Trace Spacing CV V2", metrics.get("trace_spacing_cv_v2"), "V2 补偿后道距不均匀程度"),
        ("Target Apex Error Raw", metrics.get("target_apex_error_samples_raw"), "补偿前目标顶点偏差 / sample"),
        ("Target Apex Error Atomic", metrics.get("target_apex_error_samples_atomic"), "四原子补偿后目标顶点偏差 / sample"),
        ("Target Apex Error V2", metrics.get("target_apex_error_samples_v2"), "V2 补偿后目标顶点偏差 / sample"),
        ("Target ROI Preservation Atomic", metrics.get("target_roi_energy_preservation_atomic"), "四原子目标 ROI 能量保持"),
        ("Target ROI Preservation V2", metrics.get("target_roi_energy_preservation_v2"), "V2 目标 ROI 能量保持"),
    ]
    metric_table = "\n".join(
        "<tr>"
        f"<td>{_html(name)}</td>"
        f"<td>{_html(_fmt_metric(value, 6))}</td>"
        f"<td>{_html(desc)}</td>"
        "</tr>"
        for name, value, desc in metric_rows
    )
    note_items = "\n".join(
        f"<li><strong>{_html(note.get('severity', 'info'))}</strong> "
        f"{_html(note.get('code', 'note'))}: {_html(note.get('message', ''))}</li>"
        for note in notes
    ) or "<li>无额外告警。</li>"

    image_sections: list[str] = []
    if comparison_img:
        image_sections.append(
            f"""
            <section class="band">
              <div class="section-head">
                <h2>四联 B-scan 证据图</h2>
                <p>统一灰度范围，对比 gprMax source、motion-injected raw、四原子补偿、Motion V2。</p>
              </div>
              <figure class="figure-wide">
                <img src="{_html(comparison_img)}" alt="paper motion validation comparison">
              </figure>
            </section>
            """
        )
    if bscan_img:
        image_sections.append(
            f"""
            <section class="band">
              <div class="section-head">
                <h2>B-scan 处理链路对比</h2>
                <p>用于检查运动注入和补偿后的波形结构是否仍可解释。</p>
              </div>
              <figure class="figure-wide">
                <img src="{_html(bscan_img)}" alt="motion validation bscan comparison">
              </figure>
            </section>
            """
        )
    preview_cards = ""
    for title, image_name in (("Raw 3D 预览", raw_3d), ("Motion V2 3D 预览", v2_3d)):
        if image_name:
            preview_cards += (
                f'<article class="preview"><h3>{_html(title)}</h3>'
                f'<img src="{_html(image_name)}" alt="{_html(title)}"></article>'
            )
    if preview_cards:
        image_sections.append(
            f"""
            <section class="band">
              <div class="section-head">
                <h2>三维轨迹与剖面预览</h2>
                <p>检查补偿前后航迹、剖面带和高度变化是否进入可视化链路。</p>
              </div>
              <div class="preview-grid">{preview_cards}</div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MyGPR gprMax 运动补偿验证报告</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5f6b7a;
      --line: #d8dee8;
      --panel: #ffffff;
      --paper: #f4f6f9;
      --accent: #0e766e;
      --warn: #a16207;
      --warn-soft: #fff3c4;
      --good: #166534;
      --good-soft: #dcfce7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      background: var(--paper);
      color: var(--ink);
      line-height: 1.55;
    }}
    header {{
      padding: 34px 44px 26px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: .04em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 8px 0 12px;
      font-size: 30px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    h2 {{ margin: 0 0 6px; font-size: 21px; }}
    h3 {{ margin: 0 0 10px; font-size: 16px; }}
    p {{ margin: 0; color: var(--muted); }}
    main {{ padding: 24px 44px 44px; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric-card .label {{ color: var(--muted); font-size: 13px; }}
    .metric-card .value {{ margin-top: 6px; font-size: 22px; font-weight: 700; }}
    .band {{
      margin-top: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: end;
      margin-bottom: 14px;
    }}
    .section-head p {{ max-width: 760px; }}
    .facts {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .fact {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfe;
    }}
    .fact span {{ display: block; color: var(--muted); font-size: 12px; }}
    .fact strong {{ display: block; margin-top: 4px; overflow-wrap: anywhere; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 700; background: #f8fafc; }}
    .figure-wide {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }}
    .figure-wide img, .preview img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .preview-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .preview {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 700;
    }}
    .badge.good {{ color: var(--good); background: var(--good-soft); }}
    .badge.warn {{ color: var(--warn); background: var(--warn-soft); }}
    .badge.neutral {{ color: var(--muted); background: #edf1f7; }}
    code {{
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 12px;
      background: #eef2f7;
      padding: 2px 5px;
      border-radius: 5px;
    }}
    ul {{ margin: 0; padding-left: 20px; }}
    .footer-note {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 980px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .summary-grid, .facts, .preview-grid {{ grid-template-columns: 1fr; }}
      .section-head {{ display: block; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">MyGPR / gprMax Motion Validation</div>
    <h1>UAV-GPR 运动补偿验证闭环报告</h1>
    <p>本报告使用 gprMax 地下波场作为 source，在 MyGPR 侧注入可控 UAV 运动扰动，并比较四原子补偿链与统一 Motion V2 的恢复效果。生成时间：{_html(generated_at)}</p>
    <p class="footer-note">This run uses an exaggerated demo/stress UAV motion profile for visibility and is not a field-flight baseline.</p>
    <div class="summary-grid">
      <div class="metric-card"><div class="label">Raw vs Source RMS</div><div class="value">{_html(_fmt_metric(metrics.get("raw_vs_source_rms"), 4))}</div></div>
      <div class="metric-card"><div class="label">Atomic vs Source RMS</div><div class="value">{_html(_fmt_metric(metrics.get("atomic_vs_source_rms"), 4))}</div></div>
      <div class="metric-card"><div class="label">Motion V2 vs Source RMS</div><div class="value">{_html(_fmt_metric(metrics.get("v2_vs_source_rms"), 4))}</div></div>
      <div class="metric-card"><div class="label">V2 道距 CV</div><div class="value">{_html(_fmt_metric(metrics.get("trace_spacing_cv_v2"), 4))}</div></div>
    </div>
  </header>
  <main>
    <section class="band">
      <div class="section-head">
        <h2>实验对象</h2>
        <p>验证目标是软件链路和运动补偿恢复能力，不把该合成运动扰动视为真实外业地质结论。</p>
      </div>
      <div class="facts">
        <div class="fact"><span>Scenario</span><strong>{_html(source.get("scenario_id"))}</strong></div>
        <div class="fact"><span>gprMax source shape</span><strong>{_html(shapes.get("source"))}</strong></div>
        <div class="fact"><span>Motion V2 shape</span><strong>{_html(shapes.get("v2"))}</strong></div>
        <div class="fact"><span>Trace interval</span><strong>{_html(source.get("trace_interval_m"))} m</strong></div>
        <div class="fact"><span>Time window</span><strong>{_html(source.get("total_time_ns"))} ns</strong></div>
        <div class="fact"><span>Derived long-line</span><strong>{_html(source.get("derived_longline"))}</strong></div>
        <div class="fact"><span>Height p-p</span><strong>{_html(_fmt_metric((source.get("motion_profile") or {}).get("height_agl_peak_to_peak_m"), 4))} m</strong></div>
        <div class="fact"><span>Lateral p-p</span><strong>{_html(_fmt_metric((source.get("motion_profile") or {}).get("local_y_peak_to_peak_m"), 4))} m</strong></div>
        <div class="fact"><span>Max height shift</span><strong>{_html(_fmt_metric((source.get("motion_profile") or {}).get("max_abs_height_shift_samples"), 4))} samples</strong></div>
      </div>
    </section>
    <section class="band">
      <div class="section-head">
        <h2>处理链路</h2>
        <p>四原子链先更新姿态/APC 足迹，再执行等距道距重采样，最后做高度时移和振幅归一，避免后续姿态更新破坏 speed compensation 的 trace axis。</p>
      </div>
      <table>
        <thead><tr><th>Route</th><th>Steps</th><th>说明</th></tr></thead>
        <tbody>
          <tr><td>Atomic</td><td><code>{_html(atomic_route)}</code></td><td>用于 ablation、教学展示和分项验证。</td></tr>
          <tr><td>Unified</td><td><code>motion_compensation_v2</code></td><td>推荐主线入口，统一输出 warnings、quality_flags 和 trace metadata。</td></tr>
        </tbody>
      </table>
    </section>
    <section class="band">
      <div class="section-head">
        <h2>关键指标</h2>
        <p>{_metric_delta_badge(metrics.get("raw_vs_source_rms"), metrics.get("v2_vs_source_rms"))} RMS 越低越接近未注入运动扰动的 gprMax source；道距 CV 越低表示航迹重采样越稳定。</p>
      </div>
      <table>
        <thead><tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr></thead>
        <tbody>{metric_table}</tbody>
      </table>
    </section>
    {"".join(image_sections)}
    <section class="band">
      <div class="section-head">
        <h2>Atomic 重采样解释</h2>
        <p>四原子链中的 speed compensation 也会执行等距道距重采样；atomic trace 数变化是预期结果，不是 shape 错误。</p>
      </div>
      <div class="facts">
        <div class="fact"><span>Atomic resampled</span><strong>{_html(resampling.get("atomic_resampled"))}</strong></div>
        <div class="fact"><span>Atomic source traces</span><strong>{_html(resampling.get("atomic_source_traces"))}</strong></div>
        <div class="fact"><span>Atomic target traces</span><strong>{_html(resampling.get("atomic_target_traces"))}</strong></div>
        <div class="fact"><span>Atomic spacing</span><strong>{_html(resampling.get("atomic_resample_spacing_m"))} m</strong></div>
        <div class="fact"><span>Atomic spacing mode</span><strong>{_html(resampling.get("atomic_resample_spacing_mode"))}</strong></div>
        <div class="fact"><span>RMS/ROI</span><strong>source resampled to atomic axis</strong></div>
      </div>
    </section>
    <section class="band">
      <div class="section-head">
        <h2>V2 重采样解释</h2>
        <p>Motion V2 可能改变 trace 数，这是等距道距重采样的预期结果，不应被误判为 shape 错误。</p>
      </div>
      <div class="facts">
        <div class="fact"><span>Resampled</span><strong>{_html(resampling.get("motion_v2_resampled"))}</strong></div>
        <div class="fact"><span>Source traces</span><strong>{_html(resampling.get("source_traces"))}</strong></div>
        <div class="fact"><span>Target traces</span><strong>{_html(resampling.get("target_traces"))}</strong></div>
        <div class="fact"><span>Spacing</span><strong>{_html(resampling.get("resample_spacing_m"))} m</strong></div>
        <div class="fact"><span>Target ROI</span><strong>{_html(source.get("target_roi"))}</strong></div>
        <div class="fact"><span>Target geometry</span><strong>{_html(source.get("target_geometry"))}</strong></div>
      </div>
    </section>
    <section class="band">
      <div class="section-head">
        <h2>质量告警与限制</h2>
        <p>这些信息用于判断当前证据能否进入论文/组会结论。</p>
      </div>
      <ul>{note_items}</ul>
      <p class="footer-note">限制：地下波场来自 gprMax，但 UAV 运动扰动是在 MyGPR 侧注入；该报告证明的是补偿链路可复现和指标可解释，不直接代表真实外业泛化能力。</p>
    </section>
  </main>
</body>
</html>
"""


def _write_readme(output_dir: Path, source: dict[str, Any]) -> None:
    text = f"""# gprMax Motion Validation Package

This folder contains a MyGPR motion-compensation validation package derived
from gprMax scenario `{source.get('scenario_id')}`.

The underground B-scan comes from gprMax. UAV trajectory, attitude and height
sidecars are generated by MyGPR to inject controlled motion artifacts.

This run uses an exaggerated demo/stress UAV motion profile for visibility and
is not a field-flight baseline.

Open `main.csv` in MyGPR, then run either:

- `trajectory_smoothing -> motion_compensation_attitude -> motion_compensation_speed -> motion_compensation_height`
- `motion_compensation_v2`

This is validation data, not field evidence.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a MyGPR motion validation package from a gprMax dataset."
    )
    parser.add_argument("--dataset", required=True, help="gprMax manifest JSON or dataset directory")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument(
        "--target-traces",
        type=int,
        default=None,
        help="Optionally derive a longer validation line by resampling the source B-scan to this trace count.",
    )
    args = parser.parse_args()

    result = generate_gprmax_motion_validation_package(
        args.dataset,
        args.output_dir,
        seed=args.seed,
        target_traces=args.target_traces,
    )
    print(f"Output directory: {result.output_dir}")
    print(f"Main CSV: {result.main_csv}")
    print(f"Summary: {result.summary_json}")
    print(f"Report: {result.report_md}")
    print(f"Raw/atomic/v2 shapes: {result.raw_shape} / {result.atomic_shape} / {result.v2_shape}")
    print(
        "Spacing std before/atomic-speed/atomic-final/v2: "
        f"{result.spacing_std_before_m:.6g} / "
        f"{result.spacing_std_atomic_after_speed_m:.6g} / "
        f"{result.spacing_std_atomic_m:.6g} / "
        f"{result.spacing_std_v2_m:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
