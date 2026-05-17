#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a realistic synthetic UAV-GPR motion compensation demo package."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PACKAGE_DIR = ROOT / "sample_data" / "uav_gpr_motion_demo_v1"
DEFAULT_CONFIG_PATH = ROOT / "config" / "uav_gpr_motion_demo_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "uav_gpr_motion_demo_v1"

SAMPLES = 160
TRACES = 180
TOTAL_TIME_NS = 95.0
START_TIMESTAMP_S = 2000.0
TRACE_PERIOD_S = 0.08
AIR_WAVE_SPEED_M_PER_NS = 0.299792458
ORIGIN_LONGITUDE = 104.123456
ORIGIN_LATITUDE = 30.654321


@dataclass(frozen=True)
class DemoPackageResult:
    """Generated demo package paths."""

    package_dir: Path
    config_path: Path
    package_config_path: Path
    output_dir: Path


def _repo_or_abs(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _lon_lat_from_xy(x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(ORIGIN_LATITUDE))
    longitude = ORIGIN_LONGITUDE + np.asarray(x_m, dtype=np.float64) / meters_per_deg_lon
    latitude = ORIGIN_LATITUDE + np.asarray(y_m, dtype=np.float64) / meters_per_deg_lat
    return longitude, latitude


def _add_ricker(data: np.ndarray, row: np.ndarray, amplitude: float, width: float) -> None:
    samples = np.arange(data.shape[0], dtype=np.float64)[:, None]
    row = np.asarray(row, dtype=np.float64)[None, :]
    tau = (samples - row) / float(width)
    wavelet = (1.0 - 2.0 * tau**2) * np.exp(-tau**2)
    data += float(amplitude) * wavelet


def _shift_trace(trace: np.ndarray, shift_samples: float) -> np.ndarray:
    sample_index = np.arange(trace.size, dtype=np.float64)
    return np.interp(
        sample_index - float(shift_samples),
        sample_index,
        np.asarray(trace, dtype=np.float64),
        left=0.0,
        right=0.0,
    ).astype(np.float32)


def _build_demo_payload() -> dict[str, Any]:
    rng = np.random.default_rng(20260518)
    trace = np.arange(TRACES, dtype=np.float64)
    phase = trace / max(TRACES - 1, 1)
    spacing = 0.34 + 0.035 * np.sin(2.0 * np.pi * 2.2 * phase + 0.2)
    spacing += 0.015 * rng.normal(size=TRACES)
    spacing = np.clip(spacing, 0.26, 0.43)
    local_x = np.cumsum(spacing)
    local_x -= local_x[0]
    local_y = 0.65 * np.sin(2.0 * np.pi * 0.85 * phase) + 0.08 * rng.normal(size=TRACES)

    ground_elevation = 116.0 + 0.22 * np.sin(2.0 * np.pi * 0.42 * phase)
    height_agl = 0.12 + 0.034 * np.sin(2.0 * np.pi * 1.55 * phase + 0.55)
    height_agl += 0.006 * rng.normal(size=TRACES)
    height_agl = np.clip(height_agl, 0.08, 0.16)
    flight_height = height_agl + 0.012 * np.sin(2.0 * np.pi * 3.4 * phase)
    local_z = ground_elevation + flight_height

    longitude, latitude = _lon_lat_from_xy(local_x, local_y)
    timestamps = START_TIMESTAMP_S + TRACE_PERIOD_S * trace
    roll = 2.3 * np.sin(2.0 * np.pi * 2.1 * phase + 0.4)
    pitch = 1.9 * np.cos(2.0 * np.pi * 1.6 * phase - 0.2)
    yaw = 4.0 + 2.8 * np.sin(2.0 * np.pi * 0.5 * phase)

    samples = np.arange(SAMPLES, dtype=np.float64)[:, None]
    along = phase[None, :]
    data = 0.026 * rng.normal(size=(SAMPLES, TRACES))
    data += 0.035 * np.sin(2.0 * np.pi * (samples / 8.5 + 0.09 * along))
    data += 0.020 * np.cos(2.0 * np.pi * 3.0 * along)
    data += 0.018 * rng.normal(size=(1, TRACES))

    layer_1 = 25.0 + 2.0 * np.sin(2.0 * np.pi * 0.7 * phase)
    layer_2 = 58.0 + 3.7 * np.sin(2.0 * np.pi * 0.52 * phase + 0.6)
    layer_3 = 103.0 + 5.0 * np.sin(2.0 * np.pi * 0.35 * phase - 0.4)
    _add_ricker(data, layer_1, 0.55, 2.1)
    _add_ricker(data, layer_2, -0.45, 2.8)
    _add_ricker(data, layer_3, 0.30, 3.4)

    center_trace = 0.55 * (TRACES - 1)
    pipe_apex = 67.0
    pipe_width = 18.0
    hyperbola = pipe_apex + 0.078 * (trace - center_trace) ** 2 / pipe_width
    _add_ricker(data, hyperbola, 1.10, 2.2)
    _add_ricker(data, hyperbola + 5.0, -0.62, 2.5)
    _add_ricker(data, hyperbola + 10.0, 0.34, 3.0)

    reference_height = float(np.mean(height_agl))
    dt_ns = TOTAL_TIME_NS / max(SAMPLES - 1, 1)
    time_shift_samples = 2.0 * (height_agl - reference_height) / AIR_WAVE_SPEED_M_PER_NS / dt_ns
    observed = np.empty_like(data, dtype=np.float32)
    for col in range(TRACES):
        observed[:, col] = _shift_trace(data[:, col], time_shift_samples[col])
        observed[:, col] *= np.float32(np.clip((reference_height / height_agl[col]) ** 2, 0.55, 1.85))

    return {
        "data": observed.astype(np.float32),
        "local_x_m": local_x,
        "local_y_m": local_y,
        "local_z_m": local_z,
        "longitude": longitude,
        "latitude": latitude,
        "ground_elevation_m": ground_elevation,
        "flight_height_m": flight_height,
        "height_agl_m": height_agl,
        "trace_timestamp_s": timestamps,
        "roll_deg": roll,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "reference_height_m": reference_height,
        "time_shift_samples": time_shift_samples,
        "expected_target": {
            "type": "pipe_like_hyperbola",
            "center_trace": float(center_trace),
            "apex_sample": float(pipe_apex),
            "sample_window": [58, 91],
            "trace_window": [70, 130],
        },
    }


def _write_main_csv(path: Path, payload: dict[str, Any]) -> None:
    data = np.asarray(payload["data"], dtype=np.float32)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"Number of Samples = {SAMPLES}\n")
        handle.write(f"Time windows (ns) = {TOTAL_TIME_NS:.6f}\n")
        handle.write(f"Number of Traces = {TRACES}\n")
        handle.write(f"Trace interval (m) = {float(np.mean(np.diff(payload['local_x_m']))):.6f}\n")
        writer = csv.writer(handle)
        for trace_idx in range(TRACES):
            for sample_idx in range(SAMPLES):
                writer.writerow(
                    [
                        f"{payload['longitude'][trace_idx]:.10f}",
                        f"{payload['latitude'][trace_idx]:.10f}",
                        f"{payload['ground_elevation_m'][trace_idx]:.6f}",
                        f"{data[sample_idx, trace_idx]:.8f}",
                        f"{payload['flight_height_m'][trace_idx]:.6f}",
                        f"{payload['trace_timestamp_s'][trace_idx]:.6f}",
                    ]
                )


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_sidecars(package_dir: Path, payload: dict[str, Any]) -> None:
    timestamps = np.asarray(payload["trace_timestamp_s"], dtype=np.float64)
    rtk_rows = []
    imu_rows = []
    altimeter_rows = []
    timestamp_rows = []
    for idx in range(TRACES):
        timestamp_rows.append(
            {
                "trace_index": int(idx),
                "timestamp_s": f"{timestamps[idx]:.6f}",
            }
        )
        rtk_rows.append(
            {
                "timestamp_s": f"{timestamps[idx]:.6f}",
                "longitude": f"{payload['longitude'][idx]:.10f}",
                "latitude": f"{payload['latitude'][idx]:.10f}",
                "ground_elevation_m": f"{payload['ground_elevation_m'][idx]:.6f}",
                "flight_height_m": f"{payload['flight_height_m'][idx]:.6f}",
                "local_x_m": f"{payload['local_x_m'][idx]:.6f}",
                "local_y_m": f"{payload['local_y_m'][idx]:.6f}",
                "local_z_m": f"{payload['local_z_m'][idx]:.6f}",
                "rtk_fix_type": 5,
                "satellites": 18 + int(idx % 5),
                "hdop": f"{0.55 + 0.04 * (idx % 4):.3f}",
            }
        )
        imu_rows.append(
            {
                "timestamp_s": f"{timestamps[idx]:.6f}",
                "roll_deg": f"{payload['roll_deg'][idx]:.6f}",
                "pitch_deg": f"{payload['pitch_deg'][idx]:.6f}",
                "yaw_deg": f"{payload['yaw_deg'][idx]:.6f}",
                "angular_rate_x": f"{0.12 * np.cos(idx * 0.21):.6f}",
                "angular_rate_y": f"{0.09 * np.sin(idx * 0.19):.6f}",
                "angular_rate_z": f"{0.05 * np.cos(idx * 0.11):.6f}",
            }
        )
        confidence = 0.86 + 0.10 * np.sin(idx / 17.0)
        altimeter_rows.append(
            {
                "timestamp_s": f"{timestamps[idx]:.6f}",
                "height_agl_m": f"{payload['height_agl_m'][idx]:.6f}",
                "height_source": "nar15_synthetic_demo",
                "snr": f"{18.0 + 4.0 * confidence:.3f}",
                "target_count": 1,
                "valid": 1,
                "height_confidence": f"{np.clip(confidence, 0.70, 0.98):.3f}",
            }
        )

    _write_dict_rows(package_dir / "trace_timestamps.csv", timestamp_rows)
    _write_dict_rows(package_dir / "rtk.csv", rtk_rows)
    _write_dict_rows(package_dir / "imu.csv", imu_rows)
    _write_dict_rows(package_dir / "altimeter.csv", altimeter_rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_config(package_dir: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "version": "1.0",
        "description": "Synthetic UAV-GPR motion compensation demo package.",
        "jobs": [
            {
                "id": "uav_gpr_motion_demo_v1",
                "input": _repo_or_abs(package_dir / "main.csv"),
                "rtk_path": _repo_or_abs(package_dir / "rtk.csv"),
                "imu_path": _repo_or_abs(package_dir / "imu.csv"),
                "altimeter_path": _repo_or_abs(package_dir / "altimeter.csv"),
                "recommended_profile": "motion_compensation_v2",
                "steps": [
                    {
                        "method": "motion_compensation_v2",
                        "params": {
                            "height_source": "auto",
                            "height_reference_mode": "mean",
                            "max_shift_samples": 8,
                            "max_shift_ns": 12.0,
                            "max_amplitude_scale": 1.8,
                            "resample_spacing_m": 0.34,
                        },
                    }
                ],
            }
        ],
        "output_dir": _repo_or_abs(output_dir),
    }


def _write_readme(package_dir: Path) -> None:
    readme = """# UAV-GPR Motion Demo v1

This is a synthetic UAV-GPR motion compensation demonstration dataset for MyGPR.
It is not field evidence and must not be used as an external geological
conclusion.

## Expected Visual Result

- `main.csv` is an airborne stacked CSV that reshapes to a 160 x 180 B-scan.
- The B-scan contains shallow layered reflections, a clear pipe/cylinder-like
  hyperbola, mild noise, and weak striping/background components.
- `rtk.csv` contains a lightly curved and non-equidistant UAV trajectory.
- `imu.csv` contains small roll/pitch/yaw variations.
- `altimeter.csv` contains NAR15-style AGL height variations around 0.08-0.16 m.
- After `motion_compensation_v2`, the current 3D curtain should show a slightly
  more consistent top interface/target position than the raw curtain.

## Files

- `main.csv`: stacked UAV-GPR CSV with columns
  `longitude, latitude, ground_elevation_m, amplitude, flight_height_m, trace_timestamp_s`.
- `trace_timestamps.csv`: one timestamp per trace for sidecar synchronization checks.
- `rtk.csv`: RTK sidecar with longitude/latitude and local xyz fields.
- `imu.csv`: IMU sidecar with roll/pitch/yaw.
- `altimeter.csv`: height sidecar with `height_agl_m`, SNR, target count and validity.
- `manifest.json`: dataset contract and expected target notes.
- `metadata.json`: compact generation parameters and target ROI.
- `batch_motion_v2.json`: CLI config for a smoke run.

Recommended workflow: `motion_compensation_v2`.
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")


def generate_demo_package(
    package_dir: str | Path = DEFAULT_PACKAGE_DIR,
    *,
    config_out: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> DemoPackageResult:
    """Generate the synthetic UAV-GPR motion demo package."""
    package_path = Path(package_dir).resolve()
    config_path = Path(config_out).resolve()
    output_path = Path(output_dir).resolve()
    package_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    payload = _build_demo_payload()
    _write_main_csv(package_path / "main.csv", payload)
    _write_sidecars(package_path, payload)
    _write_readme(package_path)

    metadata = {
        "schema": "uav_gpr_motion_demo_v1_metadata",
        "samples": SAMPLES,
        "traces": TRACES,
        "total_time_ns": TOTAL_TIME_NS,
        "air_wave_speed_m_per_ns": AIR_WAVE_SPEED_M_PER_NS,
        "reference_height_m": float(payload["reference_height_m"]),
        "height_agl_min_m": float(np.min(payload["height_agl_m"])),
        "height_agl_max_m": float(np.max(payload["height_agl_m"])),
        "max_abs_time_shift_samples": float(np.max(np.abs(payload["time_shift_samples"]))),
        "expected_target": payload["expected_target"],
    }
    manifest = {
        "schema": "uav_gpr_motion_demo_v1",
        "description": "Synthetic UAV-GPR 3D curtain and motion compensation demonstration dataset.",
        "data_file": "main.csv",
        "trace_timestamps_file": "trace_timestamps.csv",
        "rtk_file": "rtk.csv",
        "imu_file": "imu.csv",
        "altimeter_file": "altimeter.csv",
        "metadata_file": "metadata.json",
        "expected_target": payload["expected_target"],
        "recommended_workflow": "motion_compensation_v2",
        "recommended_params": {
            "height_source": "auto",
            "height_reference_mode": "mean",
            "max_shift_samples": 8,
            "max_shift_ns": 12.0,
            "max_amplitude_scale": 1.8,
            "resample_spacing_m": 0.34,
        },
        "notes": [
            "Synthetic only; not field evidence.",
            "Designed for MyGPR 3D preview, motion_compensation_v2 and Evidence export smoke checks.",
        ],
    }
    config = _build_config(package_path, output_path)

    _write_json(package_path / "metadata.json", metadata)
    _write_json(package_path / "manifest.json", manifest)
    _write_json(package_path / "batch_motion_v2.json", config)
    _write_json(config_path, config)

    return DemoPackageResult(
        package_dir=package_path,
        config_path=config_path,
        package_config_path=package_path / "batch_motion_v2.json",
        output_dir=output_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic UAV-GPR motion compensation demo package."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_PACKAGE_DIR))
    parser.add_argument("--config-out", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--batch-output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    result = generate_demo_package(
        args.output_dir,
        config_out=args.config_out,
        output_dir=args.batch_output_dir,
    )
    print(f"Generated package: {result.package_dir}")
    print(f"Generated config: {result.config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
