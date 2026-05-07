#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a synthetic UAV-GPR motion V2 sample package.

The package is intentionally synthetic. It validates the software contract for
main airborne CSV + RTK + IMU + NAR15/altimeter sidecars before field data is
available.
"""

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

DEFAULT_PACKAGE_DIR = ROOT / "sample_data" / "uav_gpr_motion_v2"
DEFAULT_CONFIG_PATH = ROOT / "config" / "uav_gpr_motion_v2_synthetic.json"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "uav_gpr_motion_v2_synthetic"

SAMPLES = 96
TRACES = 24
TOTAL_TIME_NS = 120.0
TRACE_INTERVAL_M = 0.55
START_TIMESTAMP_S = 1000.0
TRACE_PERIOD_S = 0.05
AIR_WAVE_SPEED_M_PER_NS = 0.299792458
ORIGIN_LONGITUDE = 104.123456
ORIGIN_LATITUDE = 30.654321


@dataclass(frozen=True)
class PackageResult:
    """Generated package paths."""

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


def _add_pulse(data: np.ndarray, row: float, col: int, pulse: np.ndarray) -> None:
    center = int(round(float(row)))
    start = center - pulse.size // 2
    if start < 0 or start + pulse.size > data.shape[0]:
        return
    data[start : start + pulse.size, col] += pulse


def _shift_trace(trace: np.ndarray, shift_samples: float) -> np.ndarray:
    sample_index = np.arange(trace.size, dtype=np.float64)
    return np.interp(
        sample_index - float(shift_samples),
        sample_index,
        np.asarray(trace, dtype=np.float64),
        left=0.0,
        right=0.0,
    ).astype(np.float32)


def _build_synthetic_line() -> dict[str, Any]:
    rng = np.random.default_rng(20260507)
    trace_idx = np.arange(TRACES, dtype=np.float64)
    phase = np.linspace(0.0, 1.0, TRACES, dtype=np.float64)
    sample_axis = np.linspace(0.0, 1.0, SAMPLES, dtype=np.float64)[:, None]

    distance_m = TRACE_INTERVAL_M * trace_idx
    lateral_m = 0.18 * np.sin(2.0 * np.pi * 0.75 * phase)
    longitude, latitude = _lon_lat_from_xy(distance_m, lateral_m)
    timestamps = START_TIMESTAMP_S + TRACE_PERIOD_S * trace_idx
    ground_elevation_m = 713.0 + 0.25 * np.sin(2.0 * np.pi * 0.35 * phase)
    height_agl_m = 1.55 + 0.18 * np.sin(2.0 * np.pi * 1.2 * phase + 0.25)
    flight_height_m = height_agl_m + 0.03 * np.cos(2.0 * np.pi * 2.4 * phase)

    roll_deg = 2.0 * np.sin(2.0 * np.pi * 1.1 * phase)
    pitch_deg = 1.5 * np.cos(2.0 * np.pi * 0.9 * phase + 0.15)
    yaw_deg = 8.0 + 1.2 * np.sin(2.0 * np.pi * 0.45 * phase)

    clean = 0.006 * rng.normal(size=(SAMPLES, TRACES))
    clean += 0.018 * np.sin(2.0 * np.pi * 1.7 * sample_axis)
    clean += 0.010 * np.exp(-2.8 * sample_axis) * np.cos(2.0 * np.pi * phase[None, :])

    direct_pulse = np.array([0.12, 0.42, 1.05, 1.88, 2.35, 1.88, 1.05, 0.42, 0.12])
    reflector_pulse = np.array([0.10, 0.28, 0.58, 1.05, 1.45, 1.05, 0.58, 0.28, 0.10])
    weak_pulse = np.array([0.04, 0.12, 0.26, 0.38, 0.26, 0.12, 0.04])

    hyperbola_center = 0.52 * (TRACES - 1)
    for col in range(TRACES):
        direct_row = 13.0 + 0.8 * np.sin(2.0 * np.pi * 0.8 * phase[col])
        reflector_row = 44.0 + 8.5 * (
            np.sqrt(1.0 + ((col - hyperbola_center) / 4.8) ** 2) - 1.0
        )
        weak_row = 68.0 + 4.0 * np.sin(2.0 * np.pi * 1.4 * phase[col])
        _add_pulse(clean, direct_row, col, direct_pulse)
        _add_pulse(clean, reflector_row, col, reflector_pulse)
        _add_pulse(clean, weak_row, col, weak_pulse)

    reference_height_m = float(np.mean(height_agl_m))
    dt_ns = TOTAL_TIME_NS / max(SAMPLES - 1, 1)
    time_shift_samples = (
        2.0 * (height_agl_m - reference_height_m) / AIR_WAVE_SPEED_M_PER_NS / dt_ns
    )
    amplitude_scale = np.clip((reference_height_m / height_agl_m) ** 2, 0.5, 2.0)

    observed = np.empty_like(clean, dtype=np.float32)
    for col in range(TRACES):
        observed[:, col] = _shift_trace(clean[:, col], time_shift_samples[col])
        observed[:, col] *= np.float32(amplitude_scale[col])

    return {
        "data": observed.astype(np.float32),
        "longitude": longitude,
        "latitude": latitude,
        "ground_elevation_m": ground_elevation_m,
        "flight_height_m": flight_height_m,
        "height_agl_m": height_agl_m,
        "trace_timestamp_s": timestamps,
        "distance_m": distance_m,
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "yaw_deg": yaw_deg,
        "time_shift_samples": time_shift_samples,
        "reference_height_m": reference_height_m,
    }


def _write_main_csv(path: Path, payload: dict[str, Any]) -> None:
    data = np.asarray(payload["data"], dtype=np.float32)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"Number of Samples = {SAMPLES}\n")
        handle.write(f"Time windows (ns) = {TOTAL_TIME_NS:.6f}\n")
        handle.write(f"Number of Traces = {TRACES}\n")
        handle.write(f"Trace interval (m) = {TRACE_INTERVAL_M:.6f}\n")
        writer = csv.writer(handle)
        for trace in range(TRACES):
            for sample in range(SAMPLES):
                writer.writerow(
                    [
                        f"{payload['longitude'][trace]:.10f}",
                        f"{payload['latitude'][trace]:.10f}",
                        f"{payload['ground_elevation_m'][trace]:.6f}",
                        f"{data[sample, trace]:.8f}",
                        f"{payload['flight_height_m'][trace]:.6f}",
                        f"{payload['trace_timestamp_s'][trace]:.6f}",
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
    longitude = np.asarray(payload["longitude"], dtype=np.float64)
    latitude = np.asarray(payload["latitude"], dtype=np.float64)
    ground = np.asarray(payload["ground_elevation_m"], dtype=np.float64)
    flight = np.asarray(payload["flight_height_m"], dtype=np.float64)
    height = np.asarray(payload["height_agl_m"], dtype=np.float64)
    roll = np.asarray(payload["roll_deg"], dtype=np.float64)
    pitch = np.asarray(payload["pitch_deg"], dtype=np.float64)
    yaw = np.asarray(payload["yaw_deg"], dtype=np.float64)

    rtk_rows = []
    imu_rows = []
    altimeter_rows = []
    for idx in range(TRACES):
        rtk_rows.append(
            {
                "timestamp_s": f"{timestamps[idx]:.6f}",
                "longitude": f"{longitude[idx]:.10f}",
                "latitude": f"{latitude[idx]:.10f}",
                "ground_elevation_m": f"{ground[idx]:.6f}",
                "flight_height_m": f"{flight[idx]:.6f}",
                "rtk_fix_type": 5,
                "satellites": 18 + int(idx % 4),
                "hdop": f"{0.65 + 0.03 * (idx % 3):.3f}",
            }
        )
        imu_rows.append(
            {
                "timestamp_s": f"{timestamps[idx]:.6f}",
                "roll_deg": f"{roll[idx]:.6f}",
                "pitch_deg": f"{pitch[idx]:.6f}",
                "yaw_deg": f"{yaw[idx]:.6f}",
                "angular_rate_x": f"{0.10 * np.cos(idx):.6f}",
                "angular_rate_y": f"{0.08 * np.sin(idx * 0.7):.6f}",
                "angular_rate_z": f"{0.05 * np.cos(idx * 0.4):.6f}",
            }
        )
        altimeter_rows.append(
            {
                "timestamp_s": f"{timestamps[idx]:.6f}",
                "height_agl_m": f"{height[idx]:.6f}",
                "height_source": "nar15_synthetic",
                "snr": f"{18.0 + 2.0 * np.sin(idx / 5.0):.3f}",
                "target_count": 1,
                "valid": 1,
            }
        )

    _write_dict_rows(package_dir / "rtk.csv", rtk_rows)
    _write_dict_rows(package_dir / "imu.csv", imu_rows)
    _write_dict_rows(package_dir / "altimeter.csv", altimeter_rows)


def _build_config(
    *,
    package_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "version": "1.0",
        "description": "Synthetic UAV-GPR motion V2 acceptance package.",
        "jobs": [
            {
                "id": "uav_gpr_motion_v2_synthetic",
                "input": _repo_or_abs(package_dir / "main.csv"),
                "rtk_path": _repo_or_abs(package_dir / "rtk.csv"),
                "imu_path": _repo_or_abs(package_dir / "imu.csv"),
                "altimeter_path": _repo_or_abs(package_dir / "altimeter.csv"),
                "recommended_profile": "motion_compensation_v2",
            }
        ],
        "output_dir": _repo_or_abs(output_dir),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_readme(package_dir: Path) -> None:
    readme = """# UAV-GPR Motion V2 Synthetic Package

This package is synthetic. It is for MyGPR software-contract testing before real field data is available.

## Files

- `main.csv`: airborne stacked GPR CSV. Columns are `longitude, latitude, ground_elevation_m, amplitude, flight_height_m, trace_timestamp_s`.
- `rtk.csv`: RTK sidecar with `timestamp_s, longitude, latitude, ground_elevation_m, flight_height_m, rtk_fix_type, satellites, hdop`.
- `imu.csv`: IMU sidecar with `timestamp_s, roll_deg, pitch_deg, yaw_deg, angular_rate_x, angular_rate_y, angular_rate_z`.
- `altimeter.csv`: NAR15-style height sidecar with `timestamp_s, height_agl_m, height_source, snr, target_count, valid`.
- `batch_motion_v2.json`: CLI batch config that runs `motion_compensation_v2`.

## CLI Check

```bash
python cli_batch.py validate --config config/uav_gpr_motion_v2_synthetic.json
python cli_batch.py run --config config/uav_gpr_motion_v2_synthetic.json --force
```

## Field Trip Acceptance Checklist

- The main GPR CSV must preserve one timestamp per trace, preferably as `trace_timestamp_s`.
- RTK records must include timestamp, longitude, latitude, fix type, satellites, and HDOP.
- IMU records must include timestamp, roll, pitch, and yaw in degrees.
- NAR15/altimeter records must include timestamp and AGL height. Keep `valid`, `target_count`, and SNR when available.
- Record lever arms between RTK antenna, IMU, radar antenna phase center, and altimeter beam center.
- Confirm all devices share a consistent time base or store enough information to align timestamps after acquisition.
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")


def generate_package(
    package_dir: str | Path = DEFAULT_PACKAGE_DIR,
    *,
    config_out: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> PackageResult:
    """Generate the synthetic sample package and return output paths."""
    package_path = Path(package_dir).resolve()
    config_path = Path(config_out).resolve()
    output_path = Path(output_dir).resolve()
    package_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    payload = _build_synthetic_line()
    _write_main_csv(package_path / "main.csv", payload)
    _write_sidecars(package_path, payload)
    _write_readme(package_path)

    config = _build_config(package_dir=package_path, output_dir=output_path)
    package_config_path = package_path / "batch_motion_v2.json"
    _write_json(package_config_path, config)
    _write_json(config_path, config)

    manifest = {
        "schema": "uav_gpr_motion_v2_synthetic_package",
        "samples": SAMPLES,
        "traces": TRACES,
        "total_time_ns": TOTAL_TIME_NS,
        "trace_interval_m": TRACE_INTERVAL_M,
        "air_wave_speed_m_per_ns": AIR_WAVE_SPEED_M_PER_NS,
        "reference_height_m": float(payload["reference_height_m"]),
        "max_abs_time_shift_samples": float(
            np.max(np.abs(payload["time_shift_samples"]))
        ),
        "files": {
            "main_csv": "main.csv",
            "rtk_csv": "rtk.csv",
            "imu_csv": "imu.csv",
            "altimeter_csv": "altimeter.csv",
            "package_config": "batch_motion_v2.json",
        },
    }
    _write_json(package_path / "manifest.json", manifest)

    return PackageResult(
        package_dir=package_path,
        config_path=config_path,
        package_config_path=package_config_path,
        output_dir=output_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic UAV-GPR motion V2 sample package."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_PACKAGE_DIR))
    parser.add_argument("--config-out", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--batch-output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    result = generate_package(
        args.output_dir,
        config_out=args.config_out,
        output_dir=args.batch_output_dir,
    )
    print(f"Generated package: {result.package_dir}")
    print(f"Generated config: {result.config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
