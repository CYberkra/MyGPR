#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the first clean GPRMAX benchmark package for MyGPR."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpr_io import read_gprmax_out  # noqa: E402


SCENARIO_ID = "cylinder_single_v1"
DEFAULT_PACKAGE_DIR = ROOT / "sample_data" / "gprmax_benchmarks" / SCENARIO_ID
SAMPLES = 220
TRACES = 81
TOTAL_TIME_NS = 18.0
TRACE_STEP_M = 0.002
DX_M = 0.002
DOMAIN_X_M = 0.240
DOMAIN_Y_M = 0.210
DOMAIN_Z_M = 0.002
CYLINDER_X_M = 0.120
CYLINDER_Y_M = 0.080
CYLINDER_RADIUS_M = 0.010


@dataclass(frozen=True)
class PackageResult:
    """Generated benchmark package paths."""

    package_dir: Path
    scenario_path: Path
    model_in_path: Path
    ground_truth_path: Path
    bscan_csv_path: Path
    preview_path: Path
    readme_path: Path


def generate_package(
    package_dir: str | Path = DEFAULT_PACKAGE_DIR,
    *,
    raw_out_path: str | Path | None = None,
) -> PackageResult:
    """Generate a clean `cylinder_single_v1` benchmark package."""
    package_root = Path(package_dir)
    package_root.mkdir(parents=True, exist_ok=True)

    model_in_path = package_root / "model.in"
    scenario_path = package_root / "scenario.json"
    ground_truth_path = package_root / "ground_truth.json"
    bscan_csv_path = package_root / "mygpr_bscan.csv"
    preview_path = package_root / "preview.png"
    readme_path = package_root / "README.md"

    if raw_out_path is not None:
        source_path = Path(raw_out_path)
        load_result = read_gprmax_out(str(source_path))
        bscan = np.asarray(load_result["data"], dtype=np.float32)
        source_info = {
            "kind": "gprmax_out",
            "path": str(source_path),
        }
        simulation = _simulation_from_load_result(load_result, bscan)
    else:
        bscan = _build_synthetic_reference_bscan()
        source_info = {
            "kind": "synthetic_reference",
            "path": None,
        }
        simulation = {
            "sample_count": int(bscan.shape[0]),
            "trace_count": int(bscan.shape[1]),
            "time_step_s": (TOTAL_TIME_NS * 1e-9) / float(bscan.shape[0]),
            "total_time_ns": TOTAL_TIME_NS,
            "trace_step_m": TRACE_STEP_M,
        }

    np.savetxt(bscan_csv_path, bscan, delimiter=",", fmt="%.8e")
    scenario = _build_scenario(source_info, simulation)
    ground_truth = _build_ground_truth(simulation)
    model_in_path.write_text(_build_model_in(), encoding="utf-8")
    scenario_path.write_text(
        json.dumps(_json_safe(scenario), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    ground_truth_path.write_text(
        json.dumps(_json_safe(ground_truth), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _save_preview(bscan, ground_truth, preview_path)
    readme_path.write_text(_build_readme(simulation, source_info), encoding="utf-8")

    return PackageResult(
        package_dir=package_root,
        scenario_path=scenario_path,
        model_in_path=model_in_path,
        ground_truth_path=ground_truth_path,
        bscan_csv_path=bscan_csv_path,
        preview_path=preview_path,
        readme_path=readme_path,
    )


def _build_synthetic_reference_bscan() -> np.ndarray:
    rng = np.random.default_rng(20260507)
    sample_axis = np.linspace(0.0, 1.0, SAMPLES, dtype=np.float64)[:, None]
    trace_axis = np.linspace(-1.0, 1.0, TRACES, dtype=np.float64)
    data = 0.025 * rng.normal(size=(SAMPLES, TRACES))
    data += 0.10 * np.sin(2.0 * np.pi * 1.2 * sample_axis)
    data += 0.035 * np.exp(-2.4 * sample_axis) * np.cos(2.0 * np.pi * trace_axis)
    data += 0.045 * np.sin(2.0 * np.pi * 3.5 * sample_axis)

    apex_sample = _default_apex_sample(SAMPLES)
    aperture = max(10.0, TRACES / 6.5)
    pulse = np.array([-0.12, -0.28, 0.18, 0.86, 1.28, 0.86, 0.18, -0.28, -0.12])
    for trace_idx in range(TRACES):
        offset = (trace_idx - (TRACES - 1) / 2.0) / aperture
        row = apex_sample + 44.0 * (np.sqrt(1.0 + offset * offset) - 1.0)
        _add_pulse(data, row, trace_idx, pulse)

    # A weak dipping reflector makes over-smoothing easier to spot.
    weak_pulse = np.array([0.04, 0.12, 0.22, 0.30, 0.22, 0.12, 0.04])
    for trace_idx in range(12, TRACES - 12):
        row = 150.0 + 0.25 * (trace_idx - TRACES / 2.0)
        _add_pulse(data, row, trace_idx, weak_pulse)

    return data.astype(np.float32)


def _add_pulse(data: np.ndarray, row: float, col: int, pulse: np.ndarray) -> None:
    center = int(round(float(row)))
    start = center - pulse.size // 2
    if start < 0 or start + pulse.size > data.shape[0]:
        return
    data[start : start + pulse.size, col] += pulse


def _simulation_from_load_result(
    load_result: dict[str, Any],
    bscan: np.ndarray,
) -> dict[str, Any]:
    time_step_s = load_result.get("time_step_s")
    total_time_ns = load_result.get("total_time_ns")
    if total_time_ns is None and time_step_s is not None:
        total_time_ns = float(time_step_s) * int(bscan.shape[0]) * 1e9
    return {
        "sample_count": int(bscan.shape[0]),
        "trace_count": int(bscan.shape[1]),
        "time_step_s": float(time_step_s) if time_step_s is not None else None,
        "total_time_ns": float(total_time_ns) if total_time_ns is not None else None,
        "trace_step_m": TRACE_STEP_M,
    }


def _build_scenario(
    source_info: dict[str, Any],
    simulation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "mygpr_gprmax_scenario_v1",
        "scenario_id": SCENARIO_ID,
        "description": "Single metal cylinder in dielectric half-space for auto-tune validation.",
        "source": source_info,
        "simulation": simulation,
        "domain_m": [DOMAIN_X_M, DOMAIN_Y_M, DOMAIN_Z_M],
        "dx_dy_dz_m": [DX_M, DX_M, DX_M],
        "materials": [
            {
                "name": "half_space",
                "relative_permittivity": 6.0,
                "conductivity_s_per_m": 0.0,
                "relative_permeability": 1.0,
                "magnetic_loss": 0.0,
            },
            {"name": "metal_cylinder", "material": "pec"},
        ],
        "antenna": {
            "waveform": "ricker",
            "center_frequency_hz": 1.5e9,
            "source": "hertzian_dipole_z",
            "source_position_m": [0.040, 0.170, 0.0],
            "receiver_position_m": [0.080, 0.170, 0.0],
            "source_step_m": [TRACE_STEP_M, 0.0, 0.0],
            "receiver_step_m": [TRACE_STEP_M, 0.0, 0.0],
        },
        "target": {
            "target_id": "metal_cylinder_01",
            "type": "metal_cylinder",
            "center_m": [CYLINDER_X_M, CYLINDER_Y_M, 0.0],
            "radius_m": CYLINDER_RADIUS_M,
        },
    }


def _build_ground_truth(simulation: dict[str, Any]) -> dict[str, Any]:
    samples = int(simulation["sample_count"])
    traces = int(simulation["trace_count"])
    total_time_ns = simulation.get("total_time_ns")
    apex_trace = max(0, min(traces // 2, traces - 1))
    apex_sample = _default_apex_sample(samples)
    if total_time_ns:
        apex_time_ns = float(apex_sample) * float(total_time_ns) / max(samples, 1)
    else:
        apex_time_ns = None
    return {
        "schema": "mygpr_gprmax_ground_truth_v1",
        "scenario_id": SCENARIO_ID,
        "targets": [
            {
                "target_id": "metal_cylinder_01",
                "type": "hyperbola",
                "source_geometry": "metal_cylinder",
                "apex_trace_idx": int(apex_trace),
                "apex_sample_idx": int(apex_sample),
                "apex_time_ns": apex_time_ns,
                "roi": _target_roi(samples, traces, apex_sample, apex_trace),
                "must_preserve": True,
                "expected_features": [
                    "hyperbola_apex",
                    "left_hyperbola_arm",
                    "right_hyperbola_arm",
                ],
            }
        ],
        "known_background": {
            "horizontal_layers": [],
            "air_ground_interface": None,
        },
        "metrics_hint": {
            "target_roi_weight": 1.0,
            "background_roi_weight": 0.5,
            "false_positive_penalty": 0.7,
        },
    }


def _default_apex_sample(samples: int) -> int:
    if samples >= 120:
        return int(round(samples * 0.36))
    return max(0, min(samples // 2, samples - 1))


def _target_roi(
    samples: int,
    traces: int,
    apex_sample: int,
    apex_trace: int,
) -> dict[str, int]:
    half_width = max(1, min(24, traces // 3))
    before = max(1, min(30, samples // 4))
    after = max(1, min(70, samples // 2))
    return {
        "time_start_idx": max(0, int(apex_sample - before)),
        "time_end_idx": min(samples, int(apex_sample + after)),
        "dist_start_idx": max(0, int(apex_trace - half_width)),
        "dist_end_idx": min(traces, int(apex_trace + half_width + 1)),
    }


def _build_model_in() -> str:
    return f"""#title: MyGPR cylinder_single_v1
#domain: {DOMAIN_X_M:.3f} {DOMAIN_Y_M:.3f} {DOMAIN_Z_M:.3f}
#dx_dy_dz: {DX_M:.3f} {DX_M:.3f} {DX_M:.3f}
#time_window: {TOTAL_TIME_NS * 1e-9:.9g}

#material: 6 0 1 0 half_space

#waveform: ricker 1 1.5e9 my_ricker
#hertzian_dipole: z 0.040 0.170 0 my_ricker
#rx: 0.080 0.170 0
#src_steps: {TRACE_STEP_M:.3f} 0 0
#rx_steps: {TRACE_STEP_M:.3f} 0 0

#box: 0 0 0 {DOMAIN_X_M:.3f} 0.170 {DOMAIN_Z_M:.3f} half_space
#cylinder: {CYLINDER_X_M:.3f} {CYLINDER_Y_M:.3f} 0 {CYLINDER_X_M:.3f} {CYLINDER_Y_M:.3f} {DOMAIN_Z_M:.3f} {CYLINDER_RADIUS_M:.3f} pec
"""


def _save_preview(
    bscan: np.ndarray,
    ground_truth: dict[str, Any],
    out_path: Path,
) -> None:
    roi = ground_truth["targets"][0]["roi"]
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=150)
    vmax = float(np.nanpercentile(np.abs(bscan), 99.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    ax.imshow(bscan, cmap="gray", aspect="auto", vmin=-vmax, vmax=vmax)
    rect = plt.Rectangle(
        (roi["dist_start_idx"], roi["time_start_idx"]),
        roi["dist_end_idx"] - roi["dist_start_idx"],
        roi["time_end_idx"] - roi["time_start_idx"],
        fill=False,
        edgecolor="tab:red",
        linewidth=1.4,
    )
    ax.add_patch(rect)
    ax.set_title("cylinder_single_v1 B-scan preview")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Sample")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _build_readme(
    simulation: dict[str, Any],
    source_info: dict[str, Any],
) -> str:
    return f"""# cylinder_single_v1

This is the first clean MyGPR GPRMAX benchmark package.

- Scenario: single PEC cylinder in a dielectric half-space.
- Source kind: {source_info.get("kind")}
- B-scan shape: {simulation.get("sample_count")} samples x {simulation.get("trace_count")} traces.
- Ground truth: `ground_truth.json`.
- MyGPR input CSV: `mygpr_bscan.csv`.

The bundled fallback B-scan is deterministic and contract-oriented. It allows
MyGPR auto-tune scoring and export paths to be tested without running gprMax.
A later optional smoke can replace `mygpr_bscan.csv` with data converted from
real gprMax `.out` files while preserving the same scenario and ground-truth
schema.
"""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, int):
        return int(value)
    return str(value)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MyGPR GPRMAX cylinder_single_v1 benchmark package."
    )
    parser.add_argument(
        "--package-dir",
        default=str(DEFAULT_PACKAGE_DIR),
        help="Output package directory.",
    )
    parser.add_argument(
        "--raw-out-path",
        default=None,
        help="Optional gprMax .out path to convert into mygpr_bscan.csv.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = generate_package(args.package_dir, raw_out_path=args.raw_out_path)
    print(f"Generated GPRMAX benchmark package: {result.package_dir}")
    print(f"Generated MyGPR CSV: {result.bscan_csv_path}")
    print(f"Generated ground truth: {result.ground_truth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
