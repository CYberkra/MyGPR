#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run deterministic algorithm output compatibility audit on current branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.methods_registry import PROCESSING_METHODS
from core.processing_engine import (
    prepare_runtime_params,
    run_processing_method,
)


HIGH_PRIORITY = [
    "set_zero_time",
    "dewow",
    "frequency_filter_1d",
    "subtracting_average_2D",
    "median_background_2D",
    "agcGain",
    "sec_gain",
    "motion_compensation_v2",
]
MEDIUM_PRIORITY = [
    "running_average_2D",
    "stolt_migration",
    "kirchhoff_migration",
    "energy_decay_gain",
    "amplitude_scale",
    "time_cut",
    "trace_qc",
    "equidistant_trace_resample",
]


def _fixture_small() -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.default_rng(42)
    ns, nt = 128, 64
    t = np.linspace(0.0, 1.0, ns, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, nt, dtype=np.float32)[None, :]
    drift = 0.35 * (t - 0.5)
    banding = 0.12 * np.sin(2 * np.pi * (x * 6.0))
    noise = 0.04 * rng.standard_normal((ns, nt)).astype(np.float32)
    arr = drift + banding + noise
    for tr in range(8, nt - 8, 7):
        center = int(24 + 0.08 * (tr - nt / 2) ** 2)
        center = max(10, min(ns - 10, center))
        arr[center - 2 : center + 2, tr] += np.array([0.2, 0.5, 0.5, 0.2], dtype=np.float32)
    header = {"total_time_ns": 100.0, "trace_interval_m": 0.05}
    meta = {}
    return arr.astype(np.float32), header, meta


def _fixture_field_like() -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.default_rng(7)
    ns, nt = 501, 2378
    t = np.linspace(0.0, 1.0, ns, dtype=np.float32)[:, None]
    arr = 0.18 * np.sin(2 * np.pi * t * 2.0) + 0.03 * rng.standard_normal((ns, nt)).astype(np.float32)
    arr += 0.08 * np.exp(-((t - 0.35) ** 2) / 0.01)
    header = {"total_time_ns": 700.0, "trace_interval_m": 0.09093}
    meta = {}
    return arr.astype(np.float32), header, meta


def _fixture_metadata_rich() -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.default_rng(123)
    ns, nt = 256, 120
    arr = (0.06 * rng.standard_normal((ns, nt)) + np.linspace(0, 0.25, ns)[:, None]).astype(np.float32)
    dist = np.cumsum(np.full(nt, 0.07, dtype=np.float64))
    ts = np.linspace(0.0, 12.0, nt, dtype=np.float64)
    height = 3.0 + 0.15 * np.sin(np.linspace(0, 4 * np.pi, nt))
    meta = {
        "trace_distance_m": dist,
        "trace_timestamp_s": ts,
        "height_agl_m": height,
        "pitch_deg": np.zeros(nt, dtype=np.float64),
        "roll_deg": np.zeros(nt, dtype=np.float64),
        "yaw_deg": np.zeros(nt, dtype=np.float64),
    }
    header = {"total_time_ns": 240.0, "trace_interval_m": 0.07, "has_airborne_metadata": True}
    return arr, header, meta


def _fixture_metadata_missing() -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    arr, _, _ = _fixture_metadata_rich()
    return arr, {}, {}


def _params_for(method_key: str) -> dict[str, Any]:
    table: dict[str, dict[str, Any]] = {
        "set_zero_time": {"first_break_threshold": 0.15},
        "dewow": {"window": 32},
        "frequency_filter_1d": {"f_low_mhz": 20.0, "f_high_mhz": 170.0},
        "subtracting_average_2D": {"ntraces": 9},
        "median_background_2D": {"ntraces": 9},
        "running_average_2D": {"window_size": 9},
        "agcGain": {"window": 48, "factor": 1.0},
        "sec_gain": {"alpha": 0.006},
        "energy_decay_gain": {"window": 48},
        "amplitude_scale": {"scale": 1.2},
        "time_cut": {"time_start_ns": 20.0, "time_end_ns": 180.0},
        "trace_qc": {"zscore_thresh": 4.0},
        "equidistant_trace_resample": {"target_spacing_m": 0.1},
        "motion_compensation_v2": {"resample_spacing_m": 0.0, "enable_resample": True},
        "stolt_migration": {"velocity": 0.1},
        "kirchhoff_migration": {"velocity": 0.1},
    }
    return dict(table.get(method_key, {}))


def _array_hash(arr: np.ndarray) -> str:
    quant = np.round(arr.astype(np.float32), 6)
    return hashlib.sha256(quant.tobytes()).hexdigest()[:16]


def _metrics(arr: np.ndarray) -> dict[str, Any]:
    x = np.asarray(arr, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(x))),
        "mean_abs": float(np.mean(np.abs(x))),
        "rms": float(np.sqrt(np.mean(x * x))),
        "hash16": _array_hash(x.astype(np.float32)),
    }


def _run_one(
    method_key: str,
    fixture_name: str,
    data: np.ndarray,
    header: dict[str, Any],
    trace_meta: dict[str, np.ndarray],
) -> dict[str, Any]:
    params = _params_for(method_key)
    out: dict[str, Any] = {
        "method_key": method_key,
        "fixture_name": fixture_name,
        "input_shape": list(data.shape),
        "params": params,
    }
    try:
        runtime_params = prepare_runtime_params(
            method_key, params, dict(header), dict(trace_meta), data.shape
        )
        def _json_safe(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(k): _json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_json_safe(v) for v in value]
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        result, meta = run_processing_method(data, method_key, runtime_params)
        arr = np.asarray(result, dtype=np.float32)
        out.update(
            {
                "status": "ok",
                "output_shape": list(arr.shape),
                "shape_changed": list(arr.shape) != list(data.shape),
                "metrics": _metrics(arr),
                "warning_count": len((meta or {}).get("runtime_warnings", []) or []),
                "runtime_warnings": _json_safe((meta or {}).get("runtime_warnings", []) or []),
                "metadata_keys_emitted": sorted(list((meta or {}).keys())),
                "effective_params": _json_safe(dict(runtime_params)),
            }
        )
    except Exception as exc:  # noqa: BLE001
        out.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "output_shape": None,
                "shape_changed": False,
                "metrics": None,
                "warning_count": 0,
                "runtime_warnings": [],
                "metadata_keys_emitted": [],
                "effective_params": {},
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--branch-label", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fixtures = {
        "small_deterministic": _fixture_small(),
        "field_like_501x2378": _fixture_field_like(),
        "metadata_rich_uav": _fixture_metadata_rich(),
        "metadata_missing": _fixture_metadata_missing(),
    }
    methods = HIGH_PRIORITY + MEDIUM_PRIORITY
    rows: list[dict[str, Any]] = []
    for method in methods:
        if method not in PROCESSING_METHODS:
            for fname, (data, _, _) in fixtures.items():
                rows.append(
                    {
                        "method_key": method,
                        "fixture_name": fname,
                        "input_shape": list(data.shape),
                        "params": _params_for(method),
                        "status": "missing_method",
                    }
                )
            continue
        for fname, (data, header, meta) in fixtures.items():
            rows.append(_run_one(method, fname, data, header, meta))

    payload = {
        "branch_label": args.branch_label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method_count": len(methods),
        "fixture_count": len(fixtures),
        "rows": rows,
    }
    (outdir / f"audit_{args.branch_label}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
