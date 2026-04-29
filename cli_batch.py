#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPR CLI Batch MVP (Phase-1) - 使用统一方法注册表

Scope:
- validate: config + inputs + method params validation
- run: sequential batch processing with minimal summary output
- resume: placeholder/basic hook (phase-2 target)

Design goals:
- 使用 methods_registry 统一方法定义，避免重复
- Keep minimal, runnable CLI main path
- Avoid report engine coupling in phase-1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "PythonModule_core"))
MODULE_DIR = os.path.join(BASE_DIR, "PythonModule")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from core.benchmark_registry import list_benchmark_sample_ids
from core.evidence_export import export_motion_compensation_benchmark
from core.gpr_io import extract_airborne_csv_payload, savecsv, save_image
from core.processing_engine import (
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.preset_profiles import GUI_PRESETS_V1, RECOMMENDED_RUN_PROFILES

# 使用统一的方法注册表
from core.methods_registry import (
    PROCESSING_METHODS,
    method_svd_background,
    method_fk_filter,
    method_hankel_svd,
    method_sliding_average,
)


# ---------- Input/header parsing (aligned with GUI logic) ----------
_HEADER_KEYS = [
    "Number of Samples",
    "Time windows (ns)",
    "Number of Traces",
    "Trace interval (m)",
]


def _parse_header_lines(lines: List[str]) -> Optional[Dict[str, float]]:
    if len(lines) < 4:
        return None
    info: Dict[str, float] = {}
    for line in lines[:4]:
        if "=" not in line:
            return None
        left, right = line.split("=", 1)
        key = left.strip()
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", right)
        if not m:
            return None
        try:
            val = float(m.group(0))
        except ValueError:
            return None
        info[key] = val
    if not all(k in info for k in _HEADER_KEYS):
        return None
    return {
        "a_scan_length": int(info["Number of Samples"]),
        "total_time_ns": float(info["Time windows (ns)"]),
        "num_traces": int(info["Number of Traces"]),
        "trace_interval_m": float(info["Trace interval (m)"]),
    }


def detect_csv_header(path: str) -> Optional[Dict[str, float]]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [f.readline().strip() for _ in range(4)]
    except OSError:
        return None
    return _parse_header_lines(lines)


def _detect_skip_lines(path: str, max_scan: int = 10) -> int:
    skip_lines = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i in range(max_scan):
            line = f.readline()
            if not line:
                break
            if "=" in line or "Samples" in line or "Traces" in line:
                skip_lines = i + 1
    return skip_lines


def load_gpr_csv(
    path: str,
    *,
    trace_timestamps_s: Any = None,
    rtk_path: str | None = None,
    imu_path: str | None = None,
) -> Tuple[
    np.ndarray,
    Optional[Dict[str, Any]],
    Optional[Dict[str, np.ndarray]],
]:
    header_info = detect_csv_header(path)
    skip_lines = _detect_skip_lines(path)

    df = pd.read_csv(path, header=None, skiprows=skip_lines)
    raw_data = df.values
    trace_metadata = None

    if header_info:
        sidecar_kwargs: Dict[str, Any] = {}
        if trace_timestamps_s is not None:
            sidecar_kwargs["trace_timestamps_s"] = np.asarray(
                trace_timestamps_s, dtype=np.float64
            )
        if rtk_path is not None:
            sidecar_kwargs["rtk_path"] = rtk_path
        if imu_path is not None:
            sidecar_kwargs["imu_path"] = imu_path
        data, trace_metadata, header_info = extract_airborne_csv_payload(
            raw_data,
            header_info,
            **sidecar_kwargs,
        )
    else:
        if rtk_path is not None or imu_path is not None:
            raise ValueError("RTK/IMU sidecars require an airborne CSV header")
        data = raw_data

    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=float(np.nanmean(data)))

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return np.asarray(data, dtype=float), header_info, trace_metadata


# 本地方法注册（用于兼容旧的本地方法调用）
LOCAL_METHODS = {
    "svd_bg": method_svd_background,
    "fk_filter": method_fk_filter,
    "hankel_svd": method_hankel_svd,
    "sliding_avg": method_sliding_average,
}


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]


# ---------- Config / validation ----------
def _coerce_param(method_key: str, param_name: str, value: Any) -> Any:
    meta_list = PROCESSING_METHODS[method_key].get("params", [])
    meta = next((m for m in meta_list if m["name"] == param_name), None)
    if meta is None:
        raise ValueError(f"Unknown param '{param_name}' for method '{method_key}'")

    if meta["type"] == "int":
        v = int(float(value))
    elif meta["type"] == "float":
        v = float(value)
    else:
        v = value

    if "min" in meta and v < meta["min"]:
        raise ValueError(f"Param {method_key}.{param_name}={v} < min({meta['min']})")
    if "max" in meta and v > meta["max"]:
        raise ValueError(f"Param {method_key}.{param_name}={v} > max({meta['max']})")
    return v


def _merge_params(
    method_key: str, raw_params: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    raw_params = raw_params or {}
    out: Dict[str, Any] = {}
    for meta in PROCESSING_METHODS[method_key].get("params", []):
        name = meta["name"]
        raw = raw_params.get(name, meta.get("default"))
        out[name] = _coerce_param(method_key, name, raw)
    unknown = sorted(
        set(raw_params.keys())
        - {m["name"] for m in PROCESSING_METHODS[method_key].get("params", [])}
    )
    if unknown:
        raise ValueError(f"Unknown params for {method_key}: {unknown}")
    return out


def _resolve_recommended_profile_methods(profile_key: str) -> List[Dict[str, Any]]:
    profile = RECOMMENDED_RUN_PROFILES.get(profile_key)
    if profile is None:
        raise ValueError(f"unknown recommended_profile: {profile_key}")

    preset_key = str(profile.get("preset_key") or "")
    preset = GUI_PRESETS_V1.get(preset_key)
    if preset is None:
        raise ValueError(
            f"recommended_profile '{profile_key}' references unknown preset_key: {preset_key}"
        )

    merged_method_params: Dict[str, Dict[str, Any]] = {}
    for method_key, params in preset.get("method_params", {}).items():
        merged_method_params[method_key] = dict(params)
    for method_key, params in profile.get("method_params", {}).items():
        merged_method_params[method_key] = dict(params)

    methods: List[Dict[str, Any]] = []
    for method_key in profile.get("order", []):
        if method_key not in PROCESSING_METHODS:
            raise ValueError(
                f"recommended_profile '{profile_key}' references unknown method: {method_key}"
            )
        methods.append(
            {
                "key": method_key,
                "params": dict(merged_method_params.get(method_key, {})),
            }
        )
    return methods


def _resolve_job_methods(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    methods = job.get("methods")
    recommended_profile = job.get("recommended_profile")

    if methods and recommended_profile:
        raise ValueError("job cannot define both methods and recommended_profile")
    if methods:
        if not isinstance(methods, list):
            raise ValueError("methods must be a non-empty list")
        return methods
    if recommended_profile:
        return _resolve_recommended_profile_methods(str(recommended_profile))
    raise ValueError("methods must be non-empty list or recommended_profile must be set")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def _safe_relpath(path: str, repo_root: str) -> str:
    """Return a repo-relative path when possible, else keep absolute Windows-safe path."""
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(repo_root)
    if os.path.splitdrive(abs_path)[0].lower() != os.path.splitdrive(abs_root)[0].lower():
        return abs_path
    return os.path.relpath(abs_path, abs_root)


def _resolve_repo_path(path: str, repo_root: str) -> str:
    """Resolve a config path relative to the repo root."""
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def _build_sidecar_loader_kwargs(job: Dict[str, Any], repo_root: str) -> Dict[str, Any]:
    """Build optional RTK/IMU sidecar loader kwargs from a CLI job."""
    kwargs: Dict[str, Any] = {}
    if job.get("trace_timestamps_s") is not None:
        kwargs["trace_timestamps_s"] = np.asarray(
            job["trace_timestamps_s"], dtype=np.float64
        )
    for field in ("rtk_path", "imu_path"):
        value = job.get(field)
        if value:
            kwargs[field] = _resolve_repo_path(str(value), repo_root)
    return kwargs


def _validate_trace_timestamps(value: Any) -> str | None:
    """Return an error message when trace_timestamps_s is not a finite 1D array."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return "trace_timestamps_s must be a numeric 1D list"
    if arr.ndim != 1 or arr.size == 0:
        return "trace_timestamps_s must be a non-empty 1D list"
    if not np.isfinite(arr).all():
        return "trace_timestamps_s must contain only finite numbers"
    return None


def validate_config(cfg: Dict[str, Any], repo_root: str) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    jobs = cfg.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        errors.append("config.jobs must be a non-empty list")
        return ValidationResult(False, errors, warnings)

    for i, job in enumerate(jobs):
        jid = job.get("id", f"job#{i}")
        input_path = job.get("input")
        benchmark_sample = job.get("benchmark_sample")
        if not input_path and not benchmark_sample:
            errors.append(f"[{jid}] missing input")
            continue
        if input_path:
            abs_input = _resolve_repo_path(str(input_path), repo_root)
            if not os.path.exists(abs_input):
                errors.append(f"[{jid}] input not found: {input_path}")
            for sidecar_field in ("rtk_path", "imu_path"):
                sidecar_path = job.get(sidecar_field)
                if sidecar_path:
                    abs_sidecar = _resolve_repo_path(str(sidecar_path), repo_root)
                    if not os.path.exists(abs_sidecar):
                        errors.append(
                            f"[{jid}] {sidecar_field} not found: {sidecar_path}"
                        )
            if job.get("trace_timestamps_s") is not None:
                timestamp_error = _validate_trace_timestamps(
                    job.get("trace_timestamps_s")
                )
                if timestamp_error:
                    errors.append(f"[{jid}] {timestamp_error}")
        if benchmark_sample and benchmark_sample not in list_benchmark_sample_ids():
            errors.append(f"[{jid}] unknown benchmark_sample: {benchmark_sample}")

        try:
            methods = _resolve_job_methods(job)
        except Exception as e:
            errors.append(f"[{jid}] {e}")
            continue

        for step_i, step in enumerate(methods):
            key = step.get("key")
            if key not in PROCESSING_METHODS:
                errors.append(f"[{jid}] step#{step_i} unknown method key: {key}")
                continue
            try:
                _merge_params(key, step.get("params"))
            except Exception as e:
                errors.append(f"[{jid}] step#{step_i} invalid params: {e}")

    return ValidationResult(len(errors) == 0, errors, warnings)


# ---------- Run pipeline ----------
def _get_core_func(module_name: str, func_name: str):
    """获取核心模块函数"""
    mod = __import__(module_name)
    return getattr(mod, func_name)


def _run_core_method(
    method_key: str,
    func_name: str,
    module_name: str,
    data: np.ndarray,
    params: Dict[str, Any],
    out_dir: str,
) -> np.ndarray:
    func = _get_core_func(module_name, func_name)

    length_trace = data.shape[0]
    start_position = 0
    end_position = data.shape[1]
    scans_per_meter = 1

    import tempfile

    temp_in_csv = os.path.join(out_dir, "temp_in.csv")
    temp_out_csv = os.path.join(out_dir, f"{method_key}_tmp_out.csv")
    temp_out_png = os.path.join(out_dir, f"{method_key}_tmp_out.png")
    savecsv(data, temp_in_csv)

    if method_key == "compensatingGain":
        gain_min = float(params.get("gain_min", 1.0))
        gain_max = float(params.get("gain_max", 6.0))
        gain_func = np.linspace(gain_min, gain_max, data.shape[0]).tolist()
        func(
            temp_in_csv,
            temp_out_csv,
            temp_out_png,
            length_trace,
            start_position,
            end_position,
            gain_func,
        )
    elif method_key == "dewow":
        window = int(params.get("window", max(1, length_trace // 4)))
        func(
            temp_in_csv,
            temp_out_csv,
            temp_out_png,
            length_trace,
            start_position,
            scans_per_meter,
            window,
        )
    elif method_key == "set_zero_time":
        new_zero_time = float(params.get("new_zero_time", 5.0))
        func(
            temp_in_csv,
            temp_out_csv,
            temp_out_png,
            length_trace,
            start_position,
            scans_per_meter,
            new_zero_time,
        )
    elif method_key == "agcGain":
        window = int(params.get("window", max(1, length_trace // 4)))
        func(
            temp_in_csv,
            temp_out_csv,
            temp_out_png,
            length_trace,
            start_position,
            scans_per_meter,
            window,
        )
    elif method_key == "subtracting_average_2D":
        ntraces = int(params.get("ntraces", 501))
        func(
            temp_in_csv,
            temp_out_csv,
            temp_out_png,
            length_trace,
            start_position,
            scans_per_meter,
            ntraces,
        )
    elif method_key == "running_average_2D":
        ntraces = int(params.get("ntraces", 9))
        func(
            temp_in_csv,
            temp_out_csv,
            temp_out_png,
            length_trace,
            start_position,
            scans_per_meter,
            ntraces,
        )
    else:
        raise ValueError(f"Unsupported core method in phase-1: {method_key}")

    out_df = pd.read_csv(temp_out_csv, header=None)
    out_data = out_df.values
    if out_data.ndim == 1:
        out_data = out_data.reshape(-1, 1)
    return out_data


def run_job(job: Dict[str, Any], repo_root: str, output_dir: str) -> Dict[str, Any]:
    benchmark_sample = job.get("benchmark_sample")
    if benchmark_sample:
        if benchmark_sample != "motion_compensation_v1":
            raise ValueError(f"unsupported benchmark_sample: {benchmark_sample}")
        if str(job.get("recommended_profile") or "") != "motion_compensation_v1":
            raise ValueError(
                "motion benchmark job requires recommended_profile=motion_compensation_v1"
            )

        summary = export_motion_compensation_benchmark(
            output_dir,
            sample_id=str(benchmark_sample),
            profile_key="motion_compensation_v1",
            seed=int(job.get("seed", 42)),
            save_images=True,
        )
        header_info = summary.get("header_info") or {}
        return {
            "job_id": job.get("id") or str(benchmark_sample),
            "benchmark_sample": str(benchmark_sample),
            "recommended_profile": job.get("recommended_profile"),
            "status": "ok",
            "steps": summary["steps"],
            "final_shape": [
                int(header_info.get("a_scan_length", 0)),
                int(header_info.get("num_traces", 0)),
            ],
            "before_png": _safe_relpath(summary["artifacts"]["before_png"], repo_root),
            "after_png": _safe_relpath(summary["artifacts"]["after_png"], repo_root),
            "difference_png": _safe_relpath(
                summary["artifacts"]["difference_png"], repo_root
            ),
            "motion_metrics_json": _safe_relpath(
                summary["artifacts"]["motion_metrics_json"], repo_root
            ),
            "corrected_trace_metadata_csv": _safe_relpath(
                summary["artifacts"]["corrected_trace_metadata_csv"], repo_root
            ),
            "summary_json": _safe_relpath(summary["summary_json"], repo_root),
            "objective_checks": summary["objective_checks"],
        }

    jid = job.get("id") or os.path.splitext(os.path.basename(job["input"]))[0]
    input_path = job["input"]
    abs_input = _resolve_repo_path(str(input_path), repo_root)

    job_out_dir = os.path.join(output_dir, jid)
    os.makedirs(job_out_dir, exist_ok=True)

    data, header_info, trace_metadata = load_gpr_csv(
        abs_input,
        **_build_sidecar_loader_kwargs(job, repo_root),
    )
    current = data
    current_header_info = merge_result_header_info(header_info, None, current.shape)
    current_trace_metadata = trace_metadata
    steps_summary: List[Dict[str, Any]] = []

    methods = _resolve_job_methods(job)
    for idx, step in enumerate(methods):
        key = step["key"]
        meta = PROCESSING_METHODS[key]
        params = _merge_params(key, step.get("params"))

        if meta["type"] == "core":
            new_data = _run_core_method(
                key, meta["func"], meta["module"], current, params, job_out_dir
            )
            current_header_info = merge_result_header_info(
                current_header_info,
                None,
                np.asarray(new_data).shape,
            )
        else:
            new_data, result_meta = run_processing_method(
                current,
                key,
                prepare_runtime_params(
                    key,
                    params,
                    current_header_info,
                    current_trace_metadata,
                    current.shape,
                ),
            )
            current_header_info = merge_result_header_info(
                current_header_info,
                result_meta,
                np.asarray(new_data).shape,
            )
            current_trace_metadata = merge_result_trace_metadata(
                current_trace_metadata,
                result_meta,
            )

        step_csv = os.path.join(job_out_dir, f"{idx:02d}_{key}.csv")
        step_png = os.path.join(job_out_dir, f"{idx:02d}_{key}.png")
        savecsv(new_data, step_csv)
        save_image(np.nan_to_num(new_data), step_png, title=f"{jid}:{key}", cmap="gray")

        steps_summary.append(
            {
                "step": idx,
                "key": key,
                "params": params,
                "output_csv": os.path.relpath(step_csv, repo_root),
                "output_png": os.path.relpath(step_png, repo_root),
                "shape": list(np.asarray(new_data).shape),
            }
        )
        current = np.asarray(new_data)

    final_csv = os.path.join(job_out_dir, "final.csv")
    final_png = os.path.join(job_out_dir, "final.png")
    savecsv(current, final_csv)
    save_image(np.nan_to_num(current), final_png, title=f"{jid}:final", cmap="gray")

    return {
        "job_id": jid,
        "input": input_path,
        "recommended_profile": job.get("recommended_profile"),
        "status": "ok",
        "steps": steps_summary,
        "final_csv": os.path.relpath(final_csv, repo_root),
        "final_png": os.path.relpath(final_png, repo_root),
        "final_shape": list(current.shape),
    }


def run_batch(cfg: Dict[str, Any], config_path: str, repo_root: str) -> int:
    output_dir_cfg = cfg.get("output_dir", "output/cli_batch")
    output_dir = (
        output_dir_cfg
        if os.path.isabs(output_dir_cfg)
        else os.path.join(repo_root, output_dir_cfg)
    )
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "config": os.path.relpath(config_path, repo_root),
        "output_dir": os.path.relpath(output_dir, repo_root),
        "results": [],
    }

    ok_count = 0
    fail_count = 0
    for job in cfg.get("jobs", []):
        jid = job.get("id", "<unknown>")
        try:
            result = run_job(job, repo_root=repo_root, output_dir=output_dir)
            summary["results"].append(result)
            ok_count += 1
            print(f"[OK] {jid}")
        except Exception as e:
            fail_count += 1
            summary["results"].append(
                {
                    "job_id": jid,
                    "input": job.get("input"),
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(limit=3),
                }
            )
            print(f"[FAIL] {jid}: {e}")

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["stats"] = {
        "ok": ok_count,
        "failed": fail_count,
        "total": ok_count + fail_count,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"summary_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary["stats"], ensure_ascii=False))
    print(f"summary_file: {os.path.relpath(summary_path, repo_root)}")

    return 0 if fail_count == 0 else 2


def cmd_validate(args) -> int:
    repo_root = os.path.abspath(args.repo_root)
    cfg = load_config(args.config)
    vr = validate_config(cfg, repo_root=repo_root)
    if vr.ok:
        print("validate: OK")
        if vr.warnings:
            for w in vr.warnings:
                print(f"[WARN] {w}")
        return 0

    print("validate: FAILED")
    for e in vr.errors:
        print(f"[ERR] {e}")
    for w in vr.warnings:
        print(f"[WARN] {w}")
    return 1


def cmd_run(args) -> int:
    repo_root = os.path.abspath(args.repo_root)
    cfg = load_config(args.config)
    vr = validate_config(cfg, repo_root=repo_root)
    if not vr.ok and not args.force:
        print("run blocked: config validation failed. Use --force to ignore.")
        for e in vr.errors:
            print(f"[ERR] {e}")
        return 1
    return run_batch(cfg, config_path=args.config, repo_root=repo_root)


def cmd_resume(args) -> int:
    print("resume: phase-1 placeholder (not implemented yet).")
    if args.summary:
        print(f"summary hint: {args.summary}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPR CLI batch MVP (phase-1)")
    sub = p.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="validate config and inputs")
    p_validate.add_argument("--config", required=True, help="path to batch config JSON")
    p_validate.add_argument(
        "--repo-root", default=BASE_DIR, help="repo root for relative paths"
    )
    p_validate.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="run batch jobs")
    p_run.add_argument("--config", required=True, help="path to batch config JSON")
    p_run.add_argument(
        "--repo-root", default=BASE_DIR, help="repo root for relative paths"
    )
    p_run.add_argument(
        "--force", action="store_true", help="run even when validation fails"
    )
    p_run.set_defaults(func=cmd_run)

    p_resume = sub.add_parser("resume", help="resume interface (phase-1 placeholder)")
    p_resume.add_argument("--summary", help="existing summary file (future use)")
    p_resume.set_defaults(func=cmd_resume)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
