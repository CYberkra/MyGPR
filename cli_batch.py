#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPR CLI Batch MVP (Phase-1) - 使用统一方法注册表

Scope:
- validate: config + inputs + method params validation
- run: sequential batch processing with minimal summary output
- resume: re-run failed jobs from an existing summary JSON

Design goals:
- 使用 methods_registry 统一方法定义，避免重复
- Keep minimal, runnable CLI main path
- Avoid report engine coupling in phase-1
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Package/root discovery is handled by Python; do not mutate sys.path.
from core.benchmark_registry import list_benchmark_sample_ids
from core.evidence_export import export_motion_compensation_benchmark
from core.gpr_io import extract_airborne_csv_payload, savecsv, save_image
from core.processing_engine import (
    merge_result_header_info,
    merge_result_trace_metadata,
)
from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor

# 使用统一的方法注册表
from core.methods_registry import (
    PROCESSING_METHODS,
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
    altimeter_path: str | None = None,
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
        if altimeter_path is not None:
            sidecar_kwargs["altimeter_path"] = altimeter_path
        data, trace_metadata, header_info = extract_airborne_csv_payload(
            raw_data,
            header_info,
            **sidecar_kwargs,
        )
    else:
        if rtk_path is not None or imu_path is not None or altimeter_path is not None:
            raise ValueError("RTK/IMU/altimeter sidecars require an airborne CSV header")
        data = raw_data

    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=float(np.nanmean(data)))

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return np.asarray(data, dtype=float), header_info, trace_metadata


OPTIONAL_METHOD_DEPENDENCIES = {
    "wavelet_2d": [("pywt", "PyWavelets")],
    "wavelet_svd": [("pywt", "PyWavelets")],
}


def _missing_optional_dependencies(method_key: str) -> list[str]:
    missing: list[str] = []
    for module_name, package_name in OPTIONAL_METHOD_DEPENDENCIES.get(method_key, []):
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


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


def _resolve_job_methods(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    methods = job.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError(
            "job 必须提供非空 methods 列表（recommended_profile 预设档已在 0.9.38 移除，"
            "请显式给出逐步算法与参数）"
        )
    return methods


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
    for field in ("rtk_path", "imu_path", "altimeter_path"):
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
        if job.get("recommended_profile") is not None:
            errors.append(
                f"[{jid}] recommended_profile 已在 0.9.38 移除：请改用 methods 列表"
                "（每项 {{'key': 算法, 'params': {{...}}}}）"
            )
            continue
        if not input_path and not benchmark_sample:
            errors.append(f"[{jid}] missing input")
            continue
        if input_path:
            abs_input = _resolve_repo_path(str(input_path), repo_root)
            if not os.path.exists(abs_input):
                errors.append(f"[{jid}] input not found: {input_path}")
            for sidecar_field in ("rtk_path", "imu_path", "altimeter_path"):
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
            missing_deps = _missing_optional_dependencies(str(key))
            if missing_deps:
                warnings.append(
                    f"[{jid}] step#{step_i} method '{key}' requires missing package(s): {', '.join(missing_deps)}"
                )
            try:
                _merge_params(key, step.get("params"))
            except Exception as e:
                errors.append(f"[{jid}] step#{step_i} invalid params: {e}")

    return ValidationResult(len(errors) == 0, errors, warnings)


# ---------- Run pipeline ----------
_NATIVE_EXECUTOR = NativeProcessingExecutor()


def _sanitize_for_native(data: np.ndarray) -> np.ndarray:
    """Native 执行器不做输入清洗；保留旧 legacy 引擎的防御性 NaN 处理。"""
    arr = np.asarray(data)
    if np.isfinite(arr).all():
        return arr
    finite = np.isfinite(arr)
    fill = float(np.mean(arr[finite])) if finite.any() else 0.0
    return np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)


def run_job(job: Dict[str, Any], repo_root: str, output_dir: str) -> Dict[str, Any]:
    benchmark_sample = job.get("benchmark_sample")
    if benchmark_sample:
        if benchmark_sample != "motion_compensation_v1":
            raise ValueError(f"unsupported benchmark_sample: {benchmark_sample}")
        summary = export_motion_compensation_benchmark(
            output_dir,
            sample_id=str(benchmark_sample),
            seed=int(job.get("seed", 42)),
            save_images=True,
        )
        header_info = summary.get("header_info") or {}
        return {
            "job_id": job.get("id") or str(benchmark_sample),
            "benchmark_sample": str(benchmark_sample),
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
        params = _merge_params(key, step.get("params"))

        result = _NATIVE_EXECUTOR.execute(
            ProcessingRequest(
                data=_sanitize_for_native(current),
                method_id=key,
                params=params,
                header_info=current_header_info,
                trace_metadata=current_trace_metadata,
            )
        )
        new_data = np.asarray(result.data)
        current_header_info = merge_result_header_info(
            current_header_info,
            result.metadata,
            new_data.shape,
        )
        current_trace_metadata = merge_result_trace_metadata(
            current_trace_metadata,
            result.metadata,
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

    result = {
        "job_id": jid,
        "input": input_path,
        "status": "ok",
        "steps": steps_summary,
        "final_csv": os.path.relpath(final_csv, repo_root),
        "final_png": os.path.relpath(final_png, repo_root),
        "final_shape": list(current.shape),
    }
    return result


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

    summary_path = _build_summary_path(output_dir)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary["stats"], ensure_ascii=False))
    print(f"summary_file: {os.path.relpath(summary_path, repo_root)}")

    return 0 if fail_count == 0 else 2


def _build_summary_path(output_dir: str) -> str:
    """Build a collision-resistant summary path for repeated batch runs."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = uuid.uuid4().hex[:8]
    return os.path.join(output_dir, f"summary_{ts}_{suffix}.json")


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
    repo_root = os.path.abspath(args.repo_root)
    if not args.summary:
        print("ERROR: --summary is required for resume.")
        return 2
    summary_path = _resolve_repo_path(str(args.summary), repo_root)
    if not os.path.isfile(summary_path):
        print(f"ERROR: summary file not found: {summary_path}")
        return 2
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            previous = json.load(f)
    except Exception as exc:
        print(f"ERROR: failed to read summary JSON: {exc}")
        return 2

    failed_ids = [
        str(item.get("job_id"))
        for item in previous.get("results", [])
        if isinstance(item, dict) and str(item.get("status", "")).lower() not in {"ok", "success"}
    ]
    failed_ids = [item for item in failed_ids if item and item != "<unknown>"]
    if not failed_ids:
        print("resume: no failed jobs found in summary.")
        return 0

    config_ref = previous.get("config")
    if not config_ref:
        print("ERROR: summary does not contain a config path; cannot resume safely.")
        return 2
    config_path = _resolve_repo_path(str(config_ref), repo_root)
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        print(f"ERROR: failed to load original config: {exc}")
        return 2

    failed_set = set(failed_ids)
    jobs = [job for job in cfg.get("jobs", []) if str(job.get("id") or os.path.splitext(os.path.basename(str(job.get("input", ""))))[0]) in failed_set]
    if not jobs:
        print("ERROR: failed job ids were not found in the original config: " + ", ".join(failed_ids))
        return 2

    output_dir_ref = previous.get("output_dir") or cfg.get("output_dir", "output/cli_batch")
    output_dir = _resolve_repo_path(str(output_dir_ref), repo_root)
    os.makedirs(output_dir, exist_ok=True)
    resumed = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "resume_of": os.path.relpath(summary_path, repo_root),
        "config": os.path.relpath(config_path, repo_root),
        "output_dir": os.path.relpath(output_dir, repo_root),
        "resumed_job_ids": failed_ids,
        "results": [],
    }
    ok_count = 0
    fail_count = 0
    for job in jobs:
        jid = str(job.get("id") or os.path.splitext(os.path.basename(str(job.get("input", ""))))[0])
        try:
            result = run_job(job, repo_root=repo_root, output_dir=output_dir)
            resumed["results"].append(result)
            ok_count += 1
            print(f"[RESUMED OK] {jid}")
        except Exception as exc:
            fail_count += 1
            resumed["results"].append({
                "job_id": jid,
                "input": job.get("input"),
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(limit=3),
            })
            print(f"[RESUMED FAIL] {jid}: {exc}")
    resumed["finished_at"] = datetime.now().isoformat(timespec="seconds")
    resumed["stats"] = {"ok": ok_count, "failed": fail_count, "total": ok_count + fail_count}
    resumed_path = _build_summary_path(output_dir)
    with open(resumed_path, "w", encoding="utf-8") as f:
        json.dump(resumed, f, ensure_ascii=False, indent=2)
    print("\n=== Resume Summary ===")
    print(json.dumps(resumed["stats"], ensure_ascii=False))
    print(f"summary_file: {os.path.relpath(resumed_path, repo_root)}")
    return 0 if fail_count == 0 else 2


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

    p_resume = sub.add_parser("resume", help="rerun failed jobs from an existing summary")
    p_resume.add_argument("--summary", required=True, help="existing summary file")
    p_resume.add_argument(
        "--repo-root", default=BASE_DIR, help="repo root for relative paths"
    )
    p_resume.set_defaults(func=cmd_resume)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
