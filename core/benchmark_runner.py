#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible benchmark runner skeleton for Stage A foundations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_registry import generate_benchmark_sample, get_benchmark_sample_spec
from .methods_registry import PROCESSING_METHODS, get_method_display_name
from .processing_engine import prepare_runtime_params, run_processing_method
from .quality_metrics import compute_benchmark_metrics
from read_file_data import save_image


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _default_params_for(method_key: str) -> dict[str, Any]:
    info = PROCESSING_METHODS[method_key]
    params = {}
    for spec in info.get("params", []):
        name = spec.get("name")
        if name:
            params[str(name)] = spec.get("default")
    return params


def _time_distance_ranges(header_info: dict[str, Any] | None, data: np.ndarray):
    if not header_info:
        return None, None
    total_time_ns = float(header_info.get("total_time_ns", data.shape[0]))
    trace_interval_m = float(header_info.get("trace_interval_m", 1.0))
    total_distance_m = trace_interval_m * max(data.shape[1] - 1, 1)
    return (0.0, total_time_ns), (0.0, total_distance_m)


def run_benchmark_sample(
    sample_id: str,
    method_keys: list[str] | None = None,
    out_dir: str | Path | None = None,
    seed: int = 42,
    save_images: bool = True,
) -> dict[str, Any]:
    """Run a deterministic benchmark sample through a method sequence."""
    spec = get_benchmark_sample_spec(sample_id)
    data, sample_meta = generate_benchmark_sample(sample_id, seed=seed)
    header_info = dict(sample_meta.get("header_info") or {})
    trace_metadata = None
    time_range, distance_range = _time_distance_ranges(header_info, data)

    run_methods = list(method_keys or spec.default_methods)
    output_root = Path(out_dir) if out_dir is not None else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "sample_id": sample_id,
        "sample_title": spec.title,
        "scenario": spec.scenario,
        "seed": int(seed),
        "shape": list(data.shape),
        "header_info": _to_jsonable(header_info),
        "focus_metrics": list(spec.focus_metrics),
        "methods": run_methods,
        "steps": [],
    }

    if output_root is not None and save_images:
        raw_png = output_root / f"{sample_id}-00-raw.png"
        save_image(
            data,
            str(raw_png),
            title=f"{spec.title} - raw",
            time_range=time_range,
            distance_range=distance_range,
        )
        summary["raw_png"] = str(raw_png)

    current = data
    zero_idx = sample_meta.get("reference_zero_idx")
    for idx, method_key in enumerate(run_methods, start=1):
        params = _default_params_for(method_key)
        runtime_params = prepare_runtime_params(
            method_key,
            params,
            header_info,
            trace_metadata,
            current.shape,
        )
        started = time.perf_counter()
        result, meta = run_processing_method(current, method_key, runtime_params)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics = compute_benchmark_metrics(current, result, zero_idx=zero_idx)

        step_record: dict[str, Any] = {
            "step_index": idx,
            "method_key": method_key,
            "method_name": get_method_display_name(method_key),
            "params": _to_jsonable(runtime_params),
            "elapsed_ms": float(elapsed_ms),
            "metrics": metrics,
            "runtime_meta": _to_jsonable(meta),
        }
        if output_root is not None and save_images:
            out_png = output_root / f"{sample_id}-{idx:02d}-{method_key}.png"
            save_image(
                result,
                str(out_png),
                title=f"{spec.title} - {method_key}",
                time_range=time_range,
                distance_range=distance_range,
            )
            step_record["output_png"] = str(out_png)

        summary["steps"].append(step_record)
        current = result

    if output_root is not None:
        final_json = output_root / f"{sample_id}-summary.json"
        final_json.write_text(
            json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["summary_json"] = str(final_json)

    return _to_jsonable(summary)
