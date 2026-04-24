#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable chain-evidence export helpers for benchmark and validation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.benchmark_registry import generate_benchmark_sample, get_benchmark_sample_spec
from core.processing_engine import prepare_runtime_params, run_processing_method
from read_file_data import save_image


STANDARD_CHAIN_SPECS: dict[str, dict[str, Any]] = {
    "conservative_default": {
        "label": "保守默认链",
        "description": "零时矫正 → 低频漂移抑制 → 背景抑制 → SEC增益",
        "steps": [
            ("set_zero_time", {"new_zero_time": 5.0}),
            ("dewow", {"window": 41}),
            ("subtracting_average_2D", {"ntraces": 501}),
            ("sec_gain", {"gain_min": 1.0, "gain_max": 4.5, "power": 1.1}),
        ],
    },
    "aggressive_gain": {
        "label": "激进增强链",
        "description": "零时矫正 → 低频漂移抑制 → 背景抑制 → AGC增益 → 尖锐杂波抑制",
        "steps": [
            ("set_zero_time", {"new_zero_time": 5.0}),
            ("dewow", {"window": 41}),
            ("subtracting_average_2D", {"ntraces": 501}),
            ("agcGain", {"window": 11}),
            ("running_average_2D", {"ntraces": 9}),
        ],
    },
}


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


def _time_distance_ranges(header_info: dict[str, Any] | None, data: np.ndarray):
    if not header_info:
        return None, None
    total_time_ns = float(header_info.get("total_time_ns", data.shape[0]))
    trace_interval_m = float(header_info.get("trace_interval_m", 1.0))
    total_distance_m = trace_interval_m * max(data.shape[1] - 1, 1)
    return (0.0, total_time_ns), (0.0, total_distance_m)


def _save_comparison(
    raw: np.ndarray, final: np.ndarray, out_path: Path, right_title: str
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(raw, cmap="gray", aspect="auto")
    axes[0].set_title("Raw")
    axes[0].set_xlabel("Trace")
    axes[0].set_ylabel("Sample")
    axes[1].imshow(final, cmap="gray", aspect="auto")
    axes[1].set_title(right_title)
    axes[1].set_xlabel("Trace")
    axes[1].set_ylabel("Sample")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def export_chain_evidence(
    *,
    data: np.ndarray,
    header_info: dict[str, Any] | None,
    bundle_name: str,
    chain_name: str,
    chain_description: str,
    steps: list[tuple[str, dict[str, Any]]],
    out_dir: str | Path,
    title_prefix: str,
    save_images: bool = True,
    trace_metadata: Any = None,
) -> dict[str, Any]:
    """Run a fixed processing chain and export reusable evidence artifacts."""
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=np.float32)
    time_range, distance_range = _time_distance_ranges(header_info, arr)

    summary: dict[str, Any] = {
        "bundle_name": bundle_name,
        "chain_name": chain_name,
        "chain_description": chain_description,
        "shape": list(arr.shape),
        "header_info": _to_jsonable(header_info),
        "steps": [],
    }

    if save_images:
        raw_png = output_root / f"{bundle_name}-00-raw.png"
        save_image(
            arr,
            str(raw_png),
            title=f"{title_prefix} - Raw",
            time_range=time_range,
            distance_range=distance_range,
        )
        summary["raw_png"] = str(raw_png)

    current = arr
    for idx, (method_key, params) in enumerate(steps, start=1):
        runtime_params = prepare_runtime_params(
            method_key,
            dict(params),
            header_info,
            trace_metadata,
            current.shape,
        )
        result, meta = run_processing_method(current, method_key, runtime_params)
        step_record: dict[str, Any] = {
            "step_index": idx,
            "method_key": method_key,
            "params": _to_jsonable(runtime_params),
            "runtime_meta": _to_jsonable(meta),
        }
        if save_images:
            out_png = output_root / f"{bundle_name}-{idx:02d}-{method_key}.png"
            save_image(
                result,
                str(out_png),
                title=f"{title_prefix} - {method_key}",
                time_range=time_range,
                distance_range=distance_range,
            )
            step_record["output_png"] = str(out_png)
        summary["steps"].append(step_record)
        current = np.asarray(result, dtype=np.float32)

    if save_images:
        comparison_png = output_root / f"{bundle_name}-raw-vs-final.png"
        _save_comparison(arr, current, comparison_png, chain_name)
        summary["comparison_png"] = str(comparison_png)

    summary_json = output_root / f"{bundle_name}-summary.json"
    summary_json.write_text(
        json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_json)
    return _to_jsonable(summary)


def export_standard_chain_for_sample(
    sample_id: str,
    chain_key: str,
    out_dir: str | Path,
    *,
    seed: int = 42,
    save_images: bool = True,
) -> dict[str, Any]:
    """Run one named standard chain against one registered benchmark sample."""
    if chain_key not in STANDARD_CHAIN_SPECS:
        raise KeyError(f"未知标准链: {chain_key}")
    spec = get_benchmark_sample_spec(sample_id)
    chain = STANDARD_CHAIN_SPECS[chain_key]
    data, sample_meta = generate_benchmark_sample(sample_id, seed=seed)
    header_info = dict(sample_meta.get("header_info") or {})
    bundle_name = f"{sample_id}-{chain_key}"
    summary = export_chain_evidence(
        data=data,
        header_info=header_info,
        bundle_name=bundle_name,
        chain_name=str(chain["label"]),
        chain_description=str(chain["description"]),
        steps=list(chain["steps"]),
        out_dir=out_dir,
        title_prefix=f"{spec.title} - {chain['label']}",
        save_images=save_images,
    )
    summary["sample_id"] = sample_id
    summary["sample_title"] = spec.title
    summary["scenario"] = spec.scenario
    summary["seed"] = int(seed)
    Path(summary["summary_json"]).write_text(
        json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _to_jsonable(summary)
