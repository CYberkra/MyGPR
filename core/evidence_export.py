#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable chain-evidence export helpers for benchmark and validation runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.benchmark_registry import generate_benchmark_sample, get_benchmark_sample_spec
from core.preset_profiles import RECOMMENDED_RUN_PROFILES
from core.processing_engine import (
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.quality_metrics import compute_motion_quality_metrics
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


def _summarize_trace_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict) or not metadata:
        return {"field_count": 0, "fields": [], "trace_count": 0}

    field_shapes: dict[str, list[int]] = {}
    field_dtypes: dict[str, str] = {}
    trace_count = 0
    for key, value in metadata.items():
        arr = np.asarray(value)
        field_shapes[str(key)] = [int(dim) for dim in arr.shape]
        field_dtypes[str(key)] = str(arr.dtype)
        if arr.ndim > 0 and arr.size > 1:
            trace_count = max(trace_count, int(arr.shape[0]))

    return {
        "field_count": len(metadata),
        "fields": sorted(str(key) for key in metadata.keys()),
        "trace_count": trace_count,
        "field_shapes": field_shapes,
        "field_dtypes": field_dtypes,
    }


def _summarize_runtime_context(runtime_params: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if "header_info" in runtime_params:
        context["header_info"] = _to_jsonable(runtime_params["header_info"])
    if "trace_metadata" in runtime_params:
        context["trace_metadata"] = _summarize_trace_metadata(
            runtime_params["trace_metadata"]
        )
    if "time_window_ns" in runtime_params:
        context["time_window_ns"] = _to_jsonable(runtime_params["time_window_ns"])
    return context


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


def _default_params_for(method_key: str) -> dict[str, Any]:
    profile = RECOMMENDED_RUN_PROFILES.get("motion_compensation_v1") or {}
    return dict((profile.get("method_params") or {}).get(method_key) or {})


def _save_trace_metadata_csv(
    trace_metadata: dict[str, Any],
    out_path: Path,
) -> None:
    if not trace_metadata:
        out_path.write_text("", encoding="utf-8")
        return

    normalized: dict[str, np.ndarray] = {}
    row_count = 0
    for key, value in trace_metadata.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = np.full(1, arr.item())
        arr = arr.reshape(-1)
        normalized[str(key)] = arr
        row_count = max(row_count, int(arr.size))

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        fieldnames = list(normalized.keys())
        writer.writerow(fieldnames)
        for row_idx in range(row_count):
            row: list[Any] = []
            for key in fieldnames:
                arr = normalized[key]
                row.append(arr[row_idx].item() if row_idx < arr.size else "")
            writer.writerow(row)


def _resample_columns_linear(data: np.ndarray, target_traces: int) -> np.ndarray:
    """Resample columns onto a target trace count for visualization-only diffs."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] == target_traces:
        return np.array(arr, copy=True)

    source_axis = np.linspace(0.0, 1.0, arr.shape[1], dtype=np.float32)
    target_axis = np.linspace(0.0, 1.0, target_traces, dtype=np.float32)
    resampled = np.empty((arr.shape[0], target_traces), dtype=np.float32)
    for row_idx in range(arr.shape[0]):
        resampled[row_idx, :] = np.interp(target_axis, source_axis, arr[row_idx, :])
    return resampled


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
            "params": _to_jsonable(params),
            "runtime_context": _summarize_runtime_context(runtime_params),
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


def export_motion_compensation_benchmark(
    out_dir: str | Path,
    *,
    sample_id: str = "motion_compensation_v1",
    profile_key: str = "motion_compensation_v1",
    seed: int = 42,
    save_images: bool = True,
) -> dict[str, Any]:
    """Run the deterministic motion-compensation benchmark and export evidence."""
    if sample_id != "motion_compensation_v1":
        raise ValueError(f"unsupported motion benchmark sample: {sample_id}")
    if profile_key != "motion_compensation_v1":
        raise ValueError(f"unsupported motion benchmark profile: {profile_key}")

    spec = get_benchmark_sample_spec(sample_id)
    profile = RECOMMENDED_RUN_PROFILES.get(profile_key)
    if not profile:
        raise KeyError(f"未知 motion profile: {profile_key}")

    raw, sample_meta = generate_benchmark_sample(sample_id, seed=seed)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    header_info = dict(sample_meta.get("header_info") or {})
    trace_metadata: dict[str, np.ndarray] = {
        str(key): np.array(value, copy=True)
        for key, value in (sample_meta.get("trace_metadata") or {}).items()
    }
    ground_truth_trace_metadata: dict[str, object] = dict(
        sample_meta.get("ground_truth_trace_metadata") or {}
    )
    metric_config = dict((sample_meta.get("expected_metrics") or {}).get("metric_config") or {})
    raw_time_range, raw_distance_range = _time_distance_ranges(header_info, raw)

    ridge_row_range = tuple(metric_config.get("ridge_row_range") or [])
    target_row_range = tuple(metric_config.get("target_row_range") or [])
    banding_trace_band_values = tuple(metric_config.get("banding_trace_band") or (0.05, 0.18))
    banding_trace_band: tuple[float, float] = (
        float(banding_trace_band_values[0]),
        float(banding_trace_band_values[1]),
    )
    banding_row_range = tuple(metric_config.get("banding_row_range") or [])

    raw_metrics = compute_motion_quality_metrics(
        raw,
        cast(dict[str, object], trace_metadata),
        ground_truth_trace_metadata,
        ground_truth_data=sample_meta.get("ground_truth_data"),
        ridge_row_range=ridge_row_range,
        target_row_range=target_row_range,
        banding_trace_band=banding_trace_band,
        banding_row_range=banding_row_range,
    )

    current = np.asarray(raw, dtype=np.float32)
    steps_summary: list[dict[str, Any]] = []
    for idx, method_key in enumerate(profile.get("order", []), start=1):
        params = _default_params_for(method_key)
        runtime_params = prepare_runtime_params(
            method_key,
            params,
            header_info,
            trace_metadata,
            current.shape,
        )
        current, runtime_meta = run_processing_method(current, method_key, runtime_params)
        header_info = merge_result_header_info(header_info, runtime_meta, current.shape)
        trace_metadata = merge_result_trace_metadata(trace_metadata, runtime_meta)
        steps_summary.append(
            {
                "step_index": idx,
                "method_key": method_key,
                "params": _to_jsonable(params),
                "runtime_context": _summarize_runtime_context(runtime_params),
                "runtime_meta": _to_jsonable(runtime_meta),
                "shape": list(np.asarray(current).shape),
            }
        )

    final_metrics = compute_motion_quality_metrics(
        current,
        cast(dict[str, object], trace_metadata),
        ground_truth_trace_metadata,
        ground_truth_data=sample_meta.get("ground_truth_data"),
        ridge_row_range=ridge_row_range,
        target_row_range=target_row_range,
        banding_trace_band=banding_trace_band,
        banding_row_range=banding_row_range,
    )
    objective_checks = {
        "ridge_improved": final_metrics["raw_ridge_rmse_samples"] < raw_metrics["raw_ridge_rmse_samples"],
        "trace_spacing_improved": final_metrics["trace_spacing_std_m"] < raw_metrics["trace_spacing_std_m"],
        "path_improved": final_metrics["path_rmse_m"] < raw_metrics["path_rmse_m"],
        "footprint_improved": final_metrics["footprint_rmse_m"] < raw_metrics["footprint_rmse_m"],
        "banding_improved": final_metrics["periodic_banding_ratio"] < raw_metrics["periodic_banding_ratio"],
        "target_preserved_or_improved": final_metrics["target_preservation_ratio"] >= raw_metrics["target_preservation_ratio"],
    }
    final_time_range, final_distance_range = _time_distance_ranges(header_info, current)

    if save_images:
        save_image(
            raw,
            str(output_root / "before.png"),
            title=f"{spec.title} - before",
            time_range=raw_time_range,
            distance_range=raw_distance_range,
        )
        save_image(
            current,
            str(output_root / "after.png"),
            title=f"{spec.title} - after",
            time_range=final_time_range,
            distance_range=final_distance_range,
        )
        raw_for_difference = _resample_columns_linear(raw, current.shape[1])
        save_image(
            np.asarray(current, dtype=np.float32) - raw_for_difference,
            str(output_root / "difference.png"),
            title=f"{spec.title} - difference",
            time_range=final_time_range,
            distance_range=final_distance_range,
            cmap="seismic",
        )

    corrected_trace_metadata_csv = output_root / "corrected_trace_metadata.csv"
    _save_trace_metadata_csv(trace_metadata, corrected_trace_metadata_csv)

    motion_metrics = {
        "sample_id": sample_id,
        "sample_title": spec.title,
        "profile_key": profile_key,
        "seed": int(seed),
        "metric_config": _to_jsonable(metric_config),
        "raw_metrics": _to_jsonable(raw_metrics),
        "final_metrics": _to_jsonable(final_metrics),
        "objective_checks": objective_checks,
    }
    motion_metrics_json = output_root / "motion_metrics.json"
    motion_metrics_json.write_text(
        json.dumps(_to_jsonable(motion_metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "sample_id": sample_id,
        "sample_title": spec.title,
        "profile_key": profile_key,
        "profile_label": profile.get("label"),
        "seed": int(seed),
        "header_info": _to_jsonable(header_info),
        "steps": steps_summary,
        "artifacts": {
            "before_png": str(output_root / "before.png"),
            "after_png": str(output_root / "after.png"),
            "difference_png": str(output_root / "difference.png"),
            "corrected_trace_metadata_csv": str(corrected_trace_metadata_csv),
            "motion_metrics_json": str(motion_metrics_json),
        },
        "objective_checks": objective_checks,
    }
    summary_json = output_root / f"{sample_id}-summary.json"
    summary_json.write_text(
        json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["summary_json"] = str(summary_json)
    summary["raw_metrics"] = _to_jsonable(raw_metrics)
    summary["final_metrics"] = _to_jsonable(final_metrics)
    summary["corrected_trace_metadata"] = _to_jsonable(trace_metadata)
    summary["raw_data"] = _to_jsonable(raw)
    summary["final_data"] = _to_jsonable(current)
    return _to_jsonable(summary)
