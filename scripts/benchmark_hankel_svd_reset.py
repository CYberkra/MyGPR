#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Hankel-SVD reset benchmark and evidence artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PythonModule.hankel_svd import method_hankel_svd


LEGACY_BENCHMARK_CASES = (
    {"name": "small", "shape": (125, 256), "seed": 7, "window_length": 31, "rank": 5},
    {"name": "medium", "shape": (251, 1024), "seed": 17, "window_length": 63, "rank": 5},
    {"name": "large", "shape": (501, 3081), "seed": 42, "window_length": 125, "rank": 5},
)


@dataclass(frozen=True)
class EvidenceCase:
    """Synthetic benchmark case with optional clean reference and target mask."""

    name: str
    clean: np.ndarray | None
    noisy: np.ndarray
    target_mask: np.ndarray | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture deterministic Hankel-SVD reset benchmark evidence."
    )
    parser.add_argument(
        "--mode",
        default="legacy",
        choices=["legacy", "rewritten", "compare"],
        help="Benchmark mode: legacy baseline, rewritten evidence, or comparison bundle.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / ".sisyphus" / "evidence" / "hankel-svd-reset" / "legacy"),
        help="Directory for JSON, CSV, and PNG benchmark artifacts.",
    )
    return parser.parse_args()


def _build_synthetic_matrix(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows, cols = shape
    time_axis = np.linspace(0.0, 1.0, rows, dtype=np.float64)[:, None]
    trace_axis = np.linspace(0.0, 1.0, cols, dtype=np.float64)[None, :]

    low_rank_signal = (
        np.sin(2.0 * np.pi * (3.0 * time_axis + 0.5 * trace_axis))
        + 0.35 * np.cos(2.0 * np.pi * (8.0 * time_axis - 0.25 * trace_axis))
        + 0.2 * np.sin(2.0 * np.pi * (time_axis @ (1.0 + 2.0 * trace_axis)))
    )
    trend = 0.15 * time_axis + 0.05 * trace_axis
    noise = rng.normal(loc=0.0, scale=0.08, size=shape)
    return (low_rank_signal + trend + noise).astype(np.float32, copy=False)


def _expand_vertical_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    expanded = mask.copy()
    for shift in range(1, radius + 1):
        expanded[shift:, :] |= mask[:-shift, :]
        expanded[:-shift, :] |= mask[shift:, :]
    return expanded


def _build_point_target_hyperbola_case() -> EvidenceCase:
    np.random.seed(7)
    samples = 96
    traces = 64
    sample_axis = np.arange(samples, dtype=np.float64)
    trace_axis = np.arange(traces, dtype=np.float64)
    trace_center = (traces - 1) / 2.0

    clean = np.zeros((samples, traces), dtype=np.float64)
    for col, lateral in enumerate(trace_axis):
        arrival = 22.0 + 0.055 * (lateral - trace_center) ** 2
        clean[:, col] += 1.40 * np.exp(-0.5 * ((sample_axis - arrival) / 1.5) ** 2)
        clean[:, col] -= 0.50 * np.exp(-0.5 * ((sample_axis - (arrival + 3.2)) / 2.2) ** 2)

    coherent_noise = 0.22 * np.sin(2.0 * np.pi * sample_axis / 13.0)[:, None]
    white_noise = 0.28 * np.random.normal(size=clean.shape)
    noisy = clean + coherent_noise + white_noise
    target_mask = _expand_vertical_mask(np.abs(clean) > 0.05, radius=3)
    return EvidenceCase("point_target_hyperbola", clean, noisy, target_mask)


def _build_horizontal_layer_case() -> EvidenceCase:
    np.random.seed(11)
    samples = 96
    traces = 64
    sample_axis = np.arange(samples, dtype=np.float64)
    trace_axis = np.arange(traces, dtype=np.float64)

    clean = np.zeros((samples, traces), dtype=np.float64)
    clean[40:43, :] += np.array([[0.60], [1.25], [0.65]], dtype=np.float64)
    clean[68:70, :] -= np.array([[0.35], [0.22]], dtype=np.float64)

    horizontal_banding = 0.20 * np.sin(2.0 * np.pi * sample_axis / 9.0)[:, None]
    lateral_trend = 0.08 * np.cos(2.0 * np.pi * trace_axis / 17.0)[None, :]
    white_noise = 0.20 * np.random.normal(size=clean.shape)
    noisy = clean + horizontal_banding + lateral_trend + white_noise
    target_mask = _expand_vertical_mask(np.abs(clean) > 0.05, radius=3)
    return EvidenceCase("horizontal_layer", clean, noisy, target_mask)


def _build_correlated_noise_case(correlation_length: int, seed: int) -> EvidenceCase:
    np.random.seed(seed)
    samples = 96
    traces = 64
    clean = np.zeros((samples, traces), dtype=np.float64)
    clean[24:27, 20:44] += np.array([[0.20], [0.80], [0.20]], dtype=np.float64)
    clean[57:60, 10:30] -= np.array([[0.12], [0.40], [0.12]], dtype=np.float64)

    white_noise = np.random.normal(size=(samples, traces + correlation_length - 1))
    kernel = np.ones(correlation_length, dtype=np.float64) / float(correlation_length)
    correlated_noise = np.array(
        [np.convolve(row, kernel, mode="valid") for row in white_noise],
        dtype=np.float64,
    )
    noisy = clean + 0.38 * correlated_noise
    return EvidenceCase(f"correlated_noise_len{correlation_length}", clean, noisy)


def _build_flat_case() -> EvidenceCase:
    matrix = np.full((64, 24), 3.5, dtype=np.float64)
    return EvidenceCase("flat_matrix", matrix, matrix.copy())


def _build_nan_case() -> EvidenceCase:
    clean = np.tile(np.linspace(-1.0, 1.0, 24, dtype=np.float64), (48, 1))
    noisy = clean.copy()
    noisy[17, 9] = np.nan
    return EvidenceCase("nan_contaminated", clean, noisy)


def _build_short_case() -> EvidenceCase:
    matrix = np.array([[1.0, -1.0, 0.5, 0.25]], dtype=np.float64)
    return EvidenceCase("very_short_trace", matrix, matrix.copy())


def _evidence_cases() -> list[EvidenceCase]:
    return [
        _build_point_target_hyperbola_case(),
        _build_horizontal_layer_case(),
        _build_correlated_noise_case(correlation_length=10, seed=17),
        _build_correlated_noise_case(correlation_length=20, seed=19),
        _build_flat_case(),
        _build_nan_case(),
        _build_short_case(),
    ]


def _checksum_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _compute_snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    signal_norm = float(np.linalg.norm(reference))
    residual_norm = float(np.linalg.norm(reference - estimate))
    if residual_norm == 0.0:
        return float("inf")
    if signal_norm == 0.0:
        return float("-inf")
    return 20.0 * np.log10(signal_norm / residual_norm)


def _compute_rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reference - estimate) ** 2)))


def _target_preservation_ratio(
    clean: np.ndarray,
    estimate: np.ndarray,
    target_mask: np.ndarray | None,
) -> float | None:
    if target_mask is None:
        return None
    clean_energy = float(np.linalg.norm(clean[target_mask]))
    if clean_energy == 0.0:
        return None
    return float(np.linalg.norm(estimate[target_mask]) / clean_energy)


def _proxy_noise_metric(matrix: np.ndarray) -> float:
    finite = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if finite.shape[0] <= 1:
        return 0.0
    return float(np.std(np.diff(finite, axis=0)))


def _legacy_hankel(trace: np.ndarray, window_length: int) -> np.ndarray:
    """Old-style fixed-window Hankel embedding for benchmark comparison only."""
    arr = np.nan_to_num(np.asarray(trace, dtype=np.float64).reshape(-1), nan=0.0)
    if arr.size <= 1:
        return arr.reshape(1, -1)
    window = int(np.clip(window_length, 1, arr.size - 1))
    rows = arr.size - window + 1
    return np.vstack([arr[start : start + window] for start in range(rows)])


def _legacy_anti_diagonal_average(hankel: np.ndarray, original_length: int) -> np.ndarray:
    """Old implementation-style anti-diagonal averaging recovery."""
    target_length = max(int(original_length), 0)
    recovered = np.zeros(target_length, dtype=np.float64)
    counts = np.zeros(target_length, dtype=np.float64)
    rows, cols = hankel.shape
    for row in range(rows):
        for col in range(cols):
            index = row + col
            if index < target_length:
                recovered[index] += hankel[row, col]
                counts[index] += 1.0
    counts[counts == 0.0] = 1.0
    return recovered / counts


def _legacy_like_denoise(data: np.ndarray, window_length: int = 32, rank: int = 5) -> np.ndarray:
    """Reproduce the old fixed-parameter anti-diagonal baseline for evidence.

    This is intentionally local to the evidence script because production code now
    points to the rewritten method. It enables same-corpus quality comparison
    without reviving the old public implementation.
    """
    arr = np.asarray(data, dtype=np.float64)
    result = np.zeros_like(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={arr.shape}")
    for col in range(arr.shape[1]):
        trace = np.nan_to_num(arr[:, col], nan=0.0, posinf=0.0, neginf=0.0)
        if trace.size <= 1:
            result[:, col] = trace
            continue
        hankel = _legacy_hankel(trace, window_length)
        U, singular_values, Vt = np.linalg.svd(hankel, full_matrices=False)
        effective_rank = int(np.clip(rank, 1, max(1, min(hankel.shape) - 1)))
        reconstructed = (U[:, :effective_rank] * singular_values[:effective_rank]) @ Vt[:effective_rank, :]
        result[:, col] = _legacy_anti_diagonal_average(reconstructed, trace.size)
    return result


def _case_metrics(case: EvidenceCase, estimate: np.ndarray) -> dict[str, Any]:
    """Compute quality/proxy metrics for one estimate."""
    metrics: dict[str, Any] = {
        "finite_output": bool(np.isfinite(estimate).all()),
        "proxy_noise_before": _proxy_noise_metric(case.noisy),
        "proxy_noise_after": _proxy_noise_metric(estimate),
    }
    if case.clean is None:
        return metrics

    clean = np.nan_to_num(case.clean, nan=0.0, posinf=0.0, neginf=0.0)
    noisy = np.nan_to_num(case.noisy, nan=0.0, posinf=0.0, neginf=0.0)
    estimate_finite = np.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0)
    snr_before = _compute_snr_db(clean, noisy)
    snr_after = _compute_snr_db(clean, estimate_finite)
    rmse_before = _compute_rmse(clean, noisy)
    rmse_after = _compute_rmse(clean, estimate_finite)
    metrics.update(
        {
            "snr_before_db": snr_before,
            "snr_after_db": snr_after,
            "snr_delta_db": snr_after - snr_before,
            "rmse_before": rmse_before,
            "rmse_after": rmse_after,
            "rmse_delta": rmse_after - rmse_before,
            "target_preservation_ratio": _target_preservation_ratio(
                clean,
                estimate_finite,
                case.target_mask,
            ),
        }
    )
    return metrics


def _run_legacy_case(case: dict[str, Any]) -> dict[str, Any]:
    data = _build_synthetic_matrix(case["shape"], case["seed"])
    tracemalloc.start()
    start = time.perf_counter()
    result, metadata = method_hankel_svd(
        data.copy(),
        window_length=case["window_length"],
        rank=case["rank"],
    )
    runtime_ms = (time.perf_counter() - start) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "name": case["name"],
        "shape": [int(case["shape"][0]), int(case["shape"][1])],
        "seed": int(case["seed"]),
        "window_length": int(case["window_length"]),
        "rank": int(case["rank"]),
        "runtime_ms": round(runtime_ms, 3),
        "peak_memory_bytes": int(peak_bytes),
        "output_checksum": _checksum_array(result),
        "metadata": _to_jsonable(metadata),
    }


def _run_evidence_case(case: EvidenceCase) -> dict[str, Any]:
    tracemalloc.start()
    start = time.perf_counter()
    result, metadata = method_hankel_svd(case.noisy.copy(), window_length=0, rank=0)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = _case_metrics(case, result)

    warnings = metadata.get("warnings", []) if isinstance(metadata, dict) else []
    return {
        "name": case.name,
        "shape": [int(case.noisy.shape[0]), int(case.noisy.shape[1])],
        "runtime_ms": round(runtime_ms, 3),
        "peak_memory_bytes": int(peak_bytes),
        "output_checksum": _checksum_array(result),
        "metrics": _to_jsonable(metrics),
        "metadata": _to_jsonable(metadata),
        "warning_count": len(warnings) if isinstance(warnings, list) else 0,
    }


def _run_legacy_evidence_case(case: EvidenceCase) -> dict[str, Any]:
    """Run same-corpus legacy-like baseline for quality comparison."""
    tracemalloc.start()
    start = time.perf_counter()
    result = _legacy_like_denoise(case.noisy.copy(), window_length=32, rank=5)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "name": case.name,
        "shape": [int(case.noisy.shape[0]), int(case.noisy.shape[1])],
        "runtime_ms": round(runtime_ms, 3),
        "peak_memory_bytes": int(peak_bytes),
        "output_checksum": _checksum_array(result),
        "metrics": _to_jsonable(_case_metrics(case, result)),
        "metadata": {
            "method": "legacy_like_hankel_svd",
            "window_length": 32,
            "rank": 5,
            "recovery_mode": "anti_diagonal_average",
            "scope": "evidence-only same-corpus baseline",
        },
    }


def _comparison_metrics(rewritten: dict[str, Any], legacy: dict[str, Any]) -> dict[str, Any]:
    """Summarize rewritten-vs-legacy quality deltas on the same corpus."""
    rewritten_metrics = rewritten.get("metrics", {})
    legacy_metrics = legacy.get("metrics", {})
    comparison: dict[str, Any] = {
        "runtime_ms_delta": rewritten.get("runtime_ms", 0.0) - legacy.get("runtime_ms", 0.0),
        "peak_memory_bytes_delta": rewritten.get("peak_memory_bytes", 0) - legacy.get("peak_memory_bytes", 0),
    }
    for key in ("snr_after_db", "snr_delta_db", "rmse_after", "rmse_delta", "target_preservation_ratio", "proxy_noise_after"):
        rewritten_value = rewritten_metrics.get(key)
        legacy_value = legacy_metrics.get(key)
        if isinstance(rewritten_value, (int, float)) and isinstance(legacy_value, (int, float)):
            comparison[f"{key}_rewritten_minus_legacy"] = rewritten_value - legacy_value
    rewritten_rmse_after = rewritten_metrics.get("rmse_after")
    legacy_rmse_after = legacy_metrics.get("rmse_after")
    if isinstance(rewritten_rmse_after, (int, float)) and isinstance(legacy_rmse_after, (int, float)):
        comparison["rmse_after_improvement_vs_legacy"] = legacy_rmse_after - rewritten_rmse_after
    rewritten_rmse_delta = rewritten_metrics.get("rmse_delta")
    legacy_rmse_delta = legacy_metrics.get("rmse_delta")
    if isinstance(rewritten_rmse_delta, (int, float)) and isinstance(legacy_rmse_delta, (int, float)):
        comparison["rmse_delta_improvement_vs_legacy"] = legacy_rmse_delta - rewritten_rmse_delta
    return _to_jsonable(comparison)


def _write_legacy_summary_csv(output_dir: Path, entries: list[dict[str, Any]]) -> None:
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "rows",
                "cols",
                "seed",
                "window_length",
                "rank",
                "runtime_ms",
                "peak_memory_bytes",
                "output_checksum",
                "metadata_json",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "name": entry["name"],
                    "rows": entry["shape"][0],
                    "cols": entry["shape"][1],
                    "seed": entry["seed"],
                    "window_length": entry["window_length"],
                    "rank": entry["rank"],
                    "runtime_ms": entry["runtime_ms"],
                    "peak_memory_bytes": entry["peak_memory_bytes"],
                    "output_checksum": entry["output_checksum"],
                    "metadata_json": json.dumps(entry["metadata"], ensure_ascii=False, sort_keys=True),
                }
            )


def _write_metrics_csv(output_dir: Path, entries: list[dict[str, Any]]) -> None:
    csv_path = output_dir / "metrics.csv"
    fieldnames = [
        "name",
        "snr_before_db",
        "snr_after_db",
        "snr_delta_db",
        "rmse_before",
        "rmse_after",
        "rmse_delta",
        "target_preservation_ratio",
        "legacy_snr_after_db",
        "legacy_snr_delta_db",
        "legacy_rmse_after",
        "legacy_rmse_delta",
        "legacy_target_preservation_ratio",
        "rewritten_minus_legacy_snr_after_db",
        "rewritten_minus_legacy_rmse_after",
        "rmse_after_improvement_vs_legacy",
        "rmse_delta_improvement_vs_legacy",
        "proxy_noise_before",
        "proxy_noise_after",
        "legacy_proxy_noise_after",
        "finite_output",
        "warning_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            metrics = entry["metrics"]
            legacy_metrics = entry.get("legacy_metrics", {})
            comparison = entry.get("comparison", {})
            writer.writerow(
                {
                    "name": entry["name"],
                    "snr_before_db": metrics.get("snr_before_db"),
                    "snr_after_db": metrics.get("snr_after_db"),
                    "snr_delta_db": metrics.get("snr_delta_db"),
                    "rmse_before": metrics.get("rmse_before"),
                    "rmse_after": metrics.get("rmse_after"),
                    "rmse_delta": metrics.get("rmse_delta"),
                    "target_preservation_ratio": metrics.get("target_preservation_ratio"),
                    "legacy_snr_after_db": legacy_metrics.get("snr_after_db"),
                    "legacy_snr_delta_db": legacy_metrics.get("snr_delta_db"),
                    "legacy_rmse_after": legacy_metrics.get("rmse_after"),
                    "legacy_rmse_delta": legacy_metrics.get("rmse_delta"),
                    "legacy_target_preservation_ratio": legacy_metrics.get("target_preservation_ratio"),
                    "rewritten_minus_legacy_snr_after_db": comparison.get("snr_after_db_rewritten_minus_legacy"),
                    "rewritten_minus_legacy_rmse_after": comparison.get("rmse_after_rewritten_minus_legacy"),
                    "rmse_after_improvement_vs_legacy": comparison.get("rmse_after_improvement_vs_legacy"),
                    "rmse_delta_improvement_vs_legacy": comparison.get("rmse_delta_improvement_vs_legacy"),
                    "proxy_noise_before": metrics.get("proxy_noise_before"),
                    "proxy_noise_after": metrics.get("proxy_noise_after"),
                    "legacy_proxy_noise_after": legacy_metrics.get("proxy_noise_after"),
                    "finite_output": metrics.get("finite_output"),
                    "warning_count": entry["warning_count"],
                }
            )


def _write_runtime_memory_csv(output_dir: Path, entries: list[dict[str, Any]]) -> None:
    csv_path = output_dir / "runtime_memory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "rows", "cols", "runtime_ms", "peak_memory_bytes"],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "name": entry["name"],
                    "rows": entry["shape"][0],
                    "cols": entry["shape"][1],
                    "runtime_ms": entry["runtime_ms"],
                    "peak_memory_bytes": entry["peak_memory_bytes"],
                }
            )


def _save_case_figure(output_dir: Path, case: EvidenceCase, entry: dict[str, Any]) -> None:
    result, _metadata = method_hankel_svd(case.noisy.copy(), window_length=0, rank=0)
    before = np.nan_to_num(case.noisy, nan=0.0, posinf=0.0, neginf=0.0)
    after = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    diff = before - after

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    panels = ((before, "Before"), (after, "After"), (diff, "Before - After"))
    for axis, (matrix, title) in zip(axes, panels):
        image = axis.imshow(matrix, aspect="auto", cmap="gray")
        axis.set_title(title)
        axis.set_xlabel("Trace")
        axis.set_ylabel("Sample")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"Hankel SVD reset: {case.name}")
    fig.savefig(output_dir / f"before_after_{entry['name']}.png", dpi=140)
    plt.close(fig)


def _load_legacy_summary(output_dir: Path) -> dict[str, Any] | None:
    candidates = [
        output_dir / "legacy" / "summary.json",
        output_dir.parent / "legacy" / "summary.json",
        ROOT / ".sisyphus" / "evidence" / "hankel-svd-reset" / "legacy" / "summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8-sig"))
    return None


def _run_legacy_mode(output_dir: Path) -> dict[str, Any]:
    entries = [_run_legacy_case(case) for case in LEGACY_BENCHMARK_CASES]
    cases = {entry["name"]: entry for entry in entries}
    summary = {
        "mode": "legacy",
        "implementation": "PythonModule.hankel_svd.method_hankel_svd",
        "artifact_root": str(output_dir),
        "cases": cases,
        "entries": entries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_legacy_summary_csv(output_dir, entries)
    return summary


def _run_rewritten_mode(output_dir: Path, *, include_legacy: bool) -> dict[str, Any]:
    cases = _evidence_cases()
    entries = [_run_evidence_case(case) for case in cases]
    legacy_quality_entries = [_run_legacy_evidence_case(case) for case in cases] if include_legacy else []
    if include_legacy:
        for entry, legacy_entry in zip(entries, legacy_quality_entries):
            entry["legacy_metrics"] = legacy_entry["metrics"]
            entry["legacy_runtime_ms"] = legacy_entry["runtime_ms"]
            entry["legacy_peak_memory_bytes"] = legacy_entry["peak_memory_bytes"]
            entry["legacy_output_checksum"] = legacy_entry["output_checksum"]
            entry["comparison"] = _comparison_metrics(entry, legacy_entry)
    for case, entry in zip(cases[:4], entries[:4]):
        _save_case_figure(output_dir, case, entry)

    summary: dict[str, Any] = {
        "mode": "compare" if include_legacy else "rewritten",
        "implementation": "PythonModule.hankel_svd.method_hankel_svd",
        "artifact_root": str(output_dir),
        "cases": {entry["name"]: entry for entry in entries},
        "entries": entries,
        "metric_note": "snr/rmse deltas compare noisy input against each estimate; compare mode also embeds same-corpus legacy-like quality metrics for rewritten-vs-legacy deltas.",
    }
    if include_legacy:
        legacy_summary = _load_legacy_summary(output_dir)
        summary["legacy_baseline"] = legacy_summary
        summary["legacy_baseline_available"] = legacy_summary is not None
        summary["legacy_quality_cases"] = {
            entry["name"]: entry for entry in legacy_quality_entries
        }
        summary["comparison"] = {
            entry["name"]: entry.get("comparison", {}) for entry in entries
        }

    (output_dir / "summary.json").write_text(
        json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_metrics_csv(output_dir, entries)
    _write_runtime_memory_csv(output_dir, entries)
    return summary


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "legacy":
        _run_legacy_mode(output_dir)
    elif args.mode == "rewritten":
        _run_rewritten_mode(output_dir, include_legacy=False)
    else:
        _run_rewritten_mode(output_dir, include_legacy=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
