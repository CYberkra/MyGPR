#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal real AutoTune candidate runner for background suppression.

This module is intentionally small and deterministic. It does not replace the
production AutoTune research pipeline. It provides a safe backend MVP for the
new AutoTune UI page:

- baseline
- mean background subtraction
- median background subtraction
- SVD low-rank background subtraction

The runner operates on the currently loaded 2D B-scan array and returns trial
records plus a recommended candidate. It writes no Evidence artifacts and does
not mutate the caller's data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class BackgroundCandidateResult:
    name: str
    method: str
    params: str
    score: float
    roi_energy_ratio: float
    background_suppression: float
    cnr_gain: float
    residual_ratio: float
    warning: str
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_float_array(data) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"AutoTune background runner expects 2D data, got shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("AutoTune background runner received empty data")
    if not np.isfinite(arr).all():
        finite = arr[np.isfinite(arr)]
        fill = float(np.nanmedian(finite)) if finite.size else 0.0
        arr = np.where(np.isfinite(arr), arr, fill)
    return arr


def _clip_roi(shape: tuple[int, int], roi: dict | None) -> tuple[slice, slice]:
    samples, traces = int(shape[0]), int(shape[1])
    if not roi:
        s0, s1 = samples // 4, max(samples // 4 + 1, samples * 3 // 4)
        t0, t1 = traces // 4, max(traces // 4 + 1, traces * 3 // 4)
    else:
        s0 = int(roi.get("sample_start", 0))
        s1 = int(roi.get("sample_end", samples))
        t0 = int(roi.get("trace_start", 0))
        t1 = int(roi.get("trace_end", traces))
    s0 = max(0, min(samples - 1, s0))
    s1 = max(s0 + 1, min(samples, s1))
    t0 = max(0, min(traces - 1, t0))
    t1 = max(t0 + 1, min(traces, t1))
    return slice(s0, s1), slice(t0, t1)


def _roi_background_masks(shape: tuple[int, int], roi_slice: tuple[slice, slice]) -> tuple[np.ndarray, np.ndarray]:
    roi_mask = np.zeros(shape, dtype=bool)
    roi_mask[roi_slice] = True
    bg_mask = ~roi_mask
    if not bg_mask.any():
        bg_mask = roi_mask
    return roi_mask, bg_mask


def _energy(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    return float(np.mean(vals * vals)) if vals.size else 0.0


def _mean_abs(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    return float(np.mean(np.abs(vals))) if vals.size else 0.0


def _cnr(arr: np.ndarray, roi_mask: np.ndarray, bg_mask: np.ndarray) -> float:
    roi_vals = arr[roi_mask]
    bg_vals = arr[bg_mask]
    denom = float(np.std(bg_vals)) + 1e-12
    return float(abs(np.mean(np.abs(roi_vals)) - np.mean(np.abs(bg_vals))) / denom)


def _apply_candidate(data: np.ndarray, method: str, rank: int | None = None) -> tuple[np.ndarray, str]:
    if method == "baseline":
        return data.copy(), "无处理"
    if method == "mean":
        background = np.mean(data, axis=1, keepdims=True)
        return data - background, "mean trace background"
    if method == "median":
        background = np.median(data, axis=1, keepdims=True)
        return data - background, "median trace background"
    if method == "sliding":
        background = _sliding_trace_background(data)
        return data - background, "sliding trace background"
    if method == "svd":
        if rank is None:
            rank = 1
        u, s, vt = _compute_svd(data)
        return _apply_svd_components(data, u, s, vt, rank)
    raise ValueError(f"Unsupported candidate method: {method}")


def _compute_svd(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one full SVD for rank-sweep reuse."""
    return np.linalg.svd(data, full_matrices=False)


def _apply_svd_components(
    data: np.ndarray,
    u: np.ndarray,
    s: np.ndarray,
    vt: np.ndarray,
    rank: int | None,
) -> tuple[np.ndarray, str]:
    rank = 1 if rank is None else max(1, int(rank))
    k = min(rank, s.size)
    low_rank = (u[:, :k] * s[:k]) @ vt[:k, :]
    return data - low_rank, f"rank={k}"


def _sliding_trace_background(data: np.ndarray, window: int | None = None) -> np.ndarray:
    """Estimate a local along-trace background without changing data shape.

    The implementation is intentionally dependency-light for GUI responsiveness.
    It pads along the trace axis and averages a small odd window.  This is not
    a paper-facing algorithm claim; it is an experimental candidate that keeps
    the UI candidate list and backend behaviour aligned.
    """
    traces = int(data.shape[1])
    if traces <= 1:
        return data.copy()
    if window is None:
        window = min(9, traces if traces % 2 == 1 else traces - 1)
        window = max(3, window)
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = max(3, min(window, traces if traces % 2 == 1 else traces - 1))
    if window <= 1:
        return np.mean(data, axis=1, keepdims=True).repeat(traces, axis=1)
    half = window // 2
    padded = np.pad(data, ((0, 0), (half, half)), mode="edge")
    cumsum = np.cumsum(padded, axis=1, dtype=np.float64)
    cumsum = np.concatenate([np.zeros((data.shape[0], 1), dtype=np.float64), cumsum], axis=1)
    return (cumsum[:, window:] - cumsum[:, :-window]) / float(window)


def _score_candidate(
    original: np.ndarray,
    candidate: np.ndarray,
    roi_mask: np.ndarray,
    bg_mask: np.ndarray,
    method: str,
) -> tuple[float, float, float, float, float, str]:
    eps = 1e-12
    raw_roi_energy = _energy(original[roi_mask]) + eps
    raw_bg_energy = _energy(original[bg_mask]) + eps
    cand_roi_energy = _energy(candidate[roi_mask])
    cand_bg_energy = _energy(candidate[bg_mask])

    roi_energy_ratio = cand_roi_energy / raw_roi_energy
    residual_ratio = cand_bg_energy / raw_bg_energy
    background_suppression = max(0.0, 1.0 - residual_ratio)

    raw_cnr = _cnr(original, roi_mask, bg_mask)
    cand_cnr = _cnr(candidate, roi_mask, bg_mask)
    cnr_gain = cand_cnr - raw_cnr

    # Conservative heuristic: preserve ROI response while reducing background.
    roi_term = min(1.0, max(0.0, roi_energy_ratio))
    suppression_term = min(1.0, max(0.0, background_suppression))
    cnr_term = 1.0 / (1.0 + np.exp(-cnr_gain))  # 0..1

    score = 0.45 * suppression_term + 0.35 * roi_term + 0.20 * cnr_term

    warning_parts = []
    if roi_energy_ratio < 0.35 and method != "baseline":
        warning_parts.append("ROI 能量削弱明显")
    if residual_ratio > 0.95 and method != "baseline":
        warning_parts.append("背景抑制有限")
    if method == "svd":
        warning_parts.append("SVD 可能削弱目标，需人工复核")
    warning = "；".join(warning_parts) if warning_parts else "需人工复核"
    return float(score), float(roi_energy_ratio), float(background_suppression), float(cnr_gain), float(residual_ratio), warning


def run_background_candidates(
    data,
    *,
    candidate_methods: Iterable[str],
    svd_ranks: Sequence[int] = (1, 2, 3),
    roi: dict | None = None,
    max_svd_traces: int = 600,
) -> list[dict]:
    """Run a minimal background-suppression candidate sweep.

    Parameters
    ----------
    data:
        2D B-scan array with shape samples × traces.
    candidate_methods:
        Iterable of method keys: baseline, mean, median, svd.
    svd_ranks:
        Ranks used only when ``svd`` is enabled.
    roi:
        Dict with sample_start/sample_end/trace_start/trace_end.
    max_svd_traces:
        Safety guard. SVD is allowed but bounded for UI responsiveness.
    """

    arr = _finite_float_array(data)
    roi_slice = _clip_roi(arr.shape, roi)
    roi_mask, bg_mask = _roi_background_masks(arr.shape, roi_slice)
    methods = list(dict.fromkeys(candidate_methods))

    supported_methods = {"baseline", "mean", "median", "svd", "sliding"}
    unsupported = [method for method in methods if method not in supported_methods]
    if unsupported:
        raise ValueError(f"Unsupported candidate method(s): {', '.join(unsupported)}")

    results: list[BackgroundCandidateResult] = []
    svd_components: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for method in methods:
        if method == "svd":
            if arr.shape[1] > max_svd_traces:
                # Still run, but users should know it may be slower.
                pass
            ranks = [max(1, int(r)) for r in svd_ranks] or [1]
            if svd_components is None:
                svd_components = _compute_svd(arr)
            u, s, vt = svd_components
            for rank in ranks:
                cand, params = _apply_svd_components(arr, u, s, vt, rank)
                score, roi_ratio, bg_sup, cnr_gain, residual_ratio, warning = _score_candidate(
                    arr, cand, roi_mask, bg_mask, "svd"
                )
                results.append(
                    BackgroundCandidateResult(
                        name=f"SVD 背景抑制 rank={rank}",
                        method="svd",
                        params=params,
                        score=score,
                        roi_energy_ratio=roi_ratio,
                        background_suppression=bg_sup,
                        cnr_gain=cnr_gain,
                        residual_ratio=residual_ratio,
                        warning=warning,
                        status="需复核" if "削弱" in warning or "SVD" in warning else "可用",
                    )
                )
        elif method in {"baseline", "mean", "median", "sliding"}:
            cand, params = _apply_candidate(arr, method)
            score, roi_ratio, bg_sup, cnr_gain, residual_ratio, warning = _score_candidate(
                arr, cand, roi_mask, bg_mask, method
            )
            name_map = {
                "baseline": "不处理基线",
                "mean": "均值背景扣除",
                "median": "中位数背景扣除",
                "sliding": "滑动窗口背景",
            }
            results.append(
                BackgroundCandidateResult(
                    name=name_map[method],
                    method=method,
                    params=params,
                    score=score,
                    roi_energy_ratio=roi_ratio,
                    background_suppression=bg_sup,
                    cnr_gain=cnr_gain,
                    residual_ratio=residual_ratio,
                    warning=warning,
                    status="基线" if method == "baseline" else ("需复核" if "削弱" in warning else "可用"),
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    ranked = []
    for i, result in enumerate(results, start=1):
        row = result.to_dict()
        row["rank"] = i
        if i == 1:
            row["status"] = "推荐候选"
        ranked.append(row)
    return ranked
