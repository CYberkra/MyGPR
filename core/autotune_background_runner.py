#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal real AutoTune candidate runner for background suppression.

This module is intentionally deterministic and side-effect free.  It does not
replace the production AutoTune research pipeline.  It provides a safe backend
MVP for the AutoTune UI page:

- baseline
- mean background subtraction
- median background subtraction
- sliding-window background subtraction
- SVD low-rank background subtraction, guarded for GUI responsiveness

The runner operates on the currently loaded 2D B-scan array and returns trial
records plus a recommended candidate.  It writes no Evidence artifacts and does
not mutate the caller's data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from core.autotune_metrics import background_score_terms_v2, score_background_candidate_v2
from core.autotune_scoring_weights import (
    DEFAULT_SCORING_METRICS,
    SUPPORTED_SCORING_METRICS,
    resolve_scoring_weights,
    resolve_target_goal,
)

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
    target_goal: str
    scoring_metrics: tuple[str, ...]
    scoring_weights: dict[str, float]
    scoring_terms: dict[str, float]
    score_version: str = "autotune_scoring_v2"
    goal_profile_weights: dict[str, float] = None
    score_breakdown: dict[str, object] = None
    selection_rule: str = "goal_profile_weighted_background_score"
    candidate_id: str | None = None
    candidate_space_hash: str | None = None
    candidate_space_profile_id: str | None = None
    candidate_space_config_version: str | None = None
    candidate_space_recipe_ids: tuple[str, ...] = ()
    candidate_source: str | None = None
    candidate_group: str | None = None
    candidate_parameters: dict[str, object] = None
    candidate_rationale: str | None = None
    candidate_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        row = asdict(self)
        row["goal_profile_weights"] = dict(self.goal_profile_weights or {})
        row["score_breakdown"] = dict(self.score_breakdown or {})
        row["candidate_parameters"] = dict(self.candidate_parameters or {})
        row["candidate_space_recipe_ids"] = list(self.candidate_space_recipe_ids or ())
        row["candidate_warnings"] = list(self.candidate_warnings or ())
        return row


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


def _optional_target_response(data, shape: tuple[int, int]) -> np.ndarray | None:
    if data is None:
        return None
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"target_response shape mismatch: expected {shape}, got {arr.shape}")
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


def _cnr(arr: np.ndarray, roi_mask: np.ndarray, bg_mask: np.ndarray) -> float:
    roi_vals = arr[roi_mask]
    bg_vals = arr[bg_mask]
    denom = float(np.std(bg_vals)) + 1e-12
    return float(abs(np.mean(np.abs(roi_vals)) - np.mean(np.abs(bg_vals))) / denom)


def _shape_similarity(original: np.ndarray, candidate: np.ndarray, roi_mask: np.ndarray) -> float:
    """Return 0..1 morphology preservation proxy inside the ROI."""
    a = np.asarray(original[roi_mask], dtype=np.float64).ravel()
    b = np.asarray(candidate[roi_mask], dtype=np.float64).ravel()
    if a.size < 2 or b.size != a.size:
        return 0.0
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    corr = float(np.dot(a, b) / denom)
    return min(1.0, max(0.0, corr))


def _rmse_similarity(candidate: np.ndarray, target: np.ndarray | None, roi_mask: np.ndarray) -> float | None:
    if target is None:
        return None
    diff = np.asarray(candidate[roi_mask] - target[roi_mask], dtype=np.float64)
    ref = np.asarray(target[roi_mask], dtype=np.float64)
    if diff.size == 0:
        return None
    rmse = float(np.sqrt(np.mean(diff * diff)))
    scale = float(np.sqrt(np.mean(ref * ref))) + 1e-12
    return float(1.0 / (1.0 + rmse / scale))


def _apply_candidate(
    data: np.ndarray,
    method: str,
    rank: int | None = None,
    *,
    window: int | None = None,
    sliding_stat: str = "mean",
) -> tuple[np.ndarray, str]:
    if method == "baseline":
        return data.copy(), "无处理"
    if method == "mean":
        background = np.mean(data, axis=1, keepdims=True)
        return data - background, "mean trace background"
    if method == "median":
        background = np.median(data, axis=1, keepdims=True)
        return data - background, "median trace background"
    if method == "sliding":
        background = _sliding_trace_background(data, window=window, stat=sliding_stat)
        w = _safe_sliding_window(data.shape[1], window)
        return data - background, f"sliding {sliding_stat} trace background window={w}"
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


def _safe_sliding_window(traces: int, window: int | None = None) -> int:
    if traces <= 1:
        return 1
    if window is None:
        window = min(9, traces if traces % 2 == 1 else traces - 1)
        window = max(3, window)
    window = int(window)
    if window % 2 == 0:
        window += 1
    upper = traces if traces % 2 == 1 else traces - 1
    return max(3, min(window, max(3, upper)))


def _sliding_trace_background(data: np.ndarray, window: int | None = None, *, stat: str = "mean") -> np.ndarray:
    """Estimate a local along-trace background without changing data shape."""
    traces = int(data.shape[1])
    if traces <= 1:
        return data.copy()
    window = _safe_sliding_window(traces, window)
    if window <= 1:
        return np.mean(data, axis=1, keepdims=True).repeat(traces, axis=1)
    half = window // 2
    padded = np.pad(data, ((0, 0), (half, half)), mode="edge")
    if stat == "median":
        out = np.empty_like(data, dtype=np.float64)
        for col in range(traces):
            out[:, col] = np.median(padded[:, col : col + window], axis=1)
        return out
    cumsum = np.cumsum(padded, axis=1, dtype=np.float64)
    cumsum = np.concatenate([np.zeros((data.shape[0], 1), dtype=np.float64), cumsum], axis=1)
    return (cumsum[:, window:] - cumsum[:, :-window]) / float(window)


def _score_candidate(
    original: np.ndarray,
    candidate: np.ndarray,
    roi_mask: np.ndarray,
    bg_mask: np.ndarray,
    method: str,
    *,
    target_goal: str | None,
    scoring_metrics: Iterable[str] | None,
    target_response: np.ndarray | None,
) -> tuple[float, float, float, float, float, str, str, tuple[str, ...], dict[str, float], dict[str, float], dict[str, float], dict, str, str]:
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

    goal, metrics, weights = resolve_scoring_weights(
        target_goal=target_goal,
        scoring_metrics=scoring_metrics,
        target_response_available=target_response is not None,
    )

    # Legacy terms are retained for old UI columns and existing tests.  The
    # final candidate score is produced by AutoTune scoring v2 below.
    legacy_terms: dict[str, float] = {
        "roi_retention": min(1.0, max(0.0, roi_energy_ratio)),
        "residual": min(1.0, max(0.0, background_suppression)),
        "cnr": float(1.0 / (1.0 + np.exp(-cnr_gain))),
        "shape": _shape_similarity(original, candidate, roi_mask),
    }
    rmse_term = _rmse_similarity(candidate, target_response, roi_mask)
    if rmse_term is not None:
        legacy_terms["rmse"] = rmse_term

    legacy_score = sum(weights[metric] * legacy_terms.get(metric, 0.0) for metric in metrics)
    v2_terms = background_score_terms_v2(
        original=original,
        candidate=candidate,
        roi_mask=roi_mask,
        bg_mask=bg_mask,
        method=method,
        cnr_gain=cnr_gain,
        roi_energy_ratio=roi_energy_ratio,
        residual_ratio=residual_ratio,
        background_suppression=background_suppression,
        target_response=target_response,
    )
    v2_breakdown = score_background_candidate_v2(
        terms=v2_terms,
        target_goal=target_goal,
        target_response_available=target_response is not None,
    )
    score = float(v2_breakdown["score"])
    terms: dict[str, float] = {
        **legacy_terms,
        **{f"v2_{key}": float(value) for key, value in v2_terms.items()},
        "legacy_score": float(legacy_score),
        "v2_score": score,
    }

    warning_parts = []
    if roi_energy_ratio < 0.35 and method != "baseline":
        warning_parts.append("关注范围能量削弱明显")
    if residual_ratio > 0.95 and method != "baseline":
        warning_parts.append("背景抑制有限")
    if cnr_gain < -0.05 and method != "baseline":
        warning_parts.append("CNR/SNR 下降")
    if terms.get("shape", 1.0) < 0.35 and method != "baseline":
        warning_parts.append("目标形态相似度低")
    if "rmse" in (scoring_metrics or ()) and target_response is None:
        warning_parts.append("RMSE 指标缺少 target_response，已自动跳过")
    if method == "svd":
        warning_parts.append("SVD 对目标响应较敏感，建议查看原始图像")
    warning = "；".join(dict.fromkeys(warning_parts)) if warning_parts else "可查看"
    return (
        float(score),
        float(roi_energy_ratio),
        float(background_suppression),
        float(cnr_gain),
        float(residual_ratio),
        warning,
        goal,
        metrics,
        weights,
        {key: float(value) for key, value in terms.items()},
        dict(v2_breakdown.get("weights", {})),
        v2_breakdown,
        str(v2_breakdown.get("scoring_version", "autotune_scoring_v2")),
        "goal_profile_weighted_background_score",
    )



def _candidate_id_from_dict(item: Mapping[str, object]) -> str | None:
    value = item.get("candidate_id")
    return str(value) if value is not None else None


def _v1_background_plan(
    data: np.ndarray,
    *,
    target_goal: str | None,
    metadata: Mapping[str, object] | None,
    target_response_available: bool,
    include_display_only_in_space: bool | None,
    include_experimental: bool,
    candidate_methods_filter: Iterable[str] | None,
    explicit_svd_ranks: Sequence[int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Generate executable background candidates from the V1 candidate-space contract.

    The public UI still exposes legacy method names. This adapter maps the richer
    V1 background declarations back to the current executable runner while
    preserving the candidate id/hash/profile metadata needed by trial tables and
    manifests.
    """
    from core.autotune_candidate_generator import generate_autotune_v1_candidates

    include_display = (not target_response_available) if include_display_only_in_space is None else bool(include_display_only_in_space)
    generated = generate_autotune_v1_candidates(
        data,
        metadata=dict(metadata or {}),
        target_goal=target_goal,
        include_display_only=include_display,
        include_experimental=include_experimental,
    )
    filter_set = {str(item).strip().lower() for item in (candidate_methods_filter or []) if str(item).strip()}
    # Empty filter means use all V1 executable background methods. Baseline is
    # added separately when requested so it remains an explicit reference row.
    background_methods = {
        "mean_background": "mean",
        "median_background": "median",
        "sliding_mean_background": "sliding",
        "sliding_median_background": "sliding",
        "svd_rank_sweep": "svd",
    }
    allowed_by_legacy = {
        "mean": {"mean_background"},
        "median": {"median_background"},
        "sliding": {"sliding_mean_background", "sliding_median_background"},
        "svd": {"svd_rank_sweep"},
    }
    allowed_v1_methods: set[str] = set()
    for legacy, methods in allowed_by_legacy.items():
        if not filter_set or legacy in filter_set:
            allowed_v1_methods.update(methods)
    planned: list[dict[str, object]] = []
    if "baseline" in filter_set:
        planned.append(
            {
                "runner_method": "baseline",
                "runner_rank": None,
                "runner_window": None,
                "runner_sliding_stat": "mean",
                "name": "不处理基线",
                "params": "无处理",
                "candidate_id": None,
                "v1_method": "baseline",
                "v1_parameters": {},
                "candidate_source": "explicit_reference",
                "candidate_group": "baseline_reference",
                "candidate_rationale": "Baseline row is kept as a no-processing reference and is not generated as a V1 processing candidate.",
                "candidate_warnings": (),
            }
        )
    explicit_rank_set = {max(1, int(rank)) for rank in explicit_svd_ranks or ()}
    for candidate in generated.candidates:
        if candidate.category != "background_suppression":
            continue
        if candidate.method not in allowed_v1_methods:
            continue
        runner_method = background_methods[candidate.method]
        params = dict(candidate.parameters or {})
        rank = int(params.get("remove_rank", 1)) if runner_method == "svd" else None
        if runner_method == "svd" and explicit_rank_set and rank not in explicit_rank_set:
            continue
        window = int(params.get("window_ntraces", 0)) or None
        stat = "median" if "median" in candidate.method else "mean"
        if candidate.method == "mean_background":
            name, param_text = "均值背景扣除", "method=mean; source=v1"
        elif candidate.method == "median_background":
            name, param_text = "中位数背景扣除", "method=median; source=v1"
        elif candidate.method == "sliding_mean_background":
            name, param_text = "滑动均值背景", f"window={window}; stat=mean; source=v1"
        elif candidate.method == "sliding_median_background":
            name, param_text = "滑动中位数背景", f"window={window}; stat=median; source=v1"
        else:
            name, param_text = f"SVD 背景抑制 rank={rank}", f"rank={rank}; source=v1"
        cdict = candidate.to_dict()
        planned.append(
            {
                "runner_method": runner_method,
                "runner_rank": rank,
                "runner_window": window,
                "runner_sliding_stat": stat,
                "name": name,
                "params": param_text,
                "candidate_id": _candidate_id_from_dict(cdict),
                "v1_method": candidate.method,
                "v1_parameters": params,
                "candidate_source": candidate.source,
                "candidate_group": candidate.candidate_group,
                "candidate_rationale": candidate.rationale,
                "candidate_warnings": tuple(candidate.warnings or ()),
            }
        )
    context = {
        "candidate_space_hash": generated.candidate_space_hash,
        "candidate_space_profile_id": generated.profile_id,
        "candidate_space_config_version": generated.config_version,
        "candidate_space_recipe_ids": tuple(generated.recipe_ids),
        "candidate_space_features": generated.features.to_dict(),
        "candidate_space_warnings": tuple(generated.warnings),
        "candidate_space_candidate_count": len(generated.candidates),
        "candidate_space_include_display_only": include_display,
        "candidate_space_include_experimental": bool(include_experimental),
    }
    return planned, context


def _legacy_background_plan(
    *,
    candidate_methods: Iterable[str],
    svd_ranks: Sequence[int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    planned: list[dict[str, object]] = []
    for method in list(dict.fromkeys(candidate_methods)):
        if method == "svd":
            ranks = [max(1, int(rank)) for rank in (svd_ranks or [1])] or [1]
            for rank in ranks:
                planned.append(
                    {
                        "runner_method": "svd",
                        "runner_rank": rank,
                        "runner_window": None,
                        "runner_sliding_stat": "mean",
                        "name": f"SVD 背景抑制 rank={rank}",
                        "params": f"rank={rank}",
                    }
                )
        else:
            planned.append(
                {
                    "runner_method": method,
                    "runner_rank": None,
                    "runner_window": None,
                    "runner_sliding_stat": "mean",
                    "name": {
                        "baseline": "不处理基线",
                        "mean": "均值背景扣除",
                        "median": "中位数背景扣除",
                        "sliding": "滑动窗口背景",
                    }.get(method, method),
                    "params": "",
                }
            )
    return planned, {}

def run_background_candidates(
    data,
    *,
    candidate_methods: Iterable[str],
    svd_ranks: Sequence[int] = (1, 2, 3),
    roi: dict | None = None,
    target_goal: str | None = "均衡推荐",
    scoring_metrics: Iterable[str] | None = DEFAULT_SCORING_METRICS,
    target_response=None,
    max_svd_traces: int = 600,
    use_v1_candidate_space: bool = False,
    metadata: Mapping[str, object] | None = None,
    include_display_only_in_space: bool | None = None,
    include_experimental_candidates: bool = False,
) -> list[dict]:
    """Run a minimal background-suppression candidate sweep.

    Parameters
    ----------
    data:
        2D B-scan array with shape samples × traces.
    candidate_methods:
        Iterable of legacy method keys: baseline, mean, median, svd, sliding.
        When ``use_v1_candidate_space=True`` this remains a UI compatibility
        filter over generated V1 background candidates.
    svd_ranks:
        Legacy/UI SVD rank filter. With V1 enabled it filters generated
        ``svd_rank_sweep`` candidates rather than creating ranks itself.
    roi:
        Dict with sample_start/sample_end/trace_start/trace_end.
    target_goal:
        UI goal label or stable alias. It selects scoring weights and, when V1
        is enabled, the profile-aware candidate caps.
    scoring_metrics:
        Enabled legacy metric keys. Weights are normalized over this subset.
    target_response:
        Optional synthetic paired label. Required for the rmse metric.
    max_svd_traces:
        Safety guard. When trace count exceeds this threshold, SVD candidates are
        skipped instead of running an unbounded SVD in the GUI path.
    use_v1_candidate_space:
        If true, build executable background candidates from
        ``core.autotune_candidate_generator`` and attach candidate-space hash /
        profile / config metadata to each trial row. Legacy default is false for
        compatibility.
    """

    arr = _finite_float_array(data)
    target = _optional_target_response(target_response, arr.shape)
    roi_slice = _clip_roi(arr.shape, roi)
    roi_mask, bg_mask = _roi_background_masks(arr.shape, roi_slice)
    methods = list(dict.fromkeys(candidate_methods))

    supported_methods = {"baseline", "mean", "median", "svd", "sliding"}
    unsupported = [method for method in methods if method not in supported_methods]
    if unsupported:
        raise ValueError(f"Unsupported candidate method(s): {', '.join(unsupported)}")

    resolved_goal, resolved_metrics, resolved_weights = resolve_scoring_weights(
        target_goal=target_goal,
        scoring_metrics=scoring_metrics,
        target_response_available=target is not None,
    )

    if use_v1_candidate_space:
        planned, candidate_context = _v1_background_plan(
            arr,
            target_goal=target_goal,
            metadata=metadata,
            target_response_available=target is not None,
            include_display_only_in_space=include_display_only_in_space,
            include_experimental=include_experimental_candidates,
            candidate_methods_filter=methods,
            explicit_svd_ranks=svd_ranks,
        )
    else:
        planned, candidate_context = _legacy_background_plan(candidate_methods=methods, svd_ranks=svd_ranks)

    results: list[BackgroundCandidateResult] = []
    svd_components: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for item in planned:
        method = str(item.get("runner_method") or "")
        rank = item.get("runner_rank")
        window = item.get("runner_window")
        sliding_stat = str(item.get("runner_sliding_stat") or "mean")
        if method == "svd" and arr.shape[1] > max_svd_traces:
            results.append(
                BackgroundCandidateResult(
                    name=str(item.get("name") or "SVD 背景抑制（已跳过）"),
                    method="svd",
                    params=f"trace_count={arr.shape[1]} > max_svd_traces={int(max_svd_traces)}",
                    score=0.0,
                    roi_energy_ratio=0.0,
                    background_suppression=0.0,
                    cnr_gain=0.0,
                    residual_ratio=1.0,
                    warning="SVD 已因 trace 数超过安全阈值跳过；可减小数据范围或调高阈值后再运行",
                    status="已跳过",
                    target_goal=resolved_goal,
                    scoring_metrics=resolved_metrics,
                    scoring_weights=resolved_weights,
                    scoring_terms={},
                    candidate_id=item.get("candidate_id"),
                    candidate_space_hash=candidate_context.get("candidate_space_hash"),
                    candidate_space_profile_id=candidate_context.get("candidate_space_profile_id"),
                    candidate_space_config_version=candidate_context.get("candidate_space_config_version"),
                    candidate_space_recipe_ids=tuple(candidate_context.get("candidate_space_recipe_ids") or ()),
                    candidate_source=item.get("candidate_source"),
                    candidate_group=item.get("candidate_group"),
                    candidate_parameters=dict(item.get("v1_parameters") or {}),
                    candidate_rationale=item.get("candidate_rationale"),
                    candidate_warnings=tuple(item.get("candidate_warnings") or ()),
                )
            )
            continue

        if method == "svd":
            if svd_components is None:
                svd_components = _compute_svd(arr)
            u, s, vt = svd_components
            cand, params = _apply_svd_components(arr, u, s, vt, int(rank or 1))
        else:
            cand, params = _apply_candidate(
                arr,
                method,
                window=int(window) if window else None,
                sliding_stat=sliding_stat,
            )
        score, roi_ratio, bg_sup, cnr_gain, residual_ratio, warning, goal, metrics, weights, terms, goal_profile_weights, score_breakdown, score_version, selection_rule = _score_candidate(
            arr,
            cand,
            roi_mask,
            bg_mask,
            method,
            target_goal=target_goal,
            scoring_metrics=scoring_metrics,
            target_response=target,
        )
        candidate_warnings = list(item.get("candidate_warnings") or ())
        if use_v1_candidate_space and candidate_warnings:
            warning = "；".join(dict.fromkeys([warning, *candidate_warnings])) if warning else "；".join(dict.fromkeys(candidate_warnings))
        status = "基线" if method == "baseline" else ("建议查看" if "削弱" in warning or "SVD" in warning or "svd_rank_may_remove_interface" in warning else "可用")
        results.append(
            BackgroundCandidateResult(
                name=str(item.get("name") or {
                    "baseline": "不处理基线",
                    "mean": "均值背景扣除",
                    "median": "中位数背景扣除",
                    "sliding": "滑动窗口背景",
                    "svd": f"SVD 背景抑制 rank={int(rank or 1)}",
                }.get(method, method)),
                method=method,
                params=str(item.get("params") or params),
                score=score,
                roi_energy_ratio=roi_ratio,
                background_suppression=bg_sup,
                cnr_gain=cnr_gain,
                residual_ratio=residual_ratio,
                warning=warning,
                status=status,
                target_goal=goal,
                scoring_metrics=metrics,
                scoring_weights=weights,
                scoring_terms=terms,
                score_version=score_version,
                goal_profile_weights=goal_profile_weights,
                score_breakdown=score_breakdown,
                selection_rule="v1_candidate_space_hash_ranked_background_score" if use_v1_candidate_space else selection_rule,
                candidate_id=item.get("candidate_id"),
                candidate_space_hash=candidate_context.get("candidate_space_hash"),
                candidate_space_profile_id=candidate_context.get("candidate_space_profile_id"),
                candidate_space_config_version=candidate_context.get("candidate_space_config_version"),
                candidate_space_recipe_ids=tuple(candidate_context.get("candidate_space_recipe_ids") or ()),
                candidate_source=item.get("candidate_source"),
                candidate_group=item.get("candidate_group"),
                candidate_parameters=dict(item.get("v1_parameters") or {}),
                candidate_rationale=item.get("candidate_rationale"),
                candidate_warnings=tuple(candidate_warnings),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    ranked = []
    for i, result in enumerate(results, start=1):
        row = result.to_dict()
        row["rank"] = i
        if candidate_context:
            row["candidate_space_context"] = {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in candidate_context.items()
            }
        if i == 1 and row.get("status") != "已跳过":
            row["status"] = "推荐候选"
        ranked.append(row)
    return ranked
