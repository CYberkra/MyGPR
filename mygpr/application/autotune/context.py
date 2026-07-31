#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROI and data-feature context construction for auto-tuning."""
from __future__ import annotations

from typing import Any

import numpy as np

from mygpr.domain.autotune.quality_metrics import (
    detect_first_break_indices,
    estimate_depth_attenuation_curve,
    estimate_lateral_correlation_length,
    estimate_singular_elbow_rank,
    extract_roi_and_context,
    hot_pixel_ratio,
    kurtosis_or_spikiness,
    low_freq_energy_ratio,
    median_first_break,
)
from mygpr.domain.autotune.models import AutoTuneContext


def _build_auto_tune_context(
    data: np.ndarray,
    header_info: dict[str, Any],
    trace_metadata: dict[str, np.ndarray],
    roi_spec: dict[str, Any],
    search_mode: str,
) -> AutoTuneContext:
    roi_mode = str((roi_spec or {}).get("mode") or "auto")
    if roi_mode == "full":
        roi_source = "full"
        roi_label = str((roi_spec or {}).get("label") or "全图")
        roi_bounds = None
        roi_data = None
        context_bounds = None
        context_data = np.asarray(data, dtype=np.float32)
    else:
        bounds = (
            (roi_spec or {}).get("bounds") if roi_mode in {"crop", "manual"} else None
        )
        roi_payload = extract_roi_and_context(data, bounds)
        roi_bounds = dict(roi_payload.get("bounds") or {})
        context_bounds = dict(roi_payload.get("context_bounds") or {})
        roi_data = np.asarray(roi_payload.get("roi_data"), dtype=np.float32)
        context_data = np.asarray(roi_payload.get("context_data"), dtype=np.float32)
        roi_source = roi_mode if bounds else "auto"
        roi_label = str(
            (roi_spec or {}).get("label")
            or (
                "当前裁剪区"
                if roi_mode == "crop" and bounds
                else "手动框选关注范围"
                if roi_mode == "manual" and bounds
                else "自动关注范围"
            )
        )
        if roi_data.size < 64 or roi_data.shape[0] < 8 or roi_data.shape[1] < 4:
            roi_source = "full"
            roi_label = "全图"
            roi_bounds = None
            roi_data = None
            context_bounds = None
            context_data = np.asarray(data, dtype=np.float32)

    features = _extract_auto_tune_features(
        np.asarray(data, dtype=np.float32), roi_data, context_data
    )
    return AutoTuneContext(
        full_data=np.asarray(data, dtype=np.float32),
        header_info=dict(header_info or {}),
        trace_metadata=dict(trace_metadata or {}),
        roi_source=roi_source,
        roi_label=roi_label,
        roi_bounds=roi_bounds,
        roi_data=roi_data,
        context_bounds=context_bounds,
        context_data=context_data,
        features=features,
        search_mode=str(search_mode or "standard"),
    )


def _extract_auto_tune_features(
    full_data: np.ndarray, roi_data: np.ndarray | None, context_data: np.ndarray
) -> dict[str, Any]:
    arr = np.asarray(full_data, dtype=np.float64)
    context = np.asarray(context_data, dtype=np.float64)
    roi = np.asarray(roi_data, dtype=np.float64) if roi_data is not None else context
    attenuation = estimate_depth_attenuation_curve(context)
    shallow = (
        float(np.mean(attenuation[: max(4, len(attenuation) // 5)]))
        if attenuation.size
        else 0.0
    )
    deep = (
        float(np.mean(attenuation[max(0, len(attenuation) * 3 // 5) :]))
        if attenuation.size
        else 0.0
    )
    fb_idx = detect_first_break_indices(context, method="threshold", threshold=0.05)
    return {
        "shape": arr.shape,
        "roi_shape": roi.shape,
        "low_freq_ratio": float(low_freq_energy_ratio(context)),
        "lateral_corr_length": int(estimate_lateral_correlation_length(context)),
        "singular_elbow_rank": int(estimate_singular_elbow_rank(context)),
        "shallow_rms": shallow,
        "deep_rms": deep,
        "attenuation_ratio": float(shallow / max(deep, 1.0e-6)) if deep > 0 else 1.0,
        "hot_pixel_ratio": float(hot_pixel_ratio(context)),
        "spikiness": float(kurtosis_or_spikiness(context)),
        "first_break_std": float(np.std(fb_idx)) if fb_idx.size else 0.0,
        "first_break_median": int(median_first_break(fb_idx)),
    }


def _get_search_plan(search_mode: str) -> dict[str, Any]:
    plans = {
        "fast": {"coarse_budget": 6, "refine_top_k": 1, "fine_budget": 4},
        "standard": {"coarse_budget": 8, "refine_top_k": 2, "fine_budget": 6},
        "thorough": {"coarse_budget": 12, "refine_top_k": 3, "fine_budget": 8},
    }
    return plans.get(str(search_mode or "standard"), plans["standard"])
