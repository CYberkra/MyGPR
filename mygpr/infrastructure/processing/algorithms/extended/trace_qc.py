#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace quality control helpers for GPR B-scan data."""

from __future__ import annotations

from typing import Any

import numpy as np

from mygpr.domain.common.scalars import to_float


def method_trace_qc(
    data: np.ndarray,
    mode: str = "mark",
    empty_rms_threshold: float = 0.0,
    spike_zscore: float = 0.0,
    manual_trace_indices: str = "",
    trace_metadata: dict[str, np.ndarray] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mark, mute, or remove bad traces.

    The default mode is intentionally non-destructive: it only reports detected
    trace indices and writes a QC mask into trace metadata.
    """

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    if arr.shape[1] == 0:
        raise ValueError("输入数据没有有效道")

    resolved_mode = str(mode or "mark").strip().lower()
    if resolved_mode not in {"mark", "mute", "remove"}:
        raise ValueError("trace_qc mode 必须是 mark、mute 或 remove")

    rms = np.sqrt(np.einsum("ij,ij->j", arr, arr, optimize=True) / arr.shape[0])
    bad_mask = np.zeros(arr.shape[1], dtype=bool)

    empty_threshold = to_float(empty_rms_threshold, default=0.0)
    if empty_threshold > 0.0:
        bad_mask |= rms <= empty_threshold

    zscore_threshold = to_float(spike_zscore, default=0.0)
    if zscore_threshold > 0.0 and arr.shape[1] >= 3:
        median = float(np.median(rms))
        mad = float(np.median(np.abs(rms - median)))
        scale = max(1.4826 * mad, 1.0e-12)
        bad_mask |= np.abs(rms - median) / scale >= zscore_threshold

    for idx in _parse_trace_indices(manual_trace_indices, arr.shape[1]):
        bad_mask[idx] = True

    bad_indices = np.flatnonzero(bad_mask).astype(np.int32)
    if resolved_mode == "remove":
        keep_mask = ~bad_mask
        if not np.any(keep_mask):
            raise ValueError("trace_qc remove 模式不能删除全部道")
        result = arr[:, keep_mask].astype(np.float32, copy=True)
        metadata_out = _filter_trace_metadata(trace_metadata or {}, keep_mask)
        metadata_out["trace_qc_bad_mask"] = bad_mask[keep_mask].astype(np.int8)
        meta_key = "trace_metadata_out"
        meta_value = metadata_out
    elif resolved_mode == "mute":
        result = np.array(arr, copy=True)
        result[:, bad_mask] = 0.0
        meta_key = "trace_metadata_updates"
        meta_value = {"trace_qc_bad_mask": bad_mask.astype(np.int8)}
    else:
        result = np.array(arr, copy=True)
        meta_key = "trace_metadata_updates"
        meta_value = {"trace_qc_bad_mask": bad_mask.astype(np.int8)}

    return result, {
        "method": "trace_qc",
        "mode": resolved_mode,
        "empty_rms_threshold": float(empty_threshold),
        "spike_zscore": float(zscore_threshold),
        "bad_trace_indices": bad_indices,
        "bad_trace_count": int(bad_indices.size),
        "input_traces": int(arr.shape[1]),
        "output_traces": int(result.shape[1]),
        meta_key: meta_value,
    }


def _parse_integer_token(value: str) -> int | None:
    token = value.strip()
    if not token:
        return None
    digits = token[1:] if token[0] in {"+", "-"} else token
    if not digits.isdigit():
        return None
    return int(token)


def _parse_trace_indices(spec: str, trace_count: int) -> list[int]:
    indices: set[int] = set()
    for part in str(spec or "").replace("，", ",").split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = _parse_integer_token(left)
            end = _parse_integer_token(right)
            if start is None or end is None:
                continue
            if start > end:
                start, end = end, start
            indices.update(idx for idx in range(start, end + 1) if 0 <= idx < trace_count)
            continue
        idx = _parse_integer_token(item)
        if idx is not None and 0 <= idx < trace_count:
            indices.add(idx)
    return sorted(indices)


def _filter_trace_metadata(
    trace_metadata: dict[str, np.ndarray], keep_mask: np.ndarray
) -> dict[str, np.ndarray]:
    filtered: dict[str, np.ndarray] = {}
    for key, value in trace_metadata.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == keep_mask.size:
            filtered[key] = np.array(arr[keep_mask], copy=True)
        else:
            filtered[key] = np.array(arr, copy=True)
    return filtered
