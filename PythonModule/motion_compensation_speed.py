#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UAV-GPR 速度误差补偿模块。

基于每道的累计距离，将非等距采样的 B-scan 沿道方向重采样到等距轴。
"""

from __future__ import annotations

from typing import Any, Tuple
import numpy as np

from core.trace_metadata_utils import (
    build_uniform_trace_distance_m,
    resample_trace_metadata,
)

from core.motion_compensation_core import apply_speed_resampling


def _derive_trace_distance_from_xy_v1(
    trace_metadata: dict, trace_count: int
) -> np.ndarray:
    """[V1 兼容] 从 local_x_m / local_y_m 推导累计距离。"""
    if "local_x_m" not in trace_metadata or "local_y_m" not in trace_metadata:
        raise ValueError("缺少 trace_distance_m，且无法从 local_x_m / local_y_m 推导")

    local_x = np.asarray(trace_metadata["local_x_m"], dtype=np.float64)
    local_y = np.asarray(trace_metadata["local_y_m"], dtype=np.float64)
    if local_x.ndim != 1 or local_y.ndim != 1:
        raise ValueError("local_x_m / local_y_m 必须为一维数组")
    if local_x.size < trace_count or local_y.size < trace_count:
        raise ValueError("local_x_m / local_y_m 长度不足，无法覆盖全部道")

    local_x = local_x[:trace_count]
    local_y = local_y[:trace_count]
    if not np.all(np.isfinite(local_x)) or not np.all(np.isfinite(local_y)):
        raise ValueError("local_x_m / local_y_m 包含非有限值")
    step = np.sqrt(np.diff(local_x) ** 2 + np.diff(local_y) ** 2)
    return np.concatenate(([0.0], np.cumsum(step))).astype(np.float32)


def _prepare_metadata_for_resampling_v1(
    trace_metadata: dict, trace_count: int, trace_distance_m: np.ndarray
) -> dict[str, np.ndarray]:
    """[V1 兼容] 复制并裁剪 metadata，确保可用于重采样。"""
    prepared: dict[str, np.ndarray] = {}
    for key, values in trace_metadata.items():
        arr = np.asarray(values)
        if arr.ndim == 0 or arr.size == 1:
            prepared[key] = np.array(arr, copy=True)
            continue
        if arr.ndim != 1:
            raise ValueError(f"trace_metadata['{key}'] 必须为一维数组")
        if arr.size < trace_count:
            raise ValueError(f"trace_metadata['{key}'] 长度不足，无法覆盖全部道")
        prepared[key] = np.array(arr[:trace_count], copy=True)

    prepared["trace_distance_m"] = np.asarray(trace_distance_m, dtype=np.float32).copy()
    return prepared


def _resample_bscan_columns_v1(
    data: np.ndarray,
    source_distance_m: np.ndarray,
    target_distance_m: np.ndarray,
) -> np.ndarray:
    """[V1 兼容] 对 B-scan 的每个采样点沿道方向做线性插值。"""
    samples = data.shape[0]
    resampled = np.empty((samples, target_distance_m.size), dtype=np.float32)
    for row in range(samples):
        resampled[row, :] = np.interp(
            target_distance_m,
            source_distance_m,
            data[row, :],
        ).astype(np.float32)
    return resampled


def _optional_positive_float_v1(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是有限正数") from None
    if not np.isfinite(parsed):
        raise ValueError(f"{name} 必须是有限正数")
    return parsed if parsed > 0.0 else None


def method_motion_compensation_speed_v1(
    data: np.ndarray,
    trace_metadata: dict | None = None,
    spacing_m: float | None = None,
    interpolation_mode: str = "linear",
    **kwargs,
) -> Tuple[np.ndarray, dict]:
    """[V1 兼容保留] 速度误差补偿：按累计距离重采样到等距道轴。"""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("速度误差补偿需要二维 B-scan 数据")

    meta: dict[str, object] = {
        "method": "motion_compensation_speed_v1",
        "interpolation_mode": str(interpolation_mode),
        "source_traces": int(arr.shape[1]),
    }

    if interpolation_mode != "linear":
        raise ValueError(
            f"interpolation_mode '{interpolation_mode}' 不受支持；V1 仅支持 'linear'"
        )

    if trace_metadata is None:
        meta["skipped"] = True
        meta["reason"] = "缺少 trace_metadata，无法进行等距重采样"
        return arr.copy(), meta

    trace_count = int(arr.shape[1])
    try:
        normalized_spacing_m = _optional_positive_float_v1(spacing_m, "spacing_m")
    except ValueError as exc:
        meta["skipped"] = True
        meta["reason"] = str(exc)
        return arr.copy(), meta

    try:
        if "trace_distance_m" in trace_metadata:
            source_distance_m = np.asarray(
                trace_metadata["trace_distance_m"], dtype=np.float64
            )
            if source_distance_m.ndim != 1 or source_distance_m.size < trace_count:
                raise ValueError("trace_metadata['trace_distance_m'] 长度不足或不是一维数组")
            source_distance_m = source_distance_m[:trace_count]
            if not np.all(np.isfinite(source_distance_m)):
                raise ValueError("trace_distance_m 包含非有限值")
            distance_source = "trace_distance_m"
        else:
            source_distance_m = _derive_trace_distance_from_xy_v1(trace_metadata, trace_count)
            distance_source = "local_xy"

        if np.any(np.diff(source_distance_m) < 0):
            raise ValueError("trace_distance_m 必须单调非递减；当前轨迹存在非单调距离")

        metadata_for_resampling = _prepare_metadata_for_resampling_v1(
            trace_metadata,
            trace_count,
            source_distance_m,
        )
        target_distance_m = build_uniform_trace_distance_m(
            source_distance_m,
            spacing_m=normalized_spacing_m,
        )
        trace_metadata_out = resample_trace_metadata(
            metadata_for_resampling,
            target_trace_distance_m=target_distance_m,
        )
    except ValueError as exc:
        meta["skipped"] = True
        meta["reason"] = str(exc)
        return arr.copy(), meta

    corrected = _resample_bscan_columns_v1(
        arr,
        np.asarray(source_distance_m, dtype=np.float64),
        np.asarray(target_distance_m, dtype=np.float64),
    )

    spacing_values = np.diff(np.asarray(target_distance_m, dtype=np.float64))
    positive_spacing = spacing_values[spacing_values > 0]
    effective_spacing_m = float(positive_spacing[0]) if positive_spacing.size else 0.0

    meta.update(
        {
            "distance_source": distance_source,
            "source_traces": int(trace_count),
            "target_traces": int(target_distance_m.size),
            "spacing_m": effective_spacing_m,
            "trace_metadata_out": trace_metadata_out,
        }
    )
    return corrected, meta


def method_motion_compensation_speed(
    data: np.ndarray,
    trace_metadata: dict | None = None,
    spacing_m: float | None = None,
    interpolation_mode: str = "linear",
    **kwargs,
) -> Tuple[np.ndarray, dict]:
    """速度误差补偿：按累计距离重采样到等距道轴。

    默认使用 V2 风格的核心逻辑。

    Args:
        data: 输入 B-scan 数据
        trace_metadata: 每道元数据
        spacing_m: 目标道间距
        interpolation_mode: 插值模式，仅支持 "linear"
        **kwargs: 兼容性参数

    Returns:
        (corrected_data, meta)
    """
    # Check if we should use V1 compatibility mode
    use_v1 = False
    if trace_metadata is None:
        use_v1 = True
    elif interpolation_mode != "linear":
        # V2 only supports linear, so use V1 for other modes (if any)
        use_v1 = True
    elif "height_agl_m" not in trace_metadata:
        # If we don't have height_agl_m, use V1 for compatibility (tests often don't include this)
        use_v1 = True

    if use_v1:
        return method_motion_compensation_speed_v1(
            data=data,
            trace_metadata=trace_metadata,
            spacing_m=spacing_m,
            interpolation_mode=interpolation_mode,
            **kwargs,
        )

    # Use new V2-style core logic
    resolved_spacing = (
        spacing_m if spacing_m and spacing_m > 0.0 else 0.0
    )

    corrected, meta = apply_speed_resampling(
        data=data,
        trace_metadata=trace_metadata,
        resample_spacing_m=resolved_spacing,
    )

    # Rename method to match expected name
    meta["method"] = "motion_compensation_speed"

    # Backward compatibility keys
    if "target_traces" not in meta:
        meta["source_traces"] = meta.get("source_traces", data.shape[1])
    if "resample_spacing_m" in meta:
        meta["spacing_m"] = meta["resample_spacing_m"]

    return corrected, meta
