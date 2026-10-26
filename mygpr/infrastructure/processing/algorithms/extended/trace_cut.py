from __future__ import annotations

from typing import Any

import numpy as np

from mygpr.domain.common.scalars import to_int


def method_trace_cut(
    data: np.ndarray,
    trace_start: int = 1,
    trace_end: int = 0,
    trace_metadata: dict[str, Any] | None = None,
    header_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"输入数据必须是2维数组，当前 shape={arr.shape}")
    trace_count = int(arr.shape[1])
    if trace_count == 0:
        raise ValueError("输入数据没有有效道")

    first = to_int(trace_start, default=1)
    last = to_int(trace_end, default=0) or trace_count
    if first < 1 or last < 1:
        raise ValueError("测道裁剪的起止道号必须从 1 开始；结束道号为 0 时表示最后一道")
    if first > trace_count or last > trace_count:
        raise ValueError(f"测道裁剪范围 {first}–{last} 超出当前数据道数 {trace_count}")
    if first > last:
        raise ValueError("测道裁剪的起始道号不能大于结束道号")

    start = first - 1
    result = arr[:, start:last].astype(np.float32, copy=True)
    source = dict(trace_metadata or {})
    source.setdefault("trace_index", np.arange(trace_count, dtype=np.int32))
    metadata_out: dict[str, Any] = {}
    for key, value in source.items():
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == trace_count:
            metadata_out[key] = np.array(array[start:last], copy=True)
        else:
            metadata_out[key] = np.array(array, copy=True)

    header = dict(header_info or {})
    distance = source.get("trace_distance_m")
    if distance is None:
        distance = header.get("distance_axis_m")
    distance = np.asarray(distance if distance is not None else [], dtype=np.float64).reshape(-1)
    if distance.size == trace_count:
        kept_distance = distance[start:last]
        finite = kept_distance[np.isfinite(kept_distance)]
        if finite.size == kept_distance.size:
            track_length = abs(float(finite[-1] - finite[0])) if finite.size > 1 else 0.0
        else:
            track_length = float(header.get("track_length_m", 0.0) or 0.0)
        differences = np.abs(np.diff(kept_distance))
        differences = differences[np.isfinite(differences) & (differences > 0)]
        interval = float(np.median(differences)) if differences.size else 0.0
    else:
        track_length = float(header.get("track_length_m", 0.0) or 0.0)
        interval = float(header.get("trace_interval_m", 0.0) or 0.0)
    offset = to_int(header.get("source_trace_offset"), default=0) + start
    return result, {
        "method": "trace_cut",
        "trace_start": first,
        "trace_end": last,
        "trace_start_index": start,
        "trace_end_index": last,
        "input_traces": trace_count,
        "output_traces": int(result.shape[1]),
        "trace_metadata_out": metadata_out,
        "header_info_updates": {
            "track_length_m": track_length,
            "trace_interval_m": interval,
            "source_trace_offset": offset,
        },
    }
