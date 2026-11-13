"""Deterministic event selection. Public trace indices are zero-based."""
from datetime import datetime

import numpy as np

from mygpr.domain.markers.mapping import TraceMapping
from mygpr.domain.markers.models import EventInterval


def interval_indices(interval: EventInterval, mapping: TraceMapping, *,
                     timestamps_s=None, time_basis_confirmed=False) -> np.ndarray:
    left, right = interval_bounds(interval, mapping, timestamps_s=timestamps_s,
                                  time_basis_confirmed=time_basis_confirmed)
    return np.arange(left, right, dtype=np.int64)


def interval_bounds(interval: EventInterval, mapping: TraceMapping, *,
                    timestamps_s=None, time_basis_confirmed=False) -> tuple[int, int]:
    """Resolve two bounds without allocating every selected index during list projection."""
    if interval.excluded:
        raise ValueError('区间已排除')
    if not interval.closed:
        raise ValueError('区间尚未闭合')
    count = len(mapping.coordinates)
    if interval.start and interval.end:
        source_id = interval.start.source_id
        if source_id == mapping.target_id:
            left = min(count, interval.start.trace + (not interval.include_start))
            right = min(count, interval.end.trace + interval.include_end)
            return int(left), int(max(left, right))
        elif source_id == mapping.source_id and mapping.kind != 'unknown':
            values = mapping.coordinates
        else:
            raise ValueError('区间无法映射到当前数据版本')
        low, high = float(interval.start.trace), float(interval.end.trace)
    else:
        if not interval.time_source_id or (interval.time_source_id != mapping.target_id
                and (interval.time_source_id != mapping.source_id or mapping.kind == 'unknown')):
            raise ValueError('区间采集时间不属于当前数据版本')
        values = np.asarray(timestamps_s, dtype=np.float64)
        if (not time_basis_confirmed or values.shape != (count,)
                or not np.all(np.isfinite(values)) or np.any(np.diff(values) < 0)):
            raise ValueError('采集时间缺失、回跳或时间基准未确认，请按道号选择')
        low = datetime.fromisoformat(interval.time_start).timestamp()
        high = datetime.fromisoformat(interval.time_end).timestamp()
    # All accepted coordinate arrays are monotonic. Search avoids an O(N) mask per record.
    left = np.searchsorted(values, low, side='left' if interval.include_start else 'right')
    right = np.searchsorted(values, high, side='right' if interval.include_end else 'left')
    return int(left), int(max(left, right))


def selection_indices(count: int, groups, mode: str) -> np.ndarray:
    if mode not in ('keep', 'exclude'):
        raise ValueError('区间操作必须是 keep 或 exclude')
    selected = np.zeros(count, dtype=bool)
    for group in groups:
        indices = np.asarray(group)
        if indices.size == 0:
            continue
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= count):
            raise ValueError('区间道号越界或格式无效')
        selected[indices] = True
    result = np.flatnonzero(selected if mode == 'keep' else ~selected)
    if not result.size:
        raise ValueError('不能生成零道成果')
    return result
