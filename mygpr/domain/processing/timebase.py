"""Sampling interval contract shared by native and compatibility executors."""
from math import isfinite, isclose, ulp
from typing import Any, Mapping
from mygpr.domain.common.scalars import to_float


def sampling_interval_s(header: Mapping[str, Any] | None, samples: int) -> float:
    header = header or {}
    if header.get('time_axis_status') == 'unknown':
        return 0.0
    explicit = to_float(header.get('time_step_s'), default=0.0)
    if isfinite(explicit) and explicit > 0:
        return explicit
    window = to_float(header.get('total_time_ns'), default=0.0)
    if window <= 0:
        window = to_float(header.get('time_window_ns'), default=0.0)
    if not isfinite(window) or window <= 0:
        return 0.0
    # Unmarked legacy headers retain their historical N-bin convention.
    denominator = max(1, samples - 1 if header.get('time_axis_endpoint', False) else samples)
    return window * 1e-9 / denominator


def require_time_axis(method_id: str, header: Mapping[str, Any] | None) -> None:
    if (header or {}).get('time_axis_status') == 'unknown' and method_id in {
        'time_cut', 'time_to_depth', 'frequency_filter_1d', 'fk_filter',
        'inverse_q', 'mixed_phase_deconvolution', 'rtm_migration', 'kirchhoff_migration', 'stolt_migration',
    }:
        raise ValueError('成果缺少有效时间轴，无法执行该时间域方法')


def snap_sample_index(value: float) -> float:
    """Remove floating-point roundoff at integer sample boundaries only."""
    nearest = round(value)
    if isclose(value, nearest, rel_tol=0.0, abs_tol=8 * ulp(max(1.0, abs(value)))):
        return float(nearest)
    return value
