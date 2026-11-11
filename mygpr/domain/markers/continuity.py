"""Reject algorithms that cross discontinuities without a segment contract."""
import numpy as np
from mygpr.domain.markers.mapping import SEGMENT_KEY

# These operators work independently on columns, or explicitly select columns.
DISCONTINUOUS_METHODS = frozenset({
    'trace_cut', 'trace_range_select', 'trace_qc', 'time_cut', 'time_to_depth',
    'amplitude_scale', 'manual_gain', 'compensatingGain', 'agcGain', 'sec_gain',
    'dewow', 'set_zero_time', 'frequency_filter_1d', 'hilbert_envelope',
    'energy_decay_gain', 'inverse_q',
})


def require_continuity(request):
    segments = request.trace_metadata.get(SEGMENT_KEY)
    if (segments is not None and np.any(np.diff(segments) != 0)
            and request.method_id not in DISCONTINUOUS_METHODS):
        raise ValueError('当前成果包含不连续道段；此算法不能跨缺口运行，请先选取单个连续区间')
    if request.method_id in ('fk_filter', 'stolt_migration', 'kirchhoff_migration', 'rtm_migration'):
        distance = request.trace_metadata.get('trace_distance_m', request.header_info.get('distance_axis_m'))
        if distance is not None and len(distance) > 1:
            delta = np.diff(np.asarray(distance, dtype=float))
            if not np.all(np.isfinite(delta)) or np.any(delta <= 0) or not np.allclose(delta, delta[0], rtol=1e-3, atol=1e-6):
                raise ValueError('此算法要求连续且等间距的测线；请先核对距离并重采样单个连续区间')

