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

