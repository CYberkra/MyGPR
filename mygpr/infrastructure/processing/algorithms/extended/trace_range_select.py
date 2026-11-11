"""Keep/exclude explicit trace ranges; no detector or inherited QC mask."""
import numpy as np

from mygpr.domain.markers.mapping import index_spans
from mygpr.domain.markers.ranges import selection_indices


def native_trace_range_select(data, params):
    matrix = np.asarray(data)
    count = matrix.shape[1]
    ranges = params.get('ranges', ())
    groups = []
    if not isinstance(ranges, (list, tuple)) or not ranges:
        raise ValueError('请提供非空的道范围集合')
    for pair in ranges:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(value) is not int for value in pair)):
            raise ValueError('道范围必须为整数起点及不包含的终点')
        start, stop = pair
        if not 0 <= start < stop <= count:
            raise ValueError('道范围越界或为空')
        groups.append(np.arange(start, stop, dtype=np.int64))
    indices = selection_indices(count, groups, params.get('mode', 'keep'))
    metadata = {}
    for key, value in params.get('_trace_metadata', {}).items():
        values = np.asarray(value)
        metadata[key] = values[indices].copy() if values.ndim and values.shape[0] == count else values.copy()
    header = params.get('_header_info', {})
    updates = {'source_trace_offset': None}
    axis = header.get('distance_axis_m')
    if axis is not None:
        axis = np.asarray(axis)
        if axis.shape != (count,):
            raise ValueError('距离轴与道数不一致')
        kept = axis[indices]
        updates['distance_axis_m'] = kept
        if np.all(np.isfinite(kept)):
            updates['track_length_m'] = float(abs(kept[-1] - kept[0]))
    return matrix[:, indices].astype(np.float32, copy=True), {
        'method': 'trace_range_select', 'selected_input_spans': index_spans(indices),
        'input_traces': count, 'output_traces': len(indices),
        'trace_metadata_out': metadata, 'header_info_updates': updates,
    }
