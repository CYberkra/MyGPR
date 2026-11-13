"""Protect trace-coordinate provenance from generic sensor interpolation."""
import numpy as np

from mygpr.domain.markers.mapping import COORDINATE_KEY, SEGMENT_KEY, TraceMapping


def update_mapping(request, output, metadata, header, trace_metadata):
    from mygpr.infrastructure.processing.algorithms.methods import AXIS_TRANSFORM_BY_METHOD

    from mygpr.domain.markers.mapping import is_mapping_key
    for key in list(trace_metadata):
        if is_mapping_key(key):
            trace_metadata.pop(key)
    maps = dict(request.header_info.get('marker_maps') or {})
    source_id = request.header_info.get('marker_source_id', '')
    if source_id and COORDINATE_KEY in request.trace_metadata:
        maps.setdefault(source_id, {'key': COORDINATE_KEY,
            'kind': request.header_info.get('marker_mapping_kind', 'unknown')})
    # Segment provenance also exists for CLI arrays with no project/source identity.
    local = TraceMapping('', '', np.arange(request.data.shape[1], dtype=float),
                         request.trace_metadata.get(SEGMENT_KEY, np.zeros(request.data.shape[1], dtype=np.int64)), 'identity')
    transformed = _transform(local, request, output, metadata, trace_metadata, AXIS_TRANSFORM_BY_METHOD)
    if transformed is not None and (maps or SEGMENT_KEY in request.trace_metadata
                                     or request.method_id == 'trace_range_select'
                                     or request.method_id == 'trace_qc' and metadata.get('mode') == 'remove'):
        trace_metadata[SEGMENT_KEY] = transformed.segments.copy()
    updated = {}
    for source_id, descriptor in maps.items():
        coords = request.trace_metadata.get(descriptor['key'])
        segments = request.trace_metadata.get(SEGMENT_KEY)
        if coords is None or segments is None or descriptor['kind'] == 'unknown':
            continue
        mapping = TraceMapping(source_id, '', coords, segments, descriptor['kind'])
        result = _transform(mapping, request, output, metadata, trace_metadata, AXIS_TRANSFORM_BY_METHOD)
        if result is None or len(result.coordinates) != output.shape[1]:
            continue
        trace_metadata[descriptor['key']] = result.coordinates.copy()
        trace_metadata[SEGMENT_KEY] = result.segments.copy()
        updated[source_id] = {**descriptor, 'kind': result.kind}
    if maps:
        header['marker_maps'] = updated
        header['marker_mapping_kind'] = 'unknown'
        root = header.get('marker_source_id')
        if root in updated:
            header['marker_mapping_kind'] = updated[root]['kind']


def _transform(mapping, request, output, metadata, trace_metadata, transforms):
    method = request.method_id
    if method == 'trace_range_select':
        indices = np.concatenate([np.arange(start, stop, dtype=np.int64)
                                  for start, stop in metadata['selected_input_spans']])
        return mapping.select(indices)
    if method == 'trace_cut':
        return mapping.select(np.arange(metadata['trace_start_index'], metadata['trace_end_index']))
    if method == 'trace_qc':
        indices = np.arange(len(mapping.coordinates))
        if metadata.get('mode') == 'remove':
            indices = np.delete(indices, metadata['bad_trace_indices'])
        return mapping.select(indices)
    transform = transforms.get(method)
    if transform in ('identity', 'sample_crop', 'time_to_depth', 'migration'):
        return mapping if output.shape[1] == len(mapping.coordinates) else None
    if transform == 'trace_resample':
        if metadata.get('skipped') is True and output.shape[1] == len(mapping.coordinates):
            return mapping
        return _resampled_mapping(mapping, request, trace_metadata, output.shape[1], metadata)
    return None


def _resampled_mapping(mapping, request, trace_metadata, count, metadata):
    before = metadata.get('marker_input_distance_m', request.trace_metadata.get('trace_distance_m'))
    after = trace_metadata.get('trace_distance_m')
    if before is None or after is None:
        return None
    before, after = np.asarray(before), np.asarray(after)
    if (before.shape != mapping.coordinates.shape or after.shape != (count,)
            or not np.all(np.isfinite(before)) or not np.all(np.isfinite(after))
            or np.any(np.diff(before) <= 0) or np.any(np.diff(after) <= 0)
            or after[0] < before[0] or after[-1] > before[-1]
            or np.any(np.diff(mapping.segments) != 0)):
        return None
    coords = np.interp(after, before, mapping.coordinates)
    return TraceMapping(mapping.source_id, '', coords, np.zeros(count, dtype=np.int64), 'resampled')
