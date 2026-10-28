"""Conservative mapping between compatible B-scan views."""
import numpy as np


def _physical_axis(view):
    label = str(view._sample_axis_label).lower()
    unit = 'ns' if 'ns' in label else 'm' if '(m)' in label else ''
    axis = view._sample_axis
    if view._showing_elevation or not unit or axis is None:
        return None, ''
    axis = np.asarray(axis, dtype=float)
    if axis.size < 2 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
        return None, ''
    return axis, unit


def compatible(source, target):
    a, unit = _physical_axis(source)
    b, other = _physical_axis(target)
    if a is None or b is None or unit != other or min(a[-1], b[-1]) <= max(a[0], b[0]):
        return False
    ids_a, ids_b = source._source_trace_ids, target._source_trace_ids
    if ids_a is not None and ids_b is not None:
        if len(np.unique(ids_a)) != len(ids_a) or len(np.unique(ids_b)) != len(ids_b):
            return False
        return bool(np.intersect1d(ids_a, ids_b).size)
    a, b = source._trace_axis_m, target._trace_axis_m
    if a is None or b is None:
        return False
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bool(len(a) > 1 and len(b) > 1 and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                and np.all(np.diff(a) > 0) and np.all(np.diff(b) > 0)
                and min(a[-1], b[-1]) > max(a[0], b[0]))


def map_point(source, target, x, y):
    sy, unit = _physical_axis(source)
    ty, other_unit = _physical_axis(target)
    if sy is None or ty is None or unit != other_unit:
        return None
    y_index = y - .5
    if not 0 <= y_index <= len(sy) - 1:
        return None
    physical_y = np.interp(y_index, np.arange(len(sy)), sy)
    if not ty[0] <= physical_y <= ty[-1]:
        return None
    target_y = float(np.interp(physical_y, ty, np.arange(len(ty)))) + .5
    source_ids, target_ids = source._source_trace_ids, target._source_trace_ids
    if source_ids is not None and target_ids is not None:
        index = int(round(x - .5))
        if not 0 <= index < len(source_ids):
            return None
        if np.count_nonzero(source_ids == source_ids[index]) != 1:
            return None
        matches = np.flatnonzero(target_ids == source_ids[index])
        if len(matches) != 1:
            return None
        return float(matches[0]) + .5, target_y
    a, b = source._trace_axis_m, target._trace_axis_m
    if a is None or b is None:
        return None
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.any(np.diff(a) <= 0) or np.any(np.diff(b) <= 0):
        return None
    index = x - .5
    if not 0 <= index <= len(a) - 1:
        return None
    distance = np.interp(index, np.arange(len(a)), a)
    if not b[0] <= distance <= b[-1]:
        return None
    return float(np.interp(distance, b, np.arange(len(b)))) + .5, target_y


def _edges(axis):
    axis = np.asarray(axis, dtype=float)
    middle = (axis[:-1] + axis[1:]) / 2
    return np.r_[axis[0] - (middle[0] - axis[0]), middle,
                 axis[-1] + (axis[-1] - middle[-1])]


def map_range(source, target, ranges):
    if not compatible(source, target):
        return None
    (left, right), (top, bottom) = ranges
    sy, _ = _physical_axis(source)
    ty, _ = _physical_axis(target)
    same_x = (source._source_trace_ids is not None and target._source_trace_ids is not None
              and np.array_equal(source._source_trace_ids, target._source_trace_ids))
    if same_x and np.array_equal(sy, ty):
        return ((max(0., left), min(float(len(source._source_trace_ids)), right)),
                (max(0., top), min(float(len(sy)), bottom)))
    se, te = _edges(sy), _edges(ty)
    low = max(float(np.interp(top, np.arange(len(se)), se)), te[0])
    high = min(float(np.interp(bottom, np.arange(len(se)), se)), te[-1])
    if high <= low:
        return None
    yrange = tuple(float(np.interp(value, te, np.arange(len(te)))) for value in (low, high))
    ids_a, ids_b = source._source_trace_ids, target._source_trace_ids
    if ids_a is not None and ids_b is not None:
        source_indices = np.arange(len(ids_a))
        visible = (source_indices + 1 > left) & (source_indices < right)
        matches = np.flatnonzero(np.isin(ids_b, ids_a[visible]))
        if not len(matches):
            return None
        xrange = (float(matches.min()), float(matches.max() + 1))
    else:
        a, b = _edges(source._trace_axis_m), _edges(target._trace_axis_m)
        low = max(float(np.interp(left, np.arange(len(a)), a)), b[0])
        high = min(float(np.interp(right, np.arange(len(a)), a)), b[-1])
        if high <= low:
            return None
        xrange = tuple(float(np.interp(value, b, np.arange(len(b)))) for value in (low, high))
    return xrange, yrange
