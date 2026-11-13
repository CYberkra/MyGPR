"""Along-track reference geometry only; no subsurface outline or depth."""
import math
from bisect import bisect_left

from mygpr.domain.markers.models import EventInterval


def track_index(track):
    points = sorted(track.points, key=lambda point: point.trace_index) if track else ()
    return tuple(point.trace_index for point in points), points


def reference_geometry(record, source_id, track, prepared=None):
    if track is None or not track.coordinate_system or track.source == 'trajectory file':
        # Legacy trajectory files can have inferred ordinal indices. Do not use them as trace anchors.
        return None
    if isinstance(record, EventInterval):
        if not record.start or not record.end or record.start.source_id != source_id:
            return None
        start = record.start.trace + (not record.include_start)
        stop = record.end.trace + record.include_end
        interval = True
    else:
        if not record.anchor or record.anchor.source_id != source_id:
            return None
        start, stop, interval = record.anchor.trace, record.anchor.trace + 1, False
    parts: list[list[list[float]]] = []
    current: list[list[float]] = []
    previous = -2
    indices, points = prepared if prepared is not None else track_index(track)
    for point in points[bisect_left(indices, start):bisect_left(indices, stop)]:
        index = point.trace_index
        if not start <= index < stop or not math.isfinite(point.x) or not math.isfinite(point.y):
            continue
        if index != previous + 1 and current:
            parts.append(current)
            current = []
        current.append([float(point.x), float(point.y)])
        previous = index
    if current:
        parts.append(current)
    if not parts:
        return None
    return {'parts': parts, 'crs': track.coordinate_system, 'source': track.source,
            'interval': interval, 'meaning': '沿测线参考范围' if interval else '沿测线参考位置'}


def projected_geometry(projection, mapping, track, prepared=None):
    """Follow retained trajectory samples; interpolate only between adjacent track samples."""
    if not track or not track.coordinate_system or track.source == 'trajectory file' or mapping.kind == 'unknown':
        return None
    indices, points = prepared if prepared is not None else track_index(track)
    groups = [range(a, b) for a, b in projection.spans] or [(p,) for p in projection.positions]
    parts = []
    for group in groups:
        current = []
        previous_root = None
        for position in group:
            left = int(position)
            weight = position - left
            root = float(mapping.coordinates[left] * (1 - weight)
                         + mapping.coordinates[min(left + 1, len(mapping.coordinates) - 1)] * weight)
            index = bisect_left(indices, root)
            xy = None
            if index < len(indices) and indices[index] == root:
                xy = [points[index].x, points[index].y]
            elif 0 < index < len(indices) and indices[index] == indices[index - 1] + 1:
                weight = root - indices[index - 1]
                xy = [getattr(points[index - 1], axis) * (1 - weight) + getattr(points[index], axis) * weight
                      for axis in ('x', 'y')]
            first = bisect_left(indices, previous_root) if previous_root is not None else index
            track_gap = any(b != a + 1 for a, b in zip(indices[first:index], indices[first + 1:index + 1]))
            if xy is None or not all(math.isfinite(v) for v in xy) or track_gap:
                if current:
                    parts.append(current)
                    current = []
            if xy is not None and all(math.isfinite(v) for v in xy):
                current.append([float(v) for v in xy])
            previous_root = root
        if current:
            parts.append(current)
    if not parts:
        return None
    interval = bool(projection.spans)
    return {'parts': parts, 'crs': track.coordinate_system, 'source': track.source,
            'interval': interval, 'meaning': '沿测线参考范围' if interval else '沿测线参考位置',
            'position_status': projection.status}
