"""Along-track reference geometry only; no subsurface outline or depth."""
import math

from mygpr.domain.markers.models import EventInterval


def reference_geometry(record, source_id, track):
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
    for point in track.points:
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
