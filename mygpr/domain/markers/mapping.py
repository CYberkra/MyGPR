"""Horizontal location mapping, separate from measured waveforms and device IDs."""
from dataclasses import dataclass
from hashlib import sha256
from functools import cached_property

import numpy as np

from mygpr.domain.markers.models import MarkerAnchor

COORDINATE_KEY = '_marker_source_coordinate'
SEGMENT_KEY = '_marker_segment'
MAPPING_KEYS = frozenset((COORDINATE_KEY, SEGMENT_KEY))


def is_mapping_key(key: str) -> bool:
    return key.startswith('_marker_')


def add_source_mapping(header, metadata, source_id, count):
    """Register the current grid without pretending it is an original grid."""
    key = '_marker_map_' + sha256(source_id.encode()).hexdigest()
    maps = dict(header.get('marker_maps') or {})
    maps[source_id] = {'key': key, 'kind': 'identity'}
    header['marker_maps'] = maps
    metadata[key] = np.arange(count, dtype=np.float64)
    metadata.setdefault(SEGMENT_KEY, np.zeros(count, dtype=np.int64))


def read_mapping(header, metadata, source_id, target_id, count):
    segments = metadata.get(SEGMENT_KEY, np.zeros(count, dtype=np.int64))
    if source_id == target_id:
        return TraceMapping(source_id, target_id, np.arange(count), segments)
    descriptor = (header.get('marker_maps') or {}).get(source_id)
    if descriptor and descriptor['key'] in metadata:
        return TraceMapping(source_id, target_id, metadata[descriptor['key']], segments, descriptor['kind'])
    if source_id == header.get('marker_source_id') and COORDINATE_KEY in metadata:
        return TraceMapping(source_id, target_id, metadata[COORDINATE_KEY], segments,
                            header.get('marker_mapping_kind', 'unknown'))
    return TraceMapping(source_id, target_id, np.arange(count), segments, 'unknown')


@dataclass(frozen=True)
class TraceMapping:
    source_id: str
    target_id: str
    coordinates: np.ndarray
    segments: np.ndarray
    kind: str = 'identity'

    def __post_init__(self):
        coords = np.array(self.coordinates, dtype=np.float64, copy=True)
        segments = np.array(self.segments, dtype=np.int64, copy=True)
        if (coords.ndim != 1 or segments.shape != coords.shape or not np.all(np.isfinite(coords))
                or np.any(coords < 0) or np.any(segments < 0)):
            raise ValueError('道映射数组无效')
        if self.kind not in ('identity', 'selection', 'resampled', 'unknown'):
            raise ValueError('道映射类型无效')
        if coords.size > 1 and (np.any(np.diff(coords) <= 0) or np.any(np.diff(segments) < 0)):
            raise ValueError('道映射必须严格单调且分段有序')
        coords.setflags(write=False)
        segments.setflags(write=False)
        object.__setattr__(self, 'coordinates', coords)
        object.__setattr__(self, 'segments', segments)

    @classmethod
    def identity(cls, source_id: str, count: int):
        return cls(source_id, source_id, np.arange(count, dtype=np.float64),
                   np.zeros(count, dtype=np.int64))

    @cached_property
    def segment_breaks(self):
        return np.flatnonzero(np.diff(self.segments) != 0) + 1

    def spans(self, start, stop):
        if start >= stop:
            return ()
        breaks = self.segment_breaks
        first, last = np.searchsorted(breaks, [start, stop], side='left')
        cuts = [start, *(int(v) for v in breaks[first:last] if v > start), stop]
        return tuple(zip(cuts[:-1], cuts[1:]))

    def select(self, indices, target_id: str = ''):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(self.coordinates)):
            raise ValueError('选择道号越界')
        if indices.size > 1 and np.any(np.diff(indices) <= 0):
            raise ValueError('选择道号必须递增且不重复')
        breaks = np.r_[False, (np.diff(indices) != 1)
                       | (np.diff(self.segments[indices]) != 0)] if indices.size else []
        return TraceMapping(self.source_id, target_id or self.target_id, self.coordinates[indices],
                            np.cumsum(breaks, dtype=np.int64),
                            kind='resampled' if self.kind == 'resampled' else 'selection')


@dataclass(frozen=True)
class MarkerProjection:
    status: str
    positions: tuple[float, ...] = ()
    spans: tuple[tuple[int, int], ...] = ()


def project_anchor(anchor: MarkerAnchor | None, mapping: TraceMapping) -> MarkerProjection:
    if anchor is None:
        return MarkerProjection('unmapped')
    count = len(mapping.coordinates)
    if anchor.source_id == mapping.target_id:
        return (MarkerProjection('exact', (float(anchor.trace),)) if anchor.trace < count
                else MarkerProjection('outside'))
    if anchor.source_id != mapping.source_id or mapping.kind == 'unknown':
        return MarkerProjection('unmapped')
    coordinates = mapping.coordinates
    index = int(np.searchsorted(coordinates, anchor.trace))
    if index < count and coordinates[index] == anchor.trace:
        return MarkerProjection('interpolated' if mapping.kind == 'resampled' else 'exact', (float(index),))
    if (mapping.kind == 'resampled' and 0 < index < count
            and mapping.segments[index - 1] == mapping.segments[index]):
        left, right = coordinates[index - 1:index + 1]
        position = index - 1 + (anchor.trace - left) / (right - left)
        return MarkerProjection('interpolated', (float(position),))
    return MarkerProjection('outside')


def index_spans(indices, segments=None) -> tuple[tuple[int, int], ...]:
    values = np.asarray(indices, dtype=np.int64)
    if not values.size:
        return ()
    split = np.diff(values) != 1
    if segments is not None:
        split |= np.diff(np.asarray(segments)[values]) != 0
    groups = np.split(values, np.flatnonzero(split) + 1)
    return tuple((int(group[0]), int(group[-1]) + 1) for group in groups)
