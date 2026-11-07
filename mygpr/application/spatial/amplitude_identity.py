"""Source identity and trace-ID validation for amplitude analysis."""
import hashlib
import struct

import numpy as np


def metadata_values(metadata, key):
    """Follow nested storage metadata without guessing values or looping."""
    values, seen = [], set()
    while hasattr(metadata, 'get') and id(metadata) not in seen:
        seen.add(id(metadata))
        value = metadata.get(key)
        if value is not None and value != '':
            values.append(value)
        metadata = metadata.get('metadata')
    return values


def track_position_fingerprint(track) -> str:
    """Hash the actual ordered trace locations and altitude metadata."""
    digest = hashlib.sha256(str(track.coordinate_system or '').encode('utf-8'))
    for point in track.points:
        values = []
        for name in ('x', 'y', 'elevation_m', 'flight_height_m', 'ground_elevation_m'):
            value = getattr(point, name, None)
            values.append(float(value) if value is not None else float('nan'))
        digest.update(struct.pack('<q5d', int(point.trace_index), *values))
    return digest.hexdigest()


def source_digest(projects, project_id, source, info):
    if source.artifact_id:
        if not hasattr(projects, 'list_artifacts'):
            return ''
        matches = [a for a in projects.list_artifacts(project_id, source.line_id)
                   if str(a.artifact_id) == source.artifact_id]
        if len(matches) != 1:
            raise ValueError(f'{source.line_id} 的处理成果不存在或不唯一')
        return str(matches[0].sha256 or '')
    hashes = metadata_values(info.metadata, 'data_sha256')
    return str(hashes[0]) if hashes else ''


def trace_ids(metadata, expected: int) -> np.ndarray:
    ids = np.asarray(metadata.get('trace_id', [])).reshape(-1)
    if len(ids) != expected:
        raise ValueError('处理成果缺少完整的原始道标识，不能映射位置')
    if ids.dtype.kind in 'fc' and not np.isfinite(ids).all():
        raise ValueError('道标识存在无效值，不能映射位置')
    ids = ids.astype(str)
    if np.any(ids == '') or len(np.unique(ids)) != len(ids):
        raise ValueError('道标识存在空值或重复，不能映射位置')
    return ids
