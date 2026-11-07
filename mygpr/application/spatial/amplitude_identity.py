"""Source identity and trace-ID validation for amplitude analysis."""
import numpy as np


def source_digest(projects, project_id, source, info):
    if source.artifact_id:
        if not hasattr(projects, 'list_artifacts'):
            return ''
        matches = [a for a in projects.list_artifacts(project_id, source.line_id)
                   if str(a.artifact_id) == source.artifact_id]
        if len(matches) != 1:
            raise ValueError(f'{source.line_id} 的处理成果不存在或不唯一')
        return str(matches[0].sha256 or '')
    metadata = info.metadata
    nested = metadata.get('metadata')
    nested = nested if hasattr(nested, 'get') else {}
    return str(metadata.get('data_sha256') or nested.get('data_sha256') or '')


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
