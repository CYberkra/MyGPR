"""Reopen immutable amplitude grids without silently recomputing their data."""
from __future__ import annotations

from dataclasses import dataclass

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.spatial.amplitude import AmplitudeResult, AmplitudeSource
from mygpr.application.spatial.curtain_preview import read_curtain_previews
from mygpr.domain.spatial.amplitude_slice import AmplitudeGrid


@dataclass(frozen=True, slots=True)
class LoadedAmplitudeSlice:
    result: AmplitudeResult
    result_id: str
    stale: bool
    warning: str = ''


def load_saved_slice(projects, project_id: str, result_id: str,
                     *, context: ExecutionContext | None = None) -> LoadedAmplitudeSlice:
    ctx = context or ExecutionContext.null()
    ctx.raise_if_cancelled()
    payload = projects.load_amplitude_result(project_id, result_id)
    records = [item for item in projects.list_spatial_results(project_id) if item.result_id == result_id]
    if len(records) != 1:
        raise ValueError('保存的空间成果不存在或不唯一')
    stale = bool(payload['stale'] or records[0].stale)
    source_lines = payload['sources']
    sources = tuple(AmplitudeSource(str(line_id), str(item.get('artifact_id') or ''),
                   str(item.get('component') or ''), str(item.get('content_sha256') or ''),
                   str(item.get('position_sha256') or '')) for line_id, item in source_lines.items())
    if not sources:
        raise ValueError('保存的切片缺少测线来源')
    grid = AmplitudeGrid(payload['values'], payload['trace_counts'], payload['line_counts'],
                         payload['x_origin_m'], payload['y_origin_m'], payload['cell_size_m'])
    curtains, warning = (), ''
    if not stale:
        try:
            curtains = read_curtain_previews(projects, project_id, sources, context=ctx)
        except (ValueError, OSError, KeyError) as exc:
            warning = f'已打开保存的切片，三维剖面不可用：{exc}'
    ctx.raise_if_cancelled()
    result = AmplitudeResult(grid, sources, payload['coordinate_system'],
                             float(payload['start_ns']), float(payload['end_ns']),
                             str(payload['attribute']), curtains, payload['source_state_token'])
    return LoadedAmplitudeSlice(result, result_id, stale, warning)
