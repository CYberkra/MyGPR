"""Read full-resolution trace blocks and associate them with measured positions."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.spatial.amplitude_identity import source_digest, trace_ids, track_position_fingerprint
from mygpr.domain.spatial.amplitude_slice import AmplitudeGrid, grid_line_values, window_attribute


@dataclass(frozen=True, slots=True)
class AmplitudeSource:
    line_id: str
    artifact_id: str = ''
    component: str = ''
    content_sha256: str = ''
    position_sha256: str = ''


@dataclass(frozen=True, slots=True)
class LineObservations:
    source: AmplitudeSource
    coordinate_system: str
    trace_indices: np.ndarray
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    source_position_sha256: str = ''


@dataclass(frozen=True, slots=True)
class AmplitudeResult:
    grid: AmplitudeGrid
    sources: tuple[AmplitudeSource, ...]
    coordinate_system: str
    start_ns: float
    end_ns: float
    attribute: str
    curtains: tuple = ()
    source_state_token: str = ''


def collect_observations(
    projects,
    project_id: str,
    sources: Sequence[AmplitudeSource],
    start_ns: float,
    end_ns: float,
    *,
    attribute: str = 'rms',
    context: ExecutionContext | None = None,
    positions: Sequence[LineObservations] | None = None,
) -> tuple[LineObservations, ...]:
    """Calculate per-trace attributes; coordinates remain in each track CRS.

    A processing result must retain unique raw trace identifiers. The caller
    must transform all coordinates to one confirmed metric CRS before gridding.
    """
    if not sources:
        raise ValueError('请选择测线')
    ctx = context or ExecutionContext.null()
    tracks = {str(track.line_id): track for track in projects.load_spatial_tracks(project_id)}
    projected = {item.source.line_id: item for item in (positions or ())}
    if len({source.line_id for source in sources}) != len(sources):
        raise ValueError('每条测线只能选择一个输入成果')
    observations = []
    for source_index, source in enumerate(sources):
        ctx.raise_if_cancelled()
        track = tracks.get(source.line_id)
        if track is None or not track.points:
            raise ValueError(f'{source.line_id} 没有有效定位轨迹')
        artifact = bool(source.artifact_id)
        info = (projects.get_artifact_dataset_info(project_id, source.line_id, source.artifact_id)
                if artifact else projects.get_dataset_info(project_id, source.line_id))
        samples, trace_count = info.shape
        metadata = info.metadata
        if (metadata.get('time_axis_status') == 'unknown' or metadata.get('is_depth')
                or metadata.get('is_elevation')):
            raise ValueError(f'{source.line_id} 没有可确认的时域时间轴')
        axis = np.asarray(metadata.get('time_axis_ns', []), dtype=np.float64)
        if axis.shape != (samples,):
            raise ValueError(f'{source.line_id} 缺少完整时间轴')
        component = str(source.component).lower()
        nested = metadata.get('metadata')
        nested = nested if hasattr(nested, 'get') else {}
        stored = str(metadata.get('store_component') or nested.get('store_component') or '').lower()
        aliases = {'magnitude': 'mag'}
        if not component:
            component = stored
        if component not in ('real', 'mag', 'magnitude'):
            raise ValueError(f'{source.line_id} 的输入分量未确认')
        if stored and aliases.get(stored, stored) != aliases.get(component, component):
            raise ValueError(f'{source.line_id} 的输入分量与存储记录不一致')
        source = replace(source, component=component,
                         content_sha256=source_digest(projects, project_id, source, info))
        projected_line = projected.get(source.line_id)
        if positions is not None and projected_line is None:
            raise ValueError(f'{source.line_id} 缺少已投影的轨迹')
        if projected_line is not None:
            position_digest = track_position_fingerprint(track)
            if (projected_line.source_position_sha256
                    and projected_line.source_position_sha256 != position_digest):
                raise ValueError(f'{source.line_id} 的轨迹已变化，请重新生成切片')
            if not (len(projected_line.trace_indices) == len(projected_line.x) == len(projected_line.y)):
                raise ValueError(f'{source.line_id} 的投影轨迹长度不匹配')
            position_map = {int(i): (float(x), float(y)) for i, x, y in
                            zip(projected_line.trace_indices, projected_line.x, projected_line.y)}
            coordinate_system = projected_line.coordinate_system
        else:
            position_map = {int(point.trace_index): (float(point.x), float(point.y)) for point in track.points}
            coordinate_system = str(track.coordinate_system)
            position_digest = track_position_fingerprint(track)
        source = replace(source, position_sha256=position_digest)
        if artifact:
            original = projects.get_dataset_info(project_id, source.line_id)
            original_ids = trace_ids(projects.read_trace_metadata(project_id, source.line_id), original.shape[1])
            result_ids = trace_ids(projects.read_artifact_trace_metadata(
                project_id, source.line_id, source.artifact_id), trace_count)
            raw_by_id = {value: index for index, value in enumerate(original_ids)}
            indices = np.array([raw_by_id.get(value, -1) for value in result_ids], dtype=np.int64)
            if np.any(indices < 0):
                raise ValueError(f'{source.line_id} 的成果道标识与原始测线不匹配')
        else:
            indices = np.arange(trace_count, dtype=np.int64)
        observations.append(_read_line_observations(
            projects, project_id, source, info, axis, indices, position_map,
            coordinate_system, start_ns, end_ns, attribute, ctx))
        ctx.report_progress(source_index + 1, len(sources), f'已处理 {source.line_id}')
    return tuple(observations)


def _read_line_observations(projects, project_id, source, info, axis, indices, position_map,
                            coordinate_system, start_ns, end_ns, attribute, context):
    samples, trace_count = info.shape
    if samples > 4_000_000:
        raise ValueError('单道采样点数过大，无法安全计算完整道包络')
    chunk = max(1, min(128, 4_000_000 // samples))
    selected_indices, xs, ys, values = [], [], [], []
    for start in range(0, trace_count, chunk):
        context.raise_if_cancelled()
        end = min(trace_count, start + chunk)
        if source.artifact_id:
            blocks = projects.iter_artifact_blocks(project_id, source.line_id, source.artifact_id,
                block_rows=samples, trace_start=start, trace_end=end)
        else:
            blocks = projects.iter_dataset_blocks(project_id, source.line_id,
                block_rows=samples, trace_start=start, trace_end=end)
        blocks = iter(blocks)
        first = next(blocks, None)
        if (first is None or first[0] != 0 or first[1] != samples
                or np.shape(first[2]) != (samples, end - start) or next(blocks, None) is not None):
            raise ValueError(f'{source.line_id} 波形分块不完整')
        calculated = window_attribute(first[2], axis, start_ns, end_ns,
                                      component=source.component, attribute=attribute)
        for local, value in enumerate(calculated):
            trace_index = start + local
            point = position_map.get(int(indices[trace_index]))
            if point is None or not np.isfinite(point).all():
                continue
            selected_indices.append(trace_index)
            xs.append(point[0])
            ys.append(point[1])
            values.append(value)
    if source.content_sha256:
        current_info = (projects.get_artifact_dataset_info(project_id, source.line_id, source.artifact_id)
                        if source.artifact_id else projects.get_dataset_info(project_id, source.line_id))
        if source_digest(projects, project_id, source, current_info) != source.content_sha256:
            raise ValueError(f'{source.line_id} 的数据在计算期间发生变化，请重新生成切片')
    return LineObservations(source, coordinate_system, np.asarray(selected_indices, dtype=np.int64),
                            np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64),
                            np.asarray(values, dtype=np.float64))


def build_amplitude_grid(
    projects, project_id: str, sources: Sequence[AmplitudeSource],
    positions: Sequence[LineObservations], start_ns: float, end_ns: float,
    *, cell_size_m: float, attribute: str = 'rms', context: ExecutionContext | None = None,
    include_curtains: bool = False,
) -> AmplitudeResult:
    """Create a bounded preview grid from positions projected by the caller."""
    crs = {str(item.coordinate_system) for item in positions}
    if len(crs) != 1 or not next(iter(crs)):
        raise ValueError('参与切片的测线必须使用同一米制坐标系')
    token_reader = getattr(projects, 'spatial_source_state_token', None)
    source_token = token_reader(project_id, [s.line_id for s in sources]) if token_reader else ''
    lines = collect_observations(projects, project_id, sources, start_ns, end_ns,
                                 attribute=attribute, context=context, positions=positions)
    if context is not None:
        context.raise_if_cancelled()
    grid = grid_line_values(
        np.concatenate([line.x for line in lines]), np.concatenate([line.y for line in lines]),
        np.concatenate([line.values for line in lines]),
        np.concatenate([np.full(len(line.values), line.source.line_id) for line in lines]),
        cell_size_m=cell_size_m, max_cells=500_000,
    )
    sources_used = tuple(line.source for line in lines)
    curtains = ()
    if include_curtains:
        from mygpr.application.spatial.curtain_preview import read_curtain_previews
        curtains = read_curtain_previews(projects, project_id, sources_used, context=context)
    if token_reader and token_reader(project_id, [s.line_id for s in sources]) != source_token:
        raise ValueError('测线或轨迹在计算期间发生变化，请重新生成切片')
    return AmplitudeResult(grid, sources_used, next(iter(crs)),
                           float(start_ns), float(end_ns), str(attribute), curtains, source_token)
