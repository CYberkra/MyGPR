"""Read full-resolution trace blocks and associate them with measured positions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.spatial.amplitude_slice import window_attribute


@dataclass(frozen=True, slots=True)
class AmplitudeSource:
    line_id: str
    artifact_id: str = ''
    component: str = ''


@dataclass(frozen=True, slots=True)
class LineObservations:
    source: AmplitudeSource
    coordinate_system: str
    trace_indices: np.ndarray
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray


def _trace_ids(metadata, expected: int) -> np.ndarray:
    ids = np.asarray(metadata.get('trace_id', [])).reshape(-1)
    if len(ids) != expected:
        raise ValueError('处理成果缺少完整的原始道标识，不能映射位置')
    # String conversion preserves exact identity across integer/string storage,
    # but never rounds a float or substitutes a row index for an identifier.
    ids = ids.astype(str)
    if np.any(ids == '') or len(np.unique(ids)) != len(ids):
        raise ValueError('道标识存在空值或重复，不能映射位置')
    return ids


def collect_observations(
    projects,
    project_id: str,
    sources: Sequence[AmplitudeSource],
    start_ns: float,
    end_ns: float,
    *,
    attribute: str = 'rms',
    context: ExecutionContext | None = None,
) -> tuple[LineObservations, ...]:
    """Calculate per-trace attributes; coordinates remain in each track CRS.

    A processing result must retain unique raw trace identifiers. The caller
    must transform all coordinates to one confirmed metric CRS before gridding.
    """
    if not sources:
        raise ValueError('请选择测线')
    ctx = context or ExecutionContext.null()
    tracks = {str(track.line_id): track for track in projects.load_spatial_tracks(project_id)}
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
        if stored and aliases.get(stored, stored) != aliases.get(component, component):
            raise ValueError(f'{source.line_id} 的输入分量与存储记录不一致')
        positions = {int(point.trace_index): (float(point.x), float(point.y)) for point in track.points}
        if artifact:
            original = projects.get_dataset_info(project_id, source.line_id)
            original_ids = _trace_ids(projects.read_trace_metadata(project_id, source.line_id), original.shape[1])
            result_ids = _trace_ids(projects.read_artifact_trace_metadata(
                project_id, source.line_id, source.artifact_id), trace_count)
            raw_by_id = {value: index for index, value in enumerate(original_ids)}
            indices = np.array([raw_by_id.get(value, -1) for value in result_ids], dtype=np.int64)
            if np.any(indices < 0):
                raise ValueError(f'{source.line_id} 的成果道标识与原始测线不匹配')
        else:
            indices = np.arange(trace_count, dtype=np.int64)
        if samples > 4_000_000:
            raise ValueError('单道采样点数过大，无法安全计算完整道包络')
        # Limit an individual full-height block to ~16 MiB of input float32.
        chunk = max(1, min(128, 4_000_000 // samples))
        selected_indices, xs, ys, values = [], [], [], []
        for start in range(0, trace_count, chunk):
            ctx.raise_if_cancelled()
            end = min(trace_count, start + chunk)
            if artifact:
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
                                          component=component, attribute=attribute)
            for local, value in enumerate(calculated):
                trace_index = start + local
                point = positions.get(int(indices[trace_index]))
                if point is None or not np.isfinite(point).all():
                    continue
                selected_indices.append(trace_index)
                xs.append(point[0])
                ys.append(point[1])
                values.append(value)
        observations.append(LineObservations(source, str(track.coordinate_system),
            np.asarray(selected_indices, dtype=np.int64), np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64), np.asarray(values, dtype=np.float64)))
        ctx.report_progress(source_index + 1, len(sources), f'已处理 {source.line_id}')
    return tuple(observations)
