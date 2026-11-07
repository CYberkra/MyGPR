"""Bounded display previews for actual measured B-scan curtains."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.spatial.amplitude_identity import trace_ids


@dataclass(frozen=True, slots=True)
class CurtainPreview:
    source: Any
    coordinate_system: str
    matrix: np.ndarray
    time_ns: np.ndarray
    x: np.ndarray
    y: np.ndarray
    anchor_z: np.ndarray
    trace_indices: np.ndarray
    connected: np.ndarray


def read_curtain_previews(projects, project_id, sources, *, context: ExecutionContext | None = None):
    """Read bounded display-only windows; never use these samples for analysis."""
    ctx = context or ExecutionContext.null()
    tracks = {str(track.line_id): track for track in projects.load_spatial_tracks(project_id)}
    result = []
    for source in sources:
        ctx.raise_if_cancelled()
        track = tracks.get(source.line_id)
        if track is None:
            raise ValueError(f'{source.line_id} 缺少轨迹')
        artifact = bool(source.artifact_id)
        if artifact:
            info = projects.get_artifact_dataset_info(project_id, source.line_id, source.artifact_id)
            raw_info = projects.get_dataset_info(project_id, source.line_id)
            raw_ids = trace_ids(projects.read_trace_metadata(project_id, source.line_id), raw_info.shape[1])
            ids = trace_ids(projects.read_artifact_trace_metadata(project_id, source.line_id, source.artifact_id), info.shape[1])
            raw_by_id = {value: index for index, value in enumerate(raw_ids)}
            source_to_raw = np.array([raw_by_id.get(value, -1) for value in ids], dtype=np.int64)
            if np.any(source_to_raw < 0):
                raise ValueError(f'{source.line_id} 的成果道标识与原始测线不匹配')
            matrix, sample_indices, trace_indices = projects.read_artifact_window(
                project_id, source.line_id, source.artifact_id, max_samples=64, max_traces=96)
        else:
            info = projects.get_dataset_info(project_id, source.line_id)
            source_to_raw = np.arange(info.shape[1], dtype=np.int64)
            matrix, sample_indices, trace_indices = projects.read_window(
                project_id, source.line_id, max_samples=64, max_traces=96)
        ctx.raise_if_cancelled()
        samples = np.asarray(sample_indices, dtype=np.int64)
        traces = np.asarray(trace_indices, dtype=np.int64)
        matrix = np.asarray(matrix, dtype=np.float32)
        axis = np.asarray(info.metadata.get('time_axis_ns', ()), dtype=np.float64)
        if len(traces) < 2:
            continue
        if (matrix.shape != (len(samples), len(traces)) or len(samples) < 2
                or axis.shape != (info.shape[0],) or np.any(samples < 0)
                or np.any(samples >= len(axis)) or np.any(traces < 0)
                or np.any(traces >= len(source_to_raw)) or np.any(np.diff(traces) <= 0)):
            raise ValueError(f'{source.line_id} 的三维剖面预览坐标无效')
        points = {int(p.trace_index): p for p in track.points}
        raw_indices = source_to_raw[traces]
        def coordinate(raw_index, field):
            point = points.get(int(raw_index))
            return float(getattr(point, field)) if point is not None else float('nan')
        x = np.array([coordinate(i, 'x') for i in raw_indices])
        y = np.array([coordinate(i, 'y') for i in raw_indices])
        z = np.array([coordinate(i, 'elevation_m') for i in raw_indices])
        # A decimated preview must not connect across skipped source positions.
        present = np.array([int(index) in points for index in source_to_raw], dtype=bool)
        links = ((np.abs(np.diff(source_to_raw)) == 1) & present[:-1] & present[1:])
        break_prefix = np.concatenate(([0], np.cumsum(~links, dtype=np.int64)))
        connected = break_prefix[traces[1:]] == break_prefix[traces[:-1]]
        result.append(CurtainPreview(source, str(track.coordinate_system), matrix, axis[samples],
                                     x, y, z, traces, connected))
    return tuple(result)
