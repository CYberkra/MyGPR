"""Headless B-scan figures for the original line and saved processing versions."""
from __future__ import annotations

import numpy as np

from mygpr.domain.markers.mapping import SEGMENT_KEY, read_mapping


def artifact_grid(store, line_id, artifact_id, dataset, raw_source):
    from core.hdf5_line_container import read_artifact_trace_metadata

    container = getattr(store.storage, 'line_container_path', None)
    trace = (read_artifact_trace_metadata(container(line_id), artifact_id) or {}) if container else {}
    count = dataset.trace_count
    segments = np.asarray(trace.get(SEGMENT_KEY, np.zeros(count, dtype=int)))
    if segments.shape != (count,) or np.any(np.diff(segments) < 0):
        raise ValueError('处理成果的分段数据与道数不一致。')
    mapping = None
    if raw_source:
        candidate = read_mapping(dataset.metadata, trace, raw_source, artifact_id, count)
        if candidate.kind != 'unknown':
            mapping = candidate
    return segments, mapping


def _edges(values, step=1.0):
    if len(values) == 1:
        return values[0] + np.array([-step, step]) / 2
    delta = np.diff(values) / 2
    return np.r_[values[0] - delta[0], values[:-1] + delta, values[-1] + delta[-1]]


def _vertical_axis(dataset):
    metadata = dataset.metadata or {}
    metadata = {**(metadata.get('metadata') or {}), **metadata}
    if metadata.get('is_elevation'):
        axis = np.asarray(metadata.get('elevation_axis_m', ()), dtype=float)
        if axis.shape == (dataset.sample_count,) and np.isfinite(axis).all() and np.all(np.diff(axis) < 0):
            return axis, '高程 (m)'
    if metadata.get('is_depth') and not metadata.get('is_elevation'):
        axis = np.asarray(dataset.depth_axis_m)
        if np.isfinite(axis).all() and np.all(np.diff(axis) > 0):
            return axis, '深度 (m)'
    if not metadata.get('is_depth') and not metadata.get('is_elevation') and metadata.get('time_axis_status') == 'known':
        axis = np.asarray(dataset.time_axis_ns)
        if np.isfinite(axis).all() and np.all(np.diff(axis) > 0):
            return axis, '双程走时 (ns)'
    return np.arange(1, dataset.sample_count + 1, dtype=float), '采样点'


def write_profile_figure(path, dataset, *, title, positions, x_label, segments, ranges=(),
                         annotation=None, x_limits=None, y_limits=None, notice='', trace_window=None):
    from matplotlib.figure import Figure
    from core.plot_font_policy import configure_matplotlib_cjk_fonts

    configure_matplotlib_cjk_fonts()
    vertical, y_label = _vertical_axis(dataset)
    fig = Figure(figsize=(10.8, 3.7), dpi=160)
    grid = fig.add_gridspec(2 if ranges else 1, 1, height_ratios=(12, 2.5) if ranges else None,
                           hspace=.1, left=.10, right=.91, bottom=.17, top=.87)
    ax = fig.add_subplot(grid[0])
    strip = fig.add_subplot(grid[1], sharex=ax) if ranges else None
    axis_for_labels = strip if strip is not None else ax
    if strip is not None:
        ax.tick_params(labelbottom=False)
        strip.set_yticks([])
        strip.set_ylim(0, 2)
        strip.set_facecolor('#f8fafc')
    row_step = max(1, -(-dataset.sample_count // 750))
    row_edges = _edges(vertical)[np.r_[np.arange(0, dataset.sample_count, row_step), dataset.sample_count]]
    left, right = trace_window or (0, dataset.trace_count)
    col_step = max(1, -(-(right - left) // 1450))
    boundaries = np.flatnonzero(np.diff(segments)) + 1
    cuts = np.r_[left, boundaries[(boundaries > left) & (boundaries < right)], right]
    # Bound the normalisation sample even when every retained trace is a separate segment.
    preview = np.asarray(dataset.matrix[::row_step, left:right:col_step], dtype=np.float32)
    scale = float(np.nanpercentile(np.abs(preview), 99.5))
    scale = scale if np.isfinite(scale) and scale > 1e-9 else 1.0
    centre = float(np.nanmean(preview))
    centre = centre if np.isfinite(centre) else 0.0
    curve = None
    if annotation is not None:
        curve = annotation.curve_samples()
        curve[~annotation.trace_semantics()['valid_trace_mask']] = np.nan
    mesh = None
    for start, stop in zip(cuts[:-1], cuts[1:]):
        offsets = np.r_[np.arange(0, stop - start, col_step), stop - start]
        column_edges = _edges(positions[start:stop])[offsets]
        block = np.asarray(dataset.matrix[::row_step, start:stop:col_step], dtype=np.float32)
        block = np.clip((block - centre) / scale, -1.0, 1.0)
        mesh = ax.pcolormesh(column_edges, row_edges, block,
                             cmap='gray', vmin=-1, vmax=1, shading='flat', rasterized=True)
        if curve is not None:
            y = np.interp(curve[start:stop], np.arange(dataset.sample_count), vertical)
            ax.plot(positions[start:stop], y, color='#60a5fa', lw=1.2)
    for cut in cuts[1:-1]:
        ax.axvline((positions[cut - 1] + positions[cut]) / 2, color='#f59e0b', lw=.85, ls=':')
    if mesh is not None:
        cax = fig.add_axes([.925, .31 if strip is not None else .22, .013, .48 if strip is not None else .59])
        fig.colorbar(mesh, cax=cax, ticks=[-1, 0, 1], label='归一化振幅')
    if strip is not None:
        span = max(float(np.ptp(x_limits if x_limits is not None else positions[left:right])), 1e-9)
        occupied = [[], []]
        hidden = 0
        for code, start, stop in ranges:
            low, high = positions[start], positions[stop - 1]
            label_width = .075 * span
            extent = (low, max(high, low + label_width))
            lane = next((index for index, entries in enumerate(occupied)
                         if all(end < extent[0] or left > extent[1] for left, end in entries)), None)
            if lane is None:
                hidden += 1
                continue
            occupied[lane].append(extent)
            strip.plot([low, high], [lane + .23] * 2, marker='|', ms=3.5, color='#1e40af', lw=1)
            strip.text(low, lane + .37, code, fontsize=7, color='#1e40af', clip_on=True)
        if hidden:
            strip.text(.99, .98, f'+{hidden}', transform=strip.transAxes,
                       ha='right', va='top', fontsize=7, color='#475569',
                       bbox={'facecolor': '#f8fafc', 'edgecolor': 'none', 'pad': 1})
    ax.set_xlim(x_limits if x_limits is not None else _edges(positions[left:right])[[0, -1]])
    if y_limits is None:
        edges = _edges(vertical)
        y_limits = (min(edges), max(edges)) if y_label == '高程 (m)' else (max(edges), min(edges))
    ax.set_ylim(y_limits)
    ax.set_ylabel(y_label)
    axis_for_labels.set_xlabel(x_label)
    fig.text(.10, .94, title, fontsize=11, fontweight='bold')
    if notice:
        fig.text(.10, .035, notice, fontsize=7, color='#475569')
    fig.savefig(path, dpi=160)
    fig.clear()


def profile_axes(raw, processed=None, mapping=None):
    metadata = raw.metadata or {}
    nested = metadata.get('metadata') or {}
    distance = np.asarray(raw.distance_axis_m, dtype=float)
    use_metres = (metadata.get('distance_axis_status', nested.get('distance_axis_status')) not in (None, '', 'unknown')
                  and np.isfinite(distance).all() and len(distance) > 1 and np.all(np.diff(distance) > 0))
    positions = distance if use_metres else np.arange(1, raw.trace_count + 1, dtype=float)
    label = '沿线距离 (m)' if use_metres else '原始道号'
    if processed is None:
        return positions, label, None, None, None
    mapped = None
    if mapping is not None and mapping.coordinates.shape == (processed.trace_count,) and mapping.coordinates.size:
        if mapping.coordinates[-1] <= raw.trace_count - 1:
            mapped = np.interp(mapping.coordinates, np.arange(raw.trace_count), positions)
    raw_y, raw_unit = _vertical_axis(raw)
    proc_y, proc_unit = _vertical_axis(processed)
    limits = None
    if raw_unit == proc_unit and raw_unit != '采样点':
        low = min(np.min(_edges(raw_y)), np.min(_edges(proc_y)))
        high = max(np.max(_edges(raw_y)), np.max(_edges(proc_y)))
        limits = (low, high) if raw_unit == '高程 (m)' else (high, low)
    return positions, label, mapped, _edges(positions)[[0, -1]] if mapped is not None else None, limits


def project_ranges(ranges, mapping, segments):
    if mapping is None:
        return []
    boundaries = np.flatnonzero(np.diff(segments)) + 1
    projected = []
    for code, start, stop in ranges:
        left = int(np.searchsorted(mapping.coordinates, start))
        right = int(np.searchsorted(mapping.coordinates, stop - 1, side='right'))
        if left < right:
            cuts = [left, *boundaries[(boundaries > left) & (boundaries < right)], right]
            projected.extend((code, a, b) for a, b in zip(cuts[:-1], cuts[1:]))
    return projected
