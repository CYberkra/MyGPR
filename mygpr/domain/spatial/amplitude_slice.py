"""Amplitude windows and observation-only grids for spatial GPR views.

Time windows are half-open [start, end). Time is the input dataset's axis;
it is not assumed to be depth or time since the surface reflection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert


@dataclass(frozen=True, slots=True)
class AmplitudeGrid:
    values: np.ndarray
    trace_counts: np.ndarray
    line_counts: np.ndarray
    x_origin_m: float
    y_origin_m: float
    cell_size_m: float


def window_attribute(
    data: np.ndarray,
    time_ns: np.ndarray,
    start_ns: float,
    end_ns: float,
    *,
    component: str,
    attribute: str = 'rms',
) -> np.ndarray:
    """Compute one amplitude per trace from complete, valid samples.

    For real data the analytic envelope uses each entire input trace. Therefore
    callers must supply all samples for each trace, even for a narrow window.
    Magnitude data already have nonnegative amplitude and are not transformed.
    A trace containing nonfinite input required by its calculation is invalid.
    """
    matrix = np.asarray(data)
    axis = np.asarray(time_ns, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise ValueError('波形必须是至少两个样点的二维矩阵')
    if axis.ndim != 1 or axis.size != matrix.shape[0] or not np.isfinite(axis).all():
        raise ValueError('时间轴与波形不匹配')
    intervals = np.diff(axis)
    # Project axes can be persisted as float32; adjacent differences then
    # accumulate quantization at the scale of the largest axis value.
    axis_atol = max(1e-9, float(np.max(np.abs(axis))) * np.finfo(np.float32).eps * 2)
    if np.any(intervals <= 0) or not np.allclose(intervals, intervals[0], rtol=1e-5, atol=axis_atol):
        raise ValueError('时间轴必须等间隔递增')
    start, end = float(start_ns), float(end_ns)
    tolerance = max(1e-9, abs(intervals[0]) * 1e-6)
    if (not np.isfinite(start) or not np.isfinite(end) or end <= start
            or start < axis[0] - tolerance or end > axis[-1] + intervals[0] + tolerance):
        raise ValueError('时间窗超出有效时间轴范围')
    selected = (axis >= start - tolerance) & (axis < end - tolerance)
    if not selected.any():
        raise ValueError('时间窗内没有采样点')
    component = str(component).lower()
    if component == 'real':
        finite = np.isfinite(matrix).all(axis=0)
        amplitude = np.full((int(selected.sum()), matrix.shape[1]), np.nan, dtype=np.float64)
        if finite.any():
            amplitude[:, finite] = np.abs(hilbert(matrix[:, finite], axis=0))[selected]
    elif component in ('mag', 'magnitude'):
        amplitude = np.asarray(matrix[selected], dtype=np.float64)
        if np.any(amplitude[np.isfinite(amplitude)] < 0):
            raise ValueError('幅值分量必须非负')
    else:
        raise ValueError('输入分量的时域振幅语义未确认')
    finite = np.isfinite(amplitude).all(axis=0)
    result = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    if attribute == 'rms':
        result[finite] = np.sqrt(np.mean(np.square(amplitude[:, finite]), axis=0))
    elif attribute == 'mean':
        result[finite] = np.mean(amplitude[:, finite], axis=0)
    elif attribute == 'max':
        result[finite] = np.max(amplitude[:, finite], axis=0)
    else:
        raise ValueError('不支持的窗口属性')
    return result


def grid_line_values(x_m, y_m, values, line_ids, *, cell_size_m: float) -> AmplitudeGrid:
    """Bin observations without interpolation, giving each line equal weight.

    Origins are anchored to integer cell boundaries for repeatable binning.
    Rows descend northwards and origin coordinates identify the first cell centre.
    """
    x, y, value = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (x_m, y_m, values))
    lines = np.asarray(line_ids, dtype=str).reshape(-1)
    if not (len(x) == len(y) == len(value) == len(lines)):
        raise ValueError('坐标、振幅与测线长度不一致')
    cell = float(cell_size_m)
    if not np.isfinite(cell) or cell <= 0:
        raise ValueError('网格尺寸必须为正')
    valid_position = np.isfinite(x) & np.isfinite(y) & (lines != '')
    x, y, value, lines = x[valid_position], y[valid_position], value[valid_position], lines[valid_position]
    if not len(value):
        raise ValueError('没有同时具备有效坐标和振幅的观测')
    xi, yi = np.floor(x / cell).astype(np.int64), np.floor(y / cell).astype(np.int64)
    xmin, xmax, ymin, ymax = int(xi.min()), int(xi.max()), int(yi.min()), int(yi.max())
    ncols, nrows = xmax - xmin + 1, ymax - ymin + 1
    if ncols * nrows > 4_000_000:
        raise ValueError('网格过大，请增大网格尺寸')
    # Extent uses all located traces, not just amplitudes available in this
    # particular time window; switching windows must not move the grid.
    valid_value = np.isfinite(value)
    flat = ((ymax - yi) * ncols + (xi - xmin))[valid_value]
    value, lines = value[valid_value], lines[valid_value]
    _, line_codes = np.unique(lines, return_inverse=True)
    pairs = np.rec.fromarrays([flat, line_codes], names='cell,line')
    unique_pairs, pair_indices, pair_counts = np.unique(pairs, return_inverse=True, return_counts=True)
    means = np.bincount(pair_indices, weights=value) / pair_counts
    line_counts = np.bincount(unique_pairs['cell'], minlength=ncols * nrows)
    trace_counts = np.bincount(flat, minlength=ncols * nrows)
    sums = np.bincount(unique_pairs['cell'], weights=means, minlength=ncols * nrows)
    result = np.full(ncols * nrows, np.nan, dtype=np.float64)
    np.divide(sums, line_counts, out=result, where=line_counts > 0)
    shape = (nrows, ncols)
    return AmplitudeGrid(result.reshape(shape), trace_counts.reshape(shape), line_counts.reshape(shape),
                         (xmin + 0.5) * cell, (ymax + 0.5) * cell, cell)
