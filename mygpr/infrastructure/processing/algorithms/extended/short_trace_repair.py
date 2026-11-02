"""Explicit short-run interpolation; marking never changes the input signal."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import median_filter

from mygpr.application.jobs.cancellation import JobCancelledError
from mygpr.domain.common.errors import InputDataError
from mygpr.domain.processing.timebase import sampling_interval_s
from mygpr.infrastructure.processing.algorithms.common import stored_axis_boundary_index


class ShortTraceDataQualityError(InputDataError, ValueError):
    """Nonfinite input is rejected with exact source trace locations, never filled."""

    error_code = 'MYGPR_SHORT_TRACE_NONFINITE_INPUT'

    def __init__(self, nonfinite_mask):
        self.nonfinite_trace_numbers = (np.flatnonzero(nonfinite_mask) + 1).tolist()
        shown = ', '.join(map(str, self.nonfinite_trace_numbers[:32]))
        suffix = '…' if len(self.nonfinite_trace_numbers) > 32 else ''
        super().__init__(f'确定数据质量问题：非有限样点（NaN/Inf），涉及道 {shown}{suffix}，'
                         f'共{len(self.nonfinite_trace_numbers)}道；处理中止，不填充、不保存派生波形。',
                         hint='按完整道号核查原始采集数据；不要通过静默填充继续统计修补。',
                         context={'method_id': 'short_trace_repair', 'quality_issue': 'nonfinite_samples',
                                  'nonfinite_trace_numbers': self.nonfinite_trace_numbers})


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    number = float(value)
    if not np.isfinite(number) or number != int(number) or not minimum <= number <= maximum:
        raise ValueError(f'{name} 必须是 {minimum}–{maximum} 范围内的整数')
    return int(number)


def _saved_mask(metadata: dict, key: str, traces: int) -> np.ndarray:
    values = np.asarray(metadata.get(key, np.zeros(traces, dtype=bool)))
    if values.shape != (traces,) or values.dtype.kind not in 'biuf' or not np.isfinite(values).all():
        raise ValueError(f'{key} 与当前道数不一致或无效')
    return values.astype(bool)


def _windows(spec: str, header: dict, samples: int) -> list[tuple[np.ndarray, dict]]:
    if not str(spec).strip():
        return [(np.arange(samples), {'sample_start': 0, 'sample_stop': samples, 'requested_ns': None})]
    if header.get('time_axis_status') == 'unknown':
        raise ValueError('检测时窗需要已知有效时间轴，不能把样点编号当作ns')
    axis = header.get('time_axis_ns')
    if axis is None:
        step = sampling_interval_s(header, samples) * 1e9
        if not np.isfinite(step) or step <= 0:
            raise ValueError('检测时窗缺少有效时间轴')
        axis = float(header.get('time_cut_offset_ns') or 0.) + np.arange(samples) * step
    source_axis = np.asarray(axis)
    axis = np.asarray(source_axis, dtype=np.float64)
    if axis.shape != (samples,) or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise ValueError('检测时间轴必须是长度与样点一致的有限递增向量')
    result = []
    for item in str(spec).replace('；', ';').split(';'):
        if not item.strip():
            continue
        try:
            start, end = map(float, item.split(':'))
        except (TypeError, ValueError) as exc:
            raise ValueError('检测时窗格式应为0:100;100:200，单位ns') from exc
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError('检测时窗必须递增且完整位于有效时间轴内')
        first = stored_axis_boundary_index(source_axis, axis, start, roundoff_ulps=4)
        stop = stored_axis_boundary_index(source_axis, axis, end, side='right', roundoff_ulps=4)
        below = start < axis[0] and stored_axis_boundary_index(source_axis, axis, start, side='right', roundoff_ulps=4) == 0
        above = end > axis[-1] and stored_axis_boundary_index(source_axis, axis, end, roundoff_ulps=4) == samples
        if below or above:
            raise ValueError('检测时窗必须完整位于有效时间轴内')
        indices = np.arange(first, stop)
        if indices.size < 4:
            raise ValueError('每个检测时窗至少需要4个有效样点')
        result.append((indices, {'requested_ns': [start, end], 'actual_ns': [float(axis[indices[0]]), float(axis[indices[-1]])],
                                 'sample_start': int(indices[0]), 'sample_stop': int(indices[-1]) + 1}))
    if not result:
        raise ValueError('检测时窗不能为空分隔符')
    return result


def _detection_windows(mode: str | None, spec: str, header: dict, samples: int):
    effective = ('manual' if str(spec).strip() else 'auto') if mode is None else mode
    if effective not in {'auto', 'manual', 'full'}:
        raise ValueError('window_mode 必须为 auto、manual 或 full')
    if effective == 'manual':
        if not str(spec).strip():
            raise ValueError('手动分窗需要填写ns时窗；无需手填时请选择自动分窗')
        windows = _windows(spec, header, samples)
        windows.sort(key=lambda value: int(value[0][0]))
        for i in range(len(windows)-1):
            a, info = windows[i]
            b = windows[i+1][0]
            overlap = np.intersect1d(a, b)
            if overlap.size > 1:
                raise ValueError('手动时窗不能重叠（允许共用一个边界样点）')
            if overlap.size:
                # Half-open ownership: a shared boundary belongs only to the later window.
                a = a[:-1]
                if a.size < 4:
                    raise ValueError('边界去重后每个手动时窗至少需要4个样点')
                axis = header.get('time_axis_ns')
                if axis is None:
                    axis = float(header.get('time_cut_offset_ns') or 0.) + np.arange(samples)*sampling_interval_s(header,samples)*1e9
                info = {**info, 'sample_stop': int(a[-1])+1,
                        'actual_ns': [float(axis[a[0]]),float(axis[a[-1]])]}
                windows[i] = (a, info)
        return effective, windows
    if effective == 'full':
        return effective, _windows('', header, samples)
    # Equal proportions of available samples: no guessed time scale or trace-specific fitting.
    cuts = np.linspace(0, samples, 4, dtype=int) if samples >= 12 else np.array([0, samples])
    axis = None
    if header.get('time_axis_status') != 'unknown':
        if header.get('time_axis_ns') is not None:
            axis = np.asarray(header['time_axis_ns'], dtype=float)
            if axis.shape != (samples,) or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
                raise ValueError('自动分窗收到无效时间轴')
        else:
            try:
                step = sampling_interval_s(header, samples) * 1e9
            except (ValueError, KeyError):
                step = None
            if step is not None and np.isfinite(step) and step > 0:
                axis = float(header.get('time_cut_offset_ns') or 0.) + np.arange(samples) * step
    if axis is not None and (axis.shape != (samples,) or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0)):
        raise ValueError('自动分窗收到无效时间轴（非有限或样点坐标重合）')
    windows = []
    for first, stop in zip(cuts, cuts[1:]):
        info = {'sample_start': int(first), 'sample_stop': int(stop), 'requested_ns': None,
                'sample_fraction': [float(first/samples), float(stop/samples)]}
        if axis is not None:
            info['actual_ns'] = [float(axis[first]), float(axis[stop-1])]
        windows.append((np.arange(first, stop), info))
    return effective, windows


def method_short_trace_repair(
    data: np.ndarray, mode: str = 'mark', window_traces: int = 31,
    robust_z: float = 8., time_windows_ns: str = '', max_run_traces: int = 4,
    window_mode: str | None = None,
    header_info: dict[str, Any] | None = None,
    trace_metadata: dict[str, np.ndarray] | None = None,
    cancel_checker=None, **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect multi-gate outliers and optionally interpolate bounded short runs.

    Arrays use (samples, traces). Auto splits available samples into three
    nonoverlapping proportions. Manual ns windows require an actual axis.
    Only explicitly selected interpolate mode alters values. This is not a
    claim of recovering an unknown reflector at an affected trace.
    """
    if np.iscomplexobj(data):
        raise ValueError('短段异常道修补需要实数时域矩阵，不能隐式丢弃复数分量')
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or not arr.shape[0] or arr.shape[1] < 3:
        raise ValueError('输入必须是非空2维矩阵，且至少有3道')
    if mode not in {'mark', 'interpolate'}:
        raise ValueError('mode 必须为 mark 或 interpolate')
    width = _integer(window_traces, 'window_traces', 3, 301)
    if width % 2 == 0:
        raise ValueError('window_traces 必须为奇数')
    max_run = _integer(max_run_traces, 'max_run_traces', 1, 4)
    z = float(robust_z)
    if not np.isfinite(z) or not 0 < z <= 100:
        raise ValueError('robust_z 必须为0到100之间的有限正数')
    effective_mode, windows = _detection_windows(window_mode, time_windows_ns, header_info or {}, arr.shape[0])
    trace_metadata = trace_metadata or {}
    previous = _saved_mask(trace_metadata, 'short_trace_repaired_mask', arr.shape[1])
    prior_unresolved = _saved_mask(trace_metadata, 'short_trace_unresolved_mask', arr.shape[1])
    blocked_anchors = (previous
        | _saved_mask(trace_metadata, 'trace_qc_bad_mask', arr.shape[1])
        | prior_unresolved
        | _saved_mask(trace_metadata, 'short_trace_local_change_mask', arr.shape[1])
        | _saved_mask(trace_metadata, 'short_trace_zero_mask', arr.shape[1]))

    def poll() -> None:
        if cancel_checker is not None and cancel_checker():
            raise JobCancelledError('短段异常道修补已取消')

    # Whole-trace data facts use every saved sample, independently of selected gates.
    # A quiet/zero time window alone is not a whole-zero trace. No amplitude floor.
    block_rows = max(1, min(64, 1_000_000 // arr.shape[1]))
    zero_mask = np.ones(arr.shape[1], dtype=bool)
    nonfinite_mask = np.zeros(arr.shape[1], dtype=bool)
    for start in range(0, arr.shape[0], block_rows):
        poll()
        block = arr[start:start + block_rows]
        zero_mask &= np.all(block == 0, axis=0)
        nonfinite_mask |= ~np.all(np.isfinite(block), axis=0)
    if nonfinite_mask.any():
        raise ShortTraceDataQualityError(nonfinite_mask)

    # Row blocks bound temporary float64 references, including for mmap input.
    # Keep the statistical score population unchanged, so other trace classes do not drift.
    sum_squares = np.zeros((len(windows), arr.shape[1]), dtype=np.float64)
    for start in range(0, arr.shape[0], block_rows):
        poll()
        stop = min(start + block_rows, arr.shape[0])
        block = np.asarray(arr[start:stop], dtype=np.float64)
        reference = median_filter(block, size=(1, width), mode='reflect')
        delta = block - reference
        for k, (indices, _) in enumerate(windows):
            selected = indices[(indices >= start) & (indices < stop)] - start
            if selected.size:
                sum_squares[k] += np.einsum('ij,ij->j', delta[selected], delta[selected])
    votes = []
    diagnostics = []
    for k, (indices, description) in enumerate(windows):
        score = np.sqrt(sum_squares[k] / indices.size)
        center = float(np.median(score))
        sigma = 1.4826 * float(np.median(abs(score - center)))
        # Zero MAD is valid; a unit-sized floor would change classifications
        # when the same uncalibrated waveform is expressed in different units.
        threshold = center + z * sigma
        flagged = score > threshold
        votes.append(flagged)
        diagnostics.append({**description, 'median_rms_difference': center, 'robust_sigma': sigma, 'threshold': threshold,
                            'input_flagged_trace_numbers': (np.flatnonzero(flagged)+1).tolist()})
    votes = np.asarray(votes)
    candidates = votes.all(axis=0) if len(windows) >= 2 or effective_mode == 'full' else np.zeros(arr.shape[1], dtype=bool)
    candidates &= ~zero_mask
    local_changes = votes.any(axis=0) & ~candidates & ~zero_mask
    blocked_anchors |= local_changes
    poll()
    result = arr.copy()
    repaired = np.zeros(arr.shape[1], dtype=bool)
    repair_targets = candidates | zero_mask
    edges = np.diff(np.r_[False, repair_targets, False].astype(np.int8))
    run_records = []
    for first, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        poll()
        eligible = bool((len(windows) >= 2 or effective_mode == 'full')
                        and first > 0 and stop < arr.shape[1] and stop - first <= max_run
                        and not blocked_anchors[first - 1] and not blocked_anchors[stop])
        changed = mode == 'interpolate' and eligible
        if changed:
            for j in range(first, stop):
                weight = (j - first + 1) / (stop - first + 1)
                result[:, j] = (1 - weight) * arr[:, first - 1].astype(np.float64) + weight * arr[:, stop].astype(np.float64)
            repaired[first:stop] = True
        run_records.append({'start_trace': int(first) + 1, 'end_trace': int(stop),
                            'repairable': eligible, 'repaired': bool(changed),
                            'statistical_candidate_count': int(candidates[first:stop].sum()),
                            'zero_trace_count': int(zero_mask[first:stop].sum())})
    warnings = []
    if zero_mask.any():
        zero_numbers = (np.flatnonzero(zero_mask)[:32] + 1).tolist()
        shown = ', '.join(map(str, zero_numbers)) + ('…' if zero_mask.sum() > 32 else '')
        warnings.append({'code': 'short_trace_zero', 'method_id': 'short_trace_repair',
                         'message': f'输入有{int(zero_mask.sum())}道整道全零（确定数据质量问题），道号：{shown}；'
                                    '已单独标记；它不同于统计候选或某一时窗安静。端部不外推。',
                         'trace_numbers': zero_numbers,
                         'trace_count': int(zero_mask.sum())})
    if effective_mode == 'full':
        warnings.append({'code': 'short_trace_full_window', 'method_id': 'short_trace_repair',
                         'message': '旧版全窗判断仅用于对照，强早波可能主导候选；不构成采集坏道证明。'})
    elif len(windows) < 2:
        warnings.append({'code': 'short_trace_local_only', 'method_id': 'short_trace_repair',
                         'message': '有效样点或独立时窗不足，仅标记局部变化，不进行整道插补。'})
    # A changed gate/threshold must not erase prior unresolved review history.
    # Only an explicit successful repair of that position clears its flag.
    unresolved = (prior_unresolved | repair_targets) & ~repaired
    if repaired.any():
        warnings.append({'code': 'short_trace_interpolated', 'method_id': 'short_trace_repair',
                         'message': '短段插值替换了标记道，可能覆盖这些道上的真实局部反射；请与原始分支和掩码对照。'})
    return result, {
        'method': 'short_trace_repair', 'mode': mode, 'window_traces': width,
        'window_mode': effective_mode, 'classification_rule': 'all_independent_windows' if effective_mode != 'full' else 'legacy_full_window',
        'manual_windows_ignored': bool(str(time_windows_ns).strip() and effective_mode != 'manual'),
        'local_change_trace_count': int(local_changes.sum()),
        'zero_trace_count': int(zero_mask.sum()),
        'robust_z': z, 'time_windows_ns': time_windows_ns, 'max_run_traces': max_run,
        'candidate_trace_count': int(candidates.sum()), 'repaired_trace_count': int(repaired.sum()),
        'unresolved_trace_count': int(unresolved.sum()), 'runs': run_records,
        'gate_diagnostics': diagnostics, 'runtime_warnings': warnings,
        'trace_metadata_updates': {
            'short_trace_candidate_mask': candidates.astype(np.int8),
            'short_trace_local_change_mask': local_changes.astype(np.int8),
            'short_trace_zero_mask': zero_mask.astype(np.int8),
            'short_trace_repaired_mask': (previous | repaired).astype(np.int8),
            'short_trace_unresolved_mask': unresolved.astype(np.int8),
        },
    }
