"""Locate trace-aligned QC and sensor faults without reading the radar matrix."""
from __future__ import annotations

import numpy as np

from mygpr.domain.project.models import LineQualityIssue


def _ranges(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def trace_quality_preview_mask(metadata, trace_count):
    """Combine aligned display masks without mutating stored bad-trace state.

    A repaired trace remains visibly auditable, but is not silently added to
    trace_qc_bad_mask (which subsequent mute/remove operations consume).
    """
    combined = np.zeros(trace_count, dtype=bool)
    found = False
    for key in ('trace_qc_bad_mask', 'short_trace_candidate_mask', 'short_trace_repaired_mask',
                'short_trace_unresolved_mask', 'short_trace_zero_mask'):
        values = np.asarray(metadata.get(key, []))
        if values.shape == (trace_count,) and values.dtype.kind in 'biuf':
            combined |= np.isfinite(values) & (values != 0)
            found = True
    local = trace_local_change_preview_mask(metadata, trace_count)
    if local is not None:
        combined &= ~local
    return combined if found else None


def trace_local_change_preview_mask(metadata, trace_count):
    """Amber local-review class; history is retained, current red evidence wins."""
    values = np.asarray(metadata.get('short_trace_local_change_mask', []))
    if values.shape != (trace_count,) or values.dtype.kind not in 'biuf':
        return None
    local = np.isfinite(values) & (values != 0)
    for key in ('trace_qc_bad_mask', 'short_trace_candidate_mask', 'short_trace_repaired_mask', 'short_trace_zero_mask'):
        mask = np.asarray(metadata.get(key, []))
        if mask.shape == (trace_count,) and mask.dtype.kind in 'biuf':
            local &= ~(np.isfinite(mask) & (mask != 0))
    return local


def locate_trace_issues(metadata, trace_count):
    """Return exact zero-based, half-open trace ranges; absent fields are not inferred."""
    issues = []

    def append(code, label, mask):
        ranges = list(_ranges(mask))
        for start, stop in ranges[:100]:
            issues.append(LineQualityIssue('warning', code, f'{label}：道 {start + 1}–{stop}。',
                                           '查看对应道，核查来源。', int(start), int(stop)))
        if len(ranges) > 100:
            issues.append(LineQualityIssue('info', code + '_truncated',
                f'{label}共有 {len(ranges)} 个区段，仅列出前 100 个。'))

    qc = np.asarray(metadata.get('trace_qc_bad_mask', []))
    if qc.shape == (trace_count,):
        append('trace_qc_bad', '已标记坏道', qc.astype(bool))
    for key, code, label in (
        ('short_trace_candidate_mask', 'short_trace_candidate', '短段异常候选'),
        ('short_trace_repaired_mask', 'short_trace_repaired', '已插补道（需对照原始）'),
        ('short_trace_unresolved_mask', 'short_trace_unresolved', '未插补候选'),
        ('short_trace_local_change_mask', 'short_trace_local_change', '局部时窗变化（仅审阅，不插补）'),
        ('short_trace_zero_mask', 'short_trace_zero', '输入整道全零（确定数据质量问题）'),
    ):
        values = np.asarray(metadata.get(key, []))
        if values.shape == (trace_count,):
            append(code, label, values.astype(bool))
    for key in ('longitude', 'latitude', 'altitude_m', 'flight_height_m', 'roll_deg', 'pitch_deg', 'yaw_deg'):
        values = np.asarray(metadata.get(key, []))
        if values.shape == (trace_count,) and values.dtype.kind in 'fiu':
            append('sensor_missing', f'{key} 缺失', ~np.isfinite(values))
    timestamps = np.asarray(metadata.get('trace_timestamp_s', []))
    if timestamps.shape == (trace_count,) and timestamps.dtype.kind in 'fiu':
        append('timestamp_missing', '时间戳缺失', ~np.isfinite(timestamps))
        delta = np.diff(timestamps.astype(np.float64))
        finite = np.isfinite(delta)
        positive = delta[finite & (delta > 0)]
        for code, label, bad in (
            ('timestamp_nonmonotonic', '时间戳非递增', finite & (delta <= 0)),
            ('timestamp_gap', '时间戳间隔超过中位数的 5 倍',
             finite & (delta > 5 * np.median(positive)) if positive.size else np.zeros(delta.shape, dtype=bool)),
        ):
            mask = np.zeros(trace_count, dtype=bool)
            indices = np.flatnonzero(bad)
            mask[indices] = True
            mask[indices + 1] = True
            append(code, label, mask)
    return issues
