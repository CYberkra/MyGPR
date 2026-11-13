"""Collect the per-line evidence shared by the report formats."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from core.acquisition_hdf5 import acquisition_trace_metadata
from mygpr.domain.markers.mapping import TraceMapping
from mygpr.domain.markers.models import EventInterval, MarkerDocument
from mygpr.domain.markers.operations import interval_boundary_issue
from mygpr.domain.markers.ranges import interval_bounds
from mygpr.domain.reporting.models import ANALYSIS_FIELDS, MANUAL_ACQUISITION_FIELDS


def _acquisition_frequency(metadata, iczt, manual):
    frequency = str(manual.get('frequency') or '未记录')
    if 'f_start' in iczt and 'f_stop' in iczt:
        frequency = f"{iczt['f_start'] / 1e6:g}–{iczt['f_stop'] / 1e6:g} MHz（采集配置）"
    elif metadata.get('nominal_frequency_mhz'):
        frequency = f"{metadata['nominal_frequency_mhz']:g} MHz（设备头标称）"
    return frequency


def acquisition_rows(dataset, line, project, manual):
    metadata = dataset.metadata if dataset is not None else {}
    metadata = {**(metadata.get('metadata') or {}), **metadata}
    iczt = (metadata.get('sfcw') or {}).get('iczt') or {}
    frequency = _acquisition_frequency(metadata, iczt, manual)
    time_known = dataset is not None and metadata.get('time_axis_status') == 'known' and dataset.time_window_ns > 0
    length = getattr(line, 'length_m', 0) or 0
    spacing = metadata.get('trace_interval_m')
    if not isinstance(spacing, (int, float)) or not np.isfinite(spacing) or spacing <= 0:
        spacing = None
    recorded_power = metadata.get('tx_power_dbm')
    date = str(metadata.get('created') or '')[:10]
    return {
        '设备': project.get('device_model') or '未记录',
        '天线': manual.get('antenna') or metadata.get('antenna_model') or '未记录',
        '平台／方式': manual.get('platform') or '未记录',
        '采集日期': manual.get('survey_date') or (f'{date}（采集文件创建日期）' if date else '未记录'),
        '频段／中心频率': frequency,
        '功率': (f'{recorded_power:g} dBm（设备设定）'
                 if isinstance(recorded_power, (int, float)) and np.isfinite(recorded_power)
                 else manual.get('power') or '未记录'),
        '频点／转换窗': (f"{iczt['n_freq']} 点 · {iczt.get('window') or '矩形窗'}" if 'n_freq' in iczt else ''),
        '存储分量': metadata.get('store_component') or '未记录',
        '时窗': f'{dataset.time_window_ns:g} ns' if time_known else '未记录',
        '采样点／道数': f'{dataset.sample_count} × {dataset.trace_count}' if dataset is not None else '未记录',
        '长度': (f'{length:g} m' if np.isfinite(length) and length > 0
                 and getattr(line, 'distance_override', '') != 'unknown' else '未知'),
        '道间距': f'{spacing:g} m' if spacing is not None else '未记录',
        '定位': getattr(line, 'rtk_status', '') or '未定位',
        '补录依据': (manual.get('source') or '人工补录') if any(manual.get(key) for key in MANUAL_ACQUISITION_FIELDS if key != 'source') else '',
    }


def choose_report_artifacts(lines, artifacts, requested):
    by_line = defaultdict(list)
    for record in artifacts:
        if record.status in ('success', 'committed') and not record.is_display_compare_transform:
            by_line[record.line_id].append(record)
    requested = requested or {}
    unknown = set(requested) - {line.line_id for line in lines}
    if unknown:
        raise ValueError(f"报告主图包含未选测线：{', '.join(sorted(unknown))}")
    chosen = {}
    for line in lines:
        candidates = by_line[line.line_id]
        if line.line_id in requested:
            artifact_id = requested[line.line_id]
            if not artifact_id:
                continue
            record = next((item for item in candidates if item.artifact_id == artifact_id), None)
            if record is None:
                raise ValueError(f'{line.line_id} 的主处理版本不在已选成果中：{artifact_id}')
        elif candidates:
            record = max(candidates, key=lambda item: (item.created_at, item.artifact_id))
        else:
            continue
        chosen[line.line_id] = record
    return chosen


def trace_timing(dataset):
    metadata = dataset.metadata or {}
    nested = metadata.get('metadata') or {}
    try:
        trace = acquisition_trace_metadata(metadata)
    except (TypeError, ValueError):
        return None, False
    times = trace.get('trace_timestamp_s', metadata.get('trace_timestamp_s', nested.get('trace_timestamp_s')))
    confirmed = 'timestamp_ms' in trace or (nested.get('trace_time_basis') or metadata.get('trace_time_basis')) == 'UTC'
    return times, confirmed


def report_events(payload, dataset):
    if not payload:
        return [], {}, []
    doc = MarkerDocument.from_dict(payload)
    codes, ranges, events = {}, [], []
    timing = trace_timing(dataset) if dataset is not None else (None, False)
    mapping = TraceMapping.identity(doc.source_id, dataset.trace_count) if dataset is not None else None
    markers = {mark.uid: mark for mark in doc.markers}
    for record in doc.records:
        code = f'{doc.line_id}-A{len(codes) + 1:02d}'
        codes[record.uid] = code
        if record.excluded:
            continue
        left = right = None
        position = '未映射到原始道'
        issue = interval_boundary_issue(record, markers) if isinstance(record, EventInterval) else ''
        if issue:
            position = issue
        elif mapping is not None and isinstance(record, EventInterval) and record.closed:
            try:
                left, right = interval_bounds(record, mapping,
                    timestamps_s=timing[0], time_basis_confirmed=timing[1])
                if left == right:
                    position = '期间无采集道' if not record.start else '范围内无采集道'
            except ValueError as exc:
                position = str(exc)
        elif isinstance(record, EventInterval) and not record.closed:
            position = '开放区间'
        elif mapping is not None and getattr(record, 'anchor', None):
            if record.anchor.source_id == doc.source_id and record.anchor.trace < dataset.trace_count:
                left, right = record.anchor.trace, record.anchor.trace + 1
        if left is not None and left < right:
            ranges.append((code, left, right))
        if isinstance(record, EventInterval) or record.note:
            events.append({
                'code': code, 'name': record.name, 'purpose': record.purpose,
                'evidence': record.evidence, 'note': record.note,
                'position': f'原始第 {left + 1}–{right} 道' if left is not None and left < right else position,
                'time_start': getattr(record, 'time_start', ''),
                'time_end': getattr(record, 'time_end', ''),
            })
    return events, codes, ranges


def processing_chain(record, index):
    """Freeze each saved run in upstream-to-downstream order."""
    chain = []
    seen = set()
    while record is not None and record.artifact_id not in seen:
        seen.add(record.artifact_id)
        params = record.params or {}
        steps = params.get('pipeline') or [{
            'method_id': record.method_id, 'label': record.method_name,
            'params': {key: value for key, value in params.items()
                       if key not in {'output_header', 'lineage', 'marker_selection'}},
        }]
        parent = str(params.get('run_input_artifact_id') or '') if 'run_input_artifact_id' in params else record.parent_artifact_id
        chain.append({
            'artifact_id': record.artifact_id,
            'parent_id': parent,
            'input_sha256': str(params.get('run_input_sha256') or ''),
            'steps': steps, 'selection': params.get('marker_selection'),
            'warnings': list(record.warnings or []),
        })
        record = index.get(parent)
    return list(reversed(chain))


def analysis_entry(profile: dict[str, Any], line_id: str, trace_count: int) -> dict:
    analysis = dict((profile.get('line_analysis') or {}).get(line_id) or {})
    first, last = int(analysis.get('first_trace') or 0), int(analysis.get('last_trace') or 0)
    if (first or last) and not 1 <= first <= last <= trace_count:
        raise ValueError(f'{line_id} 的分析起止道超出原始测线范围。')
    result = {key: str(analysis.get(key) or '').strip() for key in ANALYSIS_FIELDS}
    if first:
        result.update(first_trace=first, last_trace=last, code=f'{line_id}-F01')
    return result
