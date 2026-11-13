"""Assemble line dossiers, figure references and the selected survey map."""
from __future__ import annotations

import logging

import numpy as np

from core.report_line_content import acquisition_rows, analysis_entry, processing_chain, report_events
from core.report_profiles import _edges, artifact_grid, profile_axes, project_ranges, write_profile_figure

_LOGGER = logging.getLogger(__name__)
_DATA_ERRORS = (OSError, ValueError, TypeError, KeyError, RuntimeError)


def _annotation(store, line_id, raw, primary):
    try:
        annotation = store.load_basal_interface_annotation(line_id)
        if annotation is None:
            return None, None, ''
        source = annotation.source_result_id if annotation.source_mode == 'processed' else f'{line_id}_raw'
        if source == f'{line_id}_raw':
            if (annotation.sample_count, annotation.trace_count) == raw.matrix.shape:
                return annotation, None, ''
        elif primary and source == primary.artifact_id:
            return None, annotation, ''
        return None, None, '界面来源未纳入当前图件或数据尺寸已变化'
    except _DATA_ERRORS:
        _LOGGER.warning('报告界面读取失败：%s', line_id, exc_info=True)
        return None, None, '界面记录无法读取'


def _load_report_dataset(store, line):
    line_id = line.line_id
    dataset = None
    try:
        dataset = store.load_gpr_dataset(line_id)
        if line.time_window_override_ns > 0:
            from core.gpr_data_model import apply_time_window_override
            apply_time_window_override(dataset, line.time_window_override_ns)
        if line.distance_override == 'measured':
            dataset.distance_axis_m = np.linspace(0, line.length_m, dataset.trace_count)
        if line.distance_override in ('measured', 'unknown'):
            dataset.metadata['distance_axis_status'] = line.distance_override
    except _DATA_ERRORS:
        dataset = None
        _LOGGER.warning('报告原始数据读取失败：%s', line_id, exc_info=True)
    return dataset


def build_line_dossiers(store, lines, primary, artifacts, marker_documents, profile,
                        project, quality_rows, sections, figures_dir, check_cancel, progress_callback):
    index = {item.artifact_id: item for item in artifacts}
    quality = {row['line_id']: row for row in quality_rows}
    documents = {doc['line_id']: doc for doc in marker_documents}
    overrides = profile.get('acquisition_overrides') or {}
    featured = set(profile.get('featured_line_ids') or ())
    dossiers, figure_paths, map_ranges, record_codes = [], [], {}, {}
    for number, line in enumerate(lines, 1):
        check_cancel()
        line_id = line.line_id
        dataset = _load_report_dataset(store, line)
        document = documents.get(line_id)
        events, codes, ranges = report_events(document, dataset)
        record_codes.update(codes)
        analysis = analysis_entry(profile, line_id, dataset.trace_count if dataset is not None else 0)
        if analysis.get('code'):
            ranges.insert(0, (analysis['code'], analysis['first_trace'] - 1, analysis['last_trace']))
        map_ranges[line_id] = ranges
        artifact = primary.get(line_id)
        chain = processing_chain(artifact, index)
        for run in chain:
            selection = run.get('selection')
            if selection and selection.get('intervals'):
                run['selection'] = {**selection, 'intervals': [
                    {**event, 'report_code': codes.get(event['uid'], '')}
                    for event in selection['intervals']
                ]}
        dossier = {
            'line_id': line_id, 'name': line.name, 'featured': line_id in featured,
            'acquisition': acquisition_rows(dataset, line, project, overrides.get(line_id, {})),
            'acquisition_input': overrides.get(line_id, {}),
            'quality': quality.get(line_id, {}),
            'events': events, 'analysis': analysis, 'raw_figure': '', 'processed_figure': '',
            'detail_figures': [], 'detail_note': '', 'route_figure': '',
            'primary_artifact_id': artifact.artifact_id if artifact else '',
            'primary_name': artifact.method_name if artifact else '',
            'processing': chain if sections['include_processing_history'] else [],
            'figure_notes': [],
        }
        if dataset is None:
            dossier['figure_notes'].append('原始数据无法读取')
        elif sections['include_bscan_images']:
            paths = _write_line_figures(store, line_id, dataset, artifact, chain, dossier, document, ranges,
                                        figures_dir, sections['include_interpretation'])
            figure_paths.extend(paths)
        dossiers.append(dossier)
        if progress_callback is not None:
            progress_callback(number, max(len(lines), 1), f'整理测线 {line_id}')

    plan_name = map_note = ''
    located = set()
    if sections['include_spatial_results']:
        from core.gis_map_export import export_project_plan_map
        plan_path = figures_dir / 'project_plan_map.png'
        try:
            export_project_plan_map(
                store, plan_path, line_ids={line.line_id for line in lines}, line_ranges=map_ranges,
                include_interpretation=False, located_lines=located, require_trajectory=True,
                cancel_requested=lambda: (check_cancel(), False)[1],
                dpi=150,
            )
            plan_name = plan_path.name
            figure_paths.insert(0, plan_path)
        except (OSError, ValueError) as exc:
            map_note = str(exc) if isinstance(exc, ValueError) else '路线图文件无法写入。'
            store.append_log(f'报告 GIS 平面图未生成: {exc}')
    for dossier in dossiers:
        line_id = dossier['line_id']
        dossier['map_status'] = ('未选空间章节' if not sections['include_spatial_results'] else
                                 '已绘制轨迹' if plan_name and line_id in located else '无法在地图定位')
        if dossier['map_status'] == '已绘制轨迹':
            dossier['acquisition']['定位'] = '项目轨迹可绘制'
            if len(dossiers) == 1:
                dossier['route_figure'] = plan_name
                continue
            check_cancel()
            route_path = figures_dir / f'{line_id}_route.png'
            try:
                export_project_plan_map(
                    store, route_path, selected_line=line_id, line_ids={line_id},
                    line_ranges=map_ranges, include_interpretation=False, include_layers=False,
                    require_trajectory=True, cancel_requested=lambda: (check_cancel(), False)[1], dpi=130,
                )
                figure_paths.append(route_path)
                dossier['route_figure'] = route_path.name
            except (OSError, ValueError):
                _LOGGER.warning('报告测线图无法生成：%s', line_id, exc_info=True)
    return dossiers, figure_paths, plan_name, map_note, record_codes


def _load_processed_profile(store, line_id, raw, artifact, chain, document, notes):
    processed = mapping = None
    root = chain[0] if chain else {}
    raw_digest = raw.metadata.get('data_sha256')
    mismatched = (not root.get('parent_id') and root.get('input_sha256') and raw_digest
                  and root['input_sha256'] != raw_digest)
    if mismatched:
        notes.append('处理版本的原始输入与当前测线版本不同，未生成对照图。')
    elif artifact:
        try:
            processed = store.storage.load_processing_artifact(line_id, artifact.artifact_id, raw_dataset=raw)
        except _DATA_ERRORS:
            notes.append('所选处理版本无法读取')
            _LOGGER.warning('报告处理成果读取失败：%s', artifact.artifact_id, exc_info=True)
    segments = None
    if processed is not None:
        try:
            generation = raw.metadata.get('marker_source_version') or (raw.metadata.get('metadata') or {}).get('marker_source_version')
            raw_source = document['source_id'] if document else f'raw:{generation}' if generation else ''
            segments, mapping = artifact_grid(
                store, line_id, artifact.artifact_id, processed, raw_source)
        except _DATA_ERRORS:
            notes.append('道映射校验失败，处理剖面不可用于对照')
            _LOGGER.warning('报告成果道映射读取失败：%s', artifact.artifact_id, exc_info=True)
            processed = mapping = None
    return processed, mapping, segments


def _write_line_figures(store, line_id, raw, artifact, chain, dossier, document, ranges, directory, include_interpretation):
    notes = dossier['figure_notes']
    processed, mapping, segments = _load_processed_profile(store, line_id, raw, artifact, chain, document, notes)
    raw_ann, processed_ann, annotation_note = (
        _annotation(store, line_id, raw, artifact) if include_interpretation else (None, None, ''))
    if annotation_note:
        notes.append(annotation_note)
    if processed_ann is not None and (processed is None or
            (processed_ann.sample_count, processed_ann.trace_count) != processed.matrix.shape):
        notes.append('界面标注与所选处理剖面尺寸不一致')
        processed_ann = None
    if raw_ann is not None or processed_ann is not None:
        annotation = raw_ann or processed_ann
        notes.append(f'蓝线为已保存界面解释 · 版本 {annotation.version} · '
                     f'{"已确认" if annotation.status == "confirmed" else "待确认"}')
    x, label, mapped, x_limits, y_limits = profile_axes(raw, processed, mapping)
    if processed is not None:
        notes.append('两图分别归一化，振幅仅作图内比较')
        if mapped is None:
            notes.append('缺少可靠横向映射，处理剖面使用自身道号')
        elif len(mapped) != raw.trace_count or np.any(np.diff(segments)):
            notes.append('处理图仅绘制成果中的道，空白处无对应成果道')
        if y_limits is None:
            notes.append('纵轴按各自的数据网格显示')
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / f'{line_id}_raw.png'
    write_profile_figure(raw_path, raw, title=f'{line_id} · 原始 B-scan',
                         positions=x, x_label=label, segments=np.zeros(raw.trace_count, dtype=int),
                         ranges=ranges, annotation=raw_ann, x_limits=x_limits, y_limits=y_limits)
    dossier['raw_figure'] = raw_path.name
    figures = [raw_path]
    if processed is not None:
        processed_path = directory / f'{line_id}_processed.png'
        position = mapped if mapped is not None else np.arange(1, processed.trace_count + 1)
        notice = '橙色虚线标示处理断点' if np.any(np.diff(segments)) else ''
        write_profile_figure(
            processed_path, processed, title=f'{line_id} · 处理后 B-scan · {artifact.method_name[:32]}',
            positions=position, x_label=label if mapped is not None else '成果道号',
            segments=segments, ranges=project_ranges(ranges, mapping if mapped is not None else None, segments),
            annotation=processed_ann, x_limits=x_limits, y_limits=y_limits, notice=notice,
        )
        dossier['processed_figure'] = processed_path.name
        figures.append(processed_path)
    analysis = dossier['analysis']
    if analysis.get('code'):
        code = analysis['code']
        start, stop = analysis['first_trace'] - 1, analysis['last_trace']
        limits = _edges(x)[[start, stop]]
        detail_raw = directory / f'{code}_raw.png'
        write_profile_figure(detail_raw, raw, title=f'{code} · 原始局部剖面', positions=x, x_label=label,
                             segments=np.zeros(raw.trace_count, dtype=int), annotation=raw_ann,
                             x_limits=limits, y_limits=y_limits, trace_window=(start, stop))
        dossier['detail_figures'].append(detail_raw.name)
        figures.append(detail_raw)
        if processed is not None:
            if mapped is None:
                dossier['detail_note'] = f'{code} · 处理版本缺少横向映射，无法生成对应局部剖面。'
            else:
                left = np.searchsorted(mapping.coordinates, start)
                right = np.searchsorted(mapping.coordinates, stop - 1, side='right')
                if left >= right:
                    dossier['detail_note'] = f'{code} · 所选处理版本在此区段无保留道。'
                else:
                    detail_processed = directory / f'{code}_processed.png'
                    write_profile_figure(detail_processed, processed, title=f'{code} · 处理后局部剖面',
                                         positions=mapped, x_label=label, segments=segments,
                                         annotation=processed_ann, x_limits=limits, y_limits=y_limits,
                                         trace_window=(left, right), notice=notice)
                    dossier['detail_figures'].append(detail_processed.name)
                    figures.append(detail_processed)
    return figures
