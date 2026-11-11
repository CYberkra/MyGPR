"""Project marker use cases shared by desktop and command-line clients."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from mygpr.domain.markers.mapping import (
    MarkerProjection, SEGMENT_KEY, add_source_mapping, index_spans, project_anchor, read_mapping,
)
from mygpr.domain.markers.models import EventInterval, MarkerAnchor, MarkerDocument, MarkerRecord
from mygpr.domain.markers.operations import put_record, remove_record
from mygpr.domain.markers.ranges import interval_indices, selection_indices


def metadata_value(metadata, key, default=None):
    from collections.abc import Mapping
    while isinstance(metadata, Mapping):
        if key in metadata:
            return metadata[key]
        metadata = metadata.get('metadata')
    return default


STATUS_LABELS = {'exact': '可定位', 'interpolated': '插值定位', 'outside': '不在当前结果',
                 'unmapped': '无法定位', 'open': '未闭合', 'empty': '期间无采集道',
                 'needs_review': '需复核', 'excluded': '已排除'}


@dataclass(frozen=True)
class MarkerView:
    document: MarkerDocument
    target_id: str
    artifact_id: str
    count: int
    header: dict
    trace_metadata: dict
    rows: tuple[dict, ...]
    read_only: bool
    time_basis_confirmed: bool
    timezone: str = 'UTC'
    interpretation_links: tuple[tuple[str, str], ...] = ()

    def mapping(self, source_id):
        return read_mapping(self.header, self.trace_metadata, source_id, self.target_id, self.count)

    def indices(self, interval):
        source = interval.start.source_id if interval.start else (interval.time_source_id or self.target_id)
        return interval_indices(interval, self.mapping(source),
            timestamps_s=self.trace_metadata.get('trace_timestamp_s'),
            time_basis_confirmed=self.time_basis_confirmed)

    def anchor(self, trace: int):
        if not 0 <= trace < self.count:
            raise ValueError('道号超出当前数据范围')
        mapping = self.mapping(self.document.source_id)
        value = mapping.coordinates[trace]
        # Resampled columns remain local even where an interpolated coordinate is integral.
        if mapping.kind in ('identity', 'selection') and float(value).is_integer():
            return MarkerAnchor(mapping.source_id, int(value))
        return MarkerAnchor(self.target_id, trace)


class MarkerService:
    def __init__(self, projects):
        self.projects = projects

    def view(self, project_id, line_id, artifact_id='', *, tracks=None) -> MarkerView:
        with self.projects.acquire_lease(project_id):
            document = self.projects.read_markers(project_id, line_id)
            info = (self.projects.get_artifact_dataset_info(project_id, line_id, artifact_id)
                    if artifact_id else self.projects.get_dataset_info(project_id, line_id))
            trace = (self.projects.read_artifact_trace_metadata(project_id, line_id, artifact_id)
                     if artifact_id else self.projects.read_trace_metadata(project_id, line_id)) or {}
            header = dict(info.metadata)
            if tracks is None:
                tracks = self.projects.load_spatial_tracks(project_id)
            track = next((item for item in tracks if item.line_id == line_id), None)
            header['_marker_track'] = track
            for key in ('marker_maps', 'marker_source_id', 'marker_mapping_kind'):
                value = metadata_value(header, key)
                if value is not None:
                    header[key] = value
            target = 'artifact:' + artifact_id if artifact_id else document.source_id
            add_source_mapping(header, trace, target, info.shape[1])
            # timestamp_ms is the acquisition format's explicit UTC contract.
            times = trace.get('trace_timestamp_s')
            utc = 'timestamp_ms' in trace or metadata_value(header, 'trace_time_basis') == 'UTC'
            reliable = bool(utc and times is not None and np.shape(times) == (info.shape[1],)
                            and np.all(np.isfinite(times)) and np.all(np.diff(times) >= 0))
            summary = self.projects.get_summary(project_id)
            features = self.projects.list_interpretation_features(project_id, line_id)
            interface = self.projects.load_interface_annotation(project_id, line_id, create=False)
            links = tuple((item.feature_id, item.label) for item in features)
            if interface is not None:
                links += ((interface.annotation_id, interface.name),)
            view = MarkerView(document, target, artifact_id, info.shape[1], header, trace, (),
                              summary.read_only, reliable, document.timezone, links)
            return replace(view, rows=project_rows(view))

    def project_views(self, project_id):
        tracks = self.projects.load_spatial_tracks(project_id)
        return tuple(self.view(project_id, line.line_id, tracks=tracks)
                     for line in self.projects.list_lines(project_id) if line.trace_count > 0)

    def edit(self, view, record=None, *, delete_id=''):
        if view.read_only:
            raise ValueError('项目只读，不能编辑标记')
        if record is not None:
            anchors = (record.start, record.end) if isinstance(record, EventInterval) else (record.anchor,)
            for anchor in anchors:
                if anchor and anchor.source_id == view.target_id and anchor.trace >= view.count:
                    raise ValueError('道号超出当前数据范围')
            previous = next((item for item in view.document.records if item.uid == record.uid), None)
            existing = set(previous.links) if previous else set()
            available = {uid for uid, _ in view.interpretation_links}
            if set(record.links) - available - existing:
                raise ValueError('关联解释对象不存在，请重新加载列表')
        document = (remove_record(view.document, delete_id) if delete_id
                    else put_record(view.document, record))
        return self.projects.save_markers(document.project_id, document,
                                         expected_revision=view.document.revision)

    def preview(self, view: MarkerView, ids, mode):
        selected_ids = set(ids)
        selected = tuple(item for item in view.document.intervals if item.uid in selected_ids)
        if not selected or len(selected) != len(selected_ids):
            raise ValueError('请选择有效事件区间')
        groups = []
        for item in selected:
            indices = view.indices(item)
            if not indices.size:
                raise ValueError(f'{item.name or item.uid}：期间无采集道或不在当前结果，请从操作中移除')
            groups.append(indices)
        result = selection_indices(view.count, groups, mode)
        affected = np.unique(np.concatenate(groups))
        segments = view.trace_metadata.get(SEGMENT_KEY)
        return {
            'project_id': view.document.project_id, 'line_id': view.document.line_id,
            'input_artifact_id': view.artifact_id, 'source_id': view.target_id,
            'source_version': view.document.source_id, 'document_revision': view.document.revision,
            'mode': mode, 'intervals': [asdict(item) for item in selected],
            'ranges': index_spans(affected), 'selected_input_spans': index_spans(result),
            'output_segments': index_spans(result, segments),
            'input_count': view.count, 'affected_count': len(affected), 'output_count': len(result),
        }

    def execute(self, preview, *, context):
        current = self.view(preview['project_id'], preview['line_id'], preview['input_artifact_id'])
        if current.read_only:
            raise ValueError('项目只读，不能保存处理成果')
        if (current.document.revision != preview['document_revision']
                or current.document.source_id != preview['source_version']):
            raise ValueError('记录或源数据已变化，请重新预览')
        actual = self.preview(current, [item['uid'] for item in preview['intervals']], preview['mode'])
        if actual != preview:
            raise ValueError('处理输入已变化，请重新预览')
        return self.projects.select_marker_ranges(current, preview, context=context)


def project_rows(view):
    from mygpr.domain.markers.geometry import reference_geometry, track_index
    rows = []
    track = view.header.get('_marker_track')
    prepared = track_index(track)
    # Cache each version once; 10,000 records must not copy 10,000 mapping arrays.
    mappings = {}
    for record in view.document.records:
        if isinstance(record, MarkerRecord):
            source = record.anchor.source_id if record.anchor else ''
            if source not in mappings:
                mappings[source] = view.mapping(source)
            projection = project_anchor(record.anchor, mappings[source])
            position = (f'道 {record.anchor.trace + 1}' if record.anchor else '未定位')
        else:
            projection = _interval_projection(record, view, mappings)
            position = (f'{record.start.trace + 1}–{record.end.trace + 1} 道'
                        if record.start and record.end else '采集时间' if record.time_start else '开放范围')
        status = ('excluded' if record.excluded else 'needs_review'
                  if isinstance(record, EventInterval) and record.review_state == 'needs_review'
                  else projection.status)
        rows.append({'uid': record.uid, 'record': record, 'projection': projection,
                     'name': record.name or f'标记 {record.uid[:8]}', 'purpose': record.purpose,
                     'position': position, 'status': STATUS_LABELS[status], 'line_id': view.document.line_id,
                     'geometry': reference_geometry(record, view.document.source_id, track, prepared)})
    return tuple(rows)


def _interval_projection(record, view, mappings):
    if not record.closed:
        anchor = record.start or record.end
        if anchor:
            if anchor.source_id not in mappings:
                mappings[anchor.source_id] = view.mapping(anchor.source_id)
            mapping = mappings[anchor.source_id]
            point = project_anchor(anchor, mapping)
            return MarkerProjection('open', point.positions)
        return MarkerProjection('open')
    source = record.start.source_id if record.start else (record.time_source_id or view.target_id)
    if source not in mappings:
        mappings[source] = view.mapping(source)
    mapping = mappings[source]
    # Review state affects usability, not the visibility of saved boundaries.
    try:
        indices = interval_indices(replace(record, excluded=False, review_state='unreviewed'), mapping,
            timestamps_s=view.trace_metadata.get('trace_timestamp_s'), time_basis_confirmed=view.time_basis_confirmed)
    except ValueError:
        return MarkerProjection('unmapped')
    status = ('interpolated' if mapping.kind == 'resampled' else 'exact') if indices.size else (
        'outside' if record.start else 'empty')
    return MarkerProjection(status, spans=index_spans(indices, mapping.segments))
