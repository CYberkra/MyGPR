"""Project persistence, exact selection, provenance and immutable deliverables."""
from dataclasses import replace
import json

import numpy as np
import pytest

from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.markers.models import MarkerRecord, EventInterval
from mygpr.domain.markers.mapping import SEGMENT_KEY
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


@pytest.fixture
def project(tmp_path):
    backend = MyGPRBackend.create_default(max_workers=1)
    summary = backend.projects.create_project(tmp_path / 'project', name='事件记录')
    matrix = np.arange(96, dtype=np.float32).reshape(12, 8)
    backend.projects.save_line_dataset(summary.project_id, 'L01', matrix, length_m=7.)
    try:
        yield backend, summary.project_id, matrix, tmp_path
    finally:
        backend.shutdown()


def save_event(backend, pid, start, end, name='停车等待'):
    view = backend.markers.view(pid, 'L01')
    event = EventInterval(name=name, start=view.anchor(start), end=view.anchor(end))
    backend.markers.edit(view, event)
    return event


def test_keep_exclude_commit_exact_columns_and_reopen_mapping(project):
    backend, pid, matrix, _ = project
    first = save_event(backend, pid, 1, 2)
    second = save_event(backend, pid, 5, 6, '目标范围')
    view = backend.markers.view(pid, 'L01')
    preview = backend.markers.preview(view, [first.uid, second.uid], 'keep')
    assert preview['affected_count'] == 4
    assert preview['output_segments'] == ((1, 3), (5, 7))
    artifact = backend.markers.execute(preview, context=ExecutionContext.null())
    receipt = artifact.manifest['params']
    assert receipt['lineage_version'] == 2
    assert receipt['run_step_index'] == len(receipt['lineage']) == 1
    assert receipt['lineage'][0]['params']['mode'] == 'keep'
    assert tuple(map(tuple, receipt['lineage'][0]['params']['ranges'])) == tuple(map(tuple, preview['ranges']))
    output = backend.projects.read_artifact_dataset(pid, 'L01', artifact.artifact_id)
    np.testing.assert_array_equal(output.data, matrix[:, [1, 2, 5, 6]])
    np.testing.assert_array_equal(output.header_info['distance_axis_m'], [1, 2, 5, 6])
    np.testing.assert_array_equal(output.trace_metadata[SEGMENT_KEY], [0, 0, 1, 1])
    np.testing.assert_array_equal(backend.projects.read_dataset(pid, 'L01').data, matrix)
    projected = backend.markers.view(pid, 'L01', artifact.artifact_id)
    assert projected.rows[0]['projection'].spans == ((0, 2),)
    assert projected.rows[1]['projection'].spans == ((2, 4),)
    assert projected.rows[0]['position'] == '1–2 道'
    assert projected.rows[1]['position'] == '3–4 道'
    root = backend.projects.get_summary(pid).root_path
    backend.projects.close_project(pid)
    backend.projects.open_project(root)
    reopened = backend.markers.view(pid, 'L01', artifact.artifact_id)
    assert reopened.rows[0]['projection'] == projected.rows[0]['projection']


def test_linked_interval_uses_saved_range_until_an_endpoint_moves(project):
    backend, pid, _, _ = project
    view = backend.markers.view(pid, 'L01')
    point = MarkerRecord(name='起点', anchor=view.anchor(1))
    backend.markers.edit(view, point)
    view = backend.markers.view(pid, 'L01')
    interval = EventInterval(name='停车等待', start=point.anchor, end=view.anchor(4),
                             start_marker_id=point.uid, start_marker_revision=point.revision)
    backend.markers.edit(view, interval)

    view = backend.markers.view(pid, 'L01')
    point, = view.document.markers
    saved, = view.document.intervals
    backend.markers.edit(view, replace(point, note='文字补充'))
    view = backend.markers.view(pid, 'L01')
    assert view.document.intervals == (saved,)
    assert backend.markers.preview(view, [interval.uid], 'keep')['affected_count'] == 4

    point, = view.document.markers
    backend.markers.edit(view, replace(point, anchor=view.anchor(2)))
    view = backend.markers.view(pid, 'L01')
    assert view.rows[-1]['status'] == '起点已移动'
    assert not view.rows[-1]['projection'].spans
    with pytest.raises(ValueError, match='起点已移动'):
        backend.markers.preview(view, [interval.uid], 'keep')

    point, = view.document.markers
    backend.markers.edit(view, replace(saved, start=point.anchor, start_marker_revision=point.revision))
    view = backend.markers.view(pid, 'L01')
    assert backend.markers.preview(view, [interval.uid], 'keep')['affected_count'] == 3


def test_stale_preview_and_edit_refused_and_snapshot_stable(project):
    backend, pid, _, tmp_path = project
    event = save_event(backend, pid, 2, 3)
    old = backend.markers.view(pid, 'L01')
    preview = backend.markers.preview(old, [event.uid], 'exclude')
    backend.projects.export_markers((old,), tmp_path / 'markers.json', 'json')
    exported = (tmp_path / 'markers.json').read_bytes()
    artifact = backend.markers.execute(preview, context=ExecutionContext.null())
    backend.markers.edit(old, replace(event, note='重新复核边界'))
    with pytest.raises(ValueError, match='变化'):
        backend.markers.execute(preview, context=ExecutionContext.null())
    with pytest.raises(ValueError, match='修改'):
        backend.markers.edit(old, replace(event, note='过期编辑'))
    saved = backend.projects.list_artifacts(pid, 'L01')[0]
    assert saved.artifact_id == artifact.artifact_id
    assert saved.manifest['params']['marker_selection']['intervals'][0]['note'] == ''
    assert (tmp_path / 'markers.json').read_bytes() == exported


def test_save_failure_keeps_file_and_draft(project, monkeypatch):
    backend, pid, _, _ = project
    event = save_event(backend, pid, 1, 1)
    view = backend.markers.view(pid, 'L01')
    session = backend.projects._session(pid)
    path = session._marker_path('L01')
    original = path.read_bytes()
    draft = replace(event, note='失败后仍可编辑')
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(session._store, 'write_json', fail)
    with pytest.raises(OSError, match='disk full'):
        backend.markers.edit(view, draft)
    assert path.read_bytes() == original
    assert draft.note == '失败后仍可编辑'


def test_replaced_source_does_not_rebind_same_trace(project):
    backend, pid, matrix, _ = project
    view = backend.markers.view(pid, 'L01')
    record = MarkerRecord(name='已知地物', anchor=view.anchor(3))
    backend.markers.edit(view, record)
    backend.projects.save_line_dataset(pid, 'L01', matrix + 1)
    new = backend.markers.view(pid, 'L01')
    assert new.document.source_id != view.document.source_id
    assert new.rows[0]['projection'].status == 'unmapped'


def test_discontinuity_blocks_horizontal_algorithms_but_per_trace_gain_preserves_mapping(project):
    backend, pid, matrix, _ = project
    event = save_event(backend, pid, 3, 4)
    view = backend.markers.view(pid, 'L01')
    artifact = backend.markers.execute(backend.markers.preview(view, [event.uid], 'exclude'), context=ExecutionContext.null())
    pipeline = PipelineDefinition(name='增益', steps=[PipelineStep('amplitude_scale', {'mode': 'constant', 'scale': 2.})])
    gain = backend.project_processing.execute_pipeline(pid, 'L01', pipeline,
                input_artifact_id=artifact.artifact_id, save_intermediates=True)
    actual = backend.projects.read_artifact_dataset(pid, 'L01', gain.artifact_id)
    assert len(actual.trace_metadata[SEGMENT_KEY]) == 6
    horizontal = PipelineDefinition(name='平滑', steps=[PipelineStep('gaussian_smoothing', {})])
    with pytest.raises(ValueError, match='不连续'):
        backend.project_processing.execute_pipeline(pid, 'L01', horizontal,
            input_artifact_id=artifact.artifact_id, save_intermediates=True)


def test_csv_export_contains_every_duplicate_and_no_raw_changes(project):
    backend, pid, _, tmp_path = project
    view = backend.markers.view(pid, 'L01')
    for i in range(3):
        backend.markers.edit(view, MarkerRecord(name='=同名', anchor=view.anchor(1)))
        view = backend.markers.view(pid, 'L01')
    backend.projects.export_markers((view,), tmp_path / 'all.csv', 'csv')
    assert (tmp_path / 'all.csv').read_text(encoding='utf-8-sig').count("'=同名") == 3
    backend.projects.export_markers((view,), tmp_path / 'all.geojson', 'geojson')
    payload = json.loads((tmp_path / 'all.geojson').read_text(encoding='utf-8'))
    assert payload['unlocated_records'] == 3 and not payload['features']


def test_interpretation_links_are_validated_and_protect_dependency(project):
    backend, pid, _, _ = project
    target = backend.interpretation.add_point(pid, 'L01', trace=3, sample=2, confidence=.7, label='参考目标')
    view = backend.markers.view(pid, 'L01')
    assert (target.feature_id, '参考目标') in view.interpretation_links
    marker = MarkerRecord(name='已知范围', anchor=view.anchor(3), links=(target.feature_id,))
    backend.markers.edit(view, marker)
    with pytest.raises(ValueError, match='引用'):
        backend.interpretation.delete_feature(pid, 'L01', target.feature_id)
    view = backend.markers.view(pid, 'L01')
    backend.markers.edit(view, replace(marker, links=()))
    assert backend.interpretation.delete_feature(pid, 'L01', target.feature_id)


def test_readonly_and_invalid_anchor_refuse_mutation(project):
    backend, pid, _, _ = project
    view = backend.markers.view(pid, 'L01')
    with pytest.raises(ValueError, match='只读'):
        backend.markers.edit(replace(view, read_only=True), MarkerRecord())
    with pytest.raises(ValueError, match='道号'):
        backend.markers.edit(view, MarkerRecord(anchor=replace(view.anchor(0), trace=80)))


def test_project_timezone_and_processing_receipt_survive_reopen(project):
    backend, pid, _, _ = project
    event = save_event(backend, pid, 2, 3)
    backend.maintenance.update_metadata(pid, timezone='Asia/Shanghai')
    view = backend.markers.view(pid, 'L01')
    assert view.timezone == view.document.timezone == 'Asia/Shanghai'
    preview = backend.markers.preview(view, [event.uid], 'exclude')
    artifact = backend.markers.execute(preview, context=ExecutionContext.null())
    current = backend.markers.view(pid, 'L01')
    backend.markers.edit(current, replace(current.document.intervals[0], note='后续复核'))
    root = backend.projects.get_summary(pid).root_path
    backend.projects.close_project(pid)
    backend.projects.open_project(root)
    reopened = backend.markers.view(pid, 'L01')
    assert reopened.timezone == 'Asia/Shanghai'
    receipt = reopened.processing_history[0]
    assert receipt['artifact_id'] == artifact.artifact_id
    assert receipt['selection']['intervals'][0]['note'] == ''
    from core.field_project_errors import FieldProjectOperationError
    with pytest.raises(FieldProjectOperationError, match='时区'):
        backend.maintenance.update_metadata(pid, timezone='invalid/zone')
    assert backend.projects.get_metadata(pid).timezone == 'Asia/Shanghai'


def test_dependency_document_does_not_require_readable_raw_array(project, monkeypatch):
    backend, pid, _, _ = project
    event = save_event(backend, pid, 1, 2)
    session = backend.projects._session(pid)
    def missing(*args):
        raise FileNotFoundError('missing raw')
    monkeypatch.setattr(session, 'get_dataset_info', missing)
    document = session.read_marker_dependencies('L01')
    assert document.intervals[0].uid == event.uid
