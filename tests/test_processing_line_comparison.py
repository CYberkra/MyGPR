"""Cross-line comparisons never replace the selected processing input."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QApplication

from ui.pages.processing_page import ProcessingPage


def bundle(value):
    return SimpleNamespace(matrix=np.full((30, 40), value, dtype='float32'),
        vmin=-value, vmax=value, title=str(value), sample_axis=np.arange(30)*2.,
        sample_axis_label='时间 (ns)')


def artifact(aid):
    return SimpleNamespace(artifact_id=aid, line_id='L01', name=aid, method_id='dewow',
        method_name='Dewow', status='success', created_at='2026-10-26T12:00:00',
        manifest={'params': {'lineage_version': 2, 'step_input_artifact_id': '',
            'run_input_artifact_id': '', 'run_group_id': aid, 'run_step_index': 1,
            'artifact_kind': 'processing', 'lineage': []}})


@pytest.fixture
def page(qapp):
    page = ProcessingPage()
    page.resize(1280, 720)
    page.set_project_context('P')
    page.set_lines([SimpleNamespace(line_id=k, name=k) for k in ('L01', 'L02', 'L03')])
    page.set_line_label('L01')
    page.set_original_bundle(bundle(1))
    page.show()
    qapp.processEvents()
    yield page
    page.close()
    page.deleteLater()
    qapp.processEvents()


def comparison(page):
    assert hasattr(page, '_comparison'), 'Cross-line comparison manager missing'
    return page._comparison


def requests(page):
    records = []
    page.comparison_preview_requested.connect(lambda *args: records.append(args))
    return records


def drop(page, line='L02', edge='right', group_id=None):
    manager = comparison(page)
    group = page._workspace.group_widget(group_id)
    target = group.view._glw.viewport()
    group_point = QPoint(group.width()-5, group.height()//2) if edge == 'right' else QPoint(group.width()//2, group.height()//2)
    point = target.mapFrom(group, group_point)
    mime = manager.mime_for_line(line)
    entered = QDragEnterEvent(point, Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(target, entered)
    event = QDropEvent(QPointF(point), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(target, event)
    return event.isAccepted()


def test_real_drop_splits_and_targets_preview_without_switching_input(page, qapp):
    manager = comparison(page)
    pending = requests(page)
    original = page._workspace.primary_view()._matrix
    assert drop(page)
    assert len(page._workspace.views()) == 2
    assert page._line_id == 'L01' and page._selected_source_key == 'original'
    assert page._line_combo.currentData() == 'L01'
    assert len(pending) == 1 and pending[0][:2] == ('P', 'L02')
    manager.deliver(*pending[0], bundle(2), '')
    arrays = [view._matrix for view in page._workspace.views()]
    assert arrays[0] is original and np.all(arrays[1] == 2)
    assert 'L02' in page._workspace.tab_bar().tabText(0)
    page.line_changed.connect(page.set_line_label)
    page._workspace.source_activated_by_user.emit(page._workspace.active_key())
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 23})
    assert page._line_id == 'L02'
    assert page._selected_source_key == 'original'
    assert page._run_btn.isEnabled()


def test_repeat_drop_reuses_tab_and_center_adds_tab(page):
    manager = comparison(page)
    pending = requests(page)
    assert drop(page)
    manager.deliver(*pending[-1], bundle(2), '')
    assert drop(page)
    assert len(page._workspace.views()) == 2 and len(pending) == 1
    assert drop(page, 'L03', 'center')
    assert len(page._workspace.views()) == 2
    assert len(page._workspace.tab_keys()) == 2
    assert page._line_id == 'L01'


def test_comparison_and_split_survive_switching_processing_line(page):
    manager = comparison(page)
    pending = requests(page)
    assert drop(page)
    reference = bundle(2)
    manager.deliver(*pending[-1], reference, '')
    page.set_line_label('L03')
    page.set_original_bundle(bundle(3))
    assert len(page._workspace.views()) == 2
    assert any(view._matrix is reference.matrix for view in page._workspace.views())
    assert any(np.all(view._matrix == 3) for view in page._workspace.views())
    assert page._line_id == 'L03'
    assert page._selected_source_key == 'original'
    assert page._workspace.active_key() == 'original'


def test_close_evicts_preview_and_stale_delivery_cannot_reopen_it(page):
    manager = comparison(page)
    pending = requests(page)
    assert drop(page)
    first = pending[-1]
    key = page._workspace.active_key()
    manager.deliver(*first, bundle(2), '')
    page._workspace.close_source(key)
    assert manager.sources[key].get('bundle') is None
    manager.deliver(*first, bundle(20), '')
    assert key not in [key for gid in page._workspace.group_ids() for key in page._workspace.tab_keys(gid)]
    assert page._workspace.reopen_closed() == key
    second = pending[-1]
    assert second != first
    manager.deliver(*first, bundle(20), '')
    assert page._workspace.primary_view()._matrix is None
    manager.deliver(*second, bundle(2), '')
    assert np.all(page._workspace.primary_view()._matrix == 2)


def test_project_roundtrip_and_deleted_lines_reject_old_payloads(page):
    manager = comparison(page)
    pending = requests(page)
    assert drop(page)
    first = pending[-1]
    page.set_project_context('Q')
    page.set_project_context('P')
    manager.deliver(*first, bundle(20), '')
    assert not manager.sources
    page.set_line_label('L01')
    page.set_original_bundle(bundle(1))
    assert drop(page)
    page.set_lines([SimpleNamespace(line_id='L01', name='L01')])
    manager.deliver(*pending[-1], bundle(20), '')
    assert not manager.sources
    assert len(page._workspace.views()) == 1


def test_error_is_owned_by_reference_retry_gets_a_new_request(page):
    manager = comparison(page)
    pending = requests(page)
    assert drop(page)
    first = pending[-1]
    key = page._workspace.active_key()
    manager.deliver(*first, None, '文件暂时不可读')
    assert '文件暂时不可读' in page._workspace.primary_view()._empty_overlay._hint_label.text()
    assert not page._sources['original'].get('error')
    assert manager.retry(key)
    assert pending[-1] != first
    manager.deliver(*first, bundle(20), '')
    assert page._workspace.primary_view()._matrix is None
    manager.deliver(*pending[-1], bundle(2), '')
    assert np.all(page._workspace.primary_view()._matrix == 2)


def test_explicit_make_input_is_required_and_reference_run_is_disabled(page):
    manager = comparison(page)
    emitted, pending = [], requests(page)
    page.run_requested.connect(emitted.append)
    assert drop(page)
    manager.deliver(*pending[-1], bundle(2), '')
    page._on_run_clicked()
    assert emitted == [] and page._selected_source_key == 'original'
    page.line_changed.connect(page.set_line_label)
    key = page._workspace.active_key()
    assert manager.make_input(key)
    assert page._line_id == 'L02'
    assert page._selected_source_key == 'original'


def test_foreign_or_malformed_line_mime_is_rejected(page):
    from PyQt6.QtCore import QMimeData
    manager = comparison(page)
    other = ProcessingPage()
    try:
        other.set_project_context('P')
        other.set_lines([SimpleNamespace(line_id='L02', name='L02')])
        assert not manager.read_mime(other._comparison.mime_for_line('L02'))
        assert not manager.read_mime(QMimeData())
        assert not manager.mime_for_line('unknown').formats()
        import json
        from ui.pages.processing_comparison import MIME
        for invalid in ([], {}, 1, None):
            mime = QMimeData()
            mime.setData(MIME, json.dumps(dict(scope=manager.epoch, project='P', line=invalid)).encode())
            assert not manager.read_mime(mime)
    finally:
        other.close()


def test_local_window_and_color_follow_survive_comparison_tab_switch(page):
    from PyQt6.QtCore import QRectF
    manager, pending = comparison(page), requests(page)
    assert drop(page, edge='center')
    manager.deliver(*pending[-1], bundle(2), '')
    key = page._workspace.active_key()
    view = page._workspace.primary_view()
    view.set_view_selection(QRectF(4, 3, 10, 12))
    expected = view._plot.vb.viewRange()
    page._workspace.open_source('original')
    page._workspace.open_source(key)
    assert view._view_selection is not None
    np.testing.assert_allclose(view._plot.vb.viewRange(), expected)
    assert view._selection_follows_levels


def test_drag_current_line_from_result_does_not_change_processing_draft(page):
    page.set_artifacts([artifact('A')])
    page.select_artifact('A')
    page.set_artifact_bundle('A', bundle(5))
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 41})
    before = page.current_pipeline()
    pending = requests(page)
    assert drop(page, 'L01')
    assert page._selected_source_key == 'artifact:A'
    assert page.current_pipeline() == before
    assert len(page._workspace.views()) == 2
    assert pending[-1][:2] == ('P', 'L01')


def test_run_finishing_does_not_steal_comparison_focus(page):
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 41})
    page.set_running(True, 'job')
    page.set_run_result('job', 'new')
    assert drop(page)
    key = page._workspace.active_key()
    page.set_running(False, success=True)
    page.set_artifacts([artifact('new')])
    assert page._workspace.active_key() == key
    assert page._selected_source_key == 'original'
    assert page._completed_artifact == 'new'


def test_cross_line_drop_includes_raw_and_saved_processing_results(page):
    manager = comparison(page)
    assert drop(page, 'L02')
    manager.set_artifacts('P', 'L02', [artifact('A')])
    keys = page._workspace.tab_keys()
    assert any(key.startswith('compare:') and key.endswith(':raw') for key in keys)
    assert any(key.startswith('compare:') and key.endswith(':artifact:A') for key in keys)
    source = next(source for key, source in manager.sources.items() if key.endswith(':artifact:A'))
    assert source['presentation'].tab_title
    token = manager.pending[source['key']]
    manager.deliver_artifact('P', 'L02', 'A', token, bundle(8), '')
    assert source['bundle'].matrix.max() == 8


def test_comparison_artifact_signal_preserves_project_line_and_object_payload(page):
    from ui.controllers.project_controller import ProjectController

    manager = comparison(page)
    assert drop(page, 'L02')
    controller = ProjectController()
    controller.comparison_artifacts_ready.connect(manager.set_artifacts)
    controller.comparison_artifacts_ready.emit('P', 'L02', [artifact('A')])
    assert any(key.endswith(':artifact:A') for key in manager.sources)
    controller.comparison_artifacts_ready.emit('Q', 'L02', [artifact('STALE')])
    assert not any(key.endswith(':artifact:STALE') for key in manager.sources)


def test_cross_line_artifact_can_become_processing_input(page):
    manager = comparison(page)
    assert drop(page, 'L02')
    manager.set_artifacts('P', 'L02', [artifact('A')])
    key = next(key for key in manager.sources if key.endswith(':artifact:A'))
    focused = []
    page.artifact_focus_requested.connect(lambda *args: focused.append(args))
    assert manager.make_input(key)
    assert focused == [('L02', 'A')]
