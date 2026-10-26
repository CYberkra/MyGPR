"""Annotation sessions must stay bound to their project and survey line."""
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from ui.coordinator_processing import ProcessingChain
from ui.coordinator_project import ProjectChain


def make_coordinator():
    views = {name: Mock() for name in (
        'homeInterface', 'projectInterface', 'processingInterface',
        'interpretationInterface', 'deliveryInterface', 'spatialInterface',
    )}
    co = SimpleNamespace(
        page=views.__getitem__, file_tree=lambda: None, settings=None,
        project_controller=Mock(), interpretation_controller=Mock(),
        delivery_controller=None, current_project_id=lambda: 'P1',
        infobar=Mock(), log_message=Mock(), goto_page=Mock(),
    )
    co.project = ProjectChain(co)
    co.processing = ProcessingChain(co)
    co.jobs = SimpleNamespace(import_job_ids=set(), spatial_job_ids=set())
    co.project.current_line_id = 'L01'
    co.require_project = lambda: True
    co.require_line = lambda: co.project.current_line_id
    co.interpretation_controller.is_current_session.return_value = True
    co.interpretation_controller.has_unsaved_changes.return_value = False
    return co, views


def snapshot(project='P1', line='L01', sid='session-1'):
    return SimpleNamespace(
        project_id=project, line_id=line, session_id=sid,
        input_artifact_id='', annotation=SimpleNamespace(points=[(3, 4)]),
    )


def test_switch_line_closes_session_and_clears_annotation_context():
    co, views = make_coordinator()

    co.project.on_line_selected('L02')

    co.interpretation_controller.close_session.assert_called_once_with()
    page = views['interpretationInterface']
    page.set_session_active.assert_called_once_with(False)
    page.set_session_info.assert_called_once_with('未打开会话')
    page.set_points.assert_called_once_with([])
    page.set_bundle.assert_called_once_with(None)
    page.set_artifacts.assert_called_once_with([])
    page.set_line_label.assert_called_once_with('L02')


def test_select_same_line_keeps_current_annotation_session():
    co, views = make_coordinator()

    co.project.on_line_selected('L01')

    co.interpretation_controller.close_session.assert_not_called()
    views['interpretationInterface'].set_points.assert_not_called()


@pytest.mark.parametrize('event', ['opened', 'closed'])
def test_project_transition_invalidates_annotation_session(event):
    co, views = make_coordinator()
    if event == 'opened':
        co.project.on_project_opened(SimpleNamespace(name='new project', root_path=''))
    else:
        co.project.on_project_closed()

    co.interpretation_controller.close_session.assert_called_once_with()
    views['interpretationInterface'].set_session_active.assert_called_once_with(False)
    views['interpretationInterface'].set_points.assert_called_once_with([])
    views['interpretationInterface'].set_bundle.assert_called_once_with(None)


@pytest.mark.parametrize('callback', ['on_session_opened', 'on_session_updated'])
@pytest.mark.parametrize('wrong', ['project', 'line', 'session'])
def test_stale_annotation_snapshot_is_not_presented(callback, wrong):
    co, views = make_coordinator()
    value = snapshot(project='old-project') if wrong == 'project' else (
        snapshot(line='L00') if wrong == 'line' else snapshot(sid='old-session'))
    if wrong == 'session':
        co.interpretation_controller.is_current_session.return_value = False

    getattr(co.processing, callback)(value)

    views['interpretationInterface'].set_points.assert_not_called()
    views['interpretationInterface'].set_session_active.assert_not_called()
    co.project_controller.preview_line.assert_not_called()
    co.infobar.assert_not_called()


def test_current_snapshot_opens_session_and_requests_its_own_preview():
    co, views = make_coordinator()

    co.processing.on_session_opened(snapshot())

    views['interpretationInterface'].set_points.assert_called_once_with([(3, 4)])
    views['interpretationInterface'].set_session_active.assert_called_once_with(True)
    co.project_controller.preview_line.assert_called_once_with('L01')


@pytest.mark.parametrize('discard', [False, True])
def test_dirty_annotation_requires_explicit_discard_before_line_switch(monkeypatch, discard):
    co, views = make_coordinator()
    co.interpretation_controller.has_unsaved_changes.return_value = True
    ask = Mock(return_value=discard)
    monkeypatch.setitem(sys.modules, 'ui.dialogs', SimpleNamespace(ask_discard_interpretation=ask))

    co.project.on_line_selected('L02')

    ask.assert_called_once()
    assert co.project.current_line_id == ('L02' if discard else 'L01')
    if discard:
        co.interpretation_controller.close_session.assert_called_once()
    else:
        co.interpretation_controller.close_session.assert_not_called()
        views['interpretationInterface'].set_points.assert_not_called()
        co.project_controller.preview_line.assert_not_called()
        views['projectInterface'].select_line.assert_called_once_with('L01')


def test_dirty_annotation_can_cancel_project_change(monkeypatch):
    co, _ = make_coordinator()
    co.interpretation_controller.has_unsaved_changes.return_value = True
    monkeypatch.setitem(sys.modules, 'ui.dialogs', SimpleNamespace(
        ask_discard_interpretation=Mock(return_value=False)))

    assert not co.project.can_change_project()
    co.interpretation_controller.close_session.assert_not_called()


def test_dirty_annotation_can_cancel_replacing_its_session(monkeypatch):
    co, _ = make_coordinator()
    co.require_line = lambda: 'L01'
    co.interpretation_controller.has_unsaved_changes.return_value = True
    monkeypatch.setitem(sys.modules, 'ui.dialogs', SimpleNamespace(
        ask_discard_interpretation=Mock(return_value=False)))

    co.processing.on_open_session_requested('other-artifact')

    co.interpretation_controller.open_session.assert_not_called()


@pytest.mark.parametrize('entry', ['process', 'spatial', 'artifact', 'selector', 'original', 'reconfigure'])
def test_cancel_line_change_stops_the_requested_cross_page_action(monkeypatch, entry):
    co, views = make_coordinator()
    co.interpretation_controller.has_unsaved_changes.return_value = True
    monkeypatch.setitem(sys.modules, 'ui.dialogs', SimpleNamespace(
        ask_discard_interpretation=Mock(return_value=False)))
    actions = {
        'process': lambda: co.project.on_line_process_requested('L02'),
        'spatial': lambda: co.project.on_spatial_current_line('L02'),
        'artifact': lambda: co.project.on_artifact_focus_requested('L02', 'artifact-L02'),
        'selector': lambda: co.processing.on_processing_line_changed('L02'),
        'original': lambda: co.processing.on_original_focus_requested('L02'),
        'reconfigure': lambda: co.processing.on_reconfigure_requested('L02', 'artifact-L02'),
    }

    actions[entry]()

    assert co.project.current_line_id == 'L01'
    assert co.project.pending_focus_artifact == ''
    assert co.processing._pending_reconfigure is None
    co.goto_page.assert_not_called()
    co.infobar.assert_not_called()
    co.project_controller.preview_artifact.assert_not_called()
    views['processingInterface'].select_original.assert_not_called()


@pytest.mark.parametrize('active', [False, True])
def test_session_failure_preserves_a_live_session_for_retry(active):
    co, views = make_coordinator()
    co.interpretation_controller.has_active_session.return_value = active

    co.processing.on_session_failed('temporary save failure')

    views['interpretationInterface'].set_session_active.assert_called_once_with(active)
    views['interpretationInterface'].set_points.assert_not_called()
    co.infobar.assert_called_once()


def test_real_annotation_page_can_retry_a_failed_save(qapp, monkeypatch):
    from ui.controllers import interpretation_controller as module
    from ui.pages.interpretation_page import InterpretationPage

    queued = []
    monkeypatch.setattr(module, 'run_command', lambda command, **_: queued.append(command))
    service = Mock()
    value = snapshot()
    value.dirty = True
    service.open_session.return_value = value
    service.save_session.side_effect = [OSError('temporary write failure'), None]
    controller = module.InterpretationController()
    controller.set_backend(SimpleNamespace(backend=SimpleNamespace(interpretation_edit=service)))
    co, views = make_coordinator()
    co.interpretation_controller = controller
    page = InterpretationPage()
    views['interpretationInterface'] = page
    controller.session_opened.connect(co.processing.on_session_opened)
    controller.session_failed.connect(co.processing.on_session_failed)
    controller.saved.connect(co.processing.on_annotation_saved)
    controller.busy_changed.connect(page.set_busy)
    page.save_requested.connect(controller.save)
    try:
        controller.open_session('P1', 'L01')
        queued.pop().execute()
        page._save_btn.click()
        queued.pop().execute()
        assert controller.has_unsaved_changes()
        assert page._session_active and page._save_btn.isEnabled()
        assert page._points == [(3, 4)]
        page._save_btn.click()
        queued.pop().execute()
        assert not controller.has_unsaved_changes()
        assert page._session_active and page._points == [(3, 4)]
        assert service.save_session.call_count == 2
    finally:
        controller.close_session()
        page.close()
        page.deleteLater()


def test_real_page_cannot_save_old_points_after_line_switch(qapp):
    from ui.pages.interpretation_page import InterpretationPage

    co, views = make_coordinator()
    page = InterpretationPage()
    views['interpretationInterface'] = page
    page.set_line_label('L01')
    page.set_points([(3, 4)])
    page.set_session_active(True)
    saved = []
    page.save_requested.connect(lambda: saved.append(True))
    try:
        co.project.on_line_selected('L02')
        page._save_btn.click()
        assert page._line_label.text() == 'L02'
        assert not page._session_active
        assert page._points == []
        assert not page._save_btn.isEnabled()
        assert saved == []
    finally:
        page.close()
        page.deleteLater()
