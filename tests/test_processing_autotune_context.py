"""AutoTune recommendation ownership with synthetic previews; no algorithm run."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PyQt6')

from ui.pages.processing_page import ProcessingPage


def _bundle():
    return SimpleNamespace(
        matrix=np.ones((20, 30), dtype=np.float32), vmin=0., vmax=1., title='test',
        x_label='trace', y_label='sample', trace_axis_m=None, sample_axis=None,
        sample_axis_label='', trace_count=30, sample_count=20,
        trace_elevation_m=None, depth_axis_m=None)


@pytest.fixture
def page(qapp):
    from PyQt6.QtCore import QCoreApplication, QEvent

    widget = ProcessingPage()
    widget.set_project_context('P')
    widget.autotune_requested.connect(lambda *_: widget.set_autotune_running(True))
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def _select_line(page, line_id, win):
    page.set_line_label(line_id)
    page.set_original_bundle(_bundle())
    page._pipeline_list.add_step('dewow', 'Dewow', {'win': win})


def _finish(page, win):
    # Match the production coordinator's finish ordering.
    page.set_autotune_running(False)
    page.set_autotune_result('dewow', {'best_params': {'win': win}})


def test_new_request_clears_previous_inputs_recommendation(page):
    _select_line(page, 'L01', 3)
    page._on_autotune_clicked()
    _finish(page, 99)
    assert page._adopt_params_btn.isEnabled()

    _select_line(page, 'L02', 7)
    page._on_autotune_clicked()
    assert not page._adopt_params_btn.isEnabled()
    page._result_name_edit.setText('L02 result')
    assert not page._adopt_params_btn.isEnabled()
    assert page._autotune_result_label.text() == ''
    # The action itself must also reject an obsolete recommendation.
    page._on_adopt_params()
    assert page.current_pipeline()['steps'][0]['params'] == {'win': 7}

    _finish(page, 11)
    assert page._adopt_params_btn.isEnabled()
    page._adopt_params_btn.click()
    assert page.current_pipeline()['steps'][0]['params'] == {'win': 11}
    page.set_line_label('L01')
    page.set_original_bundle(_bundle())
    assert page.current_pipeline()['steps'][0]['params'] == {'win': 3}


def test_same_input_new_request_cannot_adopt_previous_result(page):
    _select_line(page, 'L01', 3)
    page._on_autotune_clicked()
    _finish(page, 99)
    page._on_autotune_clicked()
    page._result_name_edit.setText('second request')
    assert not page._adopt_params_btn.isEnabled()
    page._on_adopt_params()
    assert page.current_pipeline()['steps'][0]['params'] == {'win': 3}


def test_completion_keeps_submitted_input_identity_after_navigation(page):
    _select_line(page, 'L01', 3)
    page._on_autotune_clicked()
    _select_line(page, 'L02', 7)
    _finish(page, 99)
    assert not page._adopt_params_btn.isEnabled()
    page._on_adopt_params()
    assert page.current_pipeline()['steps'][0]['params'] == {'win': 7}

    page.set_line_label('L01')
    page.set_original_bundle(_bundle())
    assert page._adopt_params_btn.isEnabled()
    page._adopt_params_btn.click()
    assert page.current_pipeline()['steps'][0]['params'] == {'win': 99}


def test_recommendation_follows_exact_step_after_reorder_and_selection_change(page):
    _select_line(page, 'L01', 3)
    page._pipeline_list.add_step('dewow', 'Dewow', {'win': 7})
    submitted = []
    page.autotune_requested.connect(lambda *args: submitted.append(args))
    target = page._pipeline_list.step_uid(1)
    page._on_autotune_clicked()
    assert submitted == [('dewow', {'win': 7}, '')]
    page._pipeline_list._move_step_to(1, 0)
    page._pipeline_list.select_step(1)
    _finish(page, 99)
    assert '应用到当前第 1 步' in page._autotune_result_label.text()
    page._adopt_params_btn.click()
    assert page._pipeline_list.step_uid(0) == target
    assert [step['params']['win'] for step in page.current_pipeline()['steps']] == [99, 3]


@pytest.mark.parametrize('edit', ['remove', 'replace', 'params'])
def test_recommendation_rejects_stale_target_without_same_method_fallback(page, edit):
    _select_line(page, 'L01', 3)
    page._pipeline_list.add_step('dewow', 'Dewow', {'win': 7})
    page._on_autotune_clicked()
    if edit == 'remove':
        page._on_chain_step_removed(1)
    elif edit == 'replace':
        page._replace_step_method(1, 'dewow', {'win': 7})
    else:
        page._pipeline_list.update_step_params(1, {'win': 9})
    before = page.current_pipeline()
    _finish(page, 99)
    assert not page._adopt_params_btn.isEnabled()
    page._on_adopt_params()
    assert page.current_pipeline() == before


def test_changed_input_revision_blocks_adoption_before_preview_arrives(page):
    _select_line(page, 'L01', 3)
    page._on_autotune_clicked()
    _finish(page, 99)
    assert page._adopt_params_btn.isEnabled()
    page.set_original_revision('P', 'L01', 'new revision')
    assert not page._adopt_params_btn.isEnabled()
    page._on_adopt_params()
    assert page.current_pipeline()['steps'][0]['params'] == {'win': 3}
