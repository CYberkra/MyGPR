# -*- coding: utf-8 -*-
"""Qt workbench flow: edit a new draft, run, then branch from the saved result.

Uses synthetic previews and saved-artifact metadata; no algorithm/physical
validation is claimed by this UI regression.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6")


def _bundle(tag: float):
    matrix = np.full((60, 80), float(tag), dtype=np.float32)
    return SimpleNamespace(
        matrix=matrix, vmin=0.0, vmax=float(tag), title=f'data{tag}',
        x_label='道数', y_label='采样点', trace_axis_m=None, sample_axis=None,
        sample_axis_label='', trace_count=80, sample_count=60,
        trace_elevation_m=None, depth_axis_m=None)


def _artifact(aid, method, group, step, kind='intermediate', *, run_input='', step_input=''):
    return SimpleNamespace(
        artifact_id=aid, line_id='L01', name=f'run_{aid}',
        method_id=method, method_name=method,
        created_at=f'2026-09-26T1{step}:0{step}:00',
        manifest={'params': {'artifact_kind': kind, 'run_group_id': group,
                             'run_step_index': step, 'lineage_version': 2,
                             'run_input_artifact_id': run_input,
                             'step_input_artifact_id': step_input}})


def _run_and_feed(page, group, aids_methods, tag_base):
    run_input = page.current_pipeline()['input_artifact_id']
    previous = run_input
    artifacts = list(page._artifacts_by_id.values())
    for index, (aid, method) in enumerate(aids_methods):
        artifacts.append(_artifact(aid, method, group, index + 1,
            kind='processing' if index == len(aids_methods) - 1 else 'intermediate',
            run_input=run_input, step_input=previous))
        previous = aid
    page.set_running(True)
    page.set_artifacts(artifacts)
    for aid, _m in aids_methods:
        page.set_artifact_bundle(aid, _bundle(tag_base))
    page.set_running(False, success=True)
    # Production controller publishes the refreshed saved catalog after job
    # completion. Resolve the completion against that explicit publication.
    page.set_artifacts(artifacts)


@pytest.fixture
def page(qapp):
    from PyQt6.QtCore import QCoreApplication, QEvent
    from ui.pages.processing_page import ProcessingPage
    page = ProcessingPage()
    yield page
    page.close()
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


class TestUserPathRegression:
    """A completed run is saved history; subsequent steps form a new branch."""

    def test_full_path(self, page):
        # 1. Initial import opens an ordinary raw tab in a single view group.
        page.set_original_bundle(_bundle(1.0))
        workspace = page._workspace
        view = workspace.primary_view()
        assert workspace.tab_keys() == ['original']
        raw_matrix = view._matrix
        assert raw_matrix is not None

        # 2. Draft edits never change the raw preview or create result tabs.
        page._pipeline_list.add_step('dewow', '零时校正 (Dewow)', {})
        page._pipeline_list.add_step('sec_gain', 'SEC 增益 (AGC)', {})
        page._pipeline_list.add_step('agc', '自动增益控制 (AGC)', {})
        assert workspace.tab_keys() == ['original']
        assert workspace.primary_view() is view
        assert view._matrix is raw_matrix
        assert len(page.current_pipeline()['steps']) == 3

        # 3-4. Staying on the submitted input allows final-result activation.
        _run_and_feed(page, 'G1', [('R1', 'dewow'), ('R2', 'sec_gain'),
                                   ('R3', 'agc')], 10.0)
        assert workspace.tab_keys() == ['original', 'artifact:R3']
        assert page.current_pipeline()['input_artifact_id'] == 'R3'
        assert page.current_pipeline()['steps'] == []
        assert workspace.primary_view() is view
        assert len(workspace.views()) == 1
        first_result = view._matrix

        # 5. New draft starts at R3. Saved upstream methods are not replayed.
        page._pipeline_list.add_step('dewow', 'Dewow', {'win': 11})
        page._pipeline_list.add_step('sec_gain', 'SEC', {'alpha': 0.2})
        page._pipeline_list.add_step('agc', 'AGC', {})
        page._pipeline_list.update_step_params(1, {'alpha': 0.4})
        assert view._matrix is first_result
        assert workspace.active_key() == 'artifact:R3'
        assert page.current_pipeline()['steps'][1]['params'] == {'alpha': 0.4}

        # 6. Reordering the new draft preserves the completed run and tabs.
        page._pipeline_list._move_step_to(0, 2)
        assert view._matrix is first_result
        assert workspace.tab_keys() == ['original', 'artifact:R3']
        assert [s['method_id'] for s in page.current_pipeline()['steps']] == ['sec_gain', 'dewow', 'agc']

        # 7. The new run extends R3; all old saved results remain catalogued.
        _run_and_feed(page, 'G2', [('T1', 'sec_gain'), ('T2', 'dewow'),
                                   ('T3', 'agc')], 20.0)
        assert workspace.tab_keys() == ['original', 'artifact:R3', 'artifact:T3']
        assert page.current_pipeline()['input_artifact_id'] == 'T3'
        assert page.current_pipeline()['steps'] == []
        assert set(page._artifacts_by_id) == {'R1', 'R2', 'R3', 'T1', 'T2', 'T3'}
        assert page._artifacts_by_id['T1'].manifest['params']['step_input_artifact_id'] == 'R3'
        assert page._artifacts_by_id['T2'].manifest['params']['step_input_artifact_id'] == 'T1'
        assert workspace.primary_view() is view
        assert len(workspace.views()) == 1
        page.select_artifact('R3')
        assert page.current_pipeline()['steps'] == []  # submitted R3 draft cleared after success
        assert view._matrix is first_result
