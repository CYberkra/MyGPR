"""An explicit project open must recover a closed B-scan without losing drafts."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip('PyQt6')

from ui.coordinator_project import ProjectChain
from ui.pages.processing_page import ProcessingPage


def preview():
    return NS(matrix=np.arange(1200, dtype='float32').reshape(30, 40),
              vmin=0, vmax=1200, title='原始输入', sample_axis=None,
              sample_axis_label='')


@pytest.fixture
def context(qapp):
    page = ProcessingPage()
    page.set_project_context('P')
    page.set_lines([NS(line_id='L01', name='实测'), NS(line_id='L02', name='另一条')])
    page.set_line_label('L01')
    queued, navigation = [], []
    def noop(*args):
        return None
    other = NS(select_line=lambda *args: True, set_line_resume_available=noop,
               set_line_label=noop, set_session_active=noop, set_session_info=noop,
               set_points=noop, set_bundle=noop, set_artifacts=noop, set_velocity_failed=noop)
    pages = {'processingInterface': page, 'projectInterface': other,
             'interpretationInterface': other}
    co = NS(page=pages.__getitem__, file_tree=lambda: None, settings=None,
            current_project_id=lambda: 'P', goto_page=navigation.append, infobar=noop,
            processing=NS(velocity_token=None), interpretation_controller=None,
            project_controller=NS(preview_line=queued.append, line_sensor_summary=noop,
                                  refresh_artifacts=noop, invalidate_artifact_previews=noop))
    chain = ProjectChain(co)
    chain.current_line_id = 'L01'
    yield page, chain, queued, navigation
    page.close()
    page.deleteLater()
    qapp.processEvents()


def closed_layout(page):
    page.set_original_bundle(preview())
    page._workspace.close_source('original')
    assert page._workspace.tab_keys() == []


def test_project_open_reopens_closed_raw_tab(context):
    page, chain, queued, navigation = context
    closed_layout(page)

    chain.on_line_process_requested('L01')
    page.set_original_bundle(preview())

    assert queued == ['L01']
    assert navigation == ['processingInterface']
    assert page._workspace.tab_keys() == ['original']
    np.testing.assert_array_equal(page._workspace.primary_view()._matrix, preview().matrix)


def test_project_open_overrides_saved_empty_layout_before_async_preview(context):
    page, chain, _, _ = context
    closed_layout(page)
    page.set_line_label('L02')
    chain.current_line_id = 'L02'

    chain.on_line_process_requested('L01')
    assert page._workspace.tab_keys() == ['original']
    page.set_original_bundle(preview())
    page.set_artifacts([])

    assert page._workspace.tab_keys() == ['original']
    np.testing.assert_array_equal(page._workspace.primary_view()._matrix, preview().matrix)


def test_explicit_raw_selection_before_preview_arrives_opens_loading_tab(context):
    page, _, _, _ = context

    page.select_original()
    assert page._workspace.tab_keys() == ['original']
    page.set_original_bundle(preview())

    np.testing.assert_array_equal(page._workspace.primary_view()._matrix, preview().matrix)


def test_close_after_explicit_open_is_not_undone_by_late_preview(context):
    page, chain, _, _ = context
    closed_layout(page)
    page.set_line_label('L02')
    chain.current_line_id = 'L02'
    chain.on_line_process_requested('L01')
    assert page._workspace.tab_keys() == ['original']
    page._workspace.close_source('original')

    page.set_original_bundle(preview())
    page.set_artifacts([])

    assert page._workspace.tab_keys() == []


def test_automatic_preview_preserves_intentionally_closed_layout(context):
    page, _, _, _ = context
    closed_layout(page)
    page.set_line_label('L02')
    page.set_line_label('L01')

    page.set_original_bundle(preview())
    page.set_artifacts([])

    assert page._workspace.tab_keys() == []


def test_project_open_preserves_existing_result_input_and_draft(context):
    page, chain, _, _ = context
    page.set_original_bundle(preview())
    page.set_artifacts([NS(artifact_id='A', line_id='L01', name='保存结果',
        method_id='dewow', method_name='Dewow', status='success',
        created_at='2026-10-27T12:00:00', manifest={}, sha256='A')])
    page.select_artifact('A')
    page.set_artifact_bundle('A', preview())
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 41})
    before = page.current_pipeline()
    tabs = page._workspace.tab_keys()

    chain.on_line_process_requested('L01')
    page.set_original_bundle(preview())

    assert page._workspace.tab_keys() == tabs
    assert page._workspace.active_key() == 'artifact:A'
    assert page.current_pipeline() == before
