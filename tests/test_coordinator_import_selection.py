"""Import refresh must keep the processing selector on the imported line."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip('PyQt6')

from ui.coordinator_project import ProjectChain
from ui.pages.processing_page import ProcessingPage


def test_switching_lines_restores_saved_result_tabs_and_input(qapp):
    from tests.test_processing_workspace_flow import artifact, bundle
    page = ProcessingPage()
    other = Mock()
    pages = {name: other for name in ('projectInterface', 'interpretationInterface',
                                      'deliveryInterface', 'spatialInterface')}
    pages['processingInterface'] = page
    co = SimpleNamespace(page=pages.get, file_tree=lambda: None, project_controller=None,
                         processing=SimpleNamespace(velocity_token=None))
    chain = ProjectChain(co)
    page.set_project_context('P')
    page.set_lines([SimpleNamespace(line_id='L01'), SimpleNamespace(line_id='L02')])
    chain.on_line_selected('L01')
    page.set_original_bundle(bundle())
    page.set_artifacts([artifact('A')])
    page.select_artifact('A')
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 41})
    chain.on_line_selected('L02')
    page.set_original_bundle(bundle())
    page.set_artifacts([])
    chain.on_line_selected('L01')
    page.set_original_bundle(bundle())
    page.set_artifacts([artifact('A')])
    assert page._workspace.tab_keys() == ['original', 'artifact:A']
    assert page.current_pipeline()['input_artifact_id'] == 'A'
    assert page.current_pipeline()['steps'][0]['params'] == {'window': 41}
    page.close()
    page.deleteLater()


def test_imported_line_is_selected_after_dropdown_refresh(qapp):
    page = ProcessingPage()
    project = Mock()
    project.select_line.return_value = True
    pages = {'processingInterface': page, 'projectInterface': project,
             'interpretationInterface': Mock(), 'deliveryInterface': Mock(),
             'spatialInterface': Mock()}
    co = SimpleNamespace(page=pages.get, file_tree=lambda: None,
                         project_controller=None,
                         processing=SimpleNamespace(velocity_token=None))
    chain = ProjectChain(co)
    page.set_lines([SimpleNamespace(line_id='L01')])
    chain.current_line_id = 'L01'
    chain.pending_select_line_id = 'L02'
    chain.on_lines_updated([SimpleNamespace(line_id='L01'), SimpleNamespace(line_id='L02')])
    assert chain.current_line_id == 'L02'
    assert page._line_combo.currentData() == 'L02'
    page.close()
