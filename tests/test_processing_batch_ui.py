"""Application-only batch processing and execution-option contracts."""
from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')


def test_batch_dialog_submits_selected_lines_with_step_results_default(qapp):
    from ui.pages.processing_batch import BatchProcessingDialog

    lines = [SimpleNamespace(line_id=f'L{i:02d}', name=f'Line {i}') for i in range(1, 4)]
    dialog = BatchProcessingDialog(lines, [{'method_id': 'dewow', 'params': {'window': 23}}])
    dialog._lines_list.item(1).setCheckState(dialog._lines_list.item(1).checkState().Unchecked)
    payload = dialog.payload()
    assert payload['line_ids'] == ['L01', 'L03']
    assert payload['save_intermediates'] is True
    assert payload['input_artifact_ids'] == {}
    assert payload['steps'][0]['method_id'] == 'dewow'
    dialog.close()
    dialog.deleteLater()


def test_processing_page_pipeline_defaults_to_step_results(qapp):
    from ui.pages.processing_page import ProcessingPage

    page = ProcessingPage()
    page._pipeline_list.add_step('dewow', 'Dewow', {'window': 23})
    assert page.current_pipeline()['save_intermediates'] is True
    page._save_intermediates_check.setChecked(False)
    assert page.current_pipeline()['save_intermediates'] is False
    page.close()
    page.deleteLater()
