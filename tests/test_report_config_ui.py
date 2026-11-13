"""Report configuration and safe restore application contracts."""
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import Qt


def test_report_config_builds_complete_backend_profile(qapp):
    from ui.pages.report_config import ReportConfigDialog

    lines = [NS(line_id='L01', name='一号线'), NS(line_id='L02', name='二号线')]
    artifacts = [NS(artifact_id='A1', line_id='L01', name='去漂移')]
    spatial = [NS(result_id='R1', name='正式成果', revision=2)]
    dialog = ReportConfigDialog(lines, artifacts, spatial)
    dialog._package_name.setText('final')
    dialog._title.setText('工程报告')
    dialog._compiler.setText('甲')
    dialog._reviewer.setText('乙')
    dialog._approver.setText('丙')
    dialog._line_items['L02'].setCheckState(0, Qt.CheckState.Unchecked)
    values = dialog.payload()
    assert values['package_name'] == 'final'
    profile = values['report_profile']
    assert profile['selected_line_ids'] == ['L01']
    assert profile['selected_artifact_ids'] == ['A1']
    assert profile['spatial_result_id'] == 'R1'
    assert profile['compiler'] == '甲'
    assert profile['reviewer'] == '乙'
    assert profile['approver'] == '丙'
    assert profile['include_integrity_manifest'] is True
    dialog._artifact_items['A1'].setCheckState(0, Qt.CheckState.Unchecked)
    assert dialog.payload()['report_profile']['selected_artifact_ids'] == []
    assert '0 项处理成果' in dialog._scope_summary.text()
    dialog._line_items['L01'].setCheckState(0, Qt.CheckState.Unchecked)
    dialog.accept()
    assert not dialog._confirm.isEnabled()
    assert not dialog.result()
    dialog.close()
    dialog.deleteLater()


def test_restore_emits_new_project_destination_without_overwrite(qapp, monkeypatch, tmp_path):
    from ui.pages.delivery_page import DeliveryPage
    import ui.pages.delivery_page as module

    monkeypatch.setattr(module.file_dialogs, 'getOpenFileName',
                        lambda *args, **kwargs: (str(tmp_path / 'backup.zip'), ''))
    monkeypatch.setattr(module.file_dialogs, 'getExistingDirectory',
                        lambda *args, **kwargs: str(tmp_path / 'restored'))
    monkeypatch.setattr(module.MessageBox, 'exec', lambda self: 1)
    page = DeliveryPage()
    page._restore_name_edit.setText('RecoveredProject')
    requests = []
    page.restore_requested.connect(requests.append)
    page._on_restore_clicked()
    assert requests[0]['destination_root'].endswith('restored')
    assert requests[0]['project_dir_name'] == 'RecoveredProject'
    assert requests[0]['open_after_restore'] is True
    page.close()
    page.deleteLater()


def test_report_config_keeps_line_edits_and_requires_an_in_scope_main_version(qapp):
    from ui.pages.report_config import ReportConfigDialog

    lines = [NS(line_id=key, name=key, trace_count=60) for key in ('L01', 'L02')]
    artifacts = [NS(artifact_id=key, line_id='L01', name=key) for key in ('A1', 'A2')]
    dialog = ReportConfigDialog(lines, artifacts, [])
    selector = dialog._primary_selectors['L01']
    assert dialog._artifact_items['A1'].parent() == dialog._line_items['L01']
    selector.setCurrentIndex(selector.findData(''))
    selector.setCurrentIndex(selector.findData('A1'))
    dialog._manual_fields['frequency'].setText('10–400 MHz')
    dialog._first_trace.setValue(12)
    dialog._last_trace.setValue(20)
    dialog._analysis_fields['observation'].setPlainText('观察甲')
    dialog._content_line.setCurrentIndex(dialog._content_line.findData('L02'))
    assert not dialog._manual_fields['frequency'].text()
    dialog._content_line.setCurrentIndex(dialog._content_line.findData('L01'))
    assert dialog._first_trace.value() == 12
    assert dialog._analysis_fields['observation'].toPlainText() == '观察甲'

    dialog._artifact_items['A1'].setCheckState(0, Qt.CheckState.Unchecked)
    dialog.accept()
    assert not dialog.result()
    assert '请重新选择' in dialog._scope_error.text()
    selector.setCurrentIndex(selector.findData(''))
    profile = dialog.payload()['report_profile']
    assert profile['primary_artifacts'] == {'L01': ''}
    assert profile['line_analysis']['L01']['last_trace'] == 20
    assert profile['acquisition_overrides']['L01']['frequency'] == '10–400 MHz'
    dialog.close()
    dialog.deleteLater()
