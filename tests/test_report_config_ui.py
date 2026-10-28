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
    dialog._lines_list.item(1).setCheckState(Qt.CheckState.Unchecked)
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
