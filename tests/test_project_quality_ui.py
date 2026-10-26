"""Batch import and application quality-center contracts."""
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import Qt


def test_project_page_emits_all_selected_import_sources(qapp):
    from ui.pages.project_page import ProjectPage

    page = ProjectPage()
    page.set_project_info(NS(read_only=False))
    page._import_sources = ['D:/data/L01.h5', 'D:/data/L02.h5']
    page.file_edit.setText('2 个文件')
    requests = []
    page.import_requested.connect(requests.append)
    page._emit_import_request(preflight=False)
    assert requests[0]['sources'] == page._import_sources
    page.close()
    page.deleteLater()


def test_quality_center_displays_issues_and_filters_failed_lines(qapp):
    from ui.pages.project_quality import QualityCenterDialog

    reports = [
        NS(line_id='L01', status='passed', status_label='通过', sample_count=501, trace_count=1000,
           time_window_ns=900., length_m=120., finite_ratio=1., trajectory_points=1000,
           sampled=False, issues=()),
        NS(line_id='L02', status='warning', status_label='警告', sample_count=501, trace_count=900,
           time_window_ns=900., length_m=100., finite_ratio=.98, trajectory_points=800,
           sampled=True, issues=(NS(severity='warning', code='nan_inf',
                                  message='存在坏值', suggestion='检查原始文件'),)),
    ]
    dialog = QualityCenterDialog()
    dialog.set_reports(reports)
    assert dialog._table.rowCount() == 2
    dialog._table.selectRow(1)
    assert '存在坏值' in dialog._details.toPlainText()
    assert '检查原始文件' in dialog._details.toPlainText()
    dialog._status_filter.setCurrentIndex(dialog._status_filter.findData('warning'))
    assert dialog._table.rowCount() == 1
    assert dialog._table.item(0, 0).text() == 'L02'
    dialog.close()
    dialog.deleteLater()
