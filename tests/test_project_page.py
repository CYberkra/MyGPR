# -*- coding: utf-8 -*-
"""ProjectPage 测线号自动建议测试（UX P2-1）。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过，见 tests/conftest.py qapp 设计

from ui.pages.project_page import ProjectPage  # noqa: E402


class TestLineIdSuggestion:
    def test_suggests_next_after_existing(self, qapp):
        page = ProjectPage()
        page.line_id_edit.setText('L01')
        page.set_lines([
            SimpleNamespace(line_id='L01'),
            SimpleNamespace(line_id='L02'),
        ])
        assert page.line_id_edit.text() == 'L03'

    def test_skips_occupied_ids(self, qapp):
        page = ProjectPage()
        page.line_id_edit.setText('L01')
        page.set_lines([SimpleNamespace(line_id='L03')])
        assert page.line_id_edit.text() == 'L01'  # L01 空闲

    def test_keeps_manual_input(self, qapp):
        page = ProjectPage()
        page.line_id_edit.setText('L99')
        page.set_lines([SimpleNamespace(line_id='L01')])
        assert page.line_id_edit.text() == 'L99'

    def test_no_lines_keeps_default(self, qapp):
        page = ProjectPage()
        page.line_id_edit.setText('L01')
        page.set_lines([])
        assert page.line_id_edit.text() == 'L01'

    def test_more_than_99_lines_suggests_unused_id(self, qapp):
        page = ProjectPage()
        page.set_lines([SimpleNamespace(line_id=f'L{i:02d}') for i in range(1, 100)])
        assert page.line_id_edit.text() == 'L100'

    def test_suggestion_advances_after_each_import(self, qapp):
        page = ProjectPage()
        page.set_lines([SimpleNamespace(line_id='L01')])
        assert page.line_id_edit.text() == 'L02'
        page.set_lines([SimpleNamespace(line_id='L01'), SimpleNamespace(line_id='L02')])
        assert page.line_id_edit.text() == 'L03'


def test_sync_and_preview_require_selection(qapp):
    page = ProjectPage()
    page.set_project_info(SimpleNamespace(read_only=False))
    assert not page.sync_btn.isEnabled()
    assert not page.preview_artifact_btn.isEnabled()
    page.set_lines([SimpleNamespace(line_id='L01')])
    page.select_line('L01')
    assert page.sync_btn.isEnabled()
    page.set_artifacts([SimpleNamespace(artifact_id='A1', name='Result')])
    page._artifacts_table.selectRow(0)
    assert page.preview_artifact_btn.isEnabled()
    page.set_lines([])
    assert not page.sync_btn.isEnabled()
    assert not page.preview_artifact_btn.isEnabled()


def test_readonly_and_busy_disable_writes_and_delete_requests(qapp):
    page = ProjectPage()
    page.set_project_info(SimpleNamespace(read_only=True))
    page.set_lines([SimpleNamespace(line_id='L01')])
    page.select_line('L01')
    page.set_artifacts([SimpleNamespace(artifact_id='A1', line_id='L01')])
    requests = []
    page.artifact_delete_requested.connect(lambda *args: requests.append(args))
    assert not page.import_btn.isEnabled()
    assert not page.sync_btn.isEnabled()
    assert not page._artifacts_table.cellWidget(0, 5).isEnabled()
    page._on_artifact_delete_clicked('A1')
    assert requests == []
    page.set_project_info(SimpleNamespace(read_only=False))
    assert page.import_btn.isEnabled()
    page.set_busy(True)
    page._on_artifact_delete_clicked('A1')
    assert requests == []


def test_file_fields_fit_inside_sidebar(qapp):
    page = ProjectPage()
    page.resize(1200, 650)
    page.show()
    qapp.processEvents()
    for field in (page.file_edit, *page._sensor_edits.values()):
        row = field.parentWidget().layout()
        assert row.minimumSize().width() <= field.parentWidget().width()
    assert page.rect().contains(page.import_btn.mapTo(page, page.import_btn.rect().center()))
    page.close()
