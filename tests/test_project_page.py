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
