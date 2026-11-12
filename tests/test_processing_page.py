# -*- coding: utf-8 -*-
"""ProcessingPage AutoTune 门控测试（P2-7：运行期间禁用"开始调参"防重复提交）。"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过，见 tests/conftest.py qapp 设计

from ui.pages.processing_page import ProcessingPage  # noqa: E402
from tests.test_processing_workspace_flow import bundle  # noqa: E402


def test_run_shortcut_only_operates_in_processing_page(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLineEdit, QStackedWidget

    host = QStackedWidget()
    other = QLineEdit()
    page = ProcessingPage()
    page.set_line_label('L01')
    page.set_original_bundle(bundle())
    page._add_method_to_pipeline('dewow')
    host.addWidget(other)
    host.addWidget(page)
    host.resize(1200, 850)
    host.show()
    host.activateWindow()
    cancelled = []
    page.cancel_requested.connect(lambda: cancelled.append(True))
    runs = []
    page.run_requested.connect(runs.append)
    try:
        host.setCurrentWidget(other)
        other.setFocus()
        QTest.qWait(1)
        QTest.keyClick(other, Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier)
        assert not runs and not cancelled
        host.setCurrentWidget(page)
        page._line_combo.setFocus()
        QTest.qWait(1)
        QTest.keyClick(page._line_combo, Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier)
        assert len(runs) == 1
        page.set_running(True, 'job')
        QTest.keyClick(page._line_combo, Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier)
        page._on_run_clicked()
        assert len(runs) == 1 and cancelled == []
        page._cancel_btn.click()
        assert cancelled == [True]
    finally:
        host.close()
        host.deleteLater()


class TestAutoTuneGate:
    def test_disabled_before_method_selected(self, qapp):
        page = ProcessingPage()
        assert not page._autotune_btn.isEnabled()

    def test_enabled_after_method_selected(self, qapp):
        page = ProcessingPage()
        page.set_line_label('L01')
        page.set_original_bundle(bundle())
        page._add_method_to_pipeline('dewow')
        assert page._autotune_btn.isEnabled()

    def test_running_disables_button(self, qapp):
        page = ProcessingPage()
        page.set_line_label('L01')
        page.set_original_bundle(bundle())
        page._add_method_to_pipeline('dewow')
        page.set_autotune_running(True)
        assert not page._autotune_btn.isEnabled()

    def test_finish_reenables_button(self, qapp):
        page = ProcessingPage()
        page.set_line_label('L01')
        page.set_original_bundle(bundle())
        page._add_method_to_pipeline('dewow')
        page.set_autotune_running(True)
        page.set_autotune_running(False)
        assert page._autotune_btn.isEnabled()

    def test_switch_method_while_running_stays_disabled(self, qapp):
        page = ProcessingPage()
        page.set_line_label('L01')
        page.set_original_bundle(bundle())
        page._add_method_to_pipeline('dewow')
        page.set_autotune_running(True)
        page._add_method_to_pipeline('agcGain')
        assert not page._autotune_btn.isEnabled()
