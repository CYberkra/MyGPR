# -*- coding: utf-8 -*-
"""ProcessingPage AutoTune 门控测试（P2-7：运行期间禁用"开始调参"防重复提交）。"""
from __future__ import annotations

import pytest

from ui.pages.processing_page import ProcessingPage


class TestAutoTuneGate:
    def test_disabled_before_method_selected(self, qapp):
        page = ProcessingPage()
        assert not page._autotune_btn.isEnabled()

    def test_enabled_after_method_selected(self, qapp):
        page = ProcessingPage()
        page._on_method_selected('dewow')
        assert page._autotune_btn.isEnabled()

    def test_running_disables_button(self, qapp):
        page = ProcessingPage()
        page._on_method_selected('dewow')
        page.set_autotune_running(True)
        assert not page._autotune_btn.isEnabled()

    def test_finish_reenables_button(self, qapp):
        page = ProcessingPage()
        page._on_method_selected('dewow')
        page.set_autotune_running(True)
        page.set_autotune_running(False)
        assert page._autotune_btn.isEnabled()

    def test_switch_method_while_running_stays_disabled(self, qapp):
        page = ProcessingPage()
        page._on_method_selected('dewow')
        page.set_autotune_running(True)
        page._on_method_selected('agcGain')  # 运行中切方法也不应启用
        assert not page._autotune_btn.isEnabled()
