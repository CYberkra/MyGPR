# -*- coding: utf-8 -*-
"""InterpretationPage 数据下拉与会话门控测试（P1-3 / P1-6）。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过，见 tests/conftest.py qapp 设计

from ui.pages.interpretation_page import InterpretationPage  # noqa: E402


def _artifacts():
    return [
        SimpleNamespace(artifact_id='L03_processed_1', name='成果A'),
        SimpleNamespace(artifact_id='L03_processed_2', name='成果B'),
    ]


class TestArtifactCombo:
    def test_default_is_raw_data(self, qapp):
        page = InterpretationPage()
        page.set_artifacts(_artifacts())
        assert page._artifact_combo.currentIndex() == 0
        assert page._current_artifact_id() == ''

    def test_select_artifact_reads_back(self, qapp):
        page = InterpretationPage()
        page.set_artifacts(_artifacts())
        page._artifact_combo.setCurrentIndex(1)
        assert page._current_artifact_id() == 'L03_processed_1'
        page._artifact_combo.setCurrentIndex(2)
        assert page._current_artifact_id() == 'L03_processed_2'

    def test_reload_preserves_selection(self, qapp):
        page = InterpretationPage()
        page.set_artifacts(_artifacts())
        page._artifact_combo.setCurrentIndex(2)
        page.set_artifacts([SimpleNamespace(artifact_id='L03_processed_2', name='成果B')])
        assert page._current_artifact_id() == 'L03_processed_2'
        # 选择失效时回退到原始数据
        page.set_artifacts([SimpleNamespace(artifact_id='L03_processed_9', name='成果X')])
        assert page._current_artifact_id() == ''


class TestSessionGating:
    def test_edit_buttons_disabled_before_session(self, qapp):
        page = InterpretationPage()
        page.set_session_active(False)
        for btn in (page._auto_trace_btn, page._snap_btn, page._smooth_btn,
                    page._undo_btn, page._redo_btn, page._save_btn):
            assert not btn.isEnabled(), btn.text()

    def test_edit_buttons_enabled_after_session(self, qapp):
        page = InterpretationPage()
        page.set_session_active(True)
        for btn in (page._auto_trace_btn, page._snap_btn, page._smooth_btn,
                    page._undo_btn, page._redo_btn, page._save_btn):
            assert btn.isEnabled(), btn.text()

    def test_busy_respects_session_gate(self, qapp):
        page = InterpretationPage()
        page.set_session_active(True)
        page.set_busy(True)
        assert not page._auto_trace_btn.isEnabled()
        page.set_busy(False)
        assert page._auto_trace_btn.isEnabled()

    def test_confirm_action_emits_confirmed_status(self, qapp):
        page = InterpretationPage()
        page.set_session_active(True)
        statuses = []
        page.save_requested.connect(statuses.append)
        page._confirm_btn.click()
        assert statuses == ['confirmed']

    def test_editing_confirmed_annotation_marks_review_required(self, qapp):
        page = InterpretationPage()
        page.set_session_active(True)
        page.set_saved_status('confirmed')
        page._on_point_picked(3, 4)
        assert '需重新确认' in page._session_status_label.text()


class TestSetBundleNoneGuard:
    """P0-1 配套守卫：协调器项目直切时 set_bundle(None) 清残留——
    BScanView.set_bundle 不接受 None，必须走 clear 等价清空。"""

    def test_none_bundle_clears_view(self, qapp, monkeypatch):
        page = InterpretationPage()
        calls: list = []
        monkeypatch.setattr(page._bscan, 'clear',
                            lambda: calls.append('clear'))
        monkeypatch.setattr(page._bscan, 'set_bundle',
                            lambda b: calls.append(('set_bundle', b)))
        page.set_bundle(None)
        assert calls == ['clear']            # None 只走 clear，不碰 set_bundle
        assert page._bundle is None

    def test_real_bundle_still_reaches_view(self, qapp, monkeypatch):
        page = InterpretationPage()
        calls: list = []
        monkeypatch.setattr(page._bscan, 'clear',
                            lambda: calls.append('clear'))
        monkeypatch.setattr(page._bscan, 'set_bundle',
                            lambda b: calls.append(('set_bundle', b)))
        bundle = object()
        page.set_bundle(bundle)
        assert calls == [('set_bundle', bundle)]
        assert page._bundle is bundle
