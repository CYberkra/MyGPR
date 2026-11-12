"""Gain editing displays completed frames without committing a processing result."""
from time import monotonic

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QTimer
from PyQt6.QtTest import QTest

from ui.desktop_backend_facade import evaluate_manual_gain_preview, manual_gain_preview_bundle, method_catalog
from ui.controllers.processing_controller import build_method_dicts
from ui.pages.processing_page import ProcessingPage
from ui.widgets.manual_gain_editor import GainCurveDialog, GainCurveEditor


def bundle(scale=1):
    raw = np.random.default_rng(8).normal(size=(31, 42)).astype(np.float32) * scale
    return manual_gain_preview_bundle(raw, {"time_axis_ns": np.arange(31) * 2.0})


def wait_until(check):
    deadline = monotonic() + 3
    while not check() and monotonic() < deadline:
        QTest.qWait(10)
    assert check()


def test_live_gain_matches_processed_values_and_keeps_reference_scale(qapp):
    dialog = GainCurveDialog()
    source = bundle()
    before = source.matrix.copy()
    dialog.preview.set_input(source, "L01 · 原始输入")
    dialog.move_node(4, 1, 20)
    wait_until(lambda: dialog.ok_button.isEnabled() and dialog.preview._result is not None)
    expected = evaluate_manual_gain_preview(source.matrix, dialog.value())
    np.testing.assert_array_equal(dialog.preview.view._matrix, expected)
    np.testing.assert_allclose(dialog.preview.view._image_item.levels, [source.vmin, source.vmax])
    dialog.preview.mode.setCurrentIndex(1)
    np.testing.assert_array_equal(dialog.preview.view._matrix, before)
    dialog.preview.mode.setCurrentIndex(0)
    np.testing.assert_array_equal(dialog.preview.view._matrix, expected)
    np.testing.assert_array_equal(source.matrix, before)
    dialog.reject()


def test_continuous_moves_start_work_and_show_completed_frames_before_release(qapp, monkeypatch):
    import ui.widgets.manual_gain_preview as preview_module
    commands = []
    monkeypatch.setattr(preview_module, "run_command", lambda command, **kw: commands.append(command))
    dialog = GainCurveDialog()
    dialog.preview.set_input(bundle(), "L01")
    for value in range(10):
        dialog.move_node(4, 1, value)
        QTest.qWait(20)
    assert len(commands) == 1, "continuous moves must not indefinitely postpone the first update"
    commands[0].execute()
    assert dialog.preview._result is not None, "latest completed frame must remain visible during dragging"
    assert not dialog.ok_button.isEnabled()
    wait_until(lambda: len(commands) == 2)
    commands[1].execute()
    assert dialog.ok_button.isEnabled()
    np.testing.assert_array_equal(dialog.preview._result,
        evaluate_manual_gain_preview(dialog.preview._bundle.matrix, dialog.value()))
    dialog.reject()


def test_replacing_input_drops_a_frame_computed_for_the_old_source(qapp, monkeypatch):
    import ui.widgets.manual_gain_preview as preview_module
    commands = []
    monkeypatch.setattr(preview_module, "run_command", lambda command, **kw: commands.append(command))
    dialog = GainCurveDialog()
    dialog.preview.set_input(bundle(), "old")
    # This checks stale-result rejection, not a 130 ms scheduling deadline.
    # Wait for the real timer event; a loaded Windows CI runner can deliver it late.
    wait_until(lambda: len(commands) == 1)
    new = bundle(100)
    dialog.preview.set_input(new, "new")
    commands[0].execute()
    assert dialog.preview._result is None
    np.testing.assert_array_equal(dialog.preview.view._matrix, new.matrix)
    dialog.reject()


def test_cancel_edit_does_not_change_the_parameter_or_run_a_pending_preview(qapp, monkeypatch):
    import ui.widgets.manual_gain_preview as preview_module
    commands = []
    monkeypatch.setattr(preview_module, "run_command", lambda command, **kw: commands.append(command))
    editor = GainCurveEditor()
    before = editor.value()
    def setup(dialog):
        dialog.preview.set_input(bundle(), "L01")
        def cancel():
            dialog.move_node(2, 0.5, 24)
            dialog.reject()
        QTimer.singleShot(0, cancel)
    editor.previewRequested.connect(setup)
    editor.button.click()
    assert editor.value() == before and not commands


@pytest.fixture
def page(qapp):
    value = ProcessingPage()
    value.set_project_context("P1")
    value.set_line_label("L01")
    value.set_methods(build_method_dicts(method_catalog()))
    value.set_original_bundle(bundle())
    value._pipeline_list.add_step("dewow", "Dewow", {"window": 5})
    value._pipeline_list.add_step("manual_gain", "增益", {})
    value._pipeline_list.select_step(1)
    yield value
    value._close_gain_previews()
    value.close()
    value.deleteLater()


def test_gain_input_request_freezes_source_and_only_preceding_steps(page):
    requested, cancelled = [], []
    page.gain_preview_requested.connect(lambda *args: requested.append(args))
    page.gain_preview_cancelled.connect(cancelled.append)
    dialog = GainCurveDialog(parent=page)
    page._request_gain_preview(dialog)
    token, pid, line, artifact, prefix = requested[0]
    assert (pid, line, artifact) == ("P1", "L01", "")
    assert [step["method_id"] for step in prefix] == ["dewow"]
    page._pipeline_list.update_step_params(0, {"window": 9})
    assert not page._gain_preview_requests and cancelled == [token]
    page.set_gain_preview(token, None, "late failure")
    assert dialog.preview._bundle is None


def test_metadata_only_refresh_invalidates_a_prepared_gain_input(page, monkeypatch):
    import ui.widgets.manual_gain_preview as preview_module
    monkeypatch.setattr(preview_module, "run_command", lambda *a, **kw: None)
    tokens = []
    page.gain_preview_requested.connect(lambda token, *_: tokens.append(token))
    dialog = GainCurveDialog(parent=page)
    page._request_gain_preview(dialog)
    page.set_gain_preview(tokens[0], {"bundle": bundle(), "prefix_count": 1}, "")
    page.refresh_line_previews("L01")
    assert not page._gain_preview_requests
    assert dialog.preview._cancelled.is_set()


def test_gain_library_preview_uses_append_position_and_rejects_old_selection(page):
    requested, cancelled = [], []
    page.gain_preview_requested.connect(lambda *args: requested.append(args))
    page.gain_preview_cancelled.connect(cancelled.append)
    page._open_method_picker()
    page._method_browser.select_method("manual_gain")
    dialog = GainCurveDialog(parent=page)
    page._request_gain_preview(dialog)
    token, _, _, _, prefix = requested[0]
    assert [step["method_id"] for step in prefix] == ["dewow", "manual_gain"]
    page._method_browser.select_method("dewow")
    assert cancelled == [token]
    assert not page._gain_preview_requests
