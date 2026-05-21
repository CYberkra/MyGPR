#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workbench no-prior guard callback tests."""

from __future__ import annotations

import numpy as np

from ui.gui_workbench import WorkbenchPage, classify_workbench_method_action


def _make_page_stub() -> WorkbenchPage:
    page = WorkbenchPage.__new__(WorkbenchPage)
    page._log = lambda *args, **kwargs: None
    page._preview_seq = 0
    page._pending_preview_request = None
    page._preview_running = False
    page._preview_worker = None
    page._apply_after_preview = False
    page.preview_data = None
    page.preview_request_context = None
    page.selected_history_index = 0
    page.resolve_input_header_info = lambda _source: None
    page.resolve_input_trace_metadata = lambda _source: None
    page._build_request_context = lambda _method_id, _params, _source: {
        "method_id": _method_id,
        "source": _source,
    }
    page._update_action_buttons = lambda: None
    page.preview_info = type(
        "_PreviewInfo", (), {"setText": staticmethod(lambda _text: None)}
    )()
    return page


def test_guard_callback_blocks_action_when_false():
    page = _make_page_stub()
    page.no_prior_guard_callback = lambda _action_id: False
    assert page._guard_workbench_action("workflow_run", "guard") is False


def test_guard_callback_allows_action_when_true():
    page = _make_page_stub()
    page.no_prior_guard_callback = lambda _action_id: True
    assert page._guard_workbench_action("workflow_run", "guard") is True


def test_method_action_mapping_for_display_and_background():
    assert classify_workbench_method_action("agcGain", {}, 120) == "AGC_display_only"
    assert (
        classify_workbench_method_action("energy_decay_gain", {}, 120)
        == "conservative_energy_decay_gain_display"
    )
    assert (
        classify_workbench_method_action("subtracting_average_2D", {"ntraces": 70}, 120)
        == "background_suppression_aggressive"
    )
    assert (
        classify_workbench_method_action("subtracting_average_2D", {"ntraces": 9}, 120)
        == "background_suppression_conservative"
    )


def test_request_preview_blocks_single_method_when_guard_callback_denies():
    page = _make_page_stub()
    page.no_prior_guard_callback = (
        lambda _action_id, **_kwargs: False
    )
    start_calls = {"count": 0}
    page._start_pending_preview_request = lambda: start_calls.__setitem__(
        "count", start_calls["count"] + 1
    )

    page._request_preview(
        method_id="agcGain",
        params={},
        input_data=np.zeros((8, 16), dtype=np.float32),
        source_text="raw",
        title="preview",
        method_name="agc",
        announce=True,
    )

    assert start_calls["count"] == 0
    assert page._pending_preview_request is None


def test_request_preview_allows_single_method_when_guard_callback_allows():
    page = _make_page_stub()
    page.no_prior_guard_callback = (
        lambda _action_id, **_kwargs: True
    )
    start_calls = {"count": 0}
    page._start_pending_preview_request = lambda: start_calls.__setitem__(
        "count", start_calls["count"] + 1
    )

    page._request_preview(
        method_id="agcGain",
        params={},
        input_data=np.zeros((8, 16), dtype=np.float32),
        source_text="raw",
        title="preview",
        method_name="agc",
        announce=True,
    )

    assert start_calls["count"] == 1
    assert page._pending_preview_request is not None


def test_pure_selection_without_data_does_not_invoke_guard_callback():
    class _ParamEditor:
        def __init__(self):
            self.current_method_id = ""

        def load_method(self, method_id: str):
            self.current_method_id = method_id

    page = _make_page_stub()
    calls = {"count": 0}

    def _callback(_action_id: str, **_kwargs):
        calls["count"] += 1
        return True

    page.no_prior_guard_callback = _callback
    page.param_editor = _ParamEditor()
    page.raw_data = None
    page._on_method_selected("agcGain")

    assert calls["count"] == 0


def test_template_execution_returns_early_when_guard_blocks():
    class _WorkflowManager:
        def __init__(self) -> None:
            self.template_queried = False

        def get_template(self, _name: str):
            self.template_queried = True
            return {"name": "blocked"}

    page = _make_page_stub()
    page.raw_data = np.zeros((4, 4), dtype=np.float32)
    page.no_prior_guard_callback = lambda _action_id, **_kwargs: False
    page.workflow_manager = _WorkflowManager()
    page._on_template_execute("blocked_template")

    assert page.workflow_manager.template_queried is False


def test_template_execution_runs_when_guard_allows(monkeypatch):
    class _WorkflowManager:
        def get_template(self, _name: str):
            return {"name": "allowed"}

        def get_template_methods(self, _name: str):
            return [{"method_id": "fake_method", "params": {}}]

    class _FakeWorkflowMethod:
        @staticmethod
        def from_dict(payload: dict):
            return payload

    class _FakeExecutor:
        def __init__(self, **_kwargs):
            self.current_header_info = {"from": "executor"}
            self.current_trace_metadata = {"from": "executor"}

        def execute_all(self, current_data, _workflow_methods):
            return np.asarray(current_data, dtype=np.float32) + 1.0

    class _ParamEditor:
        @staticmethod
        def get_input_source() -> str:
            return "raw"

    page = _make_page_stub()
    page.raw_data = np.zeros((4, 4), dtype=np.float32)
    page.no_prior_guard_callback = lambda _action_id, **_kwargs: True
    page.workflow_manager = _WorkflowManager()
    page.param_editor = _ParamEditor()
    page.resolve_input_header_info = lambda _source: {"header": 1}
    page.resolve_input_trace_metadata = lambda _source: {"trace": 1}
    page.resolve_input_data = lambda _source: (np.zeros((4, 4), dtype=np.float32), "raw")

    capture = {}

    def _update_current_result(result, header_info=None, trace_metadata=None):
        capture["result"] = result
        capture["header_info"] = header_info
        capture["trace_metadata"] = trace_metadata

    page.update_current_result = _update_current_result

    import ui.gui_workbench as workbench_module

    monkeypatch.setattr(workbench_module, "WorkflowMethod", _FakeWorkflowMethod)
    monkeypatch.setattr(workbench_module, "WorkflowExecutor", _FakeExecutor)

    page._on_template_execute("allowed_template")

    assert "result" in capture
    assert capture["result"].shape == (4, 4)
    assert capture["header_info"] == {"from": "executor"}
    assert capture["trace_metadata"] == {"from": "executor"}
