#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workbench no-prior guard callback tests."""

from __future__ import annotations

import numpy as np

from ui.gui_workbench import WorkbenchPage


def _make_page_stub() -> WorkbenchPage:
    page = WorkbenchPage.__new__(WorkbenchPage)
    page._log = lambda *args, **kwargs: None
    return page


def test_guard_callback_blocks_action_when_false():
    page = _make_page_stub()
    page.no_prior_guard_callback = lambda _action_id: False
    assert page._guard_workbench_action("workflow_run", "guard") is False


def test_guard_callback_allows_action_when_true():
    page = _make_page_stub()
    page.no_prior_guard_callback = lambda _action_id: True
    assert page._guard_workbench_action("workflow_run", "guard") is True


def test_template_execution_returns_early_when_guard_blocks():
    class _WorkflowManager:
        def __init__(self) -> None:
            self.template_queried = False

        def get_template(self, _name: str):
            self.template_queried = True
            return {"name": "blocked"}

    page = _make_page_stub()
    page.raw_data = np.zeros((4, 4), dtype=np.float32)
    page.no_prior_guard_callback = lambda _action_id: False
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
    page.no_prior_guard_callback = lambda _action_id: True
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
