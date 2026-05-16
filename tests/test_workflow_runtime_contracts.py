#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workflow runtime contract tests."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication

from core.workflow_data import WorkflowMethod
from core.workflow_runtime_contracts import (
    WorkflowNodeOutput,
    WorkflowRunRequest,
    WorkflowRunResult,
)


def _get_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_workflow_run_request_preserves_signal_compatibility():
    methods = [
        WorkflowMethod("preprocessing", "dc_shift"),
        WorkflowMethod("gain", "sec_gain"),
    ]

    request = WorkflowRunRequest.from_signal_args(methods, realtime=True, run_mode="Realtime")

    assert request.methods == tuple(methods)
    assert request.realtime is True
    assert request.run_mode == "Realtime"
    assert request.as_signal_args() == (tuple(methods), True, "Realtime")


def test_workflow_run_result_can_normalize_worker_payload():
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    payload = {
        "outputs": [
            {
                "node_id": "node_001_dc_shift",
                "method_key": "dc_shift",
                "method_name": "DC 去偏",
                "data": data,
                "input_shape": [3, 4],
                "output_shape": [3, 4],
                "elapsed_ms": 2.5,
                "warnings": ["demo"],
            }
        ],
        "final_data": data,
        "final_header_info": {"Number of Samples": 3},
    }

    result = WorkflowRunResult.from_worker_payload(
        payload,
        realtime=False,
        run_mode="Run All",
    )

    assert result.realtime is False
    assert result.run_mode == "Run All"
    assert result.final_data is data
    assert result.final_header_info == {"Number of Samples": 3}
    assert result.outputs == (
        WorkflowNodeOutput(
            node_id="node_001_dc_shift",
            method_key="dc_shift",
            method_name="DC 去偏",
            data=data,
            input_shape=(3, 4),
            output_shape=(3, 4),
            elapsed_ms=2.5,
            warnings=("demo",),
        ),
    )


def test_app_pending_realtime_workflow_run_uses_request_contract(monkeypatch):
    from app_qt import GPRGuiQt

    app = _get_app()
    win = GPRGuiQt()

    class DummyWorker:
        def __init__(self):
            self.cancelled = False

        def request_cancel(self):
            self.cancelled = True

    try:
        win.shared_data.load_data(
            np.arange(24, dtype=np.float32).reshape(6, 4),
            path="demo.csv",
            source="test",
        )
        worker = DummyWorker()
        win._worker = worker
        method = WorkflowMethod(
            category="preprocessing",
            stage_id="trace_correction",
            method_id="dc_shift",
            params={"estimator": "mean", "scope": "per_trace"},
        )

        win.run_workflow_methods([method], realtime=True, run_mode="Realtime")

        assert worker.cancelled is True
        assert isinstance(win._pending_workflow_run, WorkflowRunRequest)
        assert win._pending_workflow_run.methods == (method,)
        assert win._pending_workflow_run.realtime is True
        assert win._pending_workflow_run.run_mode == "Realtime"
    finally:
        win._worker = None
        win.close()
        app.processEvents()
