#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retired Workbench compatibility shim.

The legacy Workbench UI was removed from the active MyGPR interface in
GX-UI-016.  This module intentionally keeps only the tiny compatibility surface
used by older tests/docs and migration checks.  It must not be instantiated as
an active page from ``app_qt.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PyQt6.QtWidgets import QWidget


@dataclass
class WorkflowMethod:
    """Minimal placeholder retained for monkeypatched legacy tests."""

    payload: dict

    @staticmethod
    def from_dict(payload: dict):
        return WorkflowMethod(payload)


class WorkflowExecutor:
    """Minimal placeholder retained for monkeypatched legacy tests."""

    def __init__(self, **_kwargs):
        self.current_header_info = None
        self.current_trace_metadata = None

    def execute_all(self, current_data, _workflow_methods):
        return np.asarray(current_data, dtype=np.float32)


def format_workbench_wiggle_sampling_notice(
    *,
    n_traces: int,
    shown_traces: int,
    n_samples: int,
) -> str:
    return (
        f"Wiggle 显示抽样：{shown_traces}/{n_traces} 道，{n_samples} 个采样点。"
        "该抽样仅用于显示，不改变数据。"
    )


def classify_workbench_method_action(
    method_key: str,
    params: dict | None,
    n_traces: int | None = None,
) -> str | None:
    params = params or {}
    if method_key == "agcGain":
        return "AGC_display_only"
    if method_key in {"energy_decay_gain", "sec_gain", "compensatingGain", "amplitude_scale"}:
        return "conservative_energy_decay_gain_display"
    if method_key in {"subtracting_average_2D", "running_average_2D", "svd_background"}:
        n = params.get("ntraces", params.get("window", 0)) or 0
        try:
            n = int(n)
        except Exception:
            n = 0
        if n_traces and n >= max(1, int(n_traces) // 2):
            return "background_suppression_aggressive"
        if n >= 50:
            return "background_suppression_aggressive"
        return "background_suppression_conservative"
    return None


class WorkbenchPage(QWidget):
    """Retired compatibility object, not an active UI page."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.no_prior_guard_callback = None
        self.raw_data = None
        self.preview_data = None
        self.preview_info = None
        self._pending_preview_request = None
        self._preview_running = False
        self._preview_worker = None
        self._apply_after_preview = False
        self._preview_seq = 0
        self.selected_history_index = 0

    def _log(self, *_args, **_kwargs) -> None:
        return None

    def sync_from_shared_state(self, *_args, **_kwargs) -> None:
        return None

    def _guard_workbench_action(self, action_id: str, _context: str = "") -> bool:
        callback = getattr(self, "no_prior_guard_callback", None)
        if callback is None:
            return True
        try:
            return bool(callback(action_id, show_dialog=True, allow_override=True))
        except TypeError:
            return bool(callback(action_id))

    def _build_request_context(self, method_id: str, params: dict, source: str) -> dict:
        return {"method_id": method_id, "params": params, "source": source}

    def _start_pending_preview_request(self) -> None:
        return None

    def _request_preview(
        self,
        *,
        method_id: str,
        params: dict,
        input_data,
        source_text: str,
        title: str,
        method_name: str,
        announce: bool = True,
        **_kwargs,
    ) -> None:
        action = classify_workbench_method_action(
            method_id,
            params,
            getattr(input_data, "shape", [None, None])[1] if input_data is not None else None,
        )
        if action and not self._guard_workbench_action(action, "preview"):
            self._pending_preview_request = None
            return
        self._pending_preview_request = self._build_request_context(method_id, params, source_text)
        self._start_pending_preview_request()

    def _on_method_selected(self, method_id: str) -> None:
        editor = getattr(self, "param_editor", None)
        if editor is not None and hasattr(editor, "load_method"):
            editor.load_method(method_id)

    def _on_template_execute(self, template_name: str) -> None:
        if not self._guard_workbench_action("workflow_run", "template"):
            return
        manager = getattr(self, "workflow_manager", None)
        if manager is None:
            return
        template = manager.get_template(template_name)
        methods = manager.get_template_methods(template_name) if hasattr(manager, "get_template_methods") else []
        workflow_methods = [WorkflowMethod.from_dict(m) for m in methods]
        editor = getattr(self, "param_editor", None)
        source = editor.get_input_source() if editor is not None and hasattr(editor, "get_input_source") else "raw"
        data, _source_text = self.resolve_input_data(source) if hasattr(self, "resolve_input_data") else (self.raw_data, "raw")
        if data is None:
            return
        executor = WorkflowExecutor(
            header_info=self.resolve_input_header_info(source) if hasattr(self, "resolve_input_header_info") else None,
            trace_metadata=self.resolve_input_trace_metadata(source) if hasattr(self, "resolve_input_trace_metadata") else None,
        )
        result = executor.execute_all(data, workflow_methods)
        if hasattr(self, "update_current_result"):
            self.update_current_result(
                result,
                header_info=getattr(executor, "current_header_info", None),
                trace_metadata=getattr(executor, "current_trace_metadata", None),
            )
