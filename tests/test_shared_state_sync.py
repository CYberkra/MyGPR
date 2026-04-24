#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""共享状态与工作台同步回归测试。"""

from __future__ import annotations

import os

import numpy as np
from PyQt6.QtWidgets import QApplication

from app_qt import GPRGuiQt
from core.shared_data_state import SharedDataState
from ui.gui_workbench import WorkbenchPage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_shared_state_build_result_history_tracks_formal_labels():
    state = SharedDataState()
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)

    state.load_data(raw, path="demo.csv")
    state.apply_current_data(raw + 1, push_history=True, label="dewow")
    state.apply_current_data(raw + 2, push_history=True, label="fk_filter")

    history_items = state.build_result_history()
    labels = [label for label, _ in history_items]

    assert labels == ["原始数据", "dewow", "fk_filter"]
    assert np.array_equal(history_items[0][1], raw)
    assert np.array_equal(history_items[1][1], raw + 1)
    assert np.array_equal(history_items[2][1], raw + 2)


def test_shared_state_trims_history_internally():
    state = SharedDataState()
    state.max_history = 3
    raw = np.arange(6, dtype=np.float32).reshape(2, 3)

    state.load_data(raw, path="demo.csv")
    for idx in range(5):
        state.apply_current_data(
            raw + idx + 1, push_history=True, label=f"step_{idx + 1}"
        )

    assert state.can_undo() is True
    assert len(state.history) == 3
    assert [item["label"] for item in state.history] == [
        "step_2",
        "step_3",
        "step_4",
    ]


def test_shared_state_builds_formal_compare_snapshots():
    state = SharedDataState()
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)

    state.load_data(raw, path="demo.csv")
    state.apply_current_data(raw + 1, push_history=True, label="dewow")
    state.apply_current_data(raw + 2, push_history=True, label="fk_filter")

    snapshots = state.build_formal_compare_snapshots()
    labels = [item["label"] for item in snapshots]

    assert labels == ["原始", "dewow", "当前"]
    assert np.array_equal(snapshots[0]["data"], raw)
    assert np.array_equal(snapshots[1]["data"], raw + 1)
    assert np.array_equal(snapshots[2]["data"], raw + 2)


def test_workbench_sync_uses_shared_formal_history_only():
    app = _get_app()
    state = SharedDataState()
    workbench = WorkbenchPage(None, state)
    try:
        raw = np.arange(20, dtype=np.float32).reshape(4, 5)
        state.load_data(raw, path="demo.csv")
        workbench.sync_from_shared_state({"reason": "loaded"})

        assert [label for label, _ in workbench.all_results] == ["原始数据"]

        formal_result = raw + 10
        state.apply_current_data(formal_result, push_history=True, label="dewow")
        workbench.sync_from_shared_state(
            {"reason": "current_updated", "label": "dewow"}
        )

        assert [label for label, _ in workbench.all_results] == ["原始数据", "dewow"]
        assert np.array_equal(workbench.current_result, formal_result)

        preview_result = formal_result * 2
        workbench.update_current_result(preview_result, result_name="流程模板预览")

        assert workbench.preview_data is not None
        assert np.array_equal(state.current_data, formal_result)
        assert [label for label, _ in workbench.all_results] == ["原始数据", "dewow"]

        workbench._undo()
        assert workbench.preview_data is None
        assert np.array_equal(state.current_data, formal_result)
    finally:
        workbench.close()
        app.processEvents()


def test_compare_snapshots_clear_transient_results_after_formal_update():
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(20, dtype=np.float32).reshape(4, 5)
        win.shared_data.load_data(raw, path="demo.csv", source="test")

        assert [snap["label"] for snap in win.compare_snapshots] == ["原始", "当前"]

        win._set_compare_snapshots(
            [
                {"label": "dewow", "data": raw * 0.1},
                {"label": "fk_filter", "data": raw * 0.2},
            ]
        )

        labels_with_transient = [snap["label"] for snap in win.compare_snapshots]
        assert labels_with_transient == ["原始", "当前", "dewow", "fk_filter"]

        win.shared_data.apply_current_data(raw + 1, push_history=True, label="dewow")

        labels_after_commit = [snap["label"] for snap in win.compare_snapshots]
        assert labels_after_commit == ["原始", "当前"]
    finally:
        win.close()
        app.processEvents()
