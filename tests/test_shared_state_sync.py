#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""共享状态与工作台同步回归测试。"""

from __future__ import annotations

import os

import numpy as np
import pytest
from typing import cast

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

from app_qt import GPRGuiQt
from core.methods_registry import PROCESSING_METHODS
from core.shared_data_state import SharedDataState
from ui.gui_workbench import WorkbenchPage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app() -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])


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


def test_workbench_history_input_uses_matching_metadata(tmp_path):
    os.environ["LOCALAPPDATA"] = str(tmp_path / "localappdata")
    app = _get_app()
    state = SharedDataState()
    workbench = WorkbenchPage(None, state)
    try:
        raw = np.zeros((2, 4), dtype=np.float32)
        raw_metadata = {
            "trace_distance_m": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        }
        raw_header = {"a_scan_length": 2, "num_traces": 4}
        current = np.ones((2, 3), dtype=np.float32)
        current_metadata = {
            "trace_distance_m": np.array([0.0, 1.5, 3.0], dtype=np.float32),
        }
        current_header = {"a_scan_length": 2, "num_traces": 3}

        state.load_data(
            raw,
            path="demo.csv",
            header_info=raw_header,
            trace_metadata=raw_metadata,
            source="test",
        )
        state.apply_current_data(
            current,
            push_history=True,
            label="resampled",
            header_info=current_header,
            trace_metadata=current_metadata,
        )
        workbench.sync_from_shared_state({"reason": "current_updated"})

        workbench.selected_history_index = 0
        history_data, _ = workbench.resolve_input_data("history")
        history_header = workbench.resolve_input_header_info("history")
        history_metadata = workbench.resolve_input_trace_metadata("history")
        assert history_data is not None
        assert history_header is not None
        assert history_metadata is not None
        assert history_data.shape == (2, 4)
        assert history_header["num_traces"] == 4
        assert len(history_metadata["trace_distance_m"]) == 4

        workbench.selected_history_index = 1
        current_history_data, _ = workbench.resolve_input_data("history")
        current_history_metadata = workbench.resolve_input_trace_metadata("history")
        assert current_history_data is not None
        assert current_history_metadata is not None
        assert current_history_data.shape == (2, 3)
        assert len(current_history_metadata["trace_distance_m"]) == 3
    finally:
        workbench.close()
        app.processEvents()


def test_shared_state_preserves_explicit_replacement_metadata_when_trace_count_changes():
    state = SharedDataState()
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)
    metadata = {
        "trace_index": np.array([0, 1, 2, 3], dtype=np.int32),
        "trace_distance_m": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    }

    state.load_data(raw, path="demo.csv", trace_metadata=metadata)

    resampled = np.arange(9, dtype=np.float32).reshape(3, 3)
    resampled_metadata = {
        "trace_index": np.array([0, 1, 2], dtype=np.int32),
        "trace_distance_m": np.array([0.0, 1.5, 3.0], dtype=np.float32),
        "alignment_status": np.array(["resampled", "resampled", "resampled"], dtype="<U16"),
    }

    state.apply_current_data(
        resampled,
        trace_metadata=resampled_metadata,
        label="resampled",
    )

    assert state.current_trace_metadata is not None
    current_metadata = cast(dict[str, np.ndarray], state.current_trace_metadata)
    assert np.array_equal(current_metadata["trace_index"], np.array([0, 1, 2], dtype=np.int32))
    assert np.array_equal(current_metadata["trace_distance_m"], np.array([0.0, 1.5, 3.0], dtype=np.float32))
    assert set(current_metadata["alignment_status"].tolist()) == {"resampled"}


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
        assert workbench.current_result is not None
        assert np.array_equal(workbench.current_result, formal_result)

        preview_result = formal_result * 2
        workbench.update_current_result(preview_result, result_name="流程模板预览")

        assert workbench.preview_data is not None
        assert state.current_data is not None
        assert np.array_equal(state.current_data, formal_result)
        assert [label for label, _ in workbench.all_results] == ["原始数据", "dewow"]

        workbench._undo()
        assert workbench.preview_data is None
        assert state.current_data is not None
        assert np.array_equal(state.current_data, formal_result)
    finally:
        workbench.close()
        app.processEvents()


def test_workbench_view_selector_lists_steps_and_resolves_selected_data():
    app = _get_app()
    state = SharedDataState()
    workbench = WorkbenchPage(None, state)
    try:
        raw = np.arange(20, dtype=np.float32).reshape(4, 5)
        step_one = raw + 1
        step_two = raw + 2

        state.load_data(raw, path="demo.csv", source="test")
        state.apply_current_data(step_one, push_history=True, label="dewow")
        state.apply_current_data(step_two, push_history=True, label="hankel_svd")
        workbench.sync_from_shared_state({"reason": "current_updated"})

        entries = workbench._build_view_entries()
        labels = [entry["label"] for entry in entries]

        assert labels == ["原始数据", "步骤1: dewow", "当前结果: hankel_svd"]
        assert np.array_equal(entries[0]["data"], raw)
        assert np.array_equal(entries[1]["data"], step_one)
        assert np.array_equal(entries[2]["data"], step_two)

        workbench._update_view_combo()
        assert workbench.view_combo.currentText() == "当前结果: hankel_svd"

        workbench.view_combo.setCurrentIndex(1)
        selected = workbench._get_selected_view_entry()
        assert selected is not None
        assert selected["label"] == "步骤1: dewow"
        assert np.array_equal(selected["data"], step_one)
    finally:
        workbench.close()
        app.processEvents()


def test_workbench_compare_combo_uses_same_step_entries():
    app = _get_app()
    state = SharedDataState()
    workbench = WorkbenchPage(None, state)
    try:
        raw = np.arange(20, dtype=np.float32).reshape(4, 5)
        step_one = raw + 1
        step_two = raw + 2

        state.load_data(raw, path="demo.csv", source="test")
        state.apply_current_data(step_one, push_history=True, label="dewow")
        state.apply_current_data(step_two, push_history=True, label="hankel_svd")
        workbench.sync_from_shared_state({"reason": "current_updated"})

        workbench._update_compare_combo()
        combo_labels = [
            workbench.compare_combo.itemText(index)
            for index in range(workbench.compare_combo.count())
        ]

        assert combo_labels == ["原始数据", "步骤1: dewow", "当前结果: hankel_svd"]
        workbench.compare_combo.setCurrentIndex(1)
        assert workbench._get_compare_label() == "步骤1: dewow"
        compare_data = workbench._get_compare_data()
        assert compare_data is not None
        assert np.array_equal(compare_data, step_one)
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


def test_workbench_save_result_preserves_motion_trace_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(12, dtype=np.float32).reshape(3, 4)
        header_info = {"total_time_ns": 120.0, "num_traces": 4, "a_scan_length": 3}
        trace_metadata = {
            "trace_index": np.array([0, 1, 2, 3], dtype=np.int32),
            "trace_distance_m": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        }
        resampled = np.arange(9, dtype=np.float32).reshape(3, 3)
        resampled_metadata = {
            "trace_index": np.array([0, 1, 2], dtype=np.int32),
            "trace_distance_m": np.array([0.0, 1.5, 3.0], dtype=np.float32),
        }

        def motion_stage(data, trace_metadata=None, **kwargs):
            assert trace_metadata is not None
            return resampled, {"trace_metadata_out": resampled_metadata}

        monkeypatch.setitem(
            PROCESSING_METHODS,
            "test_workbench_motion_commit",
            {
                "name": "test_workbench_motion_commit",
                "type": "local",
                "func": motion_stage,
                "params": [],
                "auto_tune_family": "motion_comp",
            },
        )

        win.shared_data.load_data(
            raw,
            path="demo.csv",
            header_info=header_info,
            trace_metadata=trace_metadata,
            source="test",
        )
        execution = win._apply_single_method(
            raw,
            "test_workbench_motion_commit",
            {},
            header_info=header_info,
            trace_metadata=trace_metadata,
        )
        win.page_workbench.set_preview_result(
            execution["preview_data"],
            title="预览: test_workbench_motion_commit",
            header_info=execution["preview_header_info"],
            trace_metadata=execution["preview_trace_metadata"],
            commit_data=execution["result_data"],
            commit_header_info=execution["result_header_info"],
            commit_trace_metadata=execution["result_trace_metadata"],
        )

        win._on_workbench_save_result()

        current_metadata = cast(dict[str, np.ndarray], win.shared_data.current_trace_metadata)
        assert win.shared_data.current_data is not None
        assert win.shared_data.current_data.shape == (3, 3)
        assert np.array_equal(current_metadata["trace_index"], np.array([0, 1, 2], dtype=np.int32))
        assert np.array_equal(
            current_metadata["trace_distance_m"],
            np.array([0.0, 1.5, 3.0], dtype=np.float32),
        )
    finally:
        win.close()
        app.processEvents()
