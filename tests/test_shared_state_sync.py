#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared state and workflow synchronization regression tests."""

from __future__ import annotations

import os
from typing import cast

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from app_qt import GPRGuiQt
from core.methods_registry import PROCESSING_METHODS
from core.shared_data_state import SharedDataState

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app() -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])


def _advanced(win: GPRGuiQt):
    return win._ensure_advanced_page()


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
        "alignment_status": np.array(
            ["resampled", "resampled", "resampled"], dtype="<U16"
        ),
    }

    state.apply_current_data(
        resampled,
        trace_metadata=resampled_metadata,
        label="resampled",
    )

    assert state.current_trace_metadata is not None
    current_metadata = cast(dict[str, np.ndarray], state.current_trace_metadata)
    assert np.array_equal(
        current_metadata["trace_index"], np.array([0, 1, 2], dtype=np.int32)
    )
    assert np.array_equal(
        current_metadata["trace_distance_m"],
        np.array([0.0, 1.5, 3.0], dtype=np.float32),
    )
    assert set(current_metadata["alignment_status"].tolist()) == {"resampled"}


def test_main_single_view_combo_selects_formal_snapshot():
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(20, dtype=np.float32).reshape(4, 5)
        step_one = raw + 1
        step_two = raw + 2

        win.shared_data.load_data(raw, path="demo.csv", source="test")
        win.shared_data.apply_current_data(step_one, push_history=True, label="dewow")
        win.shared_data.apply_current_data(
            step_two, push_history=True, label="hankel_svd"
        )
        app.processEvents()

        combo_labels = [
            _advanced(win).single_view_combo.itemText(index)
            for index in range(_advanced(win).single_view_combo.count())
        ]
        assert combo_labels == ["原始", "dewow", "当前"]
        assert _advanced(win).single_view_combo.currentText() == "当前"

        _advanced(win).single_view_combo.setCurrentText("dewow")
        selected_data, _, _ = win._get_active_plot_payload(win.data)
        assert selected_data is not None
        assert np.array_equal(selected_data, step_one)

        _advanced(win).mode_compare.setChecked(True)
        compare_data, _, _ = win._get_active_plot_payload(win.data)
        assert compare_data is not None
        assert np.array_equal(compare_data, step_two)
    finally:
        win.close()
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


def test_workflow_save_result_preserves_motion_trace_metadata(
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
            "test_workflow_motion_commit",
            {
                "name": "test_workflow_motion_commit",
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
            "test_workflow_motion_commit",
            {},
            header_info=header_info,
            trace_metadata=trace_metadata,
        )
        win._last_workflow_result = {
            "outputs": [
                {
                    "method_key": "test_workflow_motion_commit",
                    "method_name": "test_workflow_motion_commit",
                    "params": {},
                }
            ],
            "final_data": execution["result_data"],
            "final_header_info": execution["result_header_info"],
            "final_trace_metadata": execution["result_trace_metadata"],
        }
        win._last_workflow_realtime = True
        win._workflow_preview_base_state = {
            "data": raw,
            "header_info": header_info,
            "trace_metadata": trace_metadata,
            "label": "原始数据",
        }

        win.save_workflow_live_result()

        current_metadata = cast(dict[str, np.ndarray], win.shared_data.current_trace_metadata)
        assert win.shared_data.current_data is not None
        assert win.shared_data.current_data.shape == (3, 3)
        assert np.array_equal(
            current_metadata["trace_index"], np.array([0, 1, 2], dtype=np.int32)
        )
        assert np.array_equal(
            current_metadata["trace_distance_m"],
            np.array([0.0, 1.5, 3.0], dtype=np.float32),
        )
    finally:
        win.close()
        app.processEvents()
