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


def test_shared_state_load_data_keeps_independent_current_original_and_input():
    state = SharedDataState()
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)

    state.load_data(raw, path="demo.csv")
    raw[0, 0] = 999.0
    assert state.current_data is not None
    assert state.original_data is not None
    state.current_data[0, 1] = 888.0

    assert float(state.current_data[0, 0]) != 999.0
    assert float(state.original_data[0, 0]) != 999.0
    assert float(state.original_data[0, 1]) != 888.0


def test_shared_state_header_info_clone_deep_copies_ground_truth_metadata():
    state = SharedDataState()
    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    header_info = {
        "ground_truth": {
            "scenario_id": "pipe_demo",
            "targets": [{"roi": {"time_start_idx": 1}}],
        }
    }

    state.load_data(raw, path="demo.out", header_info=header_info)
    header_info["ground_truth"]["targets"][0]["roi"]["time_start_idx"] = 99

    assert state.header_info is not None
    assert state.original_header_info is not None
    assert state.header_info["ground_truth"]["targets"][0]["roi"]["time_start_idx"] == 1
    assert (
        state.original_header_info["ground_truth"]["targets"][0]["roi"]["time_start_idx"]
        == 1
    )


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


def test_shared_state_history_entries_keep_matching_metadata(tmp_path):
    os.environ["LOCALAPPDATA"] = str(tmp_path / "localappdata")
    state = SharedDataState()
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

    entries = state.build_result_history_entries()
    assert [entry["label"] for entry in entries] == ["原始数据", "resampled"]
    assert entries[0]["data"].shape == (2, 4)
    assert entries[0]["header_info"]["num_traces"] == 4
    assert len(entries[0]["trace_metadata"]["trace_distance_m"]) == 4
    assert entries[1]["data"].shape == (2, 3)
    assert entries[1]["header_info"]["num_traces"] == 3
    assert len(entries[1]["trace_metadata"]["trace_distance_m"]) == 3


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


def test_shared_state_formal_history_excludes_transient_preview_results():
    state = SharedDataState()
    raw = np.arange(20, dtype=np.float32).reshape(4, 5)
    state.load_data(raw, path="demo.csv")

    assert [label for label, _ in state.build_result_history()] == ["原始数据"]

    formal_result = raw + 10
    state.apply_current_data(formal_result, push_history=True, label="dewow")

    labels = [label for label, _ in state.build_result_history()]
    assert labels == ["原始数据", "dewow"]
    assert state.current_data is not None
    assert np.array_equal(state.current_data, formal_result)


def test_shared_state_formal_entries_list_steps_and_data():
    state = SharedDataState()
    raw = np.arange(20, dtype=np.float32).reshape(4, 5)
    step_one = raw + 1
    step_two = raw + 2

    state.load_data(raw, path="demo.csv", source="test")
    state.apply_current_data(step_one, push_history=True, label="dewow")
    state.apply_current_data(step_two, push_history=True, label="hankel_svd")

    entries = state.build_result_history_entries()
    labels = [entry["label"] for entry in entries]

    assert labels == ["原始数据", "dewow", "hankel_svd"]
    assert np.array_equal(entries[0]["data"], raw)
    assert np.array_equal(entries[1]["data"], step_one)
    assert np.array_equal(entries[2]["data"], step_two)


def test_shared_state_compare_snapshots_use_same_step_entries():
    state = SharedDataState()
    raw = np.arange(20, dtype=np.float32).reshape(4, 5)
    step_one = raw + 1
    step_two = raw + 2

    state.load_data(raw, path="demo.csv", source="test")
    state.apply_current_data(step_one, push_history=True, label="dewow")
    state.apply_current_data(step_two, push_history=True, label="hankel_svd")

    snapshots = state.build_formal_compare_snapshots()
    labels = [entry["label"] for entry in snapshots]

    assert labels == ["原始", "dewow", "当前"]
    assert np.array_equal(snapshots[1]["data"], step_one)


def test_main_single_view_combo_selects_formal_snapshot():
    app = _get_app()
    win = GPRGuiQt()
    try:
        raw = np.arange(20, dtype=np.float32).reshape(4, 5)
        step_one = raw + 1
        step_two = raw + 2

        win.shared_data.load_data(raw, path="demo.csv", source="test")
        win.shared_data.apply_current_data(step_one, push_history=True, label="dewow")
        win.shared_data.apply_current_data(step_two, push_history=True, label="hankel_svd")
        app.processEvents()

        combo_labels = [
            win.page_advanced.single_view_combo.itemText(index)
            for index in range(win.page_advanced.single_view_combo.count())
        ]
        assert combo_labels == ["原始", "dewow", "当前"]
        assert win.page_advanced.single_view_combo.currentText() == "当前"

        win.page_advanced.single_view_combo.setCurrentText("dewow")
        selected_data, _, _ = win._get_active_plot_payload(win.data)
        assert selected_data is not None
        assert np.array_equal(selected_data, step_one)

        win.page_advanced.mode_compare.setChecked(True)
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

        win._on_workbench_save_result()

        current_metadata = cast(dict[str, np.ndarray], win.shared_data.current_trace_metadata)
        assert win.shared_data.current_data is not None
        assert getattr(win, "page_workbench", None) is None
        assert execution["result_data"].shape == (3, 3)
        assert np.array_equal(
            execution["result_trace_metadata"]["trace_index"],
            np.array([0, 1, 2], dtype=np.int32),
        )
        assert np.array_equal(
            execution["result_trace_metadata"]["trace_distance_m"],
            np.array([0.0, 1.5, 3.0], dtype=np.float32),
        )
        assert win.shared_data.current_data.shape == raw.shape
        assert np.array_equal(current_metadata["trace_index"], trace_metadata["trace_index"])
    finally:
        win.close()
        app.processEvents()


def test_shared_state_skips_oversized_history_snapshot_for_memory_safety():
    state = SharedDataState()
    state.max_history_snapshot_bytes = 64
    raw = np.ones((16, 16), dtype=np.float64)

    state.load_data(raw, path="large.csv")
    state.push_history(label="oversized_step")

    assert state.can_undo() is False
    assert len(state.history) == 0
    summary = state.get_history_memory_summary()
    assert summary["stored_count"] == 0
    assert summary["pruned_count"] == 1
    assert summary["pruned_summaries"][0]["label"] == "oversized_step"
    assert summary["pruned_summaries"][0]["reason"] == "snapshot_exceeds_limit"


def test_shared_state_trims_history_by_memory_budget():
    state = SharedDataState()
    state.max_history = 10
    state.max_history_snapshot_bytes = 10_000_000
    state.max_history_bytes = 2_000
    raw = np.arange(100, dtype=np.float64).reshape(10, 10)

    state.load_data(raw, path="demo.csv")
    for idx in range(5):
        state.apply_current_data(raw + idx + 1, push_history=True, label=f"step_{idx + 1}")

    summary = state.get_history_memory_summary()
    assert summary["stored_bytes"] <= state.max_history_bytes
    assert summary["stored_count"] <= 2
    assert summary["pruned_count"] >= 1
    assert all(entry.get("data") is not None for entry in state.history)
