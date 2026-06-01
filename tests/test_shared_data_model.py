#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless tests for the pure shared data model."""

from __future__ import annotations

import importlib
import sys

import numpy as np

from core.shared_data_model import SharedDataModel


def test_shared_data_model_import_has_no_pyqt_side_effect():
    sys.modules.pop("core.shared_data_model", None)
    before = set(sys.modules)
    module = importlib.import_module("core.shared_data_model")
    added = set(sys.modules) - before

    assert hasattr(module, "SharedDataModel")
    assert "PyQt6" not in added
    assert not any(name.startswith("PyQt6.") for name in added)


def test_shared_data_model_tracks_history_without_qt():
    state = SharedDataModel()
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)

    state.load_data(raw, path="demo.csv", source="unit")
    state.apply_current_data(raw + 1, push_history=True, label="dewow")
    state.apply_current_data(raw + 2, push_history=True, label="trace_median")

    labels = [label for label, _ in state.build_result_history()]
    assert labels == ["原始数据", "dewow", "trace_median"]
    assert state.last_change_event is not None
    assert state.last_change_event["reason"] == "current_updated"


def test_shared_data_model_python_listener_receives_events():
    state = SharedDataModel()
    events: list[dict] = []
    state.add_change_listener(events.append)

    raw = np.ones((2, 3), dtype=np.float32)
    state.load_data(raw, path="demo.csv", source="listener")
    state.set_metadata(path="renamed.csv", emit=True)

    assert [event["reason"] for event in events] == ["loaded", "metadata"]
    assert events[0]["source"] == "listener"
    state.remove_change_listener(events.append)


def test_shared_data_model_history_memory_prunes_without_qt():
    state = SharedDataModel()
    state.max_history = 10
    state.max_history_bytes = 128
    raw = np.arange(100, dtype=np.float32).reshape(10, 10)

    state.load_data(raw)
    for idx in range(3):
        state.apply_current_data(raw + idx + 1, push_history=True, label=f"step_{idx}")

    summary = state.get_history_memory_summary()
    assert summary["schema"] == "mygpr.history_memory.v1"
    assert summary["pruned_count"] >= 1
    assert summary["stored_bytes"] <= state.max_history_bytes or summary["stored_count"] == 0
