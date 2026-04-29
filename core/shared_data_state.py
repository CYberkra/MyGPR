#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared application data state for main GUI and workbench."""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal


class SharedDataState(QObject):
    """Single source of truth for loaded/processed data."""

    changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_data: np.ndarray | None = None
        self.original_data: np.ndarray | None = None
        self.current_trace_metadata: dict[str, np.ndarray] | None = None
        self.original_trace_metadata: dict[str, np.ndarray] | None = None
        self.original_header_info: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self.data_path: str | None = None
        self.header_info: dict[str, Any] | None = None
        self.original_label = "原始数据"
        self.current_label = "原始数据"
        self.max_history = 10
        self.revision = 0

    def load_data(
        self,
        data: np.ndarray,
        path: str | None = None,
        header_info: dict[str, Any] | None = None,
        trace_metadata: dict[str, np.ndarray] | None = None,
        source: str = "load",
    ):
        array = np.array(data, copy=True)
        self.current_data = array.copy()
        self.original_data = array.copy()
        self.current_trace_metadata = _clone_trace_metadata(trace_metadata)
        self.original_trace_metadata = _clone_trace_metadata(trace_metadata)
        self.header_info = _clone_header_info(header_info)
        self.original_header_info = _clone_header_info(header_info)
        self.history = []
        self.data_path = path
        self.original_label = "原始数据"
        self.current_label = "原始数据"
        self.revision += 1
        self.changed.emit(
            {"reason": "loaded", "source": source, "revision": self.revision}
        )

    def push_history(self, label: str | None = None):
        if self.current_data is None:
            return
        self.history.append(
            {
                "data": np.array(self.current_data, copy=True),
                "trace_metadata": _clone_trace_metadata(self.current_trace_metadata),
                "header_info": _clone_header_info(self.header_info),
                "label": label or self.current_label or "当前结果",
            }
        )
        self._trim_history()

    def can_undo(self) -> bool:
        return bool(self.history)

    def apply_current_data(
        self,
        data: np.ndarray,
        *,
        push_history: bool = False,
        source: str = "main",
        label: str | None = None,
        trace_metadata: dict[str, np.ndarray] | None = None,
        header_info: dict[str, Any] | None = None,
    ):
        if push_history and self.current_data is not None:
            self.push_history()
        self.current_data = np.array(data, copy=True)
        self.current_label = label or self.current_label or "当前结果"
        if header_info is not None:
            self.header_info = _clone_header_info(header_info)
        if trace_metadata is not None:
            self.current_trace_metadata = _clone_trace_metadata(trace_metadata)
        elif self.current_trace_metadata is not None:
            current_traces = (
                self.current_data.shape[1] if self.current_data.ndim == 2 else None
            )
            meta_traces = (
                len(next(iter(self.current_trace_metadata.values())))
                if self.current_trace_metadata
                else None
            )
            if (
                current_traces is not None
                and meta_traces is not None
                and current_traces != meta_traces
            ):
                self.current_trace_metadata = None
        self.revision += 1
        self.changed.emit(
            {
                "reason": "current_updated",
                "source": source,
                "label": label,
                "revision": self.revision,
            }
        )

    def undo(self) -> bool:
        if not self.history:
            return False
        state = self.history.pop()
        self.current_data = state["data"]
        self.current_trace_metadata = _clone_trace_metadata(state.get("trace_metadata"))
        self.header_info = _clone_header_info(state.get("header_info"))
        self.current_label = state.get("label") or "当前结果"
        self.revision += 1
        self.changed.emit({"reason": "undo", "revision": self.revision})
        return True

    def reset_to_original(self, push_history: bool = True) -> bool:
        if self.original_data is None:
            return False
        if push_history and self.current_data is not None:
            self.push_history()
        self.current_data = np.array(self.original_data, copy=True)
        self.current_trace_metadata = _clone_trace_metadata(
            self.original_trace_metadata
        )
        self.header_info = _clone_header_info(self.original_header_info)
        self.current_label = self.original_label
        self.revision += 1
        self.changed.emit({"reason": "reset", "revision": self.revision})
        return True

    def build_result_history(self) -> list[tuple[str, np.ndarray]]:
        """构建正式结果时间线，供主界面和工作台统一展示。"""
        return [
            (str(entry["label"]), np.array(entry["data"], copy=True))
            for entry in self.build_result_history_entries()
        ]

    def build_result_history_entries(self) -> list[dict[str, Any]]:
        """Build formal result history entries with matching metadata snapshots."""
        history_items: list[dict[str, Any]] = []
        if self.original_data is None:
            return history_items

        _append_unique_history_entry(
            history_items,
            self.original_label or "原始数据",
            self.original_data,
            trace_metadata=self.original_trace_metadata,
            header_info=self.original_header_info,
        )

        for state in self.history:
            data = state.get("data")
            if data is None:
                continue
            _append_unique_history_entry(
                history_items,
                state.get("label") or f"历史结果{len(history_items)}",
                data,
                trace_metadata=state.get("trace_metadata"),
                header_info=state.get("header_info"),
            )

        if self.current_data is not None:
            _append_unique_history_entry(
                history_items,
                self.current_label
                or (self.original_label if not self.history else "当前结果"),
                self.current_data,
                trace_metadata=self.current_trace_metadata,
                header_info=self.header_info,
            )

        return history_items

    def build_formal_compare_snapshots(self) -> list[dict[str, Any]]:
        """构建正式对比快照。始终保留“原始/当前”两个锚点。"""
        snapshots: list[dict[str, Any]] = []
        if self.original_data is None:
            return snapshots

        snapshots.append(
            {
                "label": "原始",
                "data": np.array(self.original_data, copy=True),
                "trace_metadata": _clone_trace_metadata(self.original_trace_metadata),
                "header_info": _clone_header_info(self.original_header_info),
            }
        )

        for state in self.history:
            data = state.get("data")
            if data is None:
                continue
            if snapshots and np.array_equal(snapshots[-1]["data"], data):
                continue
            snapshots.append(
                {
                    "label": state.get("label") or f"历史结果{len(snapshots)}",
                    "data": np.array(data, copy=True),
                    "trace_metadata": _clone_trace_metadata(
                        state.get("trace_metadata")
                    ),
                    "header_info": _clone_header_info(state.get("header_info")),
                }
            )

        if self.current_data is not None:
            snapshots.append(
                {
                    "label": "当前",
                    "data": np.array(self.current_data, copy=True),
                    "trace_metadata": _clone_trace_metadata(
                        self.current_trace_metadata
                    ),
                    "header_info": _clone_header_info(self.header_info),
                }
            )

        return snapshots

    def set_metadata(
        self,
        *,
        path: str | None = None,
        header_info: dict[str, Any] | None = None,
        trace_metadata: dict[str, np.ndarray] | None = None,
        emit: bool = False,
    ):
        if path is not None:
            self.data_path = path
        if header_info is not None:
            self.header_info = _clone_header_info(header_info)
        if trace_metadata is not None:
            self.current_trace_metadata = _clone_trace_metadata(trace_metadata)
        if emit:
            self.changed.emit({"reason": "metadata", "revision": self.revision})

    def _trim_history(self) -> None:
        overflow = len(self.history) - int(self.max_history)
        if overflow > 0:
            del self.history[:overflow]


def _clone_trace_metadata(
    metadata: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray] | None:
    if metadata is None:
        return None
    return {k: np.array(v, copy=True) for k, v in metadata.items()}


def _clone_header_info(header_info: dict[str, Any] | None) -> dict[str, Any] | None:
    if header_info is None:
        return None
    cloned: dict[str, Any] = {}
    for key, value in header_info.items():
        cloned[key] = (
            np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        )
    return cloned


def _append_unique_history_item(
    items: list[tuple[str, np.ndarray]], label: str, data: np.ndarray
) -> None:
    candidate = np.array(data, copy=True)
    if items and np.array_equal(items[-1][1], candidate):
        items[-1] = (label, candidate)
        return
    items.append((label, candidate))


def _append_unique_history_entry(
    items: list[dict[str, Any]],
    label: str,
    data: np.ndarray,
    *,
    trace_metadata: dict[str, np.ndarray] | None,
    header_info: dict[str, Any] | None,
) -> None:
    candidate = {
        "label": label,
        "data": np.array(data, copy=True),
        "trace_metadata": _clone_trace_metadata(trace_metadata),
        "header_info": _clone_header_info(header_info),
    }
    if items and np.array_equal(items[-1]["data"], candidate["data"]):
        items[-1] = candidate
        return
    items.append(candidate)
