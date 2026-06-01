#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure Python shared data model for MyGPR.

This module intentionally has no Qt/PyQt dependency. GUI notification is
provided by ``ui.shared_data_qt_adapter``. Keep processing state, history
pruning, metadata cloning, and replay evidence logic here so CLI/headless tests
can exercise the data model without a QApplication.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np
def _env_int(name: str, default: int) -> int:
    try:
        value = int(str(os.environ.get(name, "")).strip())
        return value if value >= 0 else int(default)
    except Exception:
        return int(default)


DEFAULT_MAX_HISTORY_STEPS = _env_int("MYGPR_HISTORY_MAX_STEPS", 10)
DEFAULT_MAX_HISTORY_BYTES = _env_int("MYGPR_HISTORY_MAX_BYTES", 256 * 1024 * 1024)
DEFAULT_MAX_HISTORY_SNAPSHOT_BYTES = _env_int("MYGPR_HISTORY_MAX_SNAPSHOT_BYTES", 128 * 1024 * 1024)
DEFAULT_MAX_PRUNED_HISTORY_SUMMARIES = _env_int("MYGPR_HISTORY_MAX_PRUNED_SUMMARIES", 50)


class SharedDataModel:
    """Single source of truth for loaded/processed data.

    The model exposes the same state/mutation methods historically used by
    ``SharedDataState`` but emits no Qt signals. Lightweight Python listeners
    may be registered for headless consumers.
    """

    def __init__(self):
        self._change_listeners: list[Any] = []
        self.last_change_event: dict[str, Any] | None = None
        self.current_data: np.ndarray | None = None
        self.original_data: np.ndarray | None = None
        self.current_trace_metadata: dict[str, np.ndarray] | None = None
        self.original_trace_metadata: dict[str, np.ndarray] | None = None
        self.original_header_info: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self.pruned_history_summaries: list[dict[str, Any]] = []
        self.replay_package: dict[str, Any] | None = None
        self.data_path: str | None = None
        self.header_info: dict[str, Any] | None = None
        self.original_label = "原始数据"
        self.current_label = "原始数据"
        self.max_history = DEFAULT_MAX_HISTORY_STEPS
        self.max_history_bytes = DEFAULT_MAX_HISTORY_BYTES
        self.max_history_snapshot_bytes = DEFAULT_MAX_HISTORY_SNAPSHOT_BYTES
        self.max_pruned_history_summaries = DEFAULT_MAX_PRUNED_HISTORY_SUMMARIES
        self.revision = 0

    def add_change_listener(self, callback: Any) -> None:
        """Register a Python callback receiving change-event dictionaries."""
        if callback not in self._change_listeners:
            self._change_listeners.append(callback)

    def remove_change_listener(self, callback: Any) -> None:
        """Remove a previously registered Python change callback."""
        try:
            self._change_listeners.remove(callback)
        except ValueError:
            pass

    def _notify_changed(self, event: dict[str, Any]) -> None:
        """Notify headless listeners. Qt adapters override and also emit a signal."""
        payload = dict(event)
        self.last_change_event = payload
        for callback in list(self._change_listeners):
            try:
                callback(payload)
            except Exception:
                # Listener failures must not corrupt the application data state.
                continue

    def load_data(
        self,
        data: np.ndarray,
        path: str | None = None,
        header_info: dict[str, Any] | None = None,
        trace_metadata: dict[str, np.ndarray] | None = None,
        source: str = "load",
    ):
        array = np.asarray(data)
        self.current_data = np.array(array, copy=True)
        self.original_data = np.array(array, copy=True)
        self.current_trace_metadata = _clone_trace_metadata(trace_metadata)
        self.original_trace_metadata = _clone_trace_metadata(trace_metadata)
        self.header_info = _clone_header_info(header_info)
        self.original_header_info = _clone_header_info(header_info)
        self.history = []
        self.pruned_history_summaries = []
        self.data_path = path
        self.original_label = "原始数据"
        self.current_label = "原始数据"
        self.revision += 1
        self._refresh_replay_package()
        self._notify_changed(
            {"reason": "loaded", "source": source, "revision": self.revision}
        )

    def push_history(self, label: str | None = None):
        if self.current_data is None:
            return
        entry = {
            "data": np.array(self.current_data, copy=True),
            "trace_metadata": _clone_trace_metadata(self.current_trace_metadata),
            "header_info": _clone_header_info(self.header_info),
            "label": label or self.current_label or "当前结果",
        }
        entry["memory_bytes"] = _history_entry_nbytes(entry)
        entry["array_summary"] = _summarize_array(entry.get("data"))

        max_snapshot_bytes = int(getattr(self, "max_history_snapshot_bytes", 0) or 0)
        if max_snapshot_bytes > 0 and entry["memory_bytes"] > max_snapshot_bytes:
            self._record_pruned_history(entry, reason="snapshot_exceeds_limit")
            self._refresh_replay_package()
            return

        self.history.append(entry)
        self._trim_history()
        self._refresh_replay_package()

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
        self._refresh_replay_package()
        self._notify_changed(
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
        self._refresh_replay_package()
        self._notify_changed({"reason": "undo", "revision": self.revision})
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
        self._refresh_replay_package()
        self._notify_changed({"reason": "reset", "revision": self.revision})
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
            self._refresh_replay_package()
            self._notify_changed({"reason": "metadata", "revision": self.revision})
        else:
            self._refresh_replay_package()

    def _trim_history(self) -> None:
        max_history = int(getattr(self, "max_history", DEFAULT_MAX_HISTORY_STEPS) or 0)
        if max_history < 0:
            max_history = DEFAULT_MAX_HISTORY_STEPS
        while max_history and len(self.history) > max_history:
            removed = self.history.pop(0)
            self._record_pruned_history(removed, reason="history_count_limit")
        if max_history == 0:
            while self.history:
                removed = self.history.pop(0)
                self._record_pruned_history(removed, reason="history_disabled")

        max_bytes = int(getattr(self, "max_history_bytes", DEFAULT_MAX_HISTORY_BYTES) or 0)
        if max_bytes > 0:
            while self.history and self._history_memory_bytes() > max_bytes:
                removed = self.history.pop(0)
                self._record_pruned_history(removed, reason="history_memory_limit")

    def _history_memory_bytes(self) -> int:
        return int(sum(_history_entry_nbytes(entry) for entry in self.history))

    def _record_pruned_history(self, entry: dict[str, Any], *, reason: str) -> None:
        self.pruned_history_summaries.append(_build_pruned_history_summary(entry, reason=reason))
        max_items = int(getattr(self, "max_pruned_history_summaries", DEFAULT_MAX_PRUNED_HISTORY_SUMMARIES) or 0)
        if max_items > 0 and len(self.pruned_history_summaries) > max_items:
            del self.pruned_history_summaries[: len(self.pruned_history_summaries) - max_items]

    def get_history_memory_summary(self) -> dict[str, Any]:
        stored_bytes = self._history_memory_bytes()
        stored_entries = []
        for idx, entry in enumerate(self.history):
            stored_entries.append(
                {
                    "index": idx,
                    "label": str(entry.get("label") or "历史结果"),
                    "memory_bytes": int(_history_entry_nbytes(entry)),
                    "array": _summarize_array(entry.get("data")),
                }
            )
        return {
            "schema": "mygpr.history_memory.v1",
            "stored_count": int(len(self.history)),
            "stored_bytes": int(stored_bytes),
            "max_history_steps": int(getattr(self, "max_history", DEFAULT_MAX_HISTORY_STEPS) or 0),
            "max_history_bytes": int(getattr(self, "max_history_bytes", DEFAULT_MAX_HISTORY_BYTES) or 0),
            "max_history_snapshot_bytes": int(getattr(self, "max_history_snapshot_bytes", DEFAULT_MAX_HISTORY_SNAPSHOT_BYTES) or 0),
            "pruned_count": int(len(self.pruned_history_summaries)),
            "stored_entries": stored_entries,
            "pruned_summaries": copy.deepcopy(self.pruned_history_summaries),
            "policy_note": "Only stored_entries with data are undoable/clickable. Pruned summaries preserve audit context without keeping full arrays in memory.",
        }

    def build_replay_evidence_package(self) -> dict[str, Any]:
        """Build an in-memory replay evidence package without touching disk."""
        snapshots = self.build_result_history_entries()
        package: dict[str, Any] = {
            "package_type": "mygpr_replay_evidence",
            "schema_version": 1,
            "storage": "memory_only_until_user_export",
            "revision": int(self.revision),
            "data_path": self.data_path,
            "original_label": self.original_label,
            "current_label": self.current_label,
            "history_count": int(len(self.history)),
            "pruned_history_count": int(len(self.pruned_history_summaries)),
            "snapshot_count": int(len(snapshots)),
            "history_memory": self.get_history_memory_summary(),
            "has_original": bool(self.original_data is not None),
            "has_current": bool(self.current_data is not None),
            "original": _build_snapshot_payload(
                self.original_label,
                self.original_data,
                self.original_header_info,
                self.original_trace_metadata,
            ),
            "current": _build_snapshot_payload(
                self.current_label,
                self.current_data,
                self.header_info,
                self.current_trace_metadata,
            ),
            "history_entries": [],
            "snapshots": [],
        }
        if self.original_data is not None:
            package["history_entries"] = [
                _build_snapshot_payload(
                    entry.get("label") or "历史结果",
                    entry.get("data"),
                    entry.get("header_info"),
                    entry.get("trace_metadata"),
                )
                for entry in self.history
                if entry.get("data") is not None
            ]
        package["snapshots"] = [
            _build_snapshot_payload(
                entry.get("label") or "历史结果",
                entry.get("data"),
                entry.get("header_info"),
                entry.get("trace_metadata"),
                index=idx,
                role=_snapshot_role(idx, len(snapshots)),
            )
            for idx, entry in enumerate(snapshots)
            if entry.get("data") is not None
        ]
        package["summary"] = {
            "history_labels": [
                str(entry.get("label") or "历史结果") for entry in package["history_entries"]
            ],
            "snapshot_labels": [
                str(entry.get("label") or "历史结果") for entry in package["snapshots"]
            ],
            "original_stats": _summarize_array(self.original_data),
            "current_stats": _summarize_array(self.current_data),
            "history_memory": self.get_history_memory_summary(),
        }
        return package

    def get_replay_evidence_package(self) -> dict[str, Any] | None:
        """Return the latest in-memory replay evidence package."""
        if self.replay_package is None:
            self._refresh_replay_package()
        return self.replay_package

    def _refresh_replay_package(self) -> None:
        """Refresh the in-memory replay evidence package."""
        if self.original_data is None or self.current_data is None:
            self.replay_package = None
            return
        self.replay_package = self.build_replay_evidence_package()


def _array_nbytes(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(np.asarray(value).nbytes)
    except Exception:
        return 0


def _history_entry_nbytes(entry: dict[str, Any]) -> int:
    total = _array_nbytes(entry.get("data"))
    metadata = entry.get("trace_metadata") or {}
    if isinstance(metadata, dict):
        for value in metadata.values():
            total += _array_nbytes(value)
    header = entry.get("header_info") or {}
    if isinstance(header, dict):
        for value in header.values():
            if isinstance(value, np.ndarray):
                total += _array_nbytes(value)
    return int(total)


def _build_pruned_history_summary(entry: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "label": str(entry.get("label") or "历史结果"),
        "reason": str(reason),
        "memory_bytes": int(_history_entry_nbytes(entry)),
        "array": _summarize_array(entry.get("data")),
        "header_summary": _summarize_header_info(entry.get("header_info")),
        "trace_metadata_summary": _summarize_trace_metadata(entry.get("trace_metadata")),
    }


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
            np.array(value, copy=True)
            if isinstance(value, np.ndarray)
            else copy.deepcopy(value)
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


def _build_snapshot_payload(
    label: str,
    data: np.ndarray | None,
    header_info: dict[str, Any] | None,
    trace_metadata: dict[str, np.ndarray] | None,
    *,
    index: int | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    arr = None if data is None else np.array(data, copy=True)
    payload: dict[str, Any] = {
        "label": str(label),
        "data": arr,
        "header_info": _clone_header_info(header_info),
        "trace_metadata": _clone_trace_metadata(trace_metadata),
    }
    if index is not None:
        payload["index"] = int(index)
    if role is not None:
        payload["role"] = role
    payload["summary"] = _summarize_array(arr)
    payload["header_summary"] = _summarize_header_info(header_info)
    payload["trace_metadata_summary"] = _summarize_trace_metadata(trace_metadata)
    return payload


def _snapshot_role(index: int, total: int) -> str:
    if index == 0:
        return "original"
    if index == total - 1:
        return "current"
    return "history"


def _summarize_array(data: np.ndarray | None) -> dict[str, Any]:
    if data is None:
        return {"present": False}

    arr = np.asarray(data)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        min_value = float(np.min(finite))
        max_value = float(np.max(finite))
        mean_value = float(np.mean(finite))
        std_value = float(np.std(finite))
    else:
        min_value = max_value = mean_value = std_value = float("nan")
    return {
        "present": True,
        "shape": [int(dim) for dim in arr.shape],
        "dtype": str(arr.dtype),
        "finite_ratio": float(finite.size / arr.size) if arr.size else 0.0,
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "std": std_value,
    }


def _summarize_header_info(header_info: dict[str, Any] | None) -> dict[str, Any]:
    if not header_info:
        return {}
    summary: dict[str, Any] = {}
    for key, value in header_info.items():
        if isinstance(value, np.ndarray):
            summary[str(key)] = {
                "kind": "ndarray",
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype),
            }
        elif isinstance(value, (np.floating, np.integer)):
            summary[str(key)] = value.item()
        elif isinstance(value, (str, bool, int, float)) or value is None:
            summary[str(key)] = value
        else:
            summary[str(key)] = str(value)
    return summary


def _summarize_trace_metadata(
    metadata: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    if not metadata:
        return {"field_count": 0, "fields": []}
    fields: list[str] = []
    field_shapes: dict[str, list[int]] = {}
    field_dtypes: dict[str, str] = {}
    for key, value in metadata.items():
        arr = np.asarray(value)
        fields.append(str(key))
        field_shapes[str(key)] = [int(dim) for dim in arr.shape]
        field_dtypes[str(key)] = str(arr.dtype)
    return {
        "field_count": len(fields),
        "fields": sorted(fields),
        "field_shapes": field_shapes,
        "field_dtypes": field_dtypes,
    }
