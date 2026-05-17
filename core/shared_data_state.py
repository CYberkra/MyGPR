#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared application data state for main GUI and workflow views."""

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
        self.replay_package: dict[str, Any] | None = None
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
        self._refresh_replay_package()
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
        self._refresh_replay_package()
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
        self._refresh_replay_package()
        self.changed.emit({"reason": "reset", "revision": self.revision})
        return True

    def build_result_history(self) -> list[tuple[str, np.ndarray]]:
        """构建正式结果时间线，供主界面和工作流统一展示。"""
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
            self.changed.emit({"reason": "metadata", "revision": self.revision})
        else:
            self._refresh_replay_package()

    def _trim_history(self) -> None:
        overflow = len(self.history) - int(self.max_history)
        if overflow > 0:
            del self.history[:overflow]

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
            "snapshot_count": int(len(snapshots)),
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
        min_value = max_value = mean_value = std_value = None
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
        elif isinstance(value, np.floating):
            parsed = float(value.item())
            summary[str(key)] = parsed if np.isfinite(parsed) else None
        elif isinstance(value, np.integer):
            summary[str(key)] = value.item()
        elif isinstance(value, float):
            summary[str(key)] = value if np.isfinite(value) else None
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
