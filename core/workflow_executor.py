#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qt-free workflow execution engine.

The original implementation inherited ``QObject`` in the core layer.  This
version exposes small Python event hooks with the same ``connect``/``emit``
shape, so existing callers continue to work while CLI and headless consumers no
longer require PyQt6.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, List, Optional, TypeVar

import numpy as np

from core.app_errors import MyGPRError
from core.methods_registry import PROCESSING_METHODS
from core.processing_engine import (
    ProcessingEngineError,
    clone_header_info,
    clone_trace_metadata,
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.workflow_data import WorkflowMethod

T = TypeVar("T")


class EventHook(Generic[T]):
    """Minimal signal-like hook used by the pure workflow runner."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[..., Any]] = []

    def connect(self, callback: Callable[..., Any]) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def disconnect(self, callback: Callable[..., Any]) -> None:
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    def emit(self, *args: Any, **kwargs: Any) -> None:
        for callback in tuple(self._callbacks):
            callback(*args, **kwargs)


class ExecutionError(MyGPRError):
    """Workflow execution error."""


class WorkflowExecutor:
    """Sequential processing workflow executor with cooperative cancellation."""

    def __init__(
        self,
        header_info: dict | None = None,
        trace_metadata: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.step_started: EventHook[Any] = EventHook()
        self.step_finished: EventHook[Any] = EventHook()
        self.step_error: EventHook[Any] = EventHook()
        self.all_finished: EventHook[Any] = EventHook()
        self.progress_updated: EventHook[Any] = EventHook()
        self.history: list[np.ndarray] = []
        self.current_data: np.ndarray | None = None
        self.current_header_info = clone_header_info(header_info)
        self.current_trace_metadata = clone_trace_metadata(trace_metadata)
        self.is_running = False
        self._cancel_requested = False

    def execute_single(self, data: np.ndarray, method: WorkflowMethod) -> tuple[np.ndarray, dict]:
        method_id = method.method_id
        params = method.params or {}
        try:
            runtime_params = prepare_runtime_params(
                method_id,
                params,
                self.current_header_info,
                self.current_trace_metadata,
                data.shape,
            )
            result, meta = run_processing_method(
                data,
                method_id,
                runtime_params,
                cancel_checker=lambda: self._cancel_requested,
            )
            self.current_header_info = merge_result_header_info(
                self.current_header_info, meta, result.shape
            )
            self.current_trace_metadata = merge_result_trace_metadata(
                self.current_trace_metadata, meta
            )
            return result, meta
        except ProcessingEngineError as exc:
            raise ExecutionError(str(exc)) from exc

    def execute_all(self, data: np.ndarray, methods: List[WorkflowMethod]) -> np.ndarray:
        self.is_running = True
        self._cancel_requested = False
        self.current_data = np.array(data, copy=True)
        self.current_header_info = merge_result_header_info(
            self.current_header_info, None, self.current_data.shape
        )
        self.history = [np.array(data, copy=True)]
        enabled_methods = [method for method in methods if method.enabled]
        total = len(enabled_methods)
        try:
            for index, method in enumerate(enabled_methods, start=1):
                if self._cancel_requested:
                    raise ExecutionError("用户取消执行")
                method_info = PROCESSING_METHODS.get(method.method_id, {})
                method_name = method_info.get("name", method.method_id)
                self.step_started.emit(method_name, index, total)
                self.progress_updated.emit(index, total)
                try:
                    result, _ = self.execute_single(self.current_data, method)
                except Exception as exc:
                    self.step_error.emit(method_name, str(exc))
                    raise ExecutionError(f"执行 {method_name} 失败: {exc}") from exc
                self.current_data = result
                self.history.append(np.array(result, copy=True))
                self.step_finished.emit(method_name, result)
            self.all_finished.emit(self.current_data)
            return self.current_data
        finally:
            self.is_running = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def undo(self) -> Optional[np.ndarray]:
        if len(self.history) <= 1:
            return None
        self.history.pop()
        self.current_data = np.array(self.history[-1], copy=True)
        return self.current_data

    def can_undo(self) -> bool:
        return len(self.history) > 1

    def reset(self, original_data: np.ndarray) -> None:
        self.current_data = np.array(original_data, copy=True)
        self.history = [np.array(original_data, copy=True)]
        self.is_running = False
        self._cancel_requested = False


__all__ = ["EventHook", "ExecutionError", "WorkflowExecutor"]
