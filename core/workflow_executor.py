#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工作流执行引擎 - 按顺序执行方法列表"""

import numpy as np
from typing import List, Optional
from PyQt6.QtCore import QObject, pyqtSignal

from core.workflow_data import WorkflowMethod
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


class ExecutionError(Exception):
    """执行错误"""

    pass


class WorkflowExecutor(QObject):
    """工作流执行器"""

    # 信号
    step_started = pyqtSignal(str, int, int)  # (method_name, current, total)
    step_finished = pyqtSignal(str, np.ndarray)  # (method_name, result)
    step_error = pyqtSignal(str, str)  # (method_name, error_msg)
    all_finished = pyqtSignal(np.ndarray)  # (final_result)
    progress_updated = pyqtSignal(int, int)  # (current, total)

    def __init__(
        self,
        header_info: dict | None = None,
        trace_metadata: dict[str, np.ndarray] | None = None,
    ):
        super().__init__()
        self.history = []  # 执行历史
        self.current_data = None
        self.current_header_info = clone_header_info(header_info)
        self.current_trace_metadata = clone_trace_metadata(trace_metadata)
        self.is_running = False
        self._cancel_requested = False

    def execute_single(
        self, data: np.ndarray, method: WorkflowMethod
    ) -> tuple[np.ndarray, dict]:
        """执行单个方法

        Args:
            data: 输入数据
            method: 方法配置

        Returns:
            处理后的数据
        """
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
        except ProcessingEngineError as e:
            raise ExecutionError(str(e))

    def execute_all(
        self, data: np.ndarray, methods: List[WorkflowMethod]
    ) -> np.ndarray:
        """顺序执行所有方法

        Args:
            data: 原始数据
            methods: 方法列表（按顺序）

        Returns:
            最终处理结果
        """
        self.is_running = True
        self._cancel_requested = False
        self.current_data = data.copy()
        self.current_header_info = merge_result_header_info(
            self.current_header_info, None, self.current_data.shape
        )
        self.history = [data.copy()]  # 保存原始数据

        enabled_methods = [m for m in methods if m.enabled]
        total = len(enabled_methods)

        try:
            for i, method in enumerate(enabled_methods):
                if self._cancel_requested:
                    raise ExecutionError("用户取消执行")

                method_id = method.method_id
                method_info = PROCESSING_METHODS.get(method_id, {})
                method_name = method_info.get("name", method_id)

                # 发送开始信号
                self.step_started.emit(method_name, i + 1, total)
                self.progress_updated.emit(i + 1, total)

                # 执行方法
                try:
                    result, _ = self.execute_single(self.current_data, method)
                    self.current_data = result
                    self.history.append(result.copy())

                    # 发送完成信号
                    self.step_finished.emit(method_name, result)

                except Exception as e:
                    error_msg = str(e)
                    self.step_error.emit(method_name, error_msg)
                    raise ExecutionError(f"执行 {method_name} 失败: {error_msg}")

            # 全部完成
            self.all_finished.emit(self.current_data)
            return self.current_data

        finally:
            self.is_running = False

    def cancel(self):
        """请求取消执行"""
        self._cancel_requested = True

    def undo(self) -> Optional[np.ndarray]:
        """撤销上一步"""
        if len(self.history) > 1:
            self.history.pop()  # 移除当前状态
            self.current_data = self.history[-1].copy()
            return self.current_data
        return None

    def can_undo(self) -> bool:
        """是否可以撤销"""
        return len(self.history) > 1

    def reset(self, original_data: np.ndarray):
        """重置到原始数据"""
        self.current_data = original_data.copy()
        self.history = [original_data.copy()]
        self.is_running = False
        self._cancel_requested = False
