# -*- coding: utf-8 -*-
"""Qt worker objects for processing and AutoTune tasks.

These workers were split out of ``app_qt.py`` to keep threaded backend execution
separate from the main-window UI shell.
"""

from __future__ import annotations

import time

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from ui.gui_base import ProcessingCancelled, build_processing_error_message
from core.processing_engine import (
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.auto_tune import (
    auto_select_method_group,
    auto_tune_method,
    AutoTuneCancelled,
)
from core.auto_tune_comparison import run_auto_tune_comparison


class ProcessingWorker(QObject):
    """后台处理工作线程"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    # Emitted from the worker thread after a sequential processing step has
    # produced a B-scan.  The main window treats it as a display-only live
    # preview; formal data/history are committed only in ``finished``.
    step_completed = pyqtSignal(object)

    def __init__(
        self,
        base_data: np.ndarray,
        tasks: list,
        base_csv_path: str = None,
        execution_mode: str = "sequential",
        header_info: dict | None = None,
        trace_metadata: dict | None = None,
    ):
        super().__init__()
        self.base_data = np.array(base_data, copy=True)
        self.tasks = tasks
        self.base_csv_path = base_csv_path
        self.header_info = header_info or {}
        self.trace_metadata = trace_metadata or {}
        self._cancel_requested = False
        self.execution_mode = execution_mode

    def request_cancel(self):
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested)

    @staticmethod
    def _compact_result_meta(result_meta: dict | None) -> dict:
        """压缩工作线程中保留的结果元信息，避免重复挂大数组。"""
        compact_meta = dict(result_meta or {})
        compact_meta.pop("display_data", None)
        compact_meta.pop("display_header_info_updates", None)
        compact_meta.pop("display_trace_metadata", None)
        for key in list(compact_meta.keys()):
            if isinstance(compact_meta.get(key), np.ndarray):
                compact_meta.pop(key, None)
        return compact_meta

    def run(self):
        current_data = np.array(self.base_data, copy=True)
        current_header_info = merge_result_header_info(
            self.header_info, None, current_data.shape
        )
        current_trace_metadata = merge_result_trace_metadata(self.trace_metadata, None)
        current_display_data = None
        current_display_header_info = None
        current_display_trace_metadata = None
        outputs = []
        total = len(self.tasks)
        current_method_name = "未知方法"
        try:
            for i, task in enumerate(self.tasks, start=1):
                if self.is_cancel_requested():
                    raise ProcessingCancelled("用户已取消处理")

                step_t0 = time.perf_counter()
                method_key = task["method_key"]
                method = task["method"]
                current_method_name = method.get("name", method_key)
                params = dict(task.get("params", {}))
                param_source_mode = str(task.get("param_source_mode") or "manual")
                out_dir = task["out_dir"]

                self.progress.emit(
                    i - 1, total, f"开始步骤 {i}/{total}: {method['name']}"
                )

                task_input = (
                    self.base_data
                    if self.execution_mode == "independent"
                    else current_data
                )
                if param_source_mode == "auto_tune":
                    tuned_result = auto_tune_method(
                        task_input,
                        method_key,
                        header_info=current_header_info,
                        trace_metadata=current_trace_metadata,
                        base_params=params,
                        cancel_checker=self.is_cancel_requested,
                    )
                    profile_key = str(
                        tuned_result.get("recommended_profile", "balanced")
                    )
                    profile = (tuned_result.get("profiles", {}) or {}).get(
                        profile_key, {}
                    )
                    tuned_params = dict(
                        profile.get("params")
                        or tuned_result.get("recommended_params")
                        or tuned_result.get("best_params")
                        or {}
                    )
                    params.update(tuned_params)
                runtime_params = prepare_runtime_params(
                    method_key,
                    params,
                    current_header_info,
                    current_trace_metadata,
                    task_input.shape,
                )

                newdata, result_meta = run_processing_method(
                    task_input,
                    method_key,
                    runtime_params,
                    cancel_checker=self.is_cancel_requested,
                )
                display_data = result_meta.get("display_data")
                display_header_info = None
                if display_data is not None:
                    display_header_info = merge_result_header_info(
                        current_header_info,
                        {
                            "header_info_updates": result_meta.get(
                                "display_header_info_updates"
                            )
                        },
                        np.asarray(display_data).shape,
                    )
                resolved_header_info = merge_result_header_info(
                    current_header_info, result_meta, newdata.shape
                )
                resolved_trace_metadata = merge_result_trace_metadata(
                    current_trace_metadata, result_meta
                )

                if self.execution_mode != "independent":
                    current_data = newdata
                    current_header_info = resolved_header_info
                    current_trace_metadata = resolved_trace_metadata
                    current_display_data = (
                        np.array(display_data, copy=False)
                        if display_data is not None
                        else None
                    )
                    current_display_header_info = display_header_info
                    current_display_trace_metadata = result_meta.get(
                        "display_trace_metadata"
                    )
                elapsed_ms = (time.perf_counter() - step_t0) * 1000.0
                compact_meta = self._compact_result_meta(result_meta)
                item_header_info = dict(resolved_header_info or {})
                item_header_info.update(
                    {
                        "method_key": method_key,
                        "method_name": method["name"],
                        "params": dict(runtime_params),
                        "method_params": dict(runtime_params),
                        "param_source_mode": param_source_mode,
                        "elapsed_ms": elapsed_ms,
                    }
                )
                if task.get("recipe_step") is not None:
                    item_header_info["autotune_recipe_step"] = task.get("recipe_step")
                if task.get("autotune_scoring_record") is not None:
                    item_header_info["autotune_scoring_record"] = dict(task.get("autotune_scoring_record") or {})
                if task.get("autotune_recipe_plan") is not None:
                    item_header_info["autotune_recipe_plan"] = dict(task.get("autotune_recipe_plan") or {})
                outputs.append(
                    {
                        "method_key": method_key,
                        "method_name": method["name"],
                        "params": dict(runtime_params),
                        "param_source_mode": param_source_mode,
                        "elapsed_ms": elapsed_ms,
                        "data": np.array(newdata, copy=False),
                        "meta": compact_meta,
                        "runtime_warnings": compact_meta.get("runtime_warnings", []),
                        "header_info": item_header_info,
                        "trace_metadata": resolved_trace_metadata,
                        "recipe_step": task.get("recipe_step"),
                        "autotune_scoring_record": dict(task.get("autotune_scoring_record") or {}),
                    }
                )
                if self.execution_mode != "independent":
                    current_header_info = item_header_info
                    preview_data = display_data if display_data is not None else newdata
                    preview_header_info = (
                        display_header_info if display_data is not None else item_header_info
                    )
                    preview_trace_metadata = (
                        result_meta.get("display_trace_metadata")
                        if display_data is not None
                        else resolved_trace_metadata
                    )
                    self.step_completed.emit(
                        {
                            "current": i,
                            "total": total,
                            "method_key": method_key,
                            "method_name": method["name"],
                            "params": dict(runtime_params),
                            "elapsed_ms": elapsed_ms,
                            "data": np.array(preview_data, copy=True),
                            "header_info": dict(preview_header_info or {}),
                            "trace_metadata": preview_trace_metadata,
                            "execution_mode": self.execution_mode,
                            "recipe_step": task.get("recipe_step"),
                        }
                    )
                self.progress.emit(
                    i,
                    total,
                    f"完成步骤 {i}/{total}: {method['name']} ({elapsed_ms:.1f} ms)",
                )

            if self.is_cancel_requested():
                self.finished.emit(
                    {
                        "outputs": outputs,
                        "final_data": current_data,
                        "final_header_info": (outputs[-1].get("header_info") if outputs else current_header_info),
                        "final_trace_metadata": current_trace_metadata,
                        "final_display_data": current_display_data,
                        "final_display_header_info": current_display_header_info,
                        "final_display_trace_metadata": current_display_trace_metadata,
                        "cancelled": True,
                        "execution_mode": self.execution_mode,
                    }
                )
            else:
                self.finished.emit(
                    {
                        "outputs": outputs,
                        "final_data": current_data,
                        "final_header_info": (outputs[-1].get("header_info") if outputs else current_header_info),
                        "final_trace_metadata": current_trace_metadata,
                        "final_display_data": current_display_data,
                        "final_display_header_info": current_display_header_info,
                        "final_display_trace_metadata": current_display_trace_metadata,
                        "execution_mode": self.execution_mode,
                    }
                )
        except ProcessingCancelled:
            self.finished.emit(
                {
                    "outputs": outputs,
                    "final_data": current_data,
                    "final_header_info": current_header_info,
                    "final_trace_metadata": current_trace_metadata,
                    "final_display_data": current_display_data,
                    "final_display_header_info": current_display_header_info,
                    "final_display_trace_metadata": current_display_trace_metadata,
                    "cancelled": True,
                    "execution_mode": self.execution_mode,
                }
            )
        except Exception as e:
            self.error.emit(build_processing_error_message(e, current_method_name))


class AutoTuneWorker(QObject):
    """后台自动选参工作线程。"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    def __init__(
        self,
        data: np.ndarray,
        method_key: str,
        base_params: dict[str, object],
        header_info: dict | None = None,
        trace_metadata: dict | None = None,
        roi_spec: dict | None = None,
        search_mode: str = "standard",
    ):
        super().__init__()
        self.data = np.array(data, copy=True)
        self.method_key = method_key
        self.base_params = dict(base_params or {})
        self.header_info = header_info or {}
        self.trace_metadata = trace_metadata or {}
        self.roi_spec = dict(roi_spec or {})
        self.search_mode = str(search_mode or "standard")
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested)

    def run(self):
        try:
            result = auto_tune_method(
                self.data,
                self.method_key,
                header_info=self.header_info,
                trace_metadata=self.trace_metadata,
                base_params=self.base_params,
                roi_spec=self.roi_spec,
                search_mode=self.search_mode,
                progress_callback=lambda current, total, message: self.progress.emit(
                    int(current), int(total), str(message)
                ),
                cancel_checker=self.is_cancel_requested,
            )
            result["cancelled"] = self.is_cancel_requested()
            self.finished.emit(result)
        except AutoTuneCancelled:
            self.finished.emit(
                {
                    "method_key": self.method_key,
                    "cancelled": True,
                    "all_trials": [],
                }
            )
        except Exception as e:
            self.error.emit(str(e))


class AutoTuneStageWorker(QObject):
    """后台同阶段方法比较工作线程。"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    def __init__(
        self,
        data: np.ndarray,
        method_keys: list[str],
        base_params_map: dict[str, dict[str, object]],
        header_info: dict | None = None,
        trace_metadata: dict | None = None,
        roi_spec: dict | None = None,
        search_mode: str = "standard",
    ):
        super().__init__()
        self.data = np.array(data, copy=True)
        self.method_keys = list(method_keys)
        self.base_params_map = dict(base_params_map or {})
        self.header_info = header_info or {}
        self.trace_metadata = trace_metadata or {}
        self.roi_spec = dict(roi_spec or {})
        self.search_mode = str(search_mode or "standard")
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested)

    def run(self):
        try:
            result = auto_select_method_group(
                self.data,
                self.method_keys,
                header_info=self.header_info,
                trace_metadata=self.trace_metadata,
                base_params_map=self.base_params_map,
                roi_spec=self.roi_spec,
                search_mode=self.search_mode,
                progress_callback=lambda current, total, message: self.progress.emit(
                    int(current), int(total), str(message)
                ),
                cancel_checker=self.is_cancel_requested,
            )
            result["cancelled"] = self.is_cancel_requested()
            self.finished.emit(result)
        except AutoTuneCancelled:
            self.finished.emit({"cancelled": True, "candidates": []})
        except Exception as e:
            self.error.emit(str(e))


class AutoTuneComparisonWorker(QObject):
    """后台人工 baseline vs 自动选参科研对比工作线程。"""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    def __init__(
        self,
        data: np.ndarray,
        manual_params_by_method: dict[str, dict[str, object]],
        header_info: dict | None = None,
        trace_metadata: dict | None = None,
        baseline_profile_key: str = "uav_gpr_experience_baseline_v1",
        roi_spec: dict | None = None,
        search_mode: str = "standard",
    ):
        super().__init__()
        self.data = np.array(data, copy=True)
        self.manual_params_by_method = {
            str(key): dict(value or {})
            for key, value in (manual_params_by_method or {}).items()
        }
        self.header_info = header_info or {}
        self.trace_metadata = trace_metadata or {}
        self.baseline_profile_key = str(baseline_profile_key)
        self.roi_spec = dict(roi_spec or {})
        self.search_mode = str(search_mode or "standard")
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested)

    def run(self):
        try:
            result = run_auto_tune_comparison(
                self.data,
                header_info=self.header_info,
                trace_metadata=self.trace_metadata,
                manual_params_by_method=self.manual_params_by_method,
                baseline_profile_key=self.baseline_profile_key,
                roi_spec=self.roi_spec,
                search_mode=self.search_mode,
                progress_callback=lambda current, total, message: self.progress.emit(
                    int(current), int(total), str(message)
                ),
                cancel_checker=self.is_cancel_requested,
            )
            if self.is_cancel_requested():
                self.finished.emit({"cancelled": True})
                return
            self.finished.emit(result)
        except Exception as e:
            if self.is_cancel_requested():
                self.finished.emit({"cancelled": True})
                return
            self.error.emit(str(e))
