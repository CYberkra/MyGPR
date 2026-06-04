# -*- coding: utf-8 -*-
"""AutoTune worker start helpers for app_qt.GPRGuiQt."""

from __future__ import annotations

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QMessageBox

from core.methods_registry import PROCESSING_METHODS, get_auto_tune_stage, get_public_method_keys
from ui.worker_threads import AutoTuneComparisonWorker, AutoTuneStageWorker, AutoTuneWorker


class MainWindowAutoTuneStartMixin:
    def start_auto_tune_current_method(self, auto_apply_after_finish: bool = False):
        """对当前方法执行单步自动选参。"""
        self._enforce_no_prior_action_guard(
            "AutoTune",
            dialog_title="自动选参",
            allow_override=True,
            show_dialog=False,
            advisory_only=True,
        )

        if (
            self._ui_busy
            or self._worker is not None
            or self._auto_tune_worker is not None
            or self._auto_tune_stage_worker is not None
            or self._auto_tune_comparison_worker is not None
        ):
            self._pending_apply_after_auto_tune = False
            QMessageBox.information(self, "自动选参", "当前已有任务在运行，请稍候。")
            return False
        if self.data is None or self.data_path is None:
            self._pending_apply_after_auto_tune = False
            QMessageBox.warning(self, "自动选参", "请先导入数据。")
            return False

        method_key = self.page_basic.get_current_method_key()
        method_info = PROCESSING_METHODS.get(method_key, {}) if method_key else {}
        if not method_key or not method_info.get("auto_tune_enabled"):
            self._pending_apply_after_auto_tune = False
            QMessageBox.information(self, "自动选参", "当前方法暂不支持自动选参。")
            return False

        try:
            current_params = self.page_basic.get_current_params()
        except ValueError as e:
            self._pending_apply_after_auto_tune = False
            QMessageBox.warning(self, "自动选参", str(e))
            return False

        roi_mode = self.page_auto_tune.get_auto_tune_roi_mode()
        search_mode = self.page_auto_tune.get_auto_tune_search_mode()
        roi_spec = self._build_auto_tune_roi_spec(roi_mode)

        self._last_auto_tune_result = None
        self.page_basic.set_auto_tune_result_available(False)
        self.page_auto_tune.show_running(
            roi_spec.get("label", roi_spec.get("source", "全图")),
            search_mode,
        )
        self._set_busy(True, text=f"自动选参: {method_info.get('name', method_key)}")
        self._cancel_in_flight = False
        self._pending_apply_after_auto_tune = bool(auto_apply_after_finish)

        self._auto_tune_thread = QThread(self)
        self._auto_tune_worker = AutoTuneWorker(
            self.data,
            method_key,
            current_params,
            header_info=self.header_info,
            trace_metadata=self.trace_metadata,
            roi_spec=roi_spec,
            search_mode=search_mode,
        )
        self._auto_tune_worker.moveToThread(self._auto_tune_thread)
        self._auto_tune_thread.started.connect(self._auto_tune_worker.run)
        self._auto_tune_worker.progress.connect(self._on_auto_tune_progress)
        self._auto_tune_worker.finished.connect(self._on_auto_tune_finished)
        self._auto_tune_worker.error.connect(self._on_auto_tune_error)
        self._auto_tune_worker.finished.connect(self._auto_tune_thread.quit)
        self._auto_tune_worker.error.connect(self._auto_tune_thread.quit)
        self._auto_tune_thread.finished.connect(self._cleanup_auto_tune_worker)
        self._auto_tune_thread.start()
        return True

    def _get_current_stage_method_keys(self) -> list[str]:
        """获取当前方法所在 stage 的可比较方法列表。"""
        method_key = self.page_basic.get_current_method_key()
        if not method_key:
            return []
        stage = get_auto_tune_stage(method_key)
        if not stage:
            return []
        return [
            key
            for key in get_public_method_keys()
            if PROCESSING_METHODS.get(key, {}).get("auto_tune_enabled")
            and get_auto_tune_stage(key) == stage
        ]

    def start_auto_select_current_stage(self):
        """比较当前 stage 内多个可用方法并推荐最佳方法。"""
        self._enforce_no_prior_action_guard(
            "AutoTune",
            dialog_title="同阶段比较",
            allow_override=True,
            show_dialog=False,
            advisory_only=True,
        )

        if (
            self._ui_busy
            or self._worker is not None
            or self._auto_tune_worker is not None
            or self._auto_tune_stage_worker is not None
            or self._auto_tune_comparison_worker is not None
        ):
            QMessageBox.information(self, "同阶段比较", "当前已有任务在运行，请稍候。")
            return False
        if self.data is None or self.data_path is None:
            QMessageBox.warning(self, "同阶段比较", "请先导入数据。")
            return False

        method_keys = self._get_current_stage_method_keys()
        if len(method_keys) < 2:
            QMessageBox.information(
                self,
                "同阶段比较",
                "当前 stage 没有足够多的可比较方法。",
            )
            return False

        base_params_map = {}
        for key in method_keys:
            base_params_map[key] = self._resolve_method_params(key)

        roi_mode = self.page_auto_tune.get_auto_tune_roi_mode()
        search_mode = self.page_auto_tune.get_auto_tune_search_mode()
        roi_spec = self._build_auto_tune_roi_spec(roi_mode)

        self._last_auto_tune_group_result = None
        self.page_auto_tune.set_stage_compare_result(None)
        self.page_auto_tune.show_running(
            roi_spec.get("label", roi_spec.get("source", "全图")),
            f"{search_mode} | 同阶段比较",
        )
        self._set_busy(True, text="同阶段方法比较")
        self._cancel_in_flight = False

        self._auto_tune_stage_thread = QThread(self)
        self._auto_tune_stage_worker = AutoTuneStageWorker(
            self.data,
            method_keys,
            base_params_map,
            header_info=self.header_info,
            trace_metadata=self.trace_metadata,
            roi_spec=roi_spec,
            search_mode=search_mode,
        )
        self._auto_tune_stage_worker.moveToThread(self._auto_tune_stage_thread)
        self._auto_tune_stage_thread.started.connect(self._auto_tune_stage_worker.run)
        self._auto_tune_stage_worker.progress.connect(self._on_auto_tune_progress)
        self._auto_tune_stage_worker.finished.connect(self._on_auto_stage_finished)
        self._auto_tune_stage_worker.error.connect(self._on_auto_stage_error)
        self._auto_tune_stage_worker.finished.connect(self._auto_tune_stage_thread.quit)
        self._auto_tune_stage_worker.error.connect(self._auto_tune_stage_thread.quit)
        self._auto_tune_stage_thread.finished.connect(
            self._cleanup_auto_tune_stage_worker
        )
        self._auto_tune_stage_thread.start()
        return True

    def _build_manual_baseline_params_for_comparison(self) -> dict[str, dict]:
        """构建科研对比的人工 baseline 参数快照。"""
        params_map = {
            str(key): dict(value or {})
            for key, value in (self._method_param_overrides or {}).items()
        }
        current_method_key = self.page_basic.get_current_method_key()
        if not current_method_key:
            return params_map

        try:
            visible_params = self.page_basic.get_current_params()
        except ValueError:
            raise
        resolved_params = self._resolve_method_params(current_method_key)
        changed = any(
            resolved_params.get(key) != value for key, value in visible_params.items()
        )
        if changed or current_method_key in params_map:
            resolved_params.update(visible_params)
            params_map[current_method_key] = dict(resolved_params)
            self._method_param_overrides[current_method_key] = dict(resolved_params)
            self.page_basic.set_method_overrides(current_method_key, resolved_params)
        return params_map

    def start_auto_tune_comparison(self):
        """运行人工 baseline vs 自动选参科研对比。"""
        self._enforce_no_prior_action_guard(
            "AutoTune",
            dialog_title="人工/自动对比",
            allow_override=True,
            show_dialog=False,
            advisory_only=True,
        )

        if (
            self._ui_busy
            or self._worker is not None
            or self._auto_tune_worker is not None
            or self._auto_tune_stage_worker is not None
            or self._auto_tune_comparison_worker is not None
        ):
            QMessageBox.information(self, "人工/自动对比", "当前已有任务在运行，请稍候。")
            return False
        if self.data is None or self.data_path is None:
            QMessageBox.warning(self, "人工/自动对比", "请先导入数据。")
            return False

        try:
            manual_params_by_method = self._build_manual_baseline_params_for_comparison()
        except ValueError as e:
            QMessageBox.warning(self, "人工/自动对比", str(e))
            return False

        roi_mode = self.page_auto_tune.get_auto_tune_roi_mode()
        search_mode = self.page_auto_tune.get_auto_tune_search_mode()
        roi_spec = self._build_auto_tune_roi_spec(roi_mode)
        baseline_profile_key = "uav_gpr_experience_baseline_v1"

        self._last_auto_tune_comparison_result = None
        self.page_auto_tune.show_comparison_running(
            roi_spec.get("label", roi_spec.get("source", "全图")),
            search_mode,
        )
        self._set_busy(True, text="人工/自动对比")
        self._cancel_in_flight = False

        self._auto_tune_comparison_thread = QThread(self)
        self._auto_tune_comparison_worker = AutoTuneComparisonWorker(
            self.data,
            manual_params_by_method,
            header_info=self.header_info,
            trace_metadata=self.trace_metadata,
            baseline_profile_key=baseline_profile_key,
            roi_spec=roi_spec,
            search_mode=search_mode,
        )
        self._auto_tune_comparison_worker.moveToThread(
            self._auto_tune_comparison_thread
        )
        self._auto_tune_comparison_thread.started.connect(
            self._auto_tune_comparison_worker.run
        )
        self._auto_tune_comparison_worker.progress.connect(self._on_auto_tune_progress)
        self._auto_tune_comparison_worker.finished.connect(
            self._on_auto_comparison_finished
        )
        self._auto_tune_comparison_worker.error.connect(
            self._on_auto_comparison_error
        )
        self._auto_tune_comparison_worker.finished.connect(
            self._auto_tune_comparison_thread.quit
        )
        self._auto_tune_comparison_worker.error.connect(
            self._auto_tune_comparison_thread.quit
        )
        self._auto_tune_comparison_thread.finished.connect(
            self._cleanup_auto_tune_comparison_worker
        )
        self._auto_tune_comparison_thread.start()
        return True
