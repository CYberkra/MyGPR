# -*- coding: utf-8 -*-
"""Parameter-recommendation synchronization controller for MyGPR.

This controller owns the low-risk GUI synchronization layer around 参数推荐:
loaded-dataset metadata binding, manual focus area picker state, 参数推荐 result reset,
recommendation labels, and no-prior guardrail policy evaluation.  The heavy
参数推荐 worker/thread orchestration remains in ``app_qt.py`` for this first
V0.8.4 extraction; compatibility wrappers on ``GPRGuiQt`` keep older call sites
stable while reducing main-window responsibilities.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from core.no_prior_qc_policy import (
    build_no_prior_qc_policy,
    derive_no_prior_level,
    policy_to_dict,
)
from core.no_prior_ui_guardrails import (
    build_no_prior_guard_event,
    evaluate_no_prior_action,
)
from core.auto_tune_recommendation_labels import (
    assess_auto_tune_recommendation_label,
    recommendation_label_to_dict,
)


class AutoTuneSyncController:
    """Synchronize 参数推荐 UI state and no-prior guardrails for a host window."""

    def __init__(self, host):
        object.__setattr__(self, "host", host)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name == "host":
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def _resolve_target_response_for_autotune(self, *, data, path: str | None, header: dict, component: str | None):
        """Resolve a same-shape synthetic target_response for the GUI simple runner.

        Priority is deliberately conservative: explicit payload/header arrays or
        sibling ``target_response_<component>.npy`` files inside a MyGPR-readable
        run folder.  The GUI never pairs arrays across run/task folders.
        """
        shape = tuple(int(v) for v in np.asarray(data).shape[:2]) if data is not None else None
        candidates = []
        for key in ("target_response", "target_response_array"):
            value = header.get(key)
            if value is not None:
                candidates.append((value, f"header:{key}"))
        for key in ("target_response_path", "target_response_npy_path"):
            value = header.get(key)
            if value:
                candidates.append((Path(str(value)), f"header:{key}"))

        if path:
            src = Path(str(path))
            parent = src.parent
            suffix_candidates = []
            name = src.name
            if name.startswith("raw_") and name.endswith(".npy"):
                suffix_candidates.append(parent / name.replace("raw_", "target_response_", 1))
            if component:
                suffix_candidates.append(parent / f"target_response_{component}.npy")
            suffix_candidates.extend(sorted(parent.glob("target_response_*.npy")))
            seen = set()
            for candidate in suffix_candidates:
                try:
                    resolved = candidate.resolve()
                except Exception:
                    resolved = candidate
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append((candidate, str(candidate)))

        for value, label in candidates:
            try:
                if isinstance(value, (str, os.PathLike, Path)):
                    candidate_path = Path(value)
                    if not candidate_path.exists() or candidate_path.suffix.lower() != ".npy":
                        continue
                    arr = np.load(candidate_path)
                    label = str(candidate_path)
                else:
                    arr = np.asarray(value)
                arr = np.asarray(arr, dtype=np.float64)
                if shape is not None and tuple(arr.shape) != shape:
                    continue
                if not np.isfinite(arr).all():
                    finite = arr[np.isfinite(arr)]
                    fill = float(np.nanmedian(finite)) if finite.size else 0.0
                    arr = np.where(np.isfinite(arr), arr, fill)
                return arr, str(label)
            except Exception:
                continue
        return None, None

    def _sync_auto_tune_page_dataset_state(self, payload: dict | None = None) -> None:
        """Synchronize current loaded dataset metadata to AutoTuneTuningPage.

        Metadata-only binding: this does not run 参数推荐, does not change scoring,
        and does not write processing record artifacts.
        """
        page = getattr(self, "page_auto_tune", None)
        if page is None:
            return

        data = getattr(self, "data", None)
        if data is None:
            clear = getattr(page, "clear_loaded_dataset", None)
            if callable(clear):
                clear()
            return

        shape = None
        try:
            shape = tuple(int(v) for v in getattr(data, "shape", ())[:2])
        except Exception:
            shape = None

        path = getattr(self, "data_path", None)
        header = getattr(self, "header_info", None) or {}
        source = str(header.get("source") or (payload or {}).get("source") or "unknown")
        if path:
            lower_path = str(path).lower()
            if lower_path.endswith(".csv"):
                data_type = "CSV"
            elif lower_path.endswith(".out"):
                data_type = "OUT 数据"
            elif lower_path.endswith(".npy"):
                data_type = "NumPy"
            else:
                data_type = source
        else:
            data_type = source

        stage = getattr(self.shared_data, "current_label", None) or "Raw"
        component = None
        for key in ("component", "selected_component", "rx_component"):
            value = header.get(key)
            if value:
                component = str(value)
                break

        target_header = dict(header)
        if isinstance(payload, dict):
            for key in ("target_response", "target_response_array", "target_response_path", "target_response_npy_path"):
                if key in payload and key not in target_header:
                    target_header[key] = payload[key]
        target_response, target_response_label = self._resolve_target_response_for_autotune(
            data=data,
            path=path,
            header=target_header,
            component=component,
        )

        update = getattr(page, "set_loaded_dataset", None)
        if callable(update):
            update(
                file_path=path,
                data_shape=shape,
                data_type=data_type,
                component=component,
                processing_stage="原始数据" if str(stage) == "Raw" else str(stage),
                source_label=os.path.basename(path) if path else "当前数据",
                data_array=data,
                target_response_array=target_response,
                target_response_label=target_response_label,
            )

    def _clear_manual_roi(self):
        """清除当前手动框选 关注范围。"""
        self._manual_roi_values = None
        page = getattr(self, "page_auto_tune", None)
        if page is not None and hasattr(page, "set_plot_roi_picker_status"):
            try:
                page.set_plot_roi_picker_status(False)
            except Exception:
                pass
        self._update_manual_roi_status()
        if self.data is not None:
            self.plot_data(self.data)

    def _set_manual_roi_pick_enabled(self, enabled: bool):
        """启停主图手动 关注范围 框选。默认关闭，避免日常点击/拖动 B-scan 被 关注范围 逻辑干扰。"""
        self._manual_roi_pick_enabled = bool(enabled)
        if not self._manual_roi_pick_enabled:
            self._remove_drag_roi_preview()
            self._main_press_state = None
        self._sync_auto_tune_roi_picker_state()
        self._update_manual_roi_status()
        if self.data is not None:
            self.plot_data(self.data)

    def _sync_auto_tune_roi_picker_state(self):
        """同步 参数推荐 页面上的 关注范围 框选开关。"""
        page = getattr(self, "page_auto_tune", None)
        if page is None or not hasattr(page, "btn_pick_roi"):
            return
        try:
            blocker = page.btn_pick_roi.blockSignals(True)
            page.btn_pick_roi.setChecked(bool(self._manual_roi_pick_enabled))
            page.btn_pick_roi.blockSignals(blocker)
        except Exception:
            pass
        if hasattr(page, "set_plot_roi_picker_status"):
            try:
                page.set_plot_roi_picker_status(bool(self._manual_roi_pick_enabled))
            except Exception:
                pass

    def _is_manual_roi_pick_enabled(self) -> bool:
        """主图是否允许左键拖拽框选 参数推荐 手动 关注范围。"""
        return bool(getattr(self, "_manual_roi_pick_enabled", False))

    def _update_manual_roi_status(self):
        """同步显示页中的手动 关注范围 状态。"""
        if not hasattr(self, "page_advanced") or self.page_advanced is None:
            return
        if self._manual_roi_values is None:
            if self._is_manual_roi_pick_enabled():
                text = "手动 关注范围: 未设置 · 图上框选已开启"
            else:
                text = "手动 关注范围: 未设置"
            self.page_advanced.set_manual_roi_status(text, False)
            return

        vals = self._manual_roi_values
        self.page_advanced.set_manual_roi_status(
            f"手动 关注范围: X[{vals['dist_start']:.2f}, {vals['dist_end']:.2f}] | Y[{vals['time_start']:.2f}, {vals['time_end']:.2f}]",
            True,
        )
        page = getattr(self, "page_auto_tune", None)
        if page is not None and hasattr(page, "set_manual_roi_from_bounds"):
            try:
                bounds = self._get_manual_roi_bounds()
                if bounds is not None:
                    page.set_manual_roi_from_bounds(bounds, activate=True)
            except Exception:
                pass

    def _reset_auto_tune_state(self, message: str | None = None):
        """重置参数推荐结果摘要。"""
        self._last_auto_tune_result = None
        current_key = self.page_basic.get_current_method_key()
        self.page_basic.set_auto_tune_result_available(False)
        self.page_basic.set_apply_source_hint("当前未生成参数推荐结果。")
        self._last_auto_tune_group_result = None
        self._last_auto_tune_comparison_result = None
        if hasattr(self, "page_auto_tune") and self.page_auto_tune is not None:
            self.page_auto_tune.reset_for_method(current_key, message=message)

    def _build_auto_tune_roi_spec(self, roi_mode: str) -> dict:
        """构建参数推荐所用 关注范围 规格。"""
        mode = str(roi_mode or "prefer_crop")
        if self.data is None:
            return {"mode": "full", "source": "full", "label": "全图"}

        manual_bounds = self._get_manual_roi_bounds()
        if mode != "full" and manual_bounds is not None:
            return {
                "mode": "manual",
                "source": "manual",
                "label": "手动框选 关注范围",
                "bounds": manual_bounds,
            }

        crop_bounds = None
        if (
            mode in {"prefer_crop", "crop"}
            and self.page_advanced.crop_enable_var.isChecked()
        ):
            try:
                time_axis = self._build_time_axis(self.data.shape[0])
                trace_axis = self._build_trace_axis(self.data.shape[1])
                crop_bounds = self._get_crop_bounds(self.data, time_axis, trace_axis)
            except Exception:
                crop_bounds = None

        if crop_bounds:
            return {
                "mode": "crop",
                "source": "crop",
                "label": "当前裁剪区",
                "bounds": crop_bounds,
            }

        if mode == "full":
            return {"mode": "full", "source": "full", "label": "全图"}

        return {"mode": "auto", "source": "auto", "label": "自动 关注范围"}

    def _get_manual_roi_bounds(self) -> dict | None:
        """将主图手动框选 关注范围 转换为当前数据索引边界。"""
        if self.data is None or self._manual_roi_values is None:
            return None

        time_axis = self._build_time_axis(self.data.shape[0])
        trace_axis = self._build_trace_axis(self.data.shape[1])
        vals = self._manual_roi_values
        t0 = min(float(vals["time_start"]), float(vals["time_end"]))
        t1 = max(float(vals["time_start"]), float(vals["time_end"]))
        d0 = min(float(vals["dist_start"]), float(vals["dist_end"]))
        d1 = max(float(vals["dist_start"]), float(vals["dist_end"]))

        return {
            "time_start_idx": self._axis_value_to_index(
                time_axis, t0, self.data.shape[0], "left"
            ),
            "time_end_idx": self._axis_value_to_index(
                time_axis, t1, self.data.shape[0], "right"
            ),
            "dist_start_idx": self._axis_value_to_index(
                trace_axis, d0, self.data.shape[1], "left"
            ),
            "dist_end_idx": self._axis_value_to_index(
                trace_axis, d1, self.data.shape[1], "right"
            ),
        }

    def _target_prior_available_for_no_prior(self) -> bool:
        """判断当前数据是否带有可用目标先验。"""
        header = self.header_info or {}
        ground_truth = header.get("ground_truth")
        if not isinstance(ground_truth, dict):
            return False
        if ground_truth.get("targets"):
            return True
        if ground_truth.get("analysis_roi"):
            return True
        if ground_truth.get("background_rois"):
            return True
        return False

    def _attach_auto_tune_recommendation_label(self, result: dict | None) -> None:
        """为参数推荐结果附加非阻断风险标签。"""
        if not isinstance(result, dict):
            return
        method_key = str(result.get("method_key") or "")
        if not method_key:
            return
        no_prior_policy = self._build_no_prior_qc_policy(
            metrics=self._last_quality_metrics,
            airborne_qc=self._compute_airborne_qc_metrics(),
        )
        trace_count = 0
        if self.data is not None and np.asarray(self.data).ndim == 2:
            trace_count = int(np.asarray(self.data).shape[1])
        label = assess_auto_tune_recommendation_label(
            method_key=method_key,
            selected_params=result.get("recommended_params") or result.get("best_params"),
            metrics=result.get("best_metrics"),
            score=result.get("best_score"),
            no_prior_policy=no_prior_policy,
            trace_count=trace_count,
        )
        result["recommendation_label_info"] = recommendation_label_to_dict(label)

    def _log_auto_tune_recommendation_label(self, result: dict | None) -> None:
        """将参数推荐建议标签写入日志（非阻断）。"""
        if not isinstance(result, dict):
            return
        info = result.get("recommendation_label_info")
        if not isinstance(info, dict):
            return
        label = str(info.get("recommendation_label") or "--")
        severity = str(info.get("severity") or "--")
        method_name = str(result.get("method_name") or result.get("method_key") or "--")
        self._log(
            f"参数推荐 推荐标签: {method_name} | label={label} | severity={severity}"
        )
        for message in list(info.get("user_log_messages") or []):
            self._log(f"参数推荐 提示: {message}")

    def _roi_available_for_no_prior(self) -> bool:
        """判断当前是否存在可用 关注范围。"""
        return self._manual_roi_values is not None

    def _build_auto_tune_recommendation_context(self) -> dict:
        """构建最近一次 参数推荐 推荐标签上下文（用于日志/导出）。"""
        result = self._last_auto_tune_result or {}
        if not isinstance(result, dict) or not result:
            return {}
        context = {
            "method_key": result.get("method_key"),
            "method_name": result.get("method_name"),
            "recommended_params": result.get("recommended_params")
            or result.get("best_params")
            or {},
            "best_score": result.get("best_score"),
        }
        label_info = result.get("recommendation_label_info")
        if isinstance(label_info, dict):
            context["recommendation_label_info"] = dict(label_info)
        return context

    def _build_no_prior_qc_policy(
        self,
        *,
        metrics: dict | None = None,
        airborne_qc: dict | None = None,
    ) -> dict:
        """构建无先验质控策略（UI/导出使用，不触发算法执行）。"""
        if metrics is None:
            metrics = self._last_quality_metrics
        if airborne_qc is None:
            airborne_qc = self._compute_airborne_qc_metrics()

        target_prior_available = self._target_prior_available_for_no_prior()
        roi_available = self._roi_available_for_no_prior()
        metric_alerts = None
        if metrics:
            metric_alerts = {
                key: self._is_metric_alert(key, float(metrics.get(key, 0.0)))
                for key in ["focus_ratio", "hot_pixels", "spikiness", "time_ms"]
            }
        level = derive_no_prior_level(
            quality_metrics=metrics,
            metric_alerts=metric_alerts,
            airborne_qc=airborne_qc,
            runtime_warnings=self._runtime_warnings,
            target_prior_available=target_prior_available,
            roi_available=roi_available,
        )
        policy = build_no_prior_qc_policy(
            no_prior_level=level,
            target_prior_available=target_prior_available,
            roi_available=roi_available,
        )
        return policy_to_dict(policy)

    def _record_no_prior_guard_event(
        self,
        action_id: str,
        decision,
        no_prior_policy: dict,
        *,
        override_used: bool = False,
    ) -> None:
        """记录 no-prior UI guardrail 事件。"""
        event = build_no_prior_guard_event(
            action_id,
            decision,
            no_prior_policy,
            override_used=override_used,
        )
        self._no_prior_guard_events.append(event)
        if len(self._no_prior_guard_events) > 120:
            self._no_prior_guard_events = self._no_prior_guard_events[-120:]
        self._log(
            "No-prior guard: action={action}, decision={decision}, level={level}, override={override}".format(
                action=event.get("action_id"),
                decision=event.get("decision"),
                level=event.get("no_prior_level"),
                override=event.get("override_used"),
            )
        )

    def _enforce_no_prior_action_guard(
        self,
        action_id: str,
        *,
        dialog_title: str,
        allow_override: bool = True,
        show_dialog: bool = True,
        advisory_only: bool = False,
    ) -> bool:
        """执行 no-prior UI guardrail 判定并处理弹窗。"""
        no_prior_policy = self._build_no_prior_qc_policy(
            metrics=self._last_quality_metrics,
            airborne_qc=self._compute_airborne_qc_metrics(),
        )
        guard = evaluate_no_prior_action(
            action_id,
            no_prior_policy,
            allow_override=allow_override,
        )

        if advisory_only:
            if guard.log_event or guard.decision in {"blocked", "requires_confirmation", "caution"}:
                self._record_no_prior_guard_event(
                    action_id,
                    guard,
                    no_prior_policy,
                    override_used=False,
                )
            if guard.warning_text:
                self._log(f"No-prior advisory ({action_id}): {guard.warning_text}")
            return True

        if guard.decision == "blocked":
            self._record_no_prior_guard_event(
                action_id,
                guard,
                no_prior_policy,
                override_used=False,
            )
            if show_dialog:
                message = guard.warning_text or "当前操作已被无先验高风险策略阻断。"
                QMessageBox.warning(self.host, dialog_title, message)
            return False

        if guard.decision == "requires_confirmation":
            if not show_dialog:
                self._record_no_prior_guard_event(
                    action_id,
                    guard,
                    no_prior_policy,
                    override_used=False,
                )
                return False
            message = guard.warning_text or "当前操作需要人工确认。"
            choice = QMessageBox.question(
                self.host,
                dialog_title,
                message + "\n\n是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            override_used = choice == QMessageBox.StandardButton.Yes
            self._record_no_prior_guard_event(
                action_id,
                guard,
                no_prior_policy,
                override_used=override_used,
            )
            return override_used

        if guard.log_event:
            self._record_no_prior_guard_event(
                action_id,
                guard,
                no_prior_policy,
                override_used=False,
            )
        return True

    def _enforce_workbench_no_prior_action_guard(
        self,
        action_id: str,
        *,
        allow_override: bool = True,
        show_dialog: bool = True,
    ) -> bool:
        """旧工作台 no-prior guard 已退役；复用主 guard 保持兼容。"""
        return self._enforce_no_prior_action_guard(
            action_id,
            dialog_title="旧工作台无先验防护（兼容入口）",
            allow_override=allow_override,
            show_dialog=show_dialog,
        )
