# -*- coding: utf-8 -*-
"""Quality metrics and no-prior guard helpers for app_qt.GPRGuiQt."""

from __future__ import annotations

import csv
import html
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from core.preset_profiles import (
    build_profile_workflow_summary,
    compute_quality_metrics,
)


class MainWindowQualityMixin:
    def _set_quality_metrics(self, metrics: dict):
        """设置质量指标；主图区质量摘要已移除，质量页仍同步结构化 QC。"""
        self._last_quality_metrics = metrics
        airborne_qc = self._compute_airborne_qc_metrics()
        no_prior_policy = self._build_no_prior_qc_policy(
            metrics=metrics,
            airborne_qc=airborne_qc,
        )
        self._last_no_prior_qc_policy = no_prior_policy

        # V0.7.1: 兼容旧质量摘要控件，但不再要求主图区创建这些 QLabel。
        if all(hasattr(self, name) for name in (
            "quality_focus_label",
            "quality_hot_label",
            "quality_spiky_label",
            "quality_time_label",
            "quality_alert_label",
        )):
            if metrics is None:
                self.quality_focus_label.setText("focus_ratio: --")
                self.quality_hot_label.setText("hot_pixels: --")
                self.quality_spiky_label.setText("spikiness: --")
                self.quality_time_label.setText("time_ms: --")
                self.quality_alert_label.setText("阈值状态: --")
            else:
                self.quality_focus_label.setText(
                    f"focus_ratio: {metrics.get('focus_ratio', 0):.4f}"
                )
                self.quality_hot_label.setText(
                    f"hot_pixels: {metrics.get('hot_pixels', 0)}"
                )
                self.quality_spiky_label.setText(
                    f"spikiness: {metrics.get('spikiness', 0):.3f}"
                )
                self.quality_time_label.setText(f"time_ms: {metrics.get('time_ms', 0):.1f}")
                alerts = []
                for k in ["focus_ratio", "hot_pixels", "spikiness", "time_ms"]:
                    if self._is_metric_alert(k, float(metrics.get(k, 0))):
                        alerts.append(k)
                self.quality_alert_label.setText(
                    f"阈值状态: {', '.join(alerts) if alerts else '正常'}"
                )

        if airborne_qc:
            if all(hasattr(self, name) for name in (
                "quality_track_len_label",
                "quality_spacing_label",
                "quality_height_label",
                "quality_airborne_alert_label",
            )):
                self.quality_track_len_label.setText(
                    f"track_length_m: {airborne_qc['track_length_m']:.2f}"
                )
                self.quality_spacing_label.setText(
                    f"trace_spacing_cv: {airborne_qc['trace_spacing_cv']:.3f}"
                )
                self.quality_height_label.setText(
                    f"flight_height_span_m: {airborne_qc['flight_height_span_m']:.2f}"
                )
                self.quality_airborne_alert_label.setText(
                    "airborne_alerts: "
                    + (
                        ", ".join(airborne_qc.get("alerts", []))
                        if airborne_qc.get("alerts")
                        else "正常"
                    )
                )
            if hasattr(self, "page_quality") and self.page_quality is not None:
                self.page_quality.set_airborne_qc_summary(
                    self._build_airborne_qc_summary_text()
                )
                self.page_quality.set_airborne_qc_visualization(
                    self._build_airborne_qc_plot_payload()
                )
                self.page_quality.set_airborne_trajectory_visualization(
                    self._build_airborne_trajectory_plot_payload()
                )
                if getattr(self, "page_terrain3d", None) is not None:
                    self.page_terrain3d.set_airborne_georeference_3d_visualization(
                        self._build_airborne_georeference_3d_plot_payload()
                    )
                self.page_quality.set_airborne_anomaly_details(
                    self._build_airborne_anomaly_text()
                )
        else:
            if all(hasattr(self, name) for name in (
                "quality_track_len_label",
                "quality_spacing_label",
                "quality_height_label",
                "quality_airborne_alert_label",
            )):
                self.quality_track_len_label.setText("track_length_m: --")
                self.quality_spacing_label.setText("trace_spacing_cv: --")
                self.quality_height_label.setText("flight_height_span_m: --")
                self.quality_airborne_alert_label.setText("airborne_alerts: --")
            if hasattr(self, "page_quality") and self.page_quality is not None:
                self.page_quality.set_airborne_qc_summary("")
                self.page_quality.set_airborne_qc_visualization(None)
                self.page_quality.set_airborne_trajectory_visualization(None)
                if getattr(self, "page_terrain3d", None) is not None:
                    self.page_terrain3d.set_airborne_georeference_3d_visualization(None)
                self.page_quality.set_airborne_anomaly_details("")

        if all(hasattr(self, name) for name in (
            "no_prior_level_label",
            "no_prior_policy_label",
            "no_prior_blocked_label",
        )):
            blocked_actions = ", ".join(no_prior_policy.get("blocked_actions", []))
            self.no_prior_level_label.setText(
                f"no_prior_level: {no_prior_policy.get('no_prior_level', '--')}"
            )
            self.no_prior_policy_label.setText(
                "no_prior_policy: "
                + str(no_prior_policy.get("recommended_initial_policy", "--"))
            )
            self.no_prior_blocked_label.setText(
                "no_prior_blocked_actions: " + (blocked_actions or "--")
            )

    def _target_prior_available_for_no_prior(self) -> bool:
        return self.autotune_sync_controller._target_prior_available_for_no_prior()

    def _attach_auto_tune_recommendation_label(self, result: dict | None) -> None:
        return self.autotune_sync_controller._attach_auto_tune_recommendation_label(result)

    def _log_auto_tune_recommendation_label(self, result: dict | None) -> None:
        return self.autotune_sync_controller._log_auto_tune_recommendation_label(result)

    def _roi_available_for_no_prior(self) -> bool:
        return self.autotune_sync_controller._roi_available_for_no_prior()

    def _build_auto_tune_recommendation_context(self) -> dict:
        return self.autotune_sync_controller._build_auto_tune_recommendation_context()

    def _build_no_prior_qc_policy(
        self,
        *,
        metrics: dict | None = None,
        airborne_qc: dict | None = None,
    ) -> dict:
        return self.autotune_sync_controller._build_no_prior_qc_policy(
            metrics=metrics, airborne_qc=airborne_qc
        )

    def _record_no_prior_guard_event(
        self,
        action_id: str,
        decision,
        no_prior_policy: dict,
        *,
        override_used: bool = False,
    ) -> None:
        return self.autotune_sync_controller._record_no_prior_guard_event(
            action_id, decision, no_prior_policy, override_used=override_used
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
        return self.autotune_sync_controller._enforce_no_prior_action_guard(
            action_id,
            dialog_title=dialog_title,
            allow_override=allow_override,
            show_dialog=show_dialog,
            advisory_only=advisory_only,
        )

    def _enforce_workbench_no_prior_action_guard(
        self,
        action_id: str,
        *,
        allow_override: bool = True,
        show_dialog: bool = True,
    ) -> bool:
        return self.autotune_sync_controller._enforce_workbench_no_prior_action_guard(
            action_id, allow_override=allow_override, show_dialog=show_dialog
        )

    def _classify_method_guard_action(
        self,
        method_key: str,
        params: dict,
    ) -> str | None:
        """将方法执行映射到 no-prior guard action。"""
        if method_key == "agcGain":
            return "AGC_display_only"
        if method_key in {
            "energy_decay_gain",
            "sec_gain",
            "compensatingGain",
            "amplitude_scale",
        }:
            return "conservative_energy_decay_gain_display"

        if method_key in {"subtracting_average_2D", "median_background_2D", "running_average_2D"}:
            trace_count = int(self.data.shape[1]) if self.data is not None else 0
            ntraces = int(params.get("ntraces", 0) or 0)
            if trace_count > 0 and ntraces >= max(41, int(trace_count * 0.5)):
                return "background_suppression_aggressive"
            return "background_suppression_conservative"

        if method_key in {"fk_filter", "ccbs", "svd_bg"}:
            return "background_suppression_conservative"
        return None

    def _compute_airborne_qc_metrics(self) -> dict | None:
        """基于当前每道元数据计算第一批航空 QC 指标。"""
        header = self.header_info or {}
        meta = self.trace_metadata or {}
        if not header.get("has_airborne_metadata"):
            return None

        distance = np.asarray(meta.get("trace_distance_m", []), dtype=np.float64)
        flight = np.asarray(meta.get("flight_height_m", []), dtype=np.float64)
        alignment_status = np.asarray(meta.get("alignment_status", []), dtype="<U16")
        height_confidence = np.asarray(meta.get("height_confidence", []), dtype=np.float64)
        spacing = (
            np.diff(distance) if distance.size > 1 else np.array([], dtype=np.float64)
        )
        spacing_mean = float(np.mean(spacing)) if spacing.size else 0.0
        spacing_std = float(np.std(spacing)) if spacing.size else 0.0
        spacing_cv = spacing_std / spacing_mean if spacing_mean > 1e-9 else 0.0
        flight_span = float(np.max(flight) - np.min(flight)) if flight.size else 0.0
        alignment_extrapolated_traces = int(
            np.count_nonzero(alignment_status == "extrapolated")
        )
        alignment_resampled_traces = int(np.count_nonzero(alignment_status == "resampled"))
        confidence_valid = height_confidence[np.isfinite(height_confidence)]
        height_confidence_min = (
            float(np.min(confidence_valid)) if confidence_valid.size else None
        )
        height_confidence_mean = (
            float(np.mean(confidence_valid)) if confidence_valid.size else None
        )
        height_confidence_low_traces = int(np.count_nonzero(confidence_valid < 0.5))
        alignment_extrapolated_indices = np.flatnonzero(
            alignment_status == "extrapolated"
        ).tolist()
        height_confidence_low_indices = np.flatnonzero(
            np.isfinite(height_confidence) & (height_confidence < 0.5)
        ).tolist()
        spacing_outliers = (
            int(np.sum(np.abs(spacing - spacing_mean) > max(spacing_std * 2.5, 0.5)))
            if spacing.size
            else 0
        )
        spacing_outlier_indices = (
            np.flatnonzero(
                np.abs(spacing - spacing_mean) > max(spacing_std * 2.5, 0.5)
            ).tolist()
            if spacing.size
            else []
        )
        flight_outliers = (
            int(
                np.sum(
                    np.abs(flight - np.median(flight)) > max(np.std(flight) * 2.5, 0.5)
                )
            )
            if flight.size
            else 0
        )
        flight_outlier_indices = (
            np.flatnonzero(
                np.abs(flight - np.median(flight)) > max(np.std(flight) * 2.5, 0.5)
            ).tolist()
            if flight.size
            else []
        )
        alerts = []
        if spacing_cv > 0.25:
            alerts.append("spacing_cv_high")
        if flight_span > 3.0:
            alerts.append("flight_span_high")
        if spacing_outliers > 0:
            alerts.append(f"spacing_outliers={spacing_outliers}")
        if flight_outliers > 0:
            alerts.append(f"height_outliers={flight_outliers}")
        if alignment_extrapolated_traces > 0:
            alerts.append(f"alignment_extrapolated={alignment_extrapolated_traces}")
        if height_confidence_low_traces > 0:
            alerts.append(f"height_confidence_low={height_confidence_low_traces}")
        return {
            "track_length_m": float(header.get("track_length_m", 0.0)),
            "trace_spacing_cv": float(spacing_cv),
            "flight_height_span_m": float(flight_span),
            "spacing_outliers": spacing_outliers,
            "height_outliers": flight_outliers,
            "spacing_outlier_indices": spacing_outlier_indices,
            "height_outlier_indices": flight_outlier_indices,
            "alignment_extrapolated_traces": alignment_extrapolated_traces,
            "alignment_resampled_traces": alignment_resampled_traces,
            "alignment_extrapolated_indices": alignment_extrapolated_indices,
            "height_confidence_min": height_confidence_min,
            "height_confidence_mean": height_confidence_mean,
            "height_confidence_low_traces": height_confidence_low_traces,
            "height_confidence_low_indices": height_confidence_low_indices,
            "alignment_status_available": bool(alignment_status.size),
            "height_confidence_available": bool(confidence_valid.size),
            "alerts": alerts,
        }

    def _set_last_run_summary(
        self,
        run_type: str,
        label: str,
        steps: list,
        preset_key: str = None,
        profile_key: str = None,
        notes: list | None = None,
        warnings: list | None = None,
        autotune_scoring_record: dict | None = None,
        autotune_recipe_plan: dict | None = None,
    ):
        """记录最近一次真实执行摘要，供报告和诊断使用。"""
        workflow_summary = None
        if profile_key:
            try:
                workflow_summary = build_profile_workflow_summary(str(profile_key))
            except KeyError:
                workflow_summary = None
        self._last_run_summary = {
            "run_type": run_type,
            "label": label,
            "steps": steps,
            "preset_key": preset_key,
            "profile_key": profile_key,
            "notes": notes or [],
            "warnings": warnings or [],
            "autotune_scoring_record": dict(autotune_scoring_record or {}),
            "autotune_recipe_plan": dict(autotune_recipe_plan or {}),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "no_prior_qc_policy": self._build_no_prior_qc_policy(
                metrics=self._last_quality_metrics,
                airborne_qc=self._compute_airborne_qc_metrics(),
            ),
            "no_prior_guard_events": list(self._no_prior_guard_events),
        }
        if workflow_summary:
            self._last_run_summary["workflow_summary"] = workflow_summary

    def _is_metric_alert(self, metric: str, value: float) -> bool:
        """检查指标是否超出阈值"""
        thresholds = self._quality_thresholds.get(metric, {})
        min_v = thresholds.get("min")
        max_v = thresholds.get("max")
        if min_v is not None and value < min_v:
            return True
        if max_v is not None and value > max_v:
            return True
        return False

    def _save_pipeline_comparison(self, outputs: list) -> str | None:
        """导出默认/推荐流程对比图（Raw / 中间关键步 / Final）"""
        try:
            if self.original_data is None or self.data is None:
                return None
            out_dir = self._default_output_dir()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = os.path.join(out_dir, f"pipeline_compare_{ts}.png")

            raw = np.asarray(self.original_data)
            final = np.asarray(self.data)

            # 选两个中间关键步骤：第1步、倒数第2步（若存在）
            mids = []
            if len(outputs) >= 1:
                mids.append(
                    (
                        outputs[0].get("method_name", "Step1"),
                        np.asarray(outputs[0].get("data")),
                    )
                )
            if len(outputs) >= 3:
                mids.append(
                    (
                        outputs[-2].get("method_name", "StepN-1"),
                        np.asarray(outputs[-2].get("data")),
                    )
                )

            items = [("Raw", raw)] + mids + [("Final", final)]
            n = len(items)
            fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 3.6), dpi=150)
            if n == 1:
                axs = [axs]
            for ax, (title, arr) in zip(axs, items):
                arr = np.asarray(arr)
                vmax = float(np.nanmax(np.abs(arr))) if arr.size else 1.0
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1.0
                ax.imshow(
                    arr,
                    cmap="seismic",
                    aspect="auto",
                    vmin=-vmax,
                    vmax=vmax,
                    origin="upper",
                )
                ax.set_title(title)
                ax.set_xlabel("Trace")
                ax.set_ylabel("Sample")
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)
            return out_path
        except Exception as e:
            self._log(f"对比图导出失败: {e}")
            return None
