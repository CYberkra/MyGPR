# -*- coding: utf-8 -*-
"""Report, evidence and diagnostic export helpers for app_qt.GPRGuiQt."""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from core.app_paths import get_logs_dir
from core.auto_tune_comparison_export import (
    export_auto_tune_comparison_artifacts as export_auto_tune_comparison_bundle,
)
from core.evidence_export import export_replay_evidence_bundle as export_replay_evidence_zip
from core.perf_monitor import PerfMonitor
from core.uav_georeference_3d import (
    build_airborne_georeference_3d_payload,
    export_airborne_georeference_3d_bundle,
)

logger = logging.getLogger(__name__)

BASE_DIR = str(Path(__file__).resolve().parents[1])



class MainWindowExportMixin:
    def _perf(self):
        monitor = getattr(self, "perf_monitor", None)
        if monitor is None:
            monitor = PerfMonitor()
            try:
                self.perf_monitor = monitor
            except Exception:
                pass
        return monitor

    def _run_background_export(self, label: str, status_text: str, func, on_done, on_failed, *args, **kwargs):
        """Run a pure export callable in a background thread.

        Tests and headless smoke runs can force synchronous behavior by setting
        ``_sync_export_for_tests=True`` on the window.  Normal GUI use keeps the
        main thread responsive while ZIP/JSON/PNG bundles are written.
        """
        if bool(getattr(self, "_sync_export_for_tests", False)):
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                on_failed(str(exc))
                return None
            on_done(result)
            return None
        try:
            from ui.export_worker import start_export_task
        except Exception as exc:
            self._log(f"后台导出不可用，改为同步导出: {exc}")
            try:
                result = func(*args, **kwargs)
            except Exception as inner_exc:
                on_failed(str(inner_exc))
                return None
            on_done(result)
            return None
        try:
            self.status_label.setText(status_text)
        except Exception:
            pass
        try:
            self._log(f"{label}已开始，导出期间界面保持可用。")
        except Exception:
            pass
        thread, worker = start_export_task(self, func, *args, **kwargs)
        if not hasattr(self, "_background_export_tasks"):
            self._background_export_tasks = []
        task_ref = {"label": label, "thread": thread, "worker": worker}
        self._background_export_tasks.append(task_ref)

        def _cleanup():
            try:
                self._background_export_tasks.remove(task_ref)
            except Exception:
                pass

        def _done(result):
            _cleanup()
            on_done(result)

        def _failed(traceback_text):
            _cleanup()
            on_failed(traceback_text)

        worker.finished.connect(_done)
        worker.failed.connect(_failed)
        return task_ref

    def generate_report(self):
        """生成报告。

        Compatibility wrapper: implementation lives in
        ``ui.report_export_controller.ReportExportController``.
        """
        return self.report_export_controller.generate_report()

    def _build_processing_chain_export(self) -> list[dict]:
        return self.report_export_controller._build_processing_chain_export()

    def _build_report_input_identity(self) -> dict:
        return self.report_export_controller._build_report_input_identity()

    def _build_report_display_params(self) -> dict:
        return self.report_export_controller._build_report_display_params()

    def _build_report_software_version(self, timestamp: str) -> dict:
        return self.report_export_controller._build_report_software_version(timestamp)

    def _build_report_method_registry_version(self, timestamp: str) -> dict:
        return self.report_export_controller._build_report_method_registry_version(timestamp)

    def _build_report_environment_summary(self, timestamp: str) -> str:
        return self.report_export_controller._build_report_environment_summary(timestamp)

    def _build_report_roi_payload(self, timestamp: str) -> dict:
        return self.report_export_controller._build_report_roi_payload(timestamp)

    def _build_report_workflow_payload(
        self,
        timestamp: str,
        *,
        last_run: dict | None,
        processing_chain: list[dict],
        params: dict | None,
    ) -> dict:
        return self.report_export_controller._build_report_workflow_payload(
            timestamp,
            last_run=last_run,
            processing_chain=processing_chain,
            params=params,
        )

    def _build_report_figure_manifest(
        self,
        timestamp: str,
        *,
        image_path: str,
        display_settings: dict,
        bounds: dict | None = None,
    ) -> dict:
        return self.report_export_controller._build_report_figure_manifest(
            timestamp,
            image_path=image_path,
            display_settings=display_settings,
            bounds=bounds,
        )

    def _build_report_audit_note(
        self,
        timestamp: str,
        *,
        package_dir: str,
        claim_boundary: str,
        no_prior_policy: dict,
    ) -> str:
        return self.report_export_controller._build_report_audit_note(
            timestamp,
            package_dir=package_dir,
            claim_boundary=claim_boundary,
            no_prior_policy=no_prior_policy,
        )

    def _write_report_sidecars(self, *args, **kwargs) -> None:
        return self.report_export_controller._write_report_sidecars(*args, **kwargs)

    def _write_branded_report_html(self, *args, **kwargs) -> None:
        return self.report_export_controller._write_branded_report_html(*args, **kwargs)

    def _json_safe(self, value):
        """Return a JSON-serializable copy for Evidence sidecars."""
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def export_record(self):
        """导出记录"""
        text = self.page_quality.get_record_text()
        if not text:
            QMessageBox.information(self, "提示", "记录为空。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存记录", "record.txt", "Text (*.txt);;All files (*)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self._log(f"记录已导出：{path}")

    def export_auto_tune_comparison_artifacts(self):
        """导出人工 baseline vs 自动选参对比证据。"""
        result = self._last_auto_tune_comparison_result
        if result is None:
            QMessageBox.information(
                self, "无对比结果", "请先运行一次“人工/自动对比”再导出。"
            )
            return

        out_dir = self._default_output_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"auto_tune_comparison_{ts}"
        input_ref = self.data_path
        cmap = self._get_colormap()
        artifact_order = [
            "summary_json",
            "manual_png",
            "auto_png",
            "side_by_side_png",
            "params_csv",
            "metrics_csv",
            "report_md",
        ]

        def _export():
            with self._perf().span("export.autotune_comparison_bundle_ms"):
                return export_auto_tune_comparison_bundle(
                    result,
                    out_dir=out_dir,
                    bundle_name=base_name,
                    input_ref=input_ref,
                    notes=[
                        "同一输入、同一 ROI、同一锁定色标下导出的人工 baseline 与自动选参对比证据。",
                        "GPRMAX 正演数据和真实外业数据可复用该导出格式做后续验证。",
                    ],
                    cmap=cmap,
                )

        def _done(bundle):
            artifacts = dict((bundle or {}).get("artifacts") or {})
            if hasattr(self, "page_auto_tune") and self.page_auto_tune is not None:
                self.page_auto_tune.set_evidence_export_result(bundle)
            display_paths = []
            try:
                display_paths = [
                    os.path.relpath(str(artifacts[key]), BASE_DIR)
                    for key in artifact_order
                    if key in artifacts
                ]
            except ValueError:
                display_paths = [
                    str(artifacts[key]) for key in artifact_order if key in artifacts
                ]
            self._log("对比证据已导出: " + "; ".join(display_paths))
            self.status_label.setText("人工/自动对比证据导出完成")
            QMessageBox.information(
                self,
                "导出成功",
                "已导出:\n" + "\n".join(str(artifacts[key]) for key in artifact_order if key in artifacts),
            )

        def _failed(traceback_text):
            logger.error("Failed to export AutoTune comparison bundle: %s", traceback_text)
            self.status_label.setText("人工/自动对比证据导出失败")
            QMessageBox.critical(self, "导出失败", f"对比证据导出失败：\n{traceback_text}")

        return self._run_background_export(
            "人工/自动对比证据导出",
            "正在导出人工/自动对比证据...",
            _export,
            _done,
            _failed,
        )

    def open_log_directory(self):
        """打开日志目录。"""
        log_dir = get_logs_dir()
        os.startfile(log_dir)
        self._log(f"已打开日志目录：{log_dir}")

    def copy_diagnostics(self):
        """复制诊断信息到剪贴板。"""
        lines = [
            f"版本: {self.version_text}",
            f"数据路径: {self.data_path}",
            f"数据尺寸: {self.data.shape if self.data is not None else '--'}",
            f"头信息: {self.header_info}",
            f"道元数据键: {sorted((self.trace_metadata or {}).keys())}",
            f"测线摘要: {self._build_airborne_line_summary_text()}",
            f"辅助文件: {self._sidecar_files}",
            f"当前预设: {self._selected_preset_key}",
            f"上次运行: {self._last_run_summary}",
            f"质量指标: {self._last_quality_metrics}",
            f"运行告警: {self._runtime_warnings}",
            f"AutoTune推荐标签: {self._build_auto_tune_recommendation_context()}",
            f"航空质控: {self._compute_airborne_qc_metrics()}",
            (
                "无先验策略: "
                + str(
                    self._build_no_prior_qc_policy(
                        metrics=self._last_quality_metrics,
                        airborne_qc=self._compute_airborne_qc_metrics(),
                    )
                )
            ),
            f"无先验防护事件: {self._no_prior_guard_events}",
            f"日志文件: {os.path.join(get_logs_dir(), 'gpr_gui.log')}",
        ]
        lines.extend(self._build_airborne_metadata_summary())
        details = self._build_airborne_anomaly_details()
        if details:
            lines.append("航空异常明细:")
            lines.extend([str(item) for item in details])
        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        self._log("诊断信息已复制到剪贴板")

    def _pick_sidecar_file(self, kind: str):
        return self.sidecar_controller._pick_sidecar_file(kind)

    def _set_sidecar_file(self, kind: str, path=None) -> None:
        return self.sidecar_controller._set_sidecar_file(kind, path)

    def _clear_sidecar_file(self, kind: str) -> None:
        return self.sidecar_controller._clear_sidecar_file(kind)

    def _warn_sidecar_ignored(self, kind: str, reason: str) -> None:
        return self.sidecar_controller._warn_sidecar_ignored(kind, reason)

    def _store_trace_timestamps_from_metadata(self, trace_metadata) -> None:
        return self.sidecar_controller._store_trace_timestamps_from_metadata(trace_metadata)

    def _get_trace_timestamps_for_sidecars(self):
        return self.sidecar_controller._get_trace_timestamps_for_sidecars()

    def _read_csv_sidecar_manifest(self, path: str) -> dict | None:
        return self.sidecar_controller._read_csv_sidecar_manifest(path)

    def _merge_manifest_header_info(self, header_info, manifest: dict | None):
        return self.sidecar_controller._merge_manifest_header_info(header_info, manifest)

    def _manifest_sidecar_path(self, path: str, manifest: dict | None, key: str) -> str | None:
        return self.sidecar_controller._manifest_sidecar_path(path, manifest, key)

    def _read_trace_timestamps_sidecar(self, path: str, manifest: dict | None = None):
        return self.sidecar_controller._read_trace_timestamps_sidecar(path, manifest)

    def _discover_sidecar_files_for_csv(self, path: str, manifest: dict | None = None) -> dict[str, str]:
        return self.sidecar_controller._discover_sidecar_files_for_csv(path, manifest)

    def _read_explicit_trace_timestamps_from_csv(self, path: str):
        return self.sidecar_controller._read_explicit_trace_timestamps_from_csv(path)

    def _is_current_data_path(self, path: str) -> bool:
        return self.sidecar_controller._is_current_data_path(path)

    def _build_sidecar_loader_kwargs(self, path: str) -> dict:
        return self.sidecar_controller._build_sidecar_loader_kwargs(path)

    def _extract_airborne_payload_with_optional_sidecars(
        self, raw_data: np.ndarray, header_info, sidecar_kwargs: dict
    ):
        return self.sidecar_controller._extract_airborne_payload_with_optional_sidecars(
            raw_data, header_info, sidecar_kwargs
        )

    def _describe_selected_sidecars(self, sidecar_kwargs: dict) -> str:
        return self.sidecar_controller._describe_selected_sidecars(sidecar_kwargs)

    def _is_optional_sidecar_error(self, exc: ValueError) -> bool:
        return self.sidecar_controller._is_optional_sidecar_error(exc)

    def export_replay_evidence_bundle(self):
        """手动导出处理历史回放证据包。"""
        package = self.shared_data.get_replay_evidence_package()
        if not package:
            QMessageBox.information(
                self, "无可导出证据", "请先导入数据并至少完成一次处理或查看。"
            )
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        default_path = os.path.join(
            self._default_output_dir(),
            f"replay_evidence_{ts}.zip",
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出证据",
            default_path,
            "ZIP 证据包 (*.zip);;所有文件 (*)",
        )
        if not path:
            return

        selected_method_key = None
        selected_method_label = None
        try:
            selected_method_key = self.page_basic.get_current_method_key()
            selected_method_label = self.page_basic.method_combo.currentText()
        except Exception:
            pass

        export_package = dict(package)
        no_prior_policy = self._build_no_prior_qc_policy(
            metrics=self._last_quality_metrics,
            airborne_qc=self._compute_airborne_qc_metrics(),
        )
        export_package["app_context"] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "version": self.version_text,
            "data_path": self.data_path,
            "preset_key": self._selected_preset_key,
            "sidecar_files": dict(self._sidecar_files),
            "selected_method": {
                "key": selected_method_key,
                "label": selected_method_label,
            },
            "last_run_summary": dict(self._last_run_summary or {}),
            "quality_metrics": dict(self._last_quality_metrics or {}),
            "method_param_overrides": dict(self._method_param_overrides),
            "runtime_warnings": list(self._runtime_warnings),
            "auto_tune_recommendation_context": self._build_auto_tune_recommendation_context(),
            "no_prior_qc_policy": no_prior_policy,
            "no_prior_guard_events": list(self._no_prior_guard_events),
        }

        def _export():
            with self._perf().span("export.replay_evidence_zip_ms"):
                return export_replay_evidence_zip(
                    export_package,
                    path,
                    bundle_name=f"replay_evidence_{ts}",
                )

        def _done(result):
            zip_path = str((result or {}).get("zip_path") or path)
            try:
                zip_disp = os.path.relpath(zip_path, BASE_DIR)
            except ValueError:
                zip_disp = zip_path
            self._log(f"处理历史回放证据已导出: {zip_disp}")
            self.status_label.setText("处理历史回放证据导出完成")
            QMessageBox.information(self, "导出成功", f"已导出:\n{zip_path}")

        def _failed(traceback_text):
            logger.error("Failed to export replay evidence bundle: %s", traceback_text)
            self.status_label.setText("处理历史回放证据导出失败")
            QMessageBox.critical(self, "导出失败", f"证据包导出失败：\n{traceback_text}")

        return self._run_background_export(
            "处理历史回放证据导出",
            "正在导出处理历史回放证据...",
            _export,
            _done,
            _failed,
        )

    def export_airborne_georeference_3d_bundle(self):
        """导出三维地理参考预览文件。"""
        payload = self._build_airborne_georeference_3d_plot_payload()
        if isinstance(payload, dict) and "current" in payload:
            current_entry = payload.get("current") or {}
            if isinstance(current_entry, dict) and "preview" in current_entry:
                payload = current_entry
            else:
                payloads_by_lod = (
                    current_entry.get("payloads_by_lod")
                    if isinstance(current_entry, dict)
                    else None
                )
                if isinstance(payloads_by_lod, dict):
                    payload = (
                        payloads_by_lod.get("auto")
                        or payloads_by_lod.get("high")
                        or payloads_by_lod.get("medium")
                        or payloads_by_lod.get("low")
                    )
                else:
                    payload = None
        if not payload:
            QMessageBox.information(
                self, "无可导出三维预览", "请先导入航空数据并确认已有轨迹/高度元数据。"
            )
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        default_path = os.path.join(
            self._default_output_dir(),
            f"uav_georeference_3d_{ts}.vtk",
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出3D地理参考",
            default_path,
            "VTK PolyData (*.vtk);;所有文件 (*)",
        )
        if not path:
            return

        export_payload = self._json_safe(payload)

        def _export():
            with self._perf().span("export.airborne_georeference_3d_ms"):
                return export_airborne_georeference_3d_bundle(export_payload, path)

        def _done(result):
            vtk_path = str((result or {}).get("vtk_path") or path)
            csv_path = str((result or {}).get("csv_path") or Path(path).with_suffix(".csv"))
            json_path = str((result or {}).get("json_path") or Path(path).with_suffix(".json"))
            try:
                vtk_disp = os.path.relpath(vtk_path, BASE_DIR)
                csv_disp = os.path.relpath(csv_path, BASE_DIR)
                json_disp = os.path.relpath(json_path, BASE_DIR)
            except ValueError:
                vtk_disp = vtk_path
                csv_disp = csv_path
                json_disp = json_path
            self._log(f"3D 地理参考已导出: {vtk_disp}; {csv_disp}; {json_disp}")
            self.status_label.setText("3D 地理参考导出完成")
            QMessageBox.information(
                self,
                "导出成功",
                f"已导出:\n{vtk_path}\n{csv_path}\n{json_path}",
            )

        def _failed(traceback_text):
            logger.error("Failed to export airborne georeference 3D bundle: %s", traceback_text)
            self.status_label.setText("3D 地理参考导出失败")
            QMessageBox.critical(self, "导出失败", f"3D 地理参考导出失败：\n{traceback_text}")

        return self._run_background_export(
            "3D 地理参考导出",
            "正在导出 3D 地理参考...",
            _export,
            _done,
            _failed,
        )

    def export_quality_snapshot(self):
        """导出质量快照"""
        if not self._last_quality_metrics:
            QMessageBox.information(
                self, "无质量指标", "请先运行一次处理流程，再导出质量快照。"
            )
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = self._default_output_dir()
        base_name = f"quality_snapshot_{ts}"
        json_path = os.path.join(out_dir, f"{base_name}.json")
        csv_path = os.path.join(out_dir, f"{base_name}.csv")

        selected_method_key = self.page_basic.get_current_method_key()
        selected_method_label = self.page_basic.method_combo.currentText()

        alerts = {
            k: self._is_metric_alert(k, float(self._last_quality_metrics.get(k, 0.0)))
            for k in ["focus_ratio", "hot_pixels", "spikiness", "time_ms"]
        }

        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "data_path": self.data_path,
            "version": self.version_text,
            "preset_key": self._selected_preset_key,
            "line_summary_text": self._build_airborne_line_summary_text(),
            "sidecar_files": dict(self._sidecar_files),
            "airborne_metadata_summary": self._build_airborne_metadata_summary(),
            "airborne_qc": self._compute_airborne_qc_metrics(),
            "airborne_anomaly_details": self._build_airborne_anomaly_details(),
            "selected_method": {
                "key": selected_method_key,
                "label": selected_method_label,
            },
            "metrics": dict(self._last_quality_metrics),
            "thresholds": dict(self._quality_thresholds),
            "alerts": alerts,
            "last_run_summary": dict(self._last_run_summary or {}),
            "method_param_overrides": dict(self._method_param_overrides),
            "runtime_warnings": list(self._runtime_warnings),
            "auto_tune_recommendation_context": self._build_auto_tune_recommendation_context(),
            "no_prior_qc_policy": self._build_no_prior_qc_policy(
                metrics=self._last_quality_metrics,
                airborne_qc=self._compute_airborne_qc_metrics(),
            ),
            "no_prior_guard_events": list(self._no_prior_guard_events),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        rows = []
        for metric in ["focus_ratio", "hot_pixels", "spikiness", "time_ms"]:
            th = self._quality_thresholds.get(metric, {})
            rows.append(
                {
                    "metric": metric,
                    "value": self._last_quality_metrics.get(metric),
                    "threshold_min": th.get("min"),
                    "threshold_max": th.get("max"),
                    "alert": alerts.get(metric, False),
                    "preset_key": self._selected_preset_key,
                    "data_path": self.data_path,
                    "timestamp": payload["timestamp"],
                }
            )
        airborne_qc = payload.get("airborne_qc") or {}
        for metric in [
            "track_length_m",
            "trace_spacing_cv",
            "flight_height_span_m",
            "spacing_outliers",
            "height_outliers",
        ]:
            if metric in airborne_qc:
                rows.append(
                    {
                        "metric": metric,
                        "value": airborne_qc.get(metric),
                        "threshold_min": None,
                        "threshold_max": None,
                        "alert": False,
                        "preset_key": self._selected_preset_key,
                        "data_path": self.data_path,
                        "timestamp": payload["timestamp"],
                    }
                )
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        try:
            json_disp = os.path.relpath(json_path, BASE_DIR)
            csv_disp = os.path.relpath(csv_path, BASE_DIR)
        except ValueError:
            json_disp = json_path
            csv_disp = csv_path
        self._log(f"质量快照已导出: {json_disp}; {csv_disp}")
        self.status_label.setText("质量快照导出完成")
        QMessageBox.information(self, "导出成功", f"已导出:\n{json_path}\n{csv_path}")
