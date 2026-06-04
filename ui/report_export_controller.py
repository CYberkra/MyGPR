# -*- coding: utf-8 -*-
"""Report and Evidence package export controller for MyGPR.

This module intentionally keeps report-package generation out of ``app_qt.py``.
It is a thin controller around the main window state: read-only operations are
forwarded to the host window, while report-specific builders and sidecar writers
live here.  The compatibility wrappers in ``GPRGuiQt`` delegate to this class so
older tests and UI code can keep using the previous method names during the
V0.8.x staged refactor.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMessageBox

from core.export_performance import write_json_sidecars, write_text_sidecars
from core.methods_registry import PROCESSING_METHODS, get_public_method_keys
from core.runtime_warnings import format_runtime_warning_text

BASE_DIR = str(Path(__file__).resolve().parents[1])
logger = logging.getLogger(__name__)


class ReportExportController:
    """Generate MyGPR Evidence report packages for a host main window.

    The controller forwards unknown attribute access to ``host`` to keep the
    first extraction low-risk.  Report-specific helpers are implemented on this
    controller, while general GUI/data methods still live on the host.
    """

    def __init__(self, host):
        object.__setattr__(self, "host", host)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name == "host":
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def generate_report(self):
            """生成报告"""
            if self.data is None or self.data_path is None:
                QMessageBox.warning(self.host, "无数据", "请先导入数据。")
                return
            out_dir = self._default_output_dir()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            package_dir = os.path.join(out_dir, f"MyGPR_Evidence_Report_{ts}")
            os.makedirs(package_dir, exist_ok=True)

            # Canonical evidence-package paths.  These stable file names make the
            # package easier to diff, archive, and reference from reports.
            report_path = os.path.join(package_dir, "report.md")
            image_path = os.path.join(package_dir, "bscan_current_600dpi.png")
            html_path = os.path.join(package_dir, "report.html")
            manifest_path = os.path.join(package_dir, "manifest.json")
            evidence_index_path = os.path.join(package_dir, "evidence_index.json")
            workflow_path = os.path.join(package_dir, "workflow.json")
            processing_chain_path = os.path.join(package_dir, "processing_chain.json")
            params_path = os.path.join(package_dir, "params.json")
            display_settings_path = os.path.join(package_dir, "display_settings.json")
            input_identity_path = os.path.join(package_dir, "input_identity.json")
            software_version_path = os.path.join(package_dir, "software_version.json")
            method_registry_version_path = os.path.join(package_dir, "method_registry_version.json")
            environment_summary_path = os.path.join(package_dir, "environment_summary.txt")
            runtime_log_path = os.path.join(package_dir, "runtime_log.txt")
            runtime_events_path = os.path.join(package_dir, "runtime_events.json")
            warnings_path = os.path.join(package_dir, "warnings.json")
            roi_path = os.path.join(package_dir, "roi.json")
            figure_manifest_path = os.path.join(package_dir, "figure_manifest.json")
            claim_boundary_path = os.path.join(package_dir, "claim_boundary.txt")
            audit_note_path = os.path.join(package_dir, "audit_note.md")
            autotune_scoring_path = os.path.join(package_dir, "autotune_scoring_v2.json")

            # Legacy top-level Markdown alias retained for older smoke tests and
            # users accustomed to report_*.md in the selected output directory.
            legacy_report_path = os.path.join(out_dir, f"report_{ts}.md")

            lineage_view_index_before = self._lineage_view_index
            lineage_export_forced_current = lineage_view_index_before is not None
            if lineage_export_forced_current:
                # Report packages must represent the formal current result, not a
                # temporary history-step preview selected from the B-scan stepper.
                self._lineage_view_index = None
                try:
                    self.plot_data(self.data)
                    QApplication.processEvents()
                except Exception as exc:
                    self._log(f"报告导出切回当前结果视图失败: {exc}")

            try:
                with self._perf().span("export.report_figure_600dpi_ms"):
                    self.fig.savefig(image_path, dpi=600, bbox_inches="tight")
            except Exception as e:
                self._log(f"报告截图失败: {e}")

            bounds = None
            try:
                report_data = self._apply_preprocess(
                    np.asarray(self.data, dtype=np.float32)
                )
                report_time_axis = self._build_time_axis(report_data.shape[0])
                report_trace_axis = self._build_trace_axis(report_data.shape[1])
                bounds = self._get_crop_bounds(
                    report_data,
                    report_time_axis,
                    report_trace_axis,
                )
            except Exception as e:
                self._log(f"报告裁剪信息获取失败: {e}")
                bounds = None

            last_run = self._last_run_summary or {}
            method_key = self.page_basic.method_keys[
                self.page_basic.method_combo.currentIndex()
            ]
            method_name = PROCESSING_METHODS[method_key]["name"]
            try:
                params = self.page_basic.get_current_params()
            except Exception as e:
                self._log(f"参数解析失败: {e}")
                params = {}

            lines = []
            lines.append(f"# MyGPR Evidence Report ({ts})")
            lines.append("")
            lines.append("> Markdown + HTML + 600 DPI B-scan figure. This report records processing state, parameters, warnings and claim boundary for review.")
            lines.append("")
            lines.append(f"- Data file: {self.data_path}")
            lines.append("")
            lines.append("## Line Summary")
            lines.append("```")
            lines.append(self._build_airborne_line_summary_text())
            lines.append("```")
            if last_run:
                lines.append(f"- Last run: {last_run.get('label', method_name)}")
                lines.append(f"- Run type: {last_run.get('run_type', 'unknown')}")
                lines.append(f"- Run timestamp: {last_run.get('timestamp', '--')}")
                if last_run.get("preset_key"):
                    lines.append(f"- Preset: {last_run['preset_key']}")
                if last_run.get("profile_key"):
                    lines.append(f"- Profile: {last_run['profile_key']}")
                workflow_summary = last_run.get("workflow_summary") or {}
                if workflow_summary:
                    lines.append("- Workflow stages:")
                    for stage in workflow_summary.get("stages", []):
                        stage_label = stage.get("stage_label") or stage.get(
                            "stage_key", "stage"
                        )
                        method_keys = ", ".join(stage.get("method_keys", []))
                        lines.append(f"  - {stage_label}: {method_keys}")
                    sensor_warnings = workflow_summary.get(
                        "sensor_dependency_warnings", []
                    )
                    if sensor_warnings:
                        lines.append("- Sensor dependencies:")
                        for warning in sensor_warnings:
                            required_any = ", ".join(warning.get("required_any", []))
                            required_text = (
                                f" required_any={required_any}" if required_any else ""
                            )
                            lines.append(
                                "  - [{method_key}] {message}{required_text}".format(
                                    method_key=warning.get("method_key", "method"),
                                    message=warning.get("message", ""),
                                    required_text=required_text,
                                )
                            )
                steps = last_run.get("steps", [])
                if steps:
                    lines.append("- Steps:")
                    for idx, step in enumerate(steps, start=1):
                        step_name = step.get(
                            "method_name", step.get("method_key", f"step-{idx}")
                        )
                        step_params = step.get("params") or {}
                        step_ms = step.get("elapsed_ms")
                        suffix = f" | {step_ms:.1f} ms" if step_ms is not None else ""
                        lines.append(f"  - [{idx}] {step_name}{suffix}")
                        if step_params:
                            lines.append(f"    params: {step_params}")
                if last_run.get("notes"):
                    for note in last_run["notes"]:
                        lines.append(f"- Note: {note}")
                if last_run.get("warnings"):
                    lines.append("- Warnings:")
                    for warning in last_run.get("warnings", []):
                        lines.append(f"  - {format_runtime_warning_text(warning)}")
            else:
                lines.append(f"- Method: {method_name}")
                if params:
                    lines.append(f"- Params: {params}")

            lines.append(f"- Current method selection: {method_name}")
            lines.append(f"- 色图: {self._get_colormap()}")
            lines.append(f"- 显示色标: {self.page_advanced.show_cbar_var.isChecked()}")
            lines.append(f"- 显示网格: {self.page_advanced.show_grid_var.isChecked()}")
            lines.append(
                f"- 显示物理横轴（距离）: {self.page_advanced.show_physical_x_axis_var.isChecked()}"
            )
            lines.append(
                f"- 显示物理纵轴（时间/深度）: {self.page_advanced.show_physical_y_axis_var.isChecked()}"
            )
            lines.append(
                f"- Symmetric stretch: {self.page_advanced.symmetric_var.isChecked()}"
            )
            if self.page_advanced.percentile_var.isChecked():
                lines.append(
                    f"- 百分位拉伸: {self.page_advanced.percentile_var.isChecked()} (low={self.page_advanced.p_low_edit.text()}, high={self.page_advanced.p_high_edit.text()})"
                )
            else:
                lines.append(
                    f"- 百分位拉伸: {self.page_advanced.percentile_var.isChecked()}"
                )
            lines.append(f"- Normalize: {self.page_advanced.normalize_var.isChecked()}")
            lines.append(f"- Demean: {self.page_advanced.demean_var.isChecked()}")
            if bounds:
                lines.append(
                    f"- Crop: time {bounds['time_start']}~{bounds['time_end']} ; distance {bounds['dist_start']}~{bounds['dist_end']}"
                )
            else:
                lines.append("- Crop: disabled")
            quality = self._last_quality_metrics or {}
            if quality:
                lines.append(
                    "- Quality metrics: focus_ratio={focus_ratio:.4f}, hot_pixels={hot_pixels}, spikiness={spikiness:.3f}, time_ms={time_ms:.1f}".format(
                        **quality
                    )
                )
            else:
                lines.append("- Quality metrics: --")
            lines.append("")
            lines.append(f"- Evidence package: {package_dir}")
            lines.append(f"- Screenshot: {image_path}")
            lines.append(f"- HTML preview: {html_path}")
            lines.append(f"- Manifest: {manifest_path}")
            lines.append(f"- Evidence index: {evidence_index_path}")
            lines.append(f"- Workflow JSON: {workflow_path}")
            lines.append(f"- Processing chain: {processing_chain_path}")
            lines.append(f"- Runtime log: {runtime_log_path}")
            lines.append(f"- Runtime events: {runtime_events_path}")
            lines.append(f"- Params JSON: {params_path}")
            lines.append(f"- Display settings: {display_settings_path}")
            lines.append(f"- ROI JSON: {roi_path}")
            lines.append(f"- Figure manifest: {figure_manifest_path}")
            lines.append(f"- Claim boundary: {claim_boundary_path}")
            lines.append(f"- Input identity: {input_identity_path}")
            lines.append(f"- Warnings JSON: {warnings_path}")
            lines.append(f"- Audit note: {audit_note_path}")
            lines.append("")
            lines.append("## Runtime State")
            lines.append(
                f"- Data shape: {self.data.shape if self.data is not None else '--'}"
            )
            airborne_lines = self._build_airborne_metadata_summary()
            if airborne_lines:
                lines.append("- Airborne metadata:")
                for item in airborne_lines:
                    lines.append(f"  - {item}")
            airborne_qc = self._compute_airborne_qc_metrics()
            if airborne_qc:
                lines.append("- Airborne QC:")
                lines.append(f"  - track_length_m: {airborne_qc['track_length_m']:.2f}")
                lines.append(f"  - trace_spacing_cv: {airborne_qc['trace_spacing_cv']:.3f}")
                lines.append(
                    f"  - flight_height_span_m: {airborne_qc['flight_height_span_m']:.2f}"
                )
                lines.append(f"  - spacing_outliers: {airborne_qc['spacing_outliers']}")
                lines.append(f"  - height_outliers: {airborne_qc['height_outliers']}")
                lines.append(
                    "  - alerts: "
                    + (
                        ", ".join(airborne_qc.get("alerts", []))
                        if airborne_qc.get("alerts")
                        else "正常"
                    )
                )
            no_prior_policy = self._build_no_prior_qc_policy(
                metrics=self._last_quality_metrics,
                airborne_qc=airborne_qc,
            )
            lines.append("- No-prior safety policy:")
            lines.append(
                f"  - no_prior_level: {no_prior_policy.get('no_prior_level', '--')}"
            )
            lines.append(
                "  - recommended_initial_policy: "
                + str(no_prior_policy.get("recommended_initial_policy", "--"))
            )
            lines.append(
                "  - manual_review_required: "
                + str(bool(no_prior_policy.get("manual_review_required")))
            )
            lines.append(
                "  - blocked_actions: "
                + (
                    ", ".join(no_prior_policy.get("blocked_actions", []))
                    or "--"
                )
            )
            lines.append(
                "  - claim_boundary: "
                + str(no_prior_policy.get("claim_boundary") or "--")
            )
            auto_tune_ctx = self._build_auto_tune_recommendation_context()
            if auto_tune_ctx:
                lines.append("- AutoTune recommendation label:")
                lines.append(f"  - {auto_tune_ctx}")
            scoring_record = last_run.get("autotune_scoring_record") if isinstance(last_run.get("autotune_scoring_record"), dict) else {}
            if scoring_record:
                try:
                    from core.autotune_scoring_record import summarize_record

                    lines.append("- AutoTune scoring v2:")
                    for item in summarize_record(scoring_record).splitlines():
                        lines.append(f"  - {item}")
                except Exception:
                    lines.append("- AutoTune scoring v2: recorded in sidecar")
            if self._no_prior_guard_events:
                lines.append("- No-prior guard events:")
                for event in self._no_prior_guard_events[-12:]:
                    lines.append(
                        "  - {timestamp} | action={action_id} | decision={decision} | level={no_prior_level} | override={override_used} | reason={reason}".format(
                            **event
                        )
                    )
            anomaly_lines = self._build_airborne_anomaly_details()
            if anomaly_lines:
                lines.append("- Airborne anomaly details:")
                for item in anomaly_lines:
                    lines.append(
                        "  - {type}: idx={index}, distance_m={distance_m}, value={value}, lon={longitude}, lat={latitude}".format(
                            **item
                        )
                    )
            if self._runtime_warnings:
                lines.append("- Runtime warnings:")
                for warning in self._runtime_warnings:
                    lines.append(f"  - {format_runtime_warning_text(warning)}")
            lines.append("")
            lines.append("## Log")
            log_text = self.page_quality.get_record_text().strip()
            if not log_text:
                log_text = self.page_basic.info.toPlainText().strip()
            lines.append("```")
            lines.append(log_text)
            lines.append("```")

            report_text = "\n".join(lines)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            try:
                with open(legacy_report_path, "w", encoding="utf-8") as f:
                    f.write(report_text)
            except Exception as e:
                self._log(f"兼容 Markdown 报告别名写入失败: {e}")

            try:
                self._write_report_sidecars(
                    manifest_path,
                    evidence_index_path,
                    workflow_path,
                    processing_chain_path,
                    runtime_log_path,
                    runtime_events_path,
                    params_path,
                    display_settings_path,
                    claim_boundary_path,
                    input_identity_path,
                    software_version_path,
                    method_registry_version_path,
                    environment_summary_path,
                    warnings_path,
                    roi_path,
                    figure_manifest_path,
                    audit_note_path,
                    autotune_scoring_path,
                    timestamp=ts,
                    package_dir=package_dir,
                    report_path=report_path,
                    html_path=html_path,
                    image_path=image_path,
                    legacy_report_path=legacy_report_path,
                    runtime_log_text=log_text,
                    last_run=last_run,
                    no_prior_policy=no_prior_policy,
                    params=params,
                    lineage_export_forced_current=lineage_export_forced_current,
                )
            except Exception as e:
                logger.exception("Failed to write report sidecars")
                self._log(f"报告 sidecar 生成失败: {e}")

            try:
                with self._perf().span("export.report_html_ms"):
                    self._write_branded_report_html(
                        html_path,
                        ts,
                        image_path,
                        report_text,
                        last_run=last_run,
                        no_prior_policy=no_prior_policy,
                    )
                self._log(f"报告包已保存: {package_dir}")
            except Exception as e:
                logger.exception("Failed to write branded HTML report")
                self._log(f"HTML报告生成失败: {e}")
                self._log(f"报告已保存: {report_path}")

            if lineage_export_forced_current:
                try:
                    self._lineage_view_index = lineage_view_index_before
                    if lineage_view_index_before is not None:
                        self._select_lineage_step(int(lineage_view_index_before))
                    else:
                        self._update_processing_lineage_display()
                except Exception:
                    self._lineage_view_index = None
                    self._update_processing_lineage_display()

    def _build_processing_chain_export(self) -> list[dict]:
            """Build a JSON-safe processing-chain summary for report packages.

            Prefer the processing-lineage controller export so the UI stepper and
            Evidence sidecar describe the same Raw/current/warning/pruned states.
            Fall back to the historical shared-data export if the controller is not
            available, for headless compatibility.
            """
            controller = getattr(self.host, "processing_lineage_controller", None)
            if controller is not None and hasattr(controller, "build_export_steps"):
                try:
                    chain = controller.build_export_steps()
                    if chain:
                        return self._json_safe(chain)
                except Exception:
                    pass
            try:
                entries = self.shared_data.build_result_history_entries()
            except Exception:
                entries = []
            chain: list[dict] = []
            for idx, entry in enumerate(entries or []):
                data = entry.get("data")
                header = entry.get("header_info") or {}
                warnings = header.get("runtime_warnings") or header.get("warnings") or []
                item = {
                    "index": idx,
                    "role": "original" if idx == 0 else ("current" if idx == len(entries) - 1 else "history"),
                    "label": "Raw" if idx == 0 else str(entry.get("label") or f"Step {idx + 1}"),
                    "ui_status": "Raw 原始输入" if idx == 0 else ("当前正式结果" if idx == len(entries) - 1 else "已成功应用"),
                    "shape": list(getattr(data, "shape", []) or []),
                    "method_key": header.get("method_key") or header.get("display_method_key"),
                    "display_title": header.get("display_title"),
                    "warnings": warnings if isinstance(warnings, list) else [str(warnings)],
                    "has_warning": bool(warnings),
                    "has_full_data": data is not None,
                    "memory_state": "stored",
                    "display_only_preview": bool(idx != len(entries) - 1),
                    "exportable": data is not None,
                }
                params = header.get("params") or header.get("method_params")
                if isinstance(params, dict):
                    item["params"] = params
                chain.append(item)
            if not chain and self.data is not None:
                chain.append({
                    "index": 0,
                    "role": "original",
                    "label": "Raw",
                    "ui_status": "Raw 原始输入",
                    "shape": list(getattr(self.data, "shape", []) or []),
                    "method_key": None,
                    "has_warning": False,
                    "has_full_data": True,
                    "memory_state": "stored",
                    "display_only_preview": False,
                    "exportable": True,
                })
            return chain

    def _build_report_input_identity(self) -> dict:
            """Build a conservative input identity block for report sidecars."""
            path_text = str(self.data_path or "")
            payload = {
                "schema": "mygpr.input_identity.v1",
                "path": path_text,
                "basename": os.path.basename(path_text) if path_text else "",
                "exists": False,
                "size_bytes": None,
                "mtime_iso": None,
                "sha256": None,
                "sha256_note": "computed for files up to 256 MiB; omitted for missing, directory or larger inputs",
            }
            if not path_text:
                return payload
            try:
                p = Path(path_text)
                payload["exists"] = p.exists()
                if not p.exists() or not p.is_file():
                    return payload
                stat = p.stat()
                payload["size_bytes"] = int(stat.st_size)
                payload["mtime_iso"] = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
                if stat.st_size <= 256 * 1024 * 1024:
                    h = hashlib.sha256()
                    with p.open("rb") as fh:
                        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                            h.update(chunk)
                    payload["sha256"] = h.hexdigest()
            except Exception as exc:
                payload["identity_warning"] = str(exc)
            return payload

    def _build_report_display_params(self) -> dict:
            """Record display-only settings separately from processing parameters."""
            try:
                return {
                    "colormap": self._get_colormap(),
                    "show_colorbar": bool(self.page_advanced.show_cbar_var.isChecked()),
                    "show_grid": bool(self.page_advanced.show_grid_var.isChecked()),
                    "physical_x_axis": bool(self.page_advanced.show_physical_x_axis_var.isChecked()),
                    "physical_y_axis": bool(self.page_advanced.show_physical_y_axis_var.isChecked()),
                    "symmetric_stretch": bool(self.page_advanced.symmetric_var.isChecked()),
                    "percentile_stretch": bool(self.page_advanced.percentile_var.isChecked()),
                    "percentile_low": self.page_advanced.p_low_edit.text(),
                    "percentile_high": self.page_advanced.p_high_edit.text(),
                    "normalize_display": bool(self.page_advanced.normalize_var.isChecked()),
                    "demean_display": bool(self.page_advanced.demean_var.isChecked()),
                }
            except Exception as exc:
                return {"display_params_warning": str(exc)}

    def _build_report_software_version(self, timestamp: str) -> dict:
            """Build software version metadata for an Evidence package."""
            version_file = Path(BASE_DIR) / "VERSION"
            version_file_text = ""
            try:
                if version_file.exists():
                    version_file_text = version_file.read_text(encoding="utf-8").strip()
            except Exception:
                version_file_text = ""
            return {
                "schema": "mygpr.software_version.v1",
                "timestamp": timestamp,
                "software": "MyGPR",
                "window_version_text": getattr(self, "version_text", "MyGPR"),
                "version_file": version_file_text,
                "base_dir": str(BASE_DIR),
                "python_executable": sys.executable,
            }

    def _build_report_method_registry_version(self, timestamp: str) -> dict:
            """Record a lightweight fingerprint of the processing-method registry."""
            method_keys = list(PROCESSING_METHODS.keys())
            registry_digest_input = json.dumps(
                {
                    key: {
                        "name": PROCESSING_METHODS.get(key, {}).get("name"),
                        "category": PROCESSING_METHODS.get(key, {}).get("category"),
                        "auto_tune_enabled": PROCESSING_METHODS.get(key, {}).get("auto_tune_enabled"),
                    }
                    for key in method_keys
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            return {
                "schema": "mygpr.method_registry_version.v1",
                "timestamp": timestamp,
                "method_count": len(method_keys),
                "method_keys": method_keys,
                "public_method_keys": list(get_public_method_keys()),
                "registry_sha256": hashlib.sha256(registry_digest_input).hexdigest(),
            }

    def _build_report_environment_summary(self, timestamp: str) -> str:
            """Build a human-readable environment summary for reproducibility."""
            lines = [
                "MyGPR Environment Summary",
                f"timestamp: {timestamp}",
                f"platform: {platform.platform()}",
                f"python: {sys.version.replace(chr(10), ' ')}",
                f"python_executable: {sys.executable}",
                f"numpy: {getattr(np, '__version__', '--')}",
                f"pandas: {getattr(pd, '__version__', '--')}",
                f"matplotlib: {getattr(matplotlib, '__version__', '--')}",
                f"base_dir: {BASE_DIR}",
            ]
            try:
                from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR

                lines.append(f"qt: {QT_VERSION_STR}")
                lines.append(f"pyqt: {PYQT_VERSION_STR}")
            except Exception as exc:
                lines.append(f"qt: unavailable ({exc})")
            try:
                import qfluentwidgets  # type: ignore

                lines.append(f"qfluentwidgets: {getattr(qfluentwidgets, '__version__', 'installed')}")
            except Exception as exc:
                lines.append(f"qfluentwidgets: unavailable ({exc})")
            try:
                import pywt  # type: ignore

                lines.append(f"PyWavelets: {getattr(pywt, '__version__', 'installed')}")
            except Exception as exc:
                lines.append(f"PyWavelets: unavailable ({exc})")
            return "\n".join(lines) + "\n"

    def _build_report_roi_payload(self, timestamp: str) -> dict:
            """Record current ROI state separately from processing and display settings."""
            manual_values = dict(self._manual_roi_values or {}) if getattr(self, "_manual_roi_values", None) else None
            manual_bounds = None
            try:
                manual_bounds = self._get_manual_roi_bounds()
            except Exception as exc:
                manual_bounds = {"roi_bounds_warning": str(exc)}
            auto_tune_roi = None
            try:
                if hasattr(self, "page_auto_tune"):
                    roi_mode = self.page_auto_tune.get_auto_tune_roi_mode()
                    auto_tune_roi = self._build_auto_tune_roi_spec(roi_mode)
            except Exception as exc:
                auto_tune_roi = {"auto_tune_roi_warning": str(exc)}
            return {
                "schema": "mygpr.roi.v1",
                "timestamp": timestamp,
                "manual_roi_enabled": bool(getattr(self, "_manual_roi_pick_enabled", False)),
                "manual_roi_values": manual_values,
                "manual_roi_bounds": manual_bounds,
                "auto_tune_roi": auto_tune_roi,
                "roi_claim": "ROI records UI selection state only; it is not ground truth unless explicitly provided by a validated dataset contract.",
            }

    def _build_report_workflow_payload(
            self,
            timestamp: str,
            *,
            last_run: dict | None,
            processing_chain: list[dict],
            params: dict | None,
        ) -> dict:
            """Build workflow metadata for reproducibility and audit review."""
            last_run = last_run or {}
            current_method_key = None
            try:
                current_method_key = self.page_basic.method_keys[self.page_basic.method_combo.currentIndex()]
            except Exception:
                current_method_key = None
            return {
                "schema": "mygpr.workflow.v1",
                "timestamp": timestamp,
                "workflow_source": last_run.get("run_type") or "manual/current",
                "last_run_label": last_run.get("label"),
                "preset_key": last_run.get("preset_key"),
                "profile_key": last_run.get("profile_key"),
                "current_method_key": current_method_key,
                "current_method_params": params or {},
                "last_run_steps": last_run.get("steps", []),
                "workflow_summary": last_run.get("workflow_summary", {}),
                "autotune_scoring_record": last_run.get("autotune_scoring_record", {}) if isinstance(last_run.get("autotune_scoring_record"), dict) else {},
                "autotune_recipe_plan": last_run.get("autotune_recipe_plan", {}) if isinstance(last_run.get("autotune_recipe_plan"), dict) else {},
                "processing_chain_step_count": len(processing_chain),
                "processing_chain_labels": [step.get("label") for step in processing_chain],
            }

    def _build_report_figure_manifest(
            self,
            timestamp: str,
            *,
            image_path: str,
            display_settings: dict,
            bounds: dict | None = None,
        ) -> dict:
            """Describe exported figures and display-only settings."""
            image_name = os.path.basename(image_path)
            item = {
                "id": "bscan_current_600dpi",
                "file": image_name,
                "role": "current_bscan_display_export",
                "dpi": 600,
                "display_only": True,
                "colormap": display_settings.get("colormap"),
                "percentile_stretch": display_settings.get("percentile_stretch"),
                "symmetric_stretch": display_settings.get("symmetric_stretch"),
                "normalize_display": display_settings.get("normalize_display"),
                "demean_display": display_settings.get("demean_display"),
                "crop_bounds": bounds,
            }
            try:
                p = Path(image_path)
                if p.exists():
                    item["size_bytes"] = int(p.stat().st_size)
            except Exception:
                pass
            return {
                "schema": "mygpr.figure_manifest.v1",
                "timestamp": timestamp,
                "figures": [item],
                "figure_claim": "Exported B-scan PNG records the current display state; display-only operations must not be interpreted as processing evidence.",
            }

    def _build_report_audit_note(
            self,
            timestamp: str,
            *,
            package_dir: str,
            claim_boundary: str,
            no_prior_policy: dict,
        ) -> str:
            """Build a compact audit note for the report package."""
            lines = [
                f"# MyGPR Evidence Package Audit Note ({timestamp})",
                "",
                f"- Package directory: `{package_dir}`",
                f"- Data file: `{self.data_path}`",
                f"- Data shape: `{self.data.shape if self.data is not None else '--'}`",
                f"- No-prior level: `{no_prior_policy.get('no_prior_level', '--')}`",
                f"- Manual review required: `{bool(no_prior_policy.get('manual_review_required'))}`",
                "",
                "## Claim boundary",
                "",
                str(claim_boundary),
                "",
                "## Audit reminder",
                "",
                "This package is a MyGPR processing-state export. It is not a standalone proof of target correctness unless paired with validated ground truth or a synthetic paired contract.",
            ]
            return "\n".join(lines) + "\n"

    def _write_report_sidecars(
            self,
            manifest_path: str,
            evidence_index_path: str,
            workflow_path: str,
            processing_chain_path: str,
            runtime_log_path: str,
            runtime_events_path: str,
            params_path: str,
            display_settings_path: str,
            claim_boundary_path: str,
            input_identity_path: str,
            software_version_path: str,
            method_registry_version_path: str,
            environment_summary_path: str,
            warnings_path: str,
            roi_path: str,
            figure_manifest_path: str,
            audit_note_path: str,
            autotune_scoring_path: str,
            *,
            timestamp: str,
            package_dir: str,
            report_path: str,
            html_path: str,
            image_path: str,
            legacy_report_path: str = "",
            runtime_log_text: str = "",
            last_run: dict | None = None,
            no_prior_policy: dict | None = None,
            params: dict | None = None,
            lineage_export_forced_current: bool = False,
        ) -> None:
            """Write structured sidecars for the Evidence-style report package."""
            chain = self._build_processing_chain_export()
            try:
                history_memory = self.shared_data.get_history_memory_summary()
            except Exception as exc:
                history_memory = {"history_memory_warning": str(exc)}
            no_prior_policy = no_prior_policy or {}
            params = params or {}
            display_settings = self._build_report_display_params()
            input_identity = self._build_report_input_identity()
            claim_boundary = no_prior_policy.get(
                "claim_boundary",
                "Report package records current processing state only; it is not ground-truth validation.",
            )
            runtime_events_payload = self._get_structured_runtime_log_payload(timestamp)
            warnings_payload = {
                "schema": "mygpr.runtime_warnings.v1",
                "timestamp": timestamp,
                "warnings": [
                    format_runtime_warning_text(warning)
                    for warning in getattr(self, "_runtime_warnings", [])
                ],
                "raw_warnings": self._json_safe(getattr(self, "_runtime_warnings", [])),
                "no_prior_guard_events": self._json_safe(getattr(self, "_no_prior_guard_events", [])[-50:]),
            }
            autotune_scoring_record = {}
            if isinstance((last_run or {}).get("autotune_scoring_record"), dict):
                autotune_scoring_record = dict((last_run or {}).get("autotune_scoring_record") or {})
            chain_scoring_records = [
                step.get("autotune_scoring_record")
                for step in chain
                if isinstance(step.get("autotune_scoring_record"), dict) and step.get("autotune_scoring_record")
            ]
            scoring_payload = {
                "schema": "mygpr.autotune_scoring_v2_record.v1",
                "timestamp": timestamp,
                "record_source": "last_run_summary" if autotune_scoring_record else ("processing_chain" if chain_scoring_records else "not_available"),
                "autotune_scoring_record": autotune_scoring_record or (chain_scoring_records[-1] if chain_scoring_records else {}),
                "chain_scoring_records": chain_scoring_records,
                "note": "scoring v2 records explain AutoTune recommendation ranking; real no-prior records are proxy scoring, not ground truth validation.",
            }
            processing_chain_payload = {
                "schema": "mygpr.processing_chain.v2",
                "timestamp": timestamp,
                "data_file": self.data_path,
                "current_shape": list(getattr(self.data, "shape", []) or []),
                "lineage_view_index": self._lineage_view_index,
                "lineage_export_forced_current": bool(lineage_export_forced_current),
                "history_memory": history_memory,
                "steps": chain,
            }
            workflow_payload = self._build_report_workflow_payload(
                timestamp,
                last_run=last_run,
                processing_chain=chain,
                params=params,
            )
            software_payload = self._build_report_software_version(timestamp)
            method_registry_payload = self._build_report_method_registry_version(timestamp)
            environment_summary = self._build_report_environment_summary(timestamp)
            roi_payload = self._build_report_roi_payload(timestamp)
            figure_manifest = self._build_report_figure_manifest(
                timestamp,
                image_path=image_path,
                display_settings=display_settings,
            )
            audit_note_text = self._build_report_audit_note(
                timestamp,
                package_dir=package_dir,
                claim_boundary=str(claim_boundary),
                no_prior_policy=no_prior_policy,
            )
            params_payload = {
                "schema": "mygpr.report_params.v2",
                "timestamp": timestamp,
                "processing_params": params,
                "display_settings_file": os.path.basename(display_settings_path),
                "last_run": last_run or {},
                "autotune_scoring_record": autotune_scoring_record,
                "note": "processing_params are algorithm inputs; display-only settings are stored separately and must not be used as processing evidence.",
            }
            display_payload = {
                "schema": "mygpr.display_settings.v1",
                "timestamp": timestamp,
                "display_only": True,
                "settings": display_settings,
            }
            artifact_files = {
                "report_markdown": os.path.basename(report_path),
                "report_html": os.path.basename(html_path),
                "bscan_current_600dpi_png": os.path.basename(image_path),
                "manifest_json": os.path.basename(manifest_path),
                "evidence_index_json": os.path.basename(evidence_index_path),
                "workflow_json": os.path.basename(workflow_path),
                "processing_chain_json": os.path.basename(processing_chain_path),
                "params_json": os.path.basename(params_path),
                "display_settings_json": os.path.basename(display_settings_path),
                "input_identity_json": os.path.basename(input_identity_path),
                "software_version_json": os.path.basename(software_version_path),
                "method_registry_version_json": os.path.basename(method_registry_version_path),
                "environment_summary_txt": os.path.basename(environment_summary_path),
                "runtime_log_txt": os.path.basename(runtime_log_path),
                "runtime_events_json": os.path.basename(runtime_events_path),
                "warnings_json": os.path.basename(warnings_path),
                "roi_json": os.path.basename(roi_path),
                "figure_manifest_json": os.path.basename(figure_manifest_path),
                "claim_boundary_txt": os.path.basename(claim_boundary_path),
                "audit_note_md": os.path.basename(audit_note_path),
                "autotune_scoring_v2_json": os.path.basename(autotune_scoring_path),
            }
            if legacy_report_path:
                artifact_files["legacy_report_markdown_alias"] = os.path.basename(legacy_report_path)
            evidence_index = {
                "schema": "mygpr.evidence_index.v2",
                "timestamp": timestamp,
                "package_role": "mygpr_report_package",
                "package_dir": os.path.basename(package_dir.rstrip(os.sep)),
                "artifacts": [
                    {"key": key, "file": file_name}
                    for key, file_name in artifact_files.items()
                ],
                "claim_boundary_file": os.path.basename(claim_boundary_path),
                "display_only_files": [os.path.basename(image_path), os.path.basename(display_settings_path)],
            }
            manifest = {
                "schema": "mygpr.report_manifest.v4",
                "timestamp": timestamp,
                "software": software_payload,
                "data_file": self.data_path,
                "input_identity": input_identity,
                "current_shape": list(getattr(self.data, "shape", []) or []),
                "artifacts": artifact_files,
                "workflow": workflow_payload,
                "processing_chain_summary": {
                    "step_count": len(chain),
                    "labels": [step.get("label") for step in chain],
                },
                "history_memory": history_memory,
                "lineage_export_forced_current": bool(lineage_export_forced_current),
                "last_run": last_run or {},
                "autotune_scoring_v2": scoring_payload,
                "current_method_params": params,
                "display_settings": display_payload,
                "roi": roi_payload,
                "no_prior_policy": no_prior_policy,
                "claim_boundary": claim_boundary,
            }

            writes = [
                (processing_chain_path, processing_chain_payload),
                (workflow_path, workflow_payload),
                (params_path, params_payload),
                (display_settings_path, display_payload),
                (input_identity_path, input_identity),
                (software_version_path, software_payload),
                (method_registry_version_path, method_registry_payload),
                (runtime_events_path, runtime_events_payload),
                (warnings_path, warnings_payload),
                (roi_path, roi_payload),
                (figure_manifest_path, figure_manifest),
                (autotune_scoring_path, scoring_payload),
                (evidence_index_path, evidence_index),
                (manifest_path, manifest),
            ]
            with self._perf().span("export.report_sidecar_json_ms"):
                write_json_sidecars(writes, json_safe=self._json_safe)
            with self._perf().span("export.report_sidecar_text_ms"):
                write_text_sidecars([
                    (runtime_log_path, runtime_log_text or ""),
                    (environment_summary_path, environment_summary),
                    (claim_boundary_path, str(claim_boundary)),
                    (audit_note_path, audit_note_text),
                ])

    def _write_branded_report_html(
            self,
            html_path: str,
            timestamp: str,
            image_path: str,
            report_text: str,
            *,
            last_run: dict | None = None,
            no_prior_policy: dict | None = None,
        ) -> None:
            """Write a lightweight branded HTML companion for the Markdown report."""
            last_run = last_run or {}
            no_prior_policy = no_prior_policy or {}
            image_name = os.path.basename(image_path)
            data_shape = "--"
            try:
                data_shape = f"{self.data.shape[0]} × {self.data.shape[1]}" if self.data is not None else "--"
            except Exception:
                data_shape = "--"
            method_label = str(last_run.get("label") or "当前方法")
            run_type = str(last_run.get("run_type") or "manual/current")
            no_prior_level = str(no_prior_policy.get("no_prior_level") or "--")
            manual_review = "是" if bool(no_prior_policy.get("manual_review_required")) else "否"
            claim_boundary = str(
                no_prior_policy.get("claim_boundary")
                or "仅作为当前数据处理状态与参数记录，不等同于 ground truth 验证。"
            )

            def esc(value) -> str:
                return html.escape(str(value), quote=True)

            html_doc = f"""<!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>MyGPR Evidence Report {esc(timestamp)}</title>
      <style>
        :root {{ --bg:#f6f8fc; --card:#ffffff; --border:#e3eaf3; --text:#172033; --muted:#64748b; --primary:#13a6a4; --soft:#e7faf7; --warn:#f59e0b; }}
        body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; }}
        .wrap {{ max-width:1120px; margin:0 auto; padding:32px 28px 48px; }}
        .hero {{ display:flex; gap:18px; align-items:center; background:linear-gradient(135deg,#ffffff,#f0fbfa); border:1px solid var(--border); border-radius:24px; padding:22px 24px; box-shadow:0 18px 46px rgba(15,23,42,.06); }}
        .mark {{ width:62px; height:62px; border-radius:18px; background:var(--soft); border:1px solid #bdebd1; display:grid; place-items:center; color:var(--primary); font-weight:800; font-size:18px; }}
        h1 {{ margin:0; font-size:25px; letter-spacing:.2px; }}
        .sub {{ color:var(--muted); margin-top:4px; }}
        .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0; }}
        .card {{ background:var(--card); border:1px solid var(--border); border-radius:18px; padding:14px 16px; }}
        .label {{ color:var(--muted); font-size:12px; margin-bottom:4px; }}
        .value {{ font-weight:800; overflow-wrap:anywhere; }}
        .figure {{ background:var(--card); border:1px solid var(--border); border-radius:22px; padding:16px; margin:18px 0; }}
        .figure img {{ width:100%; height:auto; border-radius:14px; border:1px solid var(--border); background:#fff; }}
        .boundary {{ border-left:4px solid var(--warn); background:#fff7e8; border-radius:14px; padding:12px 14px; margin:18px 0; }}
        pre {{ white-space:pre-wrap; word-break:break-word; background:#111820; color:#eaf0f8; padding:18px; border-radius:18px; overflow:auto; }}
        @media (max-width:860px) {{ .grid {{ grid-template-columns:1fr 1fr; }} .hero {{ align-items:flex-start; }} }}
      </style>
    </head>
    <body>
      <main class="wrap">
        <section class="hero">
          <div class="mark">GPR</div>
          <div>
            <h1>MyGPR Evidence Report</h1>
            <div class="sub">Generated {esc(timestamp)} · Markdown / HTML / 600 DPI B-scan figure</div>
          </div>
        </section>

        <section class="grid">
          <div class="card"><div class="label">数据尺寸</div><div class="value">{esc(data_shape)}</div></div>
          <div class="card"><div class="label">最近运行</div><div class="value">{esc(method_label)}</div></div>
          <div class="card"><div class="label">运行类型</div><div class="value">{esc(run_type)}</div></div>
          <div class="card"><div class="label">No-prior 等级</div><div class="value">{esc(no_prior_level)}</div></div>
        </section>

        <section class="figure">
          <div class="label">B-scan Figure</div>
          <img src="{esc(image_name)}" alt="B-scan figure">
        </section>

        <section class="boundary">
          <strong>Claim boundary</strong><br>
          {esc(claim_boundary)}<br>
          <span class="label">Manual review required: {esc(manual_review)}</span>
        </section>

        <section class="card">
          <div class="label">Full Markdown Record</div>
          <pre>{esc(report_text)}</pre>
        </section>
      </main>
    </body>
    </html>
    """
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_doc)
