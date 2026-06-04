# -*- coding: utf-8 -*-
"""Processing worker lifecycle helpers for app_qt.GPRGuiQt."""

from __future__ import annotations

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QMessageBox

from ui.worker_threads import ProcessingWorker


class MainWindowWorkerMixin:
    def _start_processing_worker(
        self,
        tasks: list,
        run_type: str = "single",
        restore_method_idx: int = None,
        run_label: str = None,
        preset_key: str = None,
        profile_key: str = None,
        execution_mode: str = "sequential",
        run_metadata: dict | None = None,
    ):
        """启动后台处理工作线程"""
        self._current_run_context = {
            "run_type": run_type,
            "restore_method_idx": restore_method_idx,
            "run_label": run_label,
            "preset_key": preset_key,
            "profile_key": profile_key,
            "run_metadata": dict(run_metadata or {}),
        }
        self._cancel_in_flight = False
        try:
            self.page_basic.set_apply_button_state("busy", f"正在执行 {run_type}，请等待。")
        except Exception:
            pass
        self._set_busy(True, text=f"处理中 ({run_type})...")

        self._worker_thread = QThread(self)
        self._worker = ProcessingWorker(
            self.data,
            tasks,
            self.data_path,
            execution_mode=execution_mode,
            header_info=self.header_info,
            trace_metadata=self.trace_metadata,
        )
        self._worker.moveToThread(self._worker_thread)

        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.step_completed.connect(self._on_worker_step_completed)
        self._worker_thread.started.connect(self._worker.run)

        self._worker_thread.start()

    def _on_worker_finished(self, result: dict):
        """工作线程完成回调（delegated to keep app_qt compact）。"""
        return self.processing_worker_controller.on_worker_finished(result)

    def _on_worker_error(self, error_msg: str):
        """工作线程错误回调"""
        self._set_busy(False, text="错误")
        hint = self._build_error_hint(error_msg)
        ctx = self._current_run_context or {}
        payload = self._record_structured_error(
            str(error_msg),
            category="processing",
            context={
                "run_type": ctx.get("run_type"),
                "run_label": ctx.get("run_label"),
                "profile_key": ctx.get("profile_key"),
            },
            log=False,
        )
        self._log(
            f"处理错误: {error_msg}",
            event_type="ERR",
            source="processing",
            context=payload,
        )
        self._log(f"处理建议: {hint}", event_type="WARN", source="processing")
        self.page_basic.set_apply_button_state("error", f"执行失败：{hint}")
        self._set_runtime_summary("状态：处理失败", "danger")
        QMessageBox.critical(self, "处理错误", f"{error_msg}\n\n{hint}")
        self._cleanup_worker()

    def _on_worker_progress(self, current: int, total: int, message: str):
        """工作线程进度回调"""
        self.status_label.setText(f"{message} ({current}/{total})")
        if self._progress_panel is not None:
            self._progress_panel.setVisible(True)
        if self._progress_bar is not None:
            safe_total = max(int(total), 1)
            safe_current = max(0, min(int(current), safe_total))
            self._progress_bar.setRange(0, safe_total)
            self._progress_bar.setValue(safe_current)
            self._progress_bar.setFormat(f"步骤 {safe_current}/{safe_total}")
        self._log(message)

    def _on_worker_step_completed(self, payload: object):
        """处理链路步骤完成后，将该步骤结果作为 B-scan 临时预览显示。"""
        if not isinstance(payload, dict):
            return
        if payload.get("execution_mode") == "independent":
            return
        data = payload.get("data")
        if data is None:
            return

        current = int(payload.get("current") or 0)
        total = int(payload.get("total") or 0)
        method_name = str(payload.get("method_name") or payload.get("method_key") or "处理步骤")
        label = f"实时预览 {current}/{total} · {method_name}" if total else f"实时预览 · {method_name}"

        header_info = dict(payload.get("header_info") or {})
        header_info.setdefault("display_title", label)
        header_info.setdefault("live_preview", True)
        try:
            self._set_display_override(
                data,
                header_info=header_info,
                trace_metadata=payload.get("trace_metadata"),
            )
            if getattr(self, "_plot_title_label", None) is not None:
                self._plot_title_label.setText(f"B-scan / {label}")
            if getattr(self, "_plot_stage_chip", None) is not None:
                self._set_plot_chip_tone(self._plot_stage_chip, label, "info")
            self._set_runtime_summary(f"状态：正在显示 {label}", "info")
            self.plot_data(data)
        except Exception as exc:
            self._log(
                f"实时 B-scan 预览刷新失败: {method_name} | {exc}",
                event_type="WARN",
                source="processing",
            )

    def _cleanup_worker(self):
        """清理工作线程"""
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait(5000)
            self._worker_thread = None
        self._worker = None
        self._cancel_in_flight = False
        self.page_basic.btn_cancel.setEnabled(False)
        if self._progress_bar is not None:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(0)
            self._progress_bar.setFormat("等待开始")
        if self._progress_panel is not None:
            self._progress_panel.setVisible(False)
