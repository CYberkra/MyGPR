#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Background batch-import progress/result dialog for the field workbench."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QThread, QTimer, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.field_project_operations import BatchImportItemResult, BatchImportSummary, batch_import_line_data
from core.field_project_store import FieldProjectStore


class BatchImportWorker(QObject):
    """Worker object that imports selected files in a QThread."""

    progress = pyqtSignal(int, int, object)
    log = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, store: FieldProjectStore, sources: list[str | Path]) -> None:
        super().__init__()
        self.store = store
        self.sources = [Path(source) for source in sources]
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested)

    def _on_progress(self, current: int, total: int, result: BatchImportItemResult) -> None:
        status = "成功" if result.success else "失败"
        detail = f"[{current}/{total}] {Path(result.source).name} → {result.line_id} {result.name}: {status}；{result.message}"
        if result.success:
            detail += f"；{result.shape_text}；{result.length_m:.2f}m；{result.elapsed_s:.2f}s"
        elif result.diagnosis:
            detail += f"；诊断：{result.diagnosis}"
        self.log.emit(detail)
        self.progress.emit(current, total, result)

    def run(self) -> None:
        try:
            summary = batch_import_line_data(
                self.store,
                self.sources,
                progress_callback=self._on_progress,
                cancel_requested=self._is_cancel_requested,
            )
            self.finished.emit(summary)
        except Exception as exc:  # pragma: no cover - defensive Qt boundary
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class BatchImportProgressDialog(QDialog):
    """Modal progress dialog for large CSV batch import with result table."""

    HEADERS = ["状态", "文件", "测线", "矩阵", "长度(m)", "大小(MB)", "耗时(s)", "诊断/错误"]

    def __init__(self, parent, *, store: FieldProjectStore, sources: list[str | Path], auto_start: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("批量导入测线")
        self.setModal(True)
        self.store = store
        self.sources = [Path(source) for source in sources]
        self.summary: BatchImportSummary | None = None
        self.error_message: str = ""
        self._started = False
        self._thread: QThread | None = None
        self._worker: BatchImportWorker | None = None
        self._results: list[BatchImportItemResult] = []
        self._build_ui()
        if auto_start:
            QTimer.singleShot(0, self._start_worker)

    def _is_import_running(self) -> bool:
        return bool(self._thread is not None and self._thread.isRunning() and self.summary is None and not self.error_message)

    def closeEvent(self, event) -> None:  # Qt close button / Alt+F4 protection
        if self._is_import_running():
            if self._worker is not None:
                self._worker.request_cancel()
            self.status_label.setText("已请求取消；当前文件结束前不能关闭窗口。")
            self.cancel_button.setEnabled(False)
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self) -> None:
        if self._is_import_running():
            if self._worker is not None:
                self._worker.request_cancel()
            self.status_label.setText("已请求取消；当前文件结束前不能关闭窗口。")
            self.cancel_button.setEnabled(False)
            return
        super().reject()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self.status_label = QLabel(f"准备导入 {len(self.sources)} 个文件。")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max(len(self.sources), 1))
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.result_table = QTableWidget(0, len(self.HEADERS))
        self.result_table.setHorizontalHeaderLabels(self.HEADERS)
        self.result_table.setMinimumSize(860, 250)
        self.result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setColumnWidth(0, 58)
        self.result_table.setColumnWidth(1, 190)
        self.result_table.setColumnWidth(2, 90)
        self.result_table.setColumnWidth(3, 88)
        self.result_table.setColumnWidth(4, 78)
        self.result_table.setColumnWidth(5, 80)
        self.result_table.setColumnWidth(6, 76)
        layout.addWidget(self.result_table, 1)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(120)
        self.log_view.appendPlainText("批量导入任务已创建，正在启动后台线程……")
        layout.addWidget(self.log_view)

        buttons = QHBoxLayout()
        self.open_raw_button = QPushButton("打开 raw 目录")
        self.open_raw_button.setEnabled(False)
        self.open_raw_button.clicked.connect(self._open_selected_raw_dir)
        buttons.addWidget(self.open_raw_button)

        self.open_manifest_button = QPushButton("查看 manifest")
        self.open_manifest_button.setEnabled(False)
        self.open_manifest_button.clicked.connect(self._open_selected_manifest)
        buttons.addWidget(self.open_manifest_button)

        self.copy_error_button = QPushButton("复制诊断")
        self.copy_error_button.setEnabled(False)
        self.copy_error_button.clicked.connect(self._copy_selected_diagnosis)
        buttons.addWidget(self.copy_error_button)

        buttons.addStretch(1)
        self.cancel_button = QPushButton("取消后续导入")
        self.cancel_button.clicked.connect(self._cancel_or_close)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.result_table.itemSelectionChanged.connect(self._update_action_buttons)

    def _start_worker(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = QThread(self)
        self._worker = BatchImportWorker(self.store, self.sources)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()
        self.status_label.setText("正在后台导入，请勿关闭项目目录……")

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _on_progress(self, current: int, total: int, result: object) -> None:
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(current)
        if isinstance(result, BatchImportItemResult):
            self._append_result_row(result)
        self.status_label.setText(f"正在导入：{current} / {total}")

    def _append_result_row(self, result: BatchImportItemResult) -> None:
        self._results.append(result)
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)
        values = [
            "成功" if result.success else "失败",
            Path(result.source).name,
            f"{result.line_id} {result.name}",
            result.shape_text,
            f"{result.length_m:.2f}" if result.length_m else "--",
            f"{result.file_size_mb:.3f}" if result.file_size_mb else "--",
            f"{result.elapsed_s:.2f}" if result.elapsed_s else "--",
            result.diagnosis or result.message,
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if col == 0:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.result_table.setItem(row, col, item)
        self.result_table.scrollToBottom()
        self._update_action_buttons()

    def _selected_result(self) -> BatchImportItemResult | None:
        selected = self.result_table.selectionModel().selectedRows() if self.result_table.selectionModel() else []
        if not selected:
            return self._results[-1] if self._results else None
        row = selected[0].row()
        if 0 <= row < len(self._results):
            return self._results[row]
        return None

    def _update_action_buttons(self) -> None:
        result = self._selected_result()
        self.open_raw_button.setEnabled(bool(result and result.raw_dir and Path(result.raw_dir).exists()))
        self.open_manifest_button.setEnabled(bool(result and result.manifest_path and Path(result.manifest_path).exists()))
        self.copy_error_button.setEnabled(bool(result and (result.diagnosis or result.message)))

    def _open_selected_raw_dir(self) -> None:
        result = self._selected_result()
        if result and result.raw_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(result.raw_dir))

    def _open_selected_manifest(self) -> None:
        result = self._selected_result()
        if result and result.manifest_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(result.manifest_path))

    def _copy_selected_diagnosis(self) -> None:
        result = self._selected_result()
        if result is None:
            return
        text = (
            f"文件：{result.source}\n"
            f"测线：{result.line_id} {result.name}\n"
            f"状态：{'成功' if result.success else '失败'}\n"
            f"信息：{result.message}\n"
            f"诊断：{result.diagnosis or '--'}"
        )
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "复制诊断", "已复制当前导入记录的诊断信息。")

    def _on_finished(self, summary: object) -> None:
        self.summary = summary if isinstance(summary, BatchImportSummary) else None
        if self.summary is not None:
            self.status_label.setText(f"批量导入完成：成功 {self.summary.succeeded}/{self.summary.total}，失败 {self.summary.failed}。")
            self.log_view.appendPlainText("\n".join(self.summary.to_log_lines()))
        self.cancel_button.setText("关闭")
        self._update_action_buttons()

    def _on_failed(self, message: str) -> None:
        self.error_message = message
        self.status_label.setText("批量导入任务失败。")
        self.log_view.appendPlainText(message)
        self.cancel_button.setText("关闭")

    def _cancel_or_close(self) -> None:
        if self.summary is not None or self.error_message:
            self.accept() if self.summary is not None else self.reject()
            return
        if self._worker is not None:
            self._worker.request_cancel()
        self.status_label.setText("已请求取消；当前文件结束后跳过后续文件。")
        self.cancel_button.setEnabled(False)


__all__ = ["BatchImportProgressDialog", "BatchImportWorker"]
