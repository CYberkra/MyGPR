#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Field deliverable checks, report, and export page."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.delivery_service import DeliveryService
from core.project_service import ProjectService
from core.user_labels import qc_code_label, severity_label


class DeliveryPage(QWidget):
    package_built = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: ProjectService | None = None
        self.checks: dict | None = None
        layout = QVBoxLayout(self)
        top = QFrame()
        top_layout = QHBoxLayout(top)
        self.status_label = QLabel("未加载项目")
        self.package_name = QLineEdit("项目成果")
        self.check_button = QPushButton("运行成果检查")
        self.build_button = QPushButton("生成成果报告")
        top_layout.addWidget(QLabel("成果报告"))
        top_layout.addWidget(self.status_label)
        top_layout.addStretch(1)
        top_layout.addWidget(self.package_name)
        top_layout.addWidget(self.check_button)
        top_layout.addWidget(self.build_button)
        layout.addWidget(top)
        self.check_table = QTableWidget(0, 4)
        self.check_table.setHorizontalHeaderLabels(["等级", "检查内容", "测线", "说明"])
        self.check_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.check_table, 2)
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        layout.addWidget(self.preview, 1)
        self.check_button.clicked.connect(self.run_checks)
        self.build_button.clicked.connect(lambda: self._build_from_ui())
        self._sync_controls()

    def open_project(self, project: ProjectService) -> None:
        self.project = project
        self.run_checks()
        self._sync_controls()

    def run_checks(self) -> dict:
        if self.project is None:
            raise RuntimeError("未打开项目")
        self.checks = DeliveryService(self.project).run_checks()
        items = self.checks["items"]
        display_items = items or [
            {
                "severity": "info",
                "code": "delivery_checks_passed",
                "line_id": None,
                "message": "成果检查已通过，可以生成成果报告。",
            }
        ]
        self.check_table.setRowCount(len(display_items))
        for row, item in enumerate(display_items):
            code_text = (
                "成果检查通过"
                if item["code"] == "delivery_checks_passed"
                else qc_code_label(item["code"])
            )
            values = (
                severity_label(item["severity"]),
                code_text,
                item.get("line_id") or "--",
                item["message"],
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                if column == 0:
                    cell.setData(Qt.ItemDataRole.UserRole, item["severity"])
                elif column == 1:
                    cell.setData(Qt.ItemDataRole.UserRole, item["code"])
                self.check_table.setItem(row, column, cell)
        summary = self.checks["summary"]
        self.status_label.setText(
            f"阻断 {summary['error_count']} · 待复核 {summary['warning_count']} · "
            f"处理结果 {summary['result_count']} · 目标标注 {summary['interpretation_count']}"
        )
        self.preview.setPlainText(
            "\n".join(
                [
                    f"测线：{summary['line_count']}",
                    f"处理结果：{summary['result_count']}",
                    f"目标标注：{summary['interpretation_count']}",
                    f"阻断错误：{summary['error_count']}",
                    f"待复核警告：{summary['warning_count']}",
                ]
            )
        )
        self._sync_controls()
        return self.checks

    def build_package(self, name: str) -> Path:
        if self.project is None:
            raise RuntimeError("未打开项目")
        package = DeliveryService(self.project).build_package(name)
        self.preview.setPlainText((package / "report.md").read_text(encoding="utf-8"))
        self.status_label.setText(f"已生成：{package.name}")
        self.package_built.emit(str(package))
        self.status_changed.emit(f"交付成果已生成：{package}")
        return package

    def _build_from_ui(self) -> None:
        try:
            self.build_package(self.package_name.text())
        except Exception as exc:
            QMessageBox.critical(self, "交付成果生成失败", str(exc))

    def _sync_controls(self) -> None:
        loaded = self.project is not None
        self.check_button.setEnabled(loaded)
        can_build = bool(
            loaded
            and self.checks
            and self.checks.get("summary", {}).get("error_count", 1) == 0
        )
        self.build_button.setEnabled(can_build)


__all__ = ["DeliveryPage"]
