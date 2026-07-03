#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gprMax simulation/validation workspace page.

This page deliberately exposes the existing gprMax campaign backend through a
safe UI boundary: dry-run validation, scene readiness review, and reproducible
command preview.  It does not launch long-running gprMax simulations from the
GUI; users can copy the generated command into a prepared gprMax shell.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.gprmax_campaign import Campaign, CampaignValidationResult, load_campaign_yaml, validate_campaign
from core.gprmax_campaign.schema import VALIDATION_INVALID, VALIDATION_READY, VALIDATION_WARNING


class SimulationValidationPage(QWidget):
    """Read-only gprMax campaign validation and run-command planning UI."""

    status_changed = pyqtSignal(str)
    campaign_loaded = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.campaign: Campaign | None = None
        self.validation_result: CampaignValidationResult | None = None
        self._selected_scene_id: str | None = None
        self._gpu_device_parse_error: str | None = None
        self._build_ui()
        self._sync_controls()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top = QFrame()
        top.setObjectName("simulationValidationToolbar")
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(10, 8, 10, 8)
        top_layout.setSpacing(6)

        title_row = QHBoxLayout()
        title = QLabel("仿真与验证")
        title.setObjectName("simulationValidationTitle")
        self.status_label = QLabel("未加载 campaign YAML")
        self.status_label.setObjectName("simulationValidationStatus")
        title_row.addWidget(title)
        title_row.addWidget(self.status_label)
        title_row.addStretch(1)
        top_layout.addLayout(title_row)

        path_row = QHBoxLayout()
        self.campaign_path_edit = QLineEdit()
        self.campaign_path_edit.setPlaceholderText("选择 gprMax campaign YAML（仅干跑验证，不自动执行仿真）")
        self.browse_button = QPushButton("选择 YAML")
        self.validate_button = QPushButton("载入并干跑检查")
        self.copy_command_button = QPushButton("复制命令")
        path_row.addWidget(QLabel("Campaign"))
        path_row.addWidget(self.campaign_path_edit, 1)
        path_row.addWidget(self.browse_button)
        path_row.addWidget(self.validate_button)
        path_row.addWidget(self.copy_command_button)
        top_layout.addLayout(path_row)

        options_row = QHBoxLayout()
        self.scene_combo = QComboBox()
        self.variant_combo = QComboBox()
        self.variant_combo.addItem("原始含目标 raw_with_target", "raw_with_target")
        self.variant_combo.addItem("背景 background_only", "background_only")
        self.num_runs_spin = QSpinBox()
        self.num_runs_spin.setRange(1, 9999)
        self.num_runs_spin.setValue(1)
        self.num_runs_spin.setToolTip("传给 gprMax 的 -n N；1 表示单次运行。")
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(0, 24 * 60 * 60)
        self.timeout_spin.setValue(0)
        self.timeout_spin.setSuffix(" s")
        self.timeout_spin.setToolTip("0 表示不添加 --timeout-seconds。")
        self.gpu_check = QCheckBox("GPU")
        self.gpu_devices_edit = QLineEdit()
        self.gpu_devices_edit.setPlaceholderText("设备，如 0 或 0 1")
        self.gpu_devices_edit.setMaximumWidth(130)
        self.gprmax_python_edit = QLineEdit()
        self.gprmax_python_edit.setPlaceholderText("可选：外部 gprMax Python")
        self.gprmax_python_edit.setMinimumWidth(180)
        options_row.addWidget(QLabel("场景"))
        options_row.addWidget(self.scene_combo, 1)
        options_row.addWidget(QLabel("变体"))
        options_row.addWidget(self.variant_combo)
        options_row.addWidget(QLabel("N"))
        options_row.addWidget(self.num_runs_spin)
        options_row.addWidget(QLabel("超时"))
        options_row.addWidget(self.timeout_spin)
        options_row.addWidget(self.gpu_check)
        options_row.addWidget(self.gpu_devices_edit)
        options_row.addWidget(self.gprmax_python_edit, 1)
        top_layout.addLayout(options_row)
        layout.addWidget(top)

        body = QHBoxLayout()
        body.setContentsMargins(10, 10, 10, 10)
        body.setSpacing(10)

        left_panel = QFrame()
        left_panel.setObjectName("simulationValidationPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("场景就绪状态"))
        self.scene_table = QTableWidget(0, 4)
        self.scene_table.setHorizontalHeaderLabels(["场景", "状态", "问题数", "标签"])
        self.scene_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.scene_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.scene_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.scene_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.scene_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        left_layout.addWidget(self.scene_table, 1)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        left_layout.addWidget(self.summary_text)
        body.addWidget(left_panel, 3)

        right_panel = QFrame()
        right_panel.setObjectName("simulationValidationPanel")
        right_layout = QVBoxLayout(right_panel)
        boundary = QLabel(
            "安全边界：本页只做 campaign 干跑验证、场景审查和可复现命令预览；"
            "不会从 GUI 直接启动长时间 gprMax，也不会写入 Evidence 或修改模型。"
        )
        boundary.setWordWrap(True)
        boundary.setObjectName("simulationValidationBoundary")
        right_layout.addWidget(boundary)
        right_layout.addWidget(QLabel("命令预览"))
        self.command_preview = QTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMinimumHeight(96)
        right_layout.addWidget(self.command_preview)
        right_layout.addWidget(QLabel("详情 / 问题"))
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        right_layout.addWidget(self.details_text, 1)
        body.addWidget(right_panel, 4)
        layout.addLayout(body, 1)

        self.browse_button.clicked.connect(self.choose_campaign)
        self.validate_button.clicked.connect(self.load_campaign_from_ui_safely)
        self.copy_command_button.clicked.connect(self.copy_command_preview)
        self.campaign_path_edit.returnPressed.connect(self.load_campaign_from_ui_safely)
        self.scene_combo.currentIndexChanged.connect(self._scene_combo_changed)
        self.variant_combo.currentIndexChanged.connect(self.refresh_command_preview)
        self.num_runs_spin.valueChanged.connect(self.refresh_command_preview)
        self.timeout_spin.valueChanged.connect(self.refresh_command_preview)
        self.gpu_check.stateChanged.connect(self.refresh_command_preview)
        self.gpu_devices_edit.textChanged.connect(self.refresh_command_preview)
        self.gprmax_python_edit.textChanged.connect(self.refresh_command_preview)
        self.scene_table.itemSelectionChanged.connect(self._table_selection_changed)

    def choose_campaign(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 gprMax campaign YAML",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self.campaign_path_edit.setText(path)
            self.load_campaign(path)

    def load_campaign(self, path: str | Path) -> CampaignValidationResult:
        """Load and dry-run validate a campaign YAML, then refresh the UI."""
        path_text = str(path or "").strip()
        if not path_text:
            raise ValueError("请选择 campaign YAML 文件。")
        try:
            campaign = load_campaign_yaml(path_text)
            result = validate_campaign(campaign)
        except Exception as exc:
            self.campaign = None
            self.validation_result = None
            self._selected_scene_id = None
            self.status_label.setText("加载失败")
            self.summary_text.setPlainText(str(exc))
            self.details_text.setPlainText(str(exc))
            self.scene_table.setRowCount(0)
            self.scene_combo.clear()
            self.command_preview.clear()
            self._sync_controls()
            self.status_changed.emit(f"gprMax campaign 加载失败：{exc}")
            raise

        self.campaign = campaign
        self.validation_result = result
        self.campaign_path_edit.setText(str(campaign.source_path))
        self._populate_validation()
        self._sync_controls()
        self.campaign_loaded.emit(str(campaign.source_path))
        self.status_changed.emit(self.status_label.text())
        return result

    def _populate_validation(self) -> None:
        if self.campaign is None or self.validation_result is None:
            return
        result = self.validation_result
        tone = {
            VALIDATION_READY: "可执行",
            VALIDATION_WARNING: "有警告",
            VALIDATION_INVALID: "不可执行",
        }.get(result.status, result.status)
        self.status_label.setText(
            f"{result.campaign_id} · {tone} · ready {result.ready_count} / "
            f"warning {result.warning_count} / invalid {result.invalid_count}"
        )
        self.summary_text.setPlainText(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

        scene_by_id = {scene.scene_id: scene for scene in self.campaign.scenes}
        self.scene_table.setRowCount(len(result.scenes))
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        for row, scene_result in enumerate(result.scenes):
            scene = scene_by_id.get(scene_result.scene_id)
            tags = ", ".join(scene.tags) if scene else ""
            values = [
                scene_result.scene_id,
                scene_result.status,
                str(len(scene_result.issues)),
                tags,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 1:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.scene_table.setItem(row, column, item)
            self.scene_combo.addItem(scene_result.scene_id, scene_result.scene_id)
        self.scene_combo.blockSignals(False)
        if result.scenes:
            selected = self._selected_scene_id if self._selected_scene_id in scene_by_id else result.scenes[0].scene_id
            self.select_scene(selected)
        else:
            self._selected_scene_id = None
            self.details_text.setPlainText("Campaign 中没有场景。")
            self.command_preview.clear()

    def select_scene(self, scene_id: str) -> None:
        """Select a scene in both the table and command planner."""
        if self.validation_result is None:
            return
        scene_ids = [item.scene_id for item in self.validation_result.scenes]
        if scene_id not in scene_ids:
            raise KeyError(scene_id)
        self._selected_scene_id = scene_id
        combo_index = self.scene_combo.findData(scene_id)
        if combo_index >= 0 and self.scene_combo.currentIndex() != combo_index:
            self.scene_combo.blockSignals(True)
            self.scene_combo.setCurrentIndex(combo_index)
            self.scene_combo.blockSignals(False)
        for row, candidate in enumerate(scene_ids):
            if candidate == scene_id:
                self.scene_table.blockSignals(True)
                self.scene_table.selectRow(row)
                self.scene_table.blockSignals(False)
                break
        self._refresh_scene_details()
        self.refresh_command_preview()
        self._sync_controls()

    def _scene_combo_changed(self) -> None:
        scene_id = self.scene_combo.currentData()
        if scene_id:
            self.select_scene(str(scene_id))

    def _table_selection_changed(self) -> None:
        selected = self.scene_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        item = self.scene_table.item(row, 0)
        if item is not None and item.text() != self._selected_scene_id:
            self.select_scene(item.text())

    def _refresh_scene_details(self) -> None:
        if self.campaign is None or self.validation_result is None or self._selected_scene_id is None:
            self.details_text.clear()
            return
        scene = next((item for item in self.campaign.scenes if item.scene_id == self._selected_scene_id), None)
        scene_result = next(
            (item for item in self.validation_result.scenes if item.scene_id == self._selected_scene_id),
            None,
        )
        payload: dict[str, Any] = {
            "campaign_id": self.campaign.campaign_id,
            "scene_id": self._selected_scene_id,
            "status": scene_result.status if scene_result else None,
            "description": scene.description if scene else "",
            "paths": {
                "raw_model": str(scene.raw_model) if scene else None,
                "background_model": str(scene.background_model) if scene else None,
                "materials": str(scene.materials) if scene else None,
                "target_roi": str(scene.target_roi) if scene else None,
                "output_root": str(self.campaign.output_root),
            },
            "expected_outputs": scene.expected_outputs if scene else None,
            "tags": scene.tags if scene else [],
            "issues": [issue.to_dict() for issue in scene_result.issues] if scene_result else [],
        }
        self.details_text.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))

    def build_run_command_preview(self) -> list[str]:
        """Return the CLI command tokens represented by the current planner state."""
        if self.campaign is None or self._selected_scene_id is None:
            return []
        if self._selected_scene_status() == VALIDATION_INVALID:
            return []
        command = [
            "python",
            "scripts/gprmax_campaign_runner.py",
            "--campaign",
            str(self.campaign.source_path),
            "--run-scene",
            self._selected_scene_id,
            "--variant",
            str(self.variant_combo.currentData()),
        ]
        if self.num_runs_spin.value() > 1:
            command.extend(["--num-runs", str(self.num_runs_spin.value())])
        if self.timeout_spin.value() > 0:
            command.extend(["--timeout-seconds", str(self.timeout_spin.value())])
        gprmax_python = self.gprmax_python_edit.text().strip()
        if gprmax_python:
            command.extend(["--gprmax-python", gprmax_python])
        gpu_devices = self._parse_gpu_devices_text()
        if self._gpu_device_parse_error:
            return []
        if self.gpu_check.isChecked() or gpu_devices:
            if gpu_devices:
                command.extend(["--gpu-devices", *[str(item) for item in gpu_devices]])
            else:
                command.append("--gpu")
        return command

    def refresh_command_preview(self) -> None:
        command = self.build_run_command_preview()
        if not command:
            if self._gpu_device_parse_error:
                self.command_preview.setPlainText(self._gpu_device_parse_error)
            elif self._selected_scene_status() == VALIDATION_INVALID:
                self.command_preview.setPlainText("当前场景未通过 dry-run 检查，不生成运行命令。")
            else:
                self.command_preview.setPlainText("载入 campaign 并选择 ready/warning 场景后生成命令预览。")
        else:
            self.command_preview.setPlainText(_format_shell_command(command))
        self._sync_controls()

    def copy_command_preview(self) -> None:
        if self._selected_scene_status() == VALIDATION_INVALID or self._gpu_device_parse_error:
            return
        text = self.command_preview.toPlainText().strip()
        if not text or text.startswith("载入 campaign") or text.startswith("GPU 设备"):
            return
        QGuiApplication.clipboard().setText(text)
        self.status_changed.emit("gprMax 命令已复制到剪贴板。")

    def _parse_gpu_devices_text(self) -> list[int]:
        text = self.gpu_devices_edit.text().replace(",", " ").strip()
        self._gpu_device_parse_error = None
        if not text:
            return []
        device_ids: list[int] = []
        invalid_tokens: list[str] = []
        for token in text.split():
            try:
                value = int(token)
            except ValueError:
                invalid_tokens.append(token)
                continue
            if value < 0:
                invalid_tokens.append(token)
                continue
            device_ids.append(value)
        if invalid_tokens:
            self._gpu_device_parse_error = (
                "GPU 设备格式无效："
                + ", ".join(invalid_tokens)
                + "。请输入非负整数，例如 0 或 0 1。"
            )
            return []
        return device_ids

    def _selected_scene_status(self) -> str | None:
        if self.validation_result is None or self._selected_scene_id is None:
            return None
        for scene_result in self.validation_result.scenes:
            if scene_result.scene_id == self._selected_scene_id:
                return scene_result.status
        return None

    def _sync_controls(self) -> None:
        loaded = self.campaign is not None and self.validation_result is not None
        self.scene_combo.setEnabled(loaded)
        self.variant_combo.setEnabled(loaded)
        self.num_runs_spin.setEnabled(loaded)
        self.timeout_spin.setEnabled(loaded)
        self.gpu_check.setEnabled(loaded)
        self.gpu_devices_edit.setEnabled(loaded)
        self.gprmax_python_edit.setEnabled(loaded)
        runnable_status = self._selected_scene_status()
        command = self.build_run_command_preview()
        can_copy = bool(loaded and not self._gpu_device_parse_error and runnable_status != VALIDATION_INVALID and command)
        self.copy_command_button.setEnabled(can_copy)

    def load_campaign_from_ui_safely(self) -> None:
        """Convenience wrapper for signal/slot use with user-facing errors."""
        try:
            self.load_campaign(self.campaign_path_edit.text())
        except Exception as exc:
            QMessageBox.critical(self, "Campaign 加载失败", str(exc))


def _format_shell_command(tokens: list[str]) -> str:
    """Format command tokens for Windows CMD/PowerShell and simple POSIX shells.

    The application is delivered primarily with Windows batch launchers.  POSIX
    ``shlex.quote`` uses single quotes, which do not group paths in ``cmd.exe``.
    A conservative double-quote formatter keeps copied campaign commands usable
    for Windows users while remaining readable in PowerShell and common shells.
    """

    def quote(token: object) -> str:
        value = str(token)
        if value == "":
            return '""'
        needs_quotes = any(ch.isspace() for ch in value) or any(ch in value for ch in ['"', "'", "&", "(", ")", "^", ";"])
        if not needs_quotes:
            return value
        escaped = value.replace('"', r'\"')
        return f'"{escaped}"'

    return " ".join(quote(token) for token in tokens)


__all__ = ["SimulationValidationPage"]
