#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only research console pages for gprMax, Evidence, and AT-BG."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import FluentIcon, PushButton

from core.gprmax_model_inspector import load_scene_model
from core.research_dashboard import load_dashboard_state


CLAIM_BOUNDARY_TEXT = (
    "synthetic paired diagnostic only; background suppression only; not full AutoTune; "
    "not production scoring; not field validation; not AutoTune superiority evidence"
)


class ResearchConsolePage(QWidget):
    """Research validation dashboard integrated inside the AutoTune tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.state: dict[str, Any] = {}
        self._scene_items: list[dict[str, Any]] = []
        self._artifact_items: list[dict[str, Any]] = []
        self._model_scene_ids = [
            "scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate",
            "scene_038_gssi_ey_depth07_radius03_air_sand_interface_n80_pair_gate",
        ]
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        title_block = QWidget()
        title_layout = QVBoxLayout(title_block)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("研究验证与 Evidence")
        title.setProperty("class", "titleSmall")
        title_layout.addWidget(title)
        subtitle = QLabel("只读查看 GX-008 场景、Evidence、standard metrics、AT-BG 诊断和 gprMax 模型草稿。")
        subtitle.setWordWrap(True)
        subtitle.setProperty("class", "hintText")
        title_layout.addWidget(subtitle)
        header_layout.addWidget(title_block, 1)
        self.evidence_root_label = QLabel("Evidence 根目录：--")
        self.evidence_root_label.setWordWrap(True)
        self.evidence_root_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(self.evidence_root_label, 1)
        self.refresh_button = PushButton(FluentIcon.SYNC, "刷新")
        self.refresh_button.clicked.connect(self.refresh)
        header_layout.addWidget(self.refresh_button)
        layout.addWidget(header)

        card_box = QGroupBox("研究控制台入口")
        card_layout = QGridLayout(card_box)
        card_layout.setContentsMargins(10, 14, 10, 10)
        card_layout.setHorizontalSpacing(8)
        card_layout.setVerticalSpacing(8)
        cards = [
            ("仿真与验证 Dashboard", "查看 GX-008 场景、Evidence、metrics、preview、claim boundary。", 0),
            ("背景抑制 AutoTune", "查看 mean / median / SVD 诊断、trial table、selected parameters。", 1),
            ("Evidence Viewer", "查看 artifact manifest、report、metrics、figures 和缺失路径 warning。", 2),
            ("gprMax 模型编辑器", "只读校验模型参数、材料、目标、ROI 与 pair contract。", 3),
        ]
        for index, (title, detail, tab_index) in enumerate(cards):
            card_layout.addWidget(self._nav_card(title, detail, tab_index), index // 2, index % 2)
        layout.addWidget(card_box)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        layout.addWidget(self.tabs)

        self.dashboard_page = self._build_dashboard_page()
        self.at_bg_page = self._build_at_bg_page()
        self.evidence_page = self._build_evidence_page()
        self.model_page = self._build_model_page()

        self.tabs.addTab(self.dashboard_page, "仿真与验证 Dashboard")
        self.tabs.addTab(self.at_bg_page, "背景抑制 AutoTune")
        self.tabs.addTab(self.evidence_page, "Evidence Viewer")
        self.tabs.addTab(self.model_page, "gprMax 模型编辑器 v0")

    def _build_dashboard_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        self.status_cards = QWidget()
        self.status_cards_layout = QGridLayout(self.status_cards)
        self.status_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.status_cards_layout.setHorizontalSpacing(8)
        self.status_cards_layout.setVerticalSpacing(8)
        layout.addWidget(self.status_cards)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        scene_box = QGroupBox("GX-008 场景状态总览")
        scene_layout = QVBoxLayout(scene_box)
        self.scene_table = QTableWidget(0, 9)
        self.scene_table.setHorizontalHeaderLabels(
            ["Scene", "Soil", "Target", "Depth", "Paired", "Metrics", "AT-BG", "Backend", "Shape"]
        )
        self.scene_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.scene_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.scene_table.itemSelectionChanged.connect(self._on_scene_selected)
        scene_layout.addWidget(self.scene_table)
        left_layout.addWidget(scene_box, 2)

        self.dashboard_bottom_tabs = QTabWidget()
        self.log_text = self._readonly_text()
        self.report_text = self._readonly_text()
        self.manifest_text = self._readonly_text()
        self.metrics_text = self._readonly_text()
        self.warning_text = self._readonly_text()
        self.dashboard_bottom_tabs.addTab(self.log_text, "概览日志")
        self.dashboard_bottom_tabs.addTab(self.report_text, "Report")
        self.dashboard_bottom_tabs.addTab(self.manifest_text, "Manifest")
        self.dashboard_bottom_tabs.addTab(self.metrics_text, "Metrics")
        self.dashboard_bottom_tabs.addTab(self.warning_text, "Warnings")
        left_layout.addWidget(self.dashboard_bottom_tabs, 1)
        splitter.addWidget(left)

        inspector = QGroupBox("选中 Artifact 详情")
        inspector_layout = QVBoxLayout(inspector)
        self.artifact_detail = self._readonly_text()
        inspector_layout.addWidget(self.artifact_detail, 2)
        self.preview_row = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_row)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.addWidget(self.preview_row)
        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        self.open_report_button = PushButton(FluentIcon.DOCUMENT, "打开报告")
        self.open_metrics_button = PushButton(FluentIcon.VIEW, "打开指标")
        self.open_manifest_button = PushButton(FluentIcon.DOCUMENT, "打开 Manifest")
        self.open_folder_button = PushButton(FluentIcon.FOLDER, "打开文件夹")
        for button in [
            self.open_report_button,
            self.open_metrics_button,
            self.open_manifest_button,
            self.open_folder_button,
        ]:
            button.setEnabled(False)
            action_layout.addWidget(button)
        action_layout.addStretch(1)
        inspector_layout.addWidget(action_row)
        splitter.addWidget(inspector)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        return page

    def _build_at_bg_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        self.at_bg_overview = QLabel(CLAIM_BOUNDARY_TEXT)
        self.at_bg_overview.setWordWrap(True)
        self.at_bg_overview.setObjectName("WarningBanner")
        layout.addWidget(self.at_bg_overview)

        summary_box = QGroupBox("候选方法与多场景一致性")
        summary_layout = QVBoxLayout(summary_box)
        self.selected_table = QTableWidget(0, 5)
        self.selected_table.setHorizontalHeaderLabels(["Scene", "Trial", "Method", "Parameters", "Trials"])
        self.selected_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        summary_layout.addWidget(self.selected_table)
        layout.addWidget(summary_box)

        self.trial_table = QTableWidget(0, 8)
        self.trial_table.setHorizontalHeaderLabels(
            ["Scene", "Trial", "Method", "Parameters", "MAE", "RMSE", "PSNR", "Label"]
        )
        self.trial_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.trial_table, 1)

        rank_box = QGroupBox("Method rank summary")
        rank_layout = QVBoxLayout(rank_box)
        self.rank_table = QTableWidget(0, 7)
        self.rank_table.setHorizontalHeaderLabels(["Method", "Parameters", "Scenes", "Mean rank", "Selected", "Mean MAE", "Warnings"])
        self.rank_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        rank_layout.addWidget(self.rank_table)
        layout.addWidget(rank_box, 1)
        return page

    def _build_evidence_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)

        self.artifact_list = QListWidget()
        self.artifact_list.currentRowChanged.connect(self._on_artifact_selected)
        layout.addWidget(self.artifact_list, 1)

        detail_tabs = QTabWidget()
        self.evidence_manifest_text = self._readonly_text()
        self.evidence_report_text = self._readonly_text()
        self.evidence_metrics_text = self._readonly_text()
        self.evidence_claim_text = self._readonly_text()
        detail_tabs.addTab(self.evidence_manifest_text, "Manifest")
        detail_tabs.addTab(self.evidence_report_text, "Report")
        detail_tabs.addTab(self.evidence_metrics_text, "Metrics / Tables")
        detail_tabs.addTab(self.evidence_claim_text, "Claim Boundary")
        layout.addWidget(detail_tabs, 3)
        return page

    def _build_model_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        banner = QLabel("Model Editor v0 / read-only protected mode. 编辑、保存、dry-run、GPU 运行在后续 draft-edit 模式开放。")
        banner.setWordWrap(True)
        banner.setObjectName("InfoBanner")
        layout.addWidget(banner)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.model_scene_list = QListWidget()
        self.model_scene_list.currentRowChanged.connect(self._on_model_scene_selected)
        splitter.addWidget(self.model_scene_list)

        self.model_tabs = QTabWidget()
        self.model_overview_text = self._readonly_text()
        self.model_basic_text = self._readonly_text()
        self.model_materials_text = self._readonly_text()
        self.model_geometry_text = self._readonly_text()
        self.model_scan_text = self._readonly_text()
        self.model_roi_text = self._readonly_text()
        self.model_pair_text = self._readonly_text()
        self.model_diff_text = self._readonly_text()
        self.model_command_text = self._readonly_text()
        for name, widget in [
            ("Overview", self.model_overview_text),
            ("Basic", self.model_basic_text),
            ("Materials", self.model_materials_text),
            ("Geometry", self.model_geometry_text),
            ("Scan", self.model_scan_text),
            ("ROI", self.model_roi_text),
            ("Pair Contract", self.model_pair_text),
            ("Raw / Background Diff", self.model_diff_text),
            ("Generated Command", self.model_command_text),
        ]:
            self.model_tabs.addTab(widget, name)
        splitter.addWidget(self.model_tabs)

        inspector = QGroupBox("Validation / Inspector")
        inspector_layout = QVBoxLayout(inspector)
        self.model_inspector_text = self._readonly_text()
        inspector_layout.addWidget(self.model_inspector_text)
        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        self.copy_command_button = PushButton(FluentIcon.COPY, "复制命令")
        self.copy_command_button.clicked.connect(self._copy_model_command)
        self.disabled_save_button = PushButton(FluentIcon.SAVE, "保存草稿")
        self.disabled_save_button.setEnabled(False)
        self.disabled_run_button = PushButton(FluentIcon.PLAY, "GPU 运行")
        self.disabled_run_button.setEnabled(False)
        button_layout.addWidget(self.copy_command_button)
        button_layout.addWidget(self.disabled_save_button)
        button_layout.addWidget(self.disabled_run_button)
        inspector_layout.addWidget(button_row)
        splitter.addWidget(inspector)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        layout.addWidget(splitter, 1)
        return page

    @staticmethod
    def _readonly_text() -> QPlainTextEdit:
        widget = QPlainTextEdit()
        widget.setReadOnly(True)
        widget.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        return widget

    def _nav_card(self, title: str, detail: str, tab_index: int) -> QFrame:
        frame = QFrame()
        frame.setObjectName("ResearchNavCard")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        text = QWidget()
        text_layout = QVBoxLayout(text)
        text_layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setProperty("class", "titleSmall")
        detail_label = QLabel(detail)
        detail_label.setWordWrap(True)
        detail_label.setProperty("class", "hintText")
        text_layout.addWidget(title_label)
        text_layout.addWidget(detail_label)
        layout.addWidget(text, 1)
        button = PushButton(FluentIcon.VIEW, "查看")
        button.clicked.connect(lambda _checked=False, index=tab_index: self.tabs.setCurrentIndex(index))
        layout.addWidget(button)
        return frame

    @staticmethod
    def _status_card(title: str, value: str, detail: str, tone: str = "neutral") -> QFrame:
        frame = QFrame()
        frame.setObjectName("MetricCard")
        frame.setProperty("tone", tone if tone in {"ok", "warn", "neutral"} else "neutral")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        label_title = QLabel(title)
        label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_value = QLabel(value)
        label_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_value.setProperty("class", "titleSmall")
        label_detail = QLabel(detail)
        label_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_detail.setWordWrap(True)
        layout.addWidget(label_title)
        layout.addWidget(label_value)
        layout.addWidget(label_detail)
        return frame

    def refresh(self) -> None:
        self.state = load_dashboard_state()
        self.evidence_root_label.setText(f"Evidence 根目录：{self.state.get('evidence_root') or 'missing'}")
        self._scene_items = list(self.state.get("scene_status", []))
        self._artifact_items = list(self.state.get("gprmax_artifacts", [])) + list(self.state.get("at_bg_artifacts", []))
        self._populate_dashboard()
        self._populate_at_bg()
        self._populate_evidence()
        self._populate_model_scenes()

    def _populate_dashboard(self) -> None:
        self._clear_layout(self.status_cards_layout)
        scene_rows = self.state.get("scene_status", [])
        paired_done = sum(1 for row in scene_rows if row.get("paired_evidence") == "done")
        metrics_done = sum(1 for row in scene_rows if row.get("standard_metrics") == "done")
        at_done = sum(1 for row in scene_rows if row.get("at_bg") == "done")
        draft_count = sum(1 for row in scene_rows if row.get("paired_evidence") == "draft")
        cards = [
            ("已归档场景", f"{paired_done} / {len(scene_rows)}", "Paired Evidence", "ok" if paired_done else "warn"),
            ("标准化指标", f"{metrics_done} / {len(scene_rows)}", "standard metrics", "ok" if metrics_done else "warn"),
            ("AT-BG 诊断", f"{at_done} / {len(scene_rows)}", "mean / median / SVD", "ok" if at_done else "warn"),
            ("保留模型", str(draft_count), "scene_037 / scene_038", "neutral"),
            ("GPU Wrapper", "Ready", "通过标准入口执行", "ok"),
            ("Claim Boundary", "Active", "No field / no superiority claim", "warn"),
        ]
        for index, (title, value, detail, tone) in enumerate(cards):
            self.status_cards_layout.addWidget(self._status_card(title, value, detail, tone), 0, index)

        self.scene_table.setRowCount(len(scene_rows))
        for row_index, row in enumerate(scene_rows):
            scene_id = row.get("scene_id", "")
            soil, target, depth = self._parse_scene_tokens(scene_id)
            values = [
                scene_id,
                soil,
                target,
                depth,
                row.get("paired_evidence", ""),
                row.get("standard_metrics", ""),
                row.get("at_bg", ""),
                row.get("backend", ""),
                self._format_shape(row.get("shape")),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column in {4, 5, 6}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.scene_table.setItem(row_index, column, item)
        self.scene_table.resizeColumnsToContents()

        self.log_text.setPlainText("\n".join(self._build_log_lines()))
        self.warning_text.setPlainText("\n".join(self.state.get("warnings", [])) or "无")
        if scene_rows:
            self.scene_table.selectRow(0)

    def _populate_at_bg(self) -> None:
        at_status = self.state.get("at_bg_status", {})
        self.at_bg_overview.setText(
            "AT-BG: "
            f"{at_status.get('completed_count', 0)} / {at_status.get('total_count', 0)} artifacts complete. "
            + CLAIM_BOUNDARY_TEXT
        )
        selected = at_status.get("selected", [])
        self.selected_table.setRowCount(len(selected))
        for row_index, row in enumerate(selected):
            values = [
                row.get("scene_id", ""),
                row.get("trial_id", ""),
                row.get("method", ""),
                json.dumps(row.get("parameters") or {}, ensure_ascii=False),
                row.get("trial_count", ""),
            ]
            for column, value in enumerate(values):
                self.selected_table.setItem(row_index, column, QTableWidgetItem(str(value)))
        self.selected_table.resizeColumnsToContents()

        trial_rows = list(self.state.get("trial_rows", []))[:120]
        self.trial_table.setRowCount(len(trial_rows))
        for row_index, row in enumerate(trial_rows):
            values = [
                row.get("scene_id", ""),
                row.get("trial_id", ""),
                row.get("method", ""),
                json.dumps(row.get("parameter_set") or {}, ensure_ascii=False),
                row.get("mae", ""),
                row.get("rmse", ""),
                row.get("psnr", ""),
                row.get("recommendation_label", ""),
            ]
            for column, value in enumerate(values):
                self.trial_table.setItem(row_index, column, QTableWidgetItem(str(value)))
        self.trial_table.resizeColumnsToContents()

        rank_rows = self.state.get("method_rank_summary", [])
        self.rank_table.setRowCount(len(rank_rows))
        for row_index, row in enumerate(rank_rows):
            values = [
                row.get("method", ""),
                json.dumps(row.get("parameter_set") or {}, ensure_ascii=False),
                row.get("scenes_present", ""),
                row.get("mean_rank", ""),
                row.get("selected_count", ""),
                row.get("mean_mae", ""),
                row.get("warning_count", ""),
            ]
            for column, value in enumerate(values):
                self.rank_table.setItem(row_index, column, QTableWidgetItem(str(value)))
        self.rank_table.resizeColumnsToContents()

    def _populate_evidence(self) -> None:
        self.artifact_list.clear()
        for item in self._artifact_items:
            label = item.get("display_name") or item.get("artifact_id") or item.get("evidence_path")
            list_item = QListWidgetItem(str(label))
            list_item.setData(Qt.ItemDataRole.UserRole, item)
            self.artifact_list.addItem(list_item)
        if self.artifact_list.count():
            self.artifact_list.setCurrentRow(0)

    def _populate_model_scenes(self) -> None:
        self.model_scene_list.clear()
        for scene_id in self._model_scene_ids:
            self.model_scene_list.addItem(scene_id)
        if self.model_scene_list.count():
            self.model_scene_list.setCurrentRow(0)

    def _on_scene_selected(self) -> None:
        selected = self.scene_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        if row >= len(self._scene_items):
            return
        scene = self._scene_items[row]
        artifact = next((item for item in self.state.get("gprmax_artifacts", []) if item.get("scene_id") == scene.get("scene_id")), {})
        self._show_artifact_detail(artifact, scene)

    def _on_artifact_selected(self, row: int) -> None:
        if row < 0:
            return
        item = self.artifact_list.item(row)
        if item is None:
            return
        artifact = item.data(Qt.ItemDataRole.UserRole) or {}
        self.evidence_manifest_text.setPlainText(self._read_or_dump(artifact.get("manifest_path"), artifact))
        self.evidence_report_text.setPlainText(self._read_or_placeholder(artifact.get("report_path"), "Report missing"))
        self.evidence_metrics_text.setPlainText(self._read_or_placeholder(artifact.get("metrics_path"), "Metrics/table missing"))
        self.evidence_claim_text.setPlainText("\n".join(artifact.get("claim_boundary") or [CLAIM_BOUNDARY_TEXT]))

    def _on_model_scene_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._model_scene_ids):
            return
        model = load_scene_model(self._model_scene_ids[row]).to_dict()
        self.model_overview_text.setPlainText(self._format_model_overview(model))
        self.model_basic_text.setPlainText(self._format_model_basic(model))
        self.model_materials_text.setPlainText(model.get("materials_text") or "materials.txt missing")
        self.model_geometry_text.setPlainText(self._format_model_geometry(model))
        self.model_scan_text.setPlainText(self._format_model_scan(model))
        self.model_roi_text.setPlainText(json.dumps(model.get("roi") or {}, ensure_ascii=False, indent=2))
        self.model_pair_text.setPlainText(json.dumps(model.get("pair_contract_checks") or {}, ensure_ascii=False, indent=2))
        self.model_diff_text.setPlainText(
            "raw_with_target.in\n"
            + (model.get("raw_text") or "")
            + "\n\nbackground_only.in\n"
            + (model.get("background_text") or "")
        )
        self.model_command_text.setPlainText(model.get("generated_gpu_command") or "")
        self.model_inspector_text.setPlainText(
            f"Status: {model.get('pair_contract_status')}\n"
            f"Expected runs: {model.get('expected_num_runs')}\n"
            f"Warnings:\n" + ("\n".join(model.get("warnings") or ["无"]))
            + "\n\nMode: read-only protected mode"
            + "\nClaim boundary:\nmodel draft/read-only inspector only; not a gprMax run; not Evidence artifact."
        )

    def _show_artifact_detail(self, artifact: dict[str, Any], scene: dict[str, Any]) -> None:
        if not artifact:
            self.artifact_detail.setPlainText(json.dumps(scene, ensure_ascii=False, indent=2))
            return
        detail = {
            "artifact_id": artifact.get("artifact_id"),
            "scene_id": artifact.get("scene_id"),
            "source_commit": artifact.get("source_commit"),
            "evidence_path": artifact.get("evidence_path"),
            "backend": artifact.get("backend"),
            "raw_shape": artifact.get("raw_shape"),
            "background_shape": artifact.get("background_shape"),
            "target_response_shape": artifact.get("target_response_shape"),
            "report_path": artifact.get("report_path"),
            "metrics_path": artifact.get("metrics_path"),
            "manifest_path": artifact.get("manifest_path"),
            "claim_boundary": artifact.get("claim_boundary"),
            "warnings": artifact.get("warnings"),
        }
        self.artifact_detail.setPlainText(json.dumps(detail, ensure_ascii=False, indent=2))
        self.manifest_text.setPlainText(self._read_or_dump(artifact.get("manifest_path"), artifact))
        self.report_text.setPlainText(self._read_or_placeholder(artifact.get("report_path"), "Report missing"))
        self.metrics_text.setPlainText(self._read_or_placeholder(artifact.get("metrics_path"), "Metrics missing"))
        self._refresh_preview_row(artifact.get("preview_paths") or [])
        self._wire_open_buttons(artifact)

    def _wire_open_buttons(self, artifact: dict[str, Any]) -> None:
        for button, path in [
            (self.open_report_button, artifact.get("report_path")),
            (self.open_metrics_button, artifact.get("metrics_path")),
            (self.open_manifest_button, artifact.get("manifest_path")),
            (self.open_folder_button, artifact.get("evidence_path")),
        ]:
            try:
                button.clicked.disconnect()
            except TypeError:
                pass
            button.setEnabled(bool(path and Path(path).exists()))
            if path:
                button.clicked.connect(lambda _checked=False, p=path: self._open_path(p))

    def _refresh_preview_row(self, paths: list[str]) -> None:
        self._clear_layout(self.preview_layout)
        if not paths:
            label = QLabel("预览缺失或未归档。")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview_layout.addWidget(label)
            return
        for path in paths[:4]:
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setToolTip(path)
            pixmap = QPixmap(path)
            if pixmap.isNull():
                label.setText(Path(path).name)
            else:
                label.setPixmap(pixmap.scaled(120, 90, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.preview_layout.addWidget(label)

    def _copy_model_command(self) -> None:
        command = self.model_command_text.toPlainText().strip()
        if command:
            from PyQt6.QtWidgets import QApplication

            QApplication.clipboard().setText(command)

    @staticmethod
    def _format_shape(value: Any) -> str:
        if isinstance(value, list) and len(value) == 2:
            return f"{value[0]} x {value[1]}"
        return "-"

    @staticmethod
    def _parse_scene_tokens(scene_id: str) -> tuple[str, str, str]:
        soil = "Damp Sand" if "damp_sand" in scene_id else "Dry Sand" if "dry_sand" in scene_id else "-"
        target = "PVC" if "_pvc_" in scene_id else "PEC" if "_pec_" in scene_id else "-"
        depth = "Medium" if scene_id.endswith("_medium") else "Shallow" if scene_id.endswith("_shallow") else "-"
        return soil, target, depth

    @staticmethod
    def _read_or_placeholder(path: str | None, placeholder: str) -> str:
        if not path:
            return placeholder
        file_path = Path(path)
        if not file_path.exists():
            return f"{placeholder}: {path}"
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Binary or non-text file: {path}"
        except Exception as exc:
            return f"Failed to read {path}: {exc}"

    def _read_or_dump(self, path: str | None, fallback: dict[str, Any]) -> str:
        return self._read_or_placeholder(path, json.dumps(fallback, ensure_ascii=False, indent=2))

    @staticmethod
    def _open_path(path: str) -> None:
        if not path or not Path(path).exists():
            return
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except Exception:
            subprocess.Popen(["explorer", str(path)])

    @staticmethod
    def _clear_layout(layout: QHBoxLayout | QGridLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_log_lines(self) -> list[str]:
        lines = []
        for row in self.state.get("scene_status", []):
            lines.append(
                f"INFO scene={row.get('scene_id')} paired={row.get('paired_evidence')} "
                f"metrics={row.get('standard_metrics')} at_bg={row.get('at_bg')}"
            )
        for warning in self.state.get("warnings", []):
            lines.append(f"WARNING {warning}")
        return lines or ["INFO dashboard state loaded"]

    @staticmethod
    def _format_model_overview(model: dict[str, Any]) -> str:
        return (
            f"scene_id: {model.get('scene_id')}\n"
            f"scene_role: {model.get('scene_role')}\n"
            f"soil: {model.get('soil_type')}\n"
            f"target: {model.get('target_type')} / {model.get('target_material')}\n"
            f"depth_class: {model.get('target_depth_class')}\n"
            f"pair_contract_status: {model.get('pair_contract_status')}\n"
            f"expected_num_runs: {model.get('expected_num_runs')}\n"
            "current_status: draft or archived Evidence, depending on dashboard state"
        )

    @staticmethod
    def _format_model_basic(model: dict[str, Any]) -> str:
        keys = ["domain", "dx_dy_dz", "time_window", "waveform", "source", "receiver", "src_steps", "rx_steps"]
        return "\n".join(f"{key}: {model.get(key) or '--'}" for key in keys)

    @staticmethod
    def _format_model_geometry(model: dict[str, Any]) -> str:
        target = (model.get("manifest") or {}).get("target_design") or {}
        return json.dumps(target, ensure_ascii=False, indent=2) + "\n\nGeometry sketch: placeholder in v0."

    @staticmethod
    def _format_model_scan(model: dict[str, Any]) -> str:
        return (
            f"src_steps: {model.get('src_steps')}\n"
            f"rx_steps: {model.get('rx_steps')}\n"
            f"expected_num_runs: {model.get('expected_num_runs')}\n"
            "Note: #src_steps/#rx_steps define stepping; --num-runs is required to create B-scan traces."
        )
