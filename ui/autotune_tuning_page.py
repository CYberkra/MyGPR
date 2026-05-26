#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoTune 调参聚焦页面（MVP 占位版）。"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QGridLayout,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QCheckBox,
    QComboBox,
    QTabWidget,
)
from qfluentwidgets import PushButton, FluentIcon

from ui.gui_auto_tune_page import AutoTunePage


class AutoTuneTuningPage(QWidget):
    """面向 AutoTune 调参的聚焦 UI；保留 legacy AutoTunePage 作为兼容后端状态容器。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self._legacy_page = AutoTunePage(self)
        self._build_ui()

    _DELEGATED_NAMES = {
        "btn_auto_tune",
        "btn_compare_stage",
        "btn_compare_manual_auto",
        "btn_export_comparison",
        "btn_view_auto_tune",
        "btn_apply_stage_choice",
        "btn_open_workbench",
        "get_auto_tune_roi_mode",
        "get_auto_tune_search_mode",
        "reset_for_method",
        "show_running",
        "show_result",
        "show_error",
        "show_cancelled",
        "set_stage_compare_result",
        "set_auto_tune_method_key",
        "show_comparison_running",
        "show_comparison_result",
        "show_comparison_error",
        "set_evidence_export_result",
    }

    def __getattr__(self, name):
        # 仅代理 app_qt 需要的既有接口，避免把 parent_window 的属性查找回卷到 legacy 页面导致递归。
        if name in self._DELEGATED_NAMES:
            legacy = self.__dict__.get("_legacy_page")
            if legacy is not None:
                return getattr(legacy, name)
        raise AttributeError(name)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        title = QLabel("AutoTune 调参")
        title.setProperty("class", "sectionTitle")
        root.addWidget(title)

        root.addWidget(self._build_session_header())

        body = QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(self._build_left_panel(), 3)
        body.addWidget(self._build_center_panel(), 5)
        body.addWidget(self._build_right_panel(), 3)
        root.addLayout(body, 1)

        root.addWidget(self._build_bottom_drawer(), 2)

        # 兼容层：保留 legacy 页面实例但隐藏，不作为主 UI 展示。
        self._legacy_page.hide()
        root.addWidget(self._legacy_page)

    def _chip(self, text: str) -> QLabel:
        chip = QLabel(text)
        chip.setProperty("class", "statusChip")
        return chip

    def _build_session_header(self) -> QGroupBox:
        box = QGroupBox("AutoTune Session")
        layout = QVBoxLayout(box)
        chips = QHBoxLayout()
        for t in ["当前数据: 未载入", "AutoTune 模式: 标准", "ROI 状态: 草案", "运行状态: Idle", "Evidence 状态: 未就绪"]:
            chips.addWidget(self._chip(t))
        chips.addStretch(1)
        layout.addLayout(chips)
        btns = QHBoxLayout()
        for icon, text in [
            (FluentIcon.FOLDER, "载入数据"),
            (FluentIcon.PLAY, "运行 AutoTune"),
            (FluentIcon.SAVE_AS, "导出 Evidence"),
        ]:
            b = PushButton(icon, text)
            b.setEnabled(False)
            btns.addWidget(b)
        btns.addStretch(1)
        layout.addLayout(btns)
        return box

    def _build_left_panel(self) -> QGroupBox:
        box = QGroupBox("参数面板")
        layout = QVBoxLayout(box)
        layout.addWidget(self._section_combo("Workflow Step", ["Background Suppression", "Gain", "Dewow", "Bandpass", "Display Enhancement"]))
        layout.addWidget(self._section_combo("Candidate Space", ["no suppression", "mean background", "median background", "SVD rank", "sliding window"]))
        layout.addWidget(self._section_combo("ROI", ["trace start/end", "sample start/end", "target window", "background window", "auto ROI", "manual ROI"]))
        layout.addWidget(self._section_checks("Scoring", ["RMSE", "ROI energy retention", "outside ROI residual", "CNR/SNR", "apex stability"]))
        layout.addWidget(self._section_checks("Safety", ["no-prior warning", "display-only flag", "manual review required", "claim boundary required"]))
        layout.addStretch(1)
        return box

    def _section_combo(self, title: str, items: list[str]) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        combo = QComboBox()
        combo.addItems(items)
        combo.setEnabled(False)
        layout.addWidget(combo)
        return box

    def _section_checks(self, title: str, items: list[str]) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        for item in items:
            cb = QCheckBox(item)
            cb.setEnabled(False)
            layout.addWidget(cb)
        return box

    def _placeholder_card(self, title: str, desc: str) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        l = QVBoxLayout(card)
        t = QLabel(title)
        t.setProperty("class", "titleSmall")
        d = QLabel(desc)
        d.setWordWrap(True)
        d.setProperty("class", "hintText")
        l.addWidget(t)
        l.addWidget(d)
        l.addStretch(1)
        return card

    def _build_center_panel(self) -> QGroupBox:
        box = QGroupBox("预览工作区")
        layout = QGridLayout(box)
        layout.addWidget(self._placeholder_card("Raw/Input Preview", "原始输入预览（MVP 占位，不触发真实加载）。"), 0, 0)
        layout.addWidget(self._placeholder_card("Candidate Output Preview", "候选输出预览（MVP 占位）。"), 0, 1)
        layout.addWidget(self._placeholder_card("ROI Overlay Preview", "ROI 叠加预览（MVP 占位）。"), 1, 0, 1, 2)
        return box

    def _build_right_panel(self) -> QGroupBox:
        box = QGroupBox("结果与风险检查")
        layout = QVBoxLayout(box)
        items = [
            ("Recommended Parameters", "固定流程推荐（占位）：background_suppression=median, roi=manual_draft"),
            ("Candidate Score", "示例评分：RMSE=--, CNR=--, Apex Stability=--"),
            ("Risk Warnings", "无先验告警；需人工复核；display-only 图不用于幅值主张"),
            ("Claim Boundary", "fixed workflow only; not global optimum; not field validation"),
        ]
        for t, d in items:
            layout.addWidget(self._placeholder_card(t, d))
        return box

    def _build_bottom_drawer(self) -> QTabWidget:
        tabs = QTabWidget()
        panels = {
            "Trial Table": "trial_id | workflow_step | candidate | status | score",
            "Metrics": "RMSE / ROI retention / outside ROI residual / CNR / apex stability",
            "Logs": "AutoTune session logs placeholder",
            "Warnings": "manual review required; claim boundary required",
            "Claim Boundary": "not benchmark / not field validation / not AutoTune superiority",
        }
        for name, text in panels.items():
            p = QWidget()
            l = QVBoxLayout(p)
            te = QTextEdit()
            te.setReadOnly(True)
            te.setPlainText(text)
            l.addWidget(te)
            tabs.addTab(p, name)
        return tabs
