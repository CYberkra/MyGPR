#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exploration-oriented MyGPR project workbench."""

from __future__ import annotations

import json
import logging
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QThread, Qt, QTimer
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QInputDialog,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.command_contract import assert_command_allowed
from core.gpr_format_registry import supported_file_dialog_filter
from core.gpr_io import auto_load_data
from core.processing_session import ProcessingSessionService
from core.ingest_service import IngestService
from core.project_models import LineRecordV1, QcReportV1
from core.project_service import ProjectService, utc_now
from core.qc_service import QcService
from core.workspace_session import WorkspaceSessionService
from ui.app_branding import make_mygpr_brand_pixmap
from ui.delivery_page import DeliveryPage
from ui.interpretation_workbench_page import InterpretationWorkbenchPage
from ui.legacy_processing_bridge import LegacyProcessingBridge
from ui.processing_lab_page import ProcessingLabPage
from ui.spatial_synthesis_page import SpatialSynthesisPage
from ui.workbench_tasks import WorkbenchTaskWorker
from core.product_mode import build_workspaces, is_research_ui_enabled
from core.user_labels import (
    delivery_role_label,
    line_status_label,
    qc_code_label,
    severity_label,
    sidecar_label,
)

logger = logging.getLogger(__name__)

WORKSPACES = build_workspaces()


class MyGPRWorkbenchWindow(QMainWindow):
    """Project-first shell. Feature tabs are deliberately not used."""

    def __init__(self, version_text: str = "MyGPR 勘探定位工作台"):
        super().__init__()
        self.version_text = version_text
        self.project: ProjectService | None = None
        self.session: WorkspaceSessionService | None = None
        self.legacy_bridge: LegacyProcessingBridge | None = None
        self.selected_line_id: str | None = None
        self.research_ui_enabled = is_research_ui_enabled()
        self.active_workspace = "data_management"
        self.workspace_buttons: dict[str, QPushButton] = {}
        self.workspace_pages: dict[str, QWidget] = {}
        self._temporary_root: Path | None = None
        self._task_threads: set[QThread] = set()
        self._pending_global_layout: dict[str, Any] | None = None
        self._setup_window()
        self._setup_ui()
        self._apply_workbench_style()
        self._show_overview_document()
        self._sync_actions()

    def _setup_window(self) -> None:
        self.setWindowTitle(self.version_text)
        self.setWindowIcon(QIcon(make_mygpr_brand_pixmap(64)))
        self.resize(1440, 900)
        self.setMinimumSize(1120, 720)
        self.setStatusBar(QStatusBar(self))

    def _setup_ui(self) -> None:
        root = QWidget()
        root.setObjectName("workbenchRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_top_bar())

        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.vertical_splitter.setChildrenCollapsible(False)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)

        self.project_tree = QTreeWidget()
        self.project_tree.setObjectName("projectTree")
        self.project_tree.setHeaderLabels(["项目资源", "状态"])
        self.project_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.project_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.project_tree.itemSelectionChanged.connect(self._on_tree_selection)
        self.project_tree.setMinimumWidth(180)
        self.main_splitter.addWidget(self.project_tree)

        self.workspace_stack = QStackedWidget()
        self.workspace_stack.setObjectName("workspaceStack")
        self.workspace_stack.setMinimumWidth(500)
        self.workspace_stack.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.document_tabs = QTabWidget()
        self.document_tabs.setObjectName("dataDocumentTabs")
        self.document_tabs.setTabsClosable(True)
        self.document_tabs.tabCloseRequested.connect(self._close_document)
        self.workspace_pages["data_management"] = self.document_tabs
        self.workspace_stack.addWidget(self.document_tabs)
        self.processing_lab = ProcessingLabPage(self)
        self.processing_lab.result_saved.connect(self._on_result_saved)
        self.processing_lab.status_changed.connect(self.statusBar().showMessage)
        self.workspace_pages["processing_lab"] = self.processing_lab
        self.workspace_stack.addWidget(self.processing_lab)
        self.simulation_validation = None
        if "simulation_validation" in WORKSPACES:
            from ui.simulation_validation_page import SimulationValidationPage

            self.simulation_validation = SimulationValidationPage(self)
            self.simulation_validation.status_changed.connect(self.statusBar().showMessage)
            self.workspace_pages["simulation_validation"] = self.simulation_validation
            self.workspace_stack.addWidget(self.simulation_validation)
        self.interpretation_workbench = InterpretationWorkbenchPage(self)
        self.interpretation_workbench.interpretation_changed.connect(
            lambda _line_id: self._refresh_project_ui(reset_documents=False)
        )
        self.interpretation_workbench.status_changed.connect(self.statusBar().showMessage)
        self.workspace_pages["interpretation"] = self.interpretation_workbench
        self.workspace_stack.addWidget(self.interpretation_workbench)
        self.spatial_synthesis = SpatialSynthesisPage(self)
        self.spatial_synthesis.status_changed.connect(self.statusBar().showMessage)
        self.workspace_pages["spatial"] = self.spatial_synthesis
        self.workspace_stack.addWidget(self.spatial_synthesis)
        self.delivery_page = DeliveryPage(self)
        self.delivery_page.status_changed.connect(self.statusBar().showMessage)
        self.delivery_page.package_built.connect(self._on_delivery_package_built)
        self.workspace_pages["delivery"] = self.delivery_page
        self.workspace_stack.addWidget(self.delivery_page)
        self.main_splitter.addWidget(self.workspace_stack)

        self.inspector = self._build_inspector()
        self.inspector.setMinimumWidth(220)
        self.main_splitter.addWidget(self.inspector)
        self.main_splitter.setSizes([260, 900, 320])

        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setObjectName("bottomDrawer")
        self.bottom_tabs.setMinimumHeight(200)
        self.task_table = self._make_log_table(["任务", "状态", "详情"])
        self.warning_table = self._make_log_table(["等级", "检查内容", "说明"])
        self.evidence_table = self._make_log_table(["文件", "位置", "状态"])
        self.log_table = self._make_log_table(["时间", "事件", "详情"])
        self.bottom_tabs.addTab(self.task_table, "任务")
        self.bottom_tabs.addTab(self.warning_table, "检查提示")
        self.bottom_tabs.addTab(self.evidence_table, "交付文件")
        self.bottom_tabs.addTab(self.log_table, "日志")

        self.vertical_splitter.addWidget(self.main_splitter)
        self.vertical_splitter.addWidget(self.bottom_tabs)
        self.vertical_splitter.setSizes([700, 200])
        root_layout.addWidget(self.vertical_splitter, 1)
        self.setCentralWidget(root)
        self._restore_global_layout()

    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("workbenchTopBar")
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(10, 1, 10, 1)
        outer.setSpacing(1)
        identity_row = QHBoxLayout()
        identity_row.setSpacing(5)
        self.project_label = QLabel("未打开项目")
        self.project_label.setObjectName("projectIdentity")
        self.line_label = QLabel("测线：--")
        self.line_label.setObjectName("lineIdentity")
        identity_row.addWidget(self.project_label)
        identity_row.addWidget(self.line_label)
        identity_row.addSpacing(14)
        for key, text in WORKSPACES.items():
            button = QPushButton(text)
            button.setCheckable(True)
            button.setObjectName("workspaceButton")
            button.setToolTip(f"切换到{text}页面")
            button.clicked.connect(lambda _checked=False, value=key: self.switch_workspace(value))
            self.workspace_buttons[key] = button
            identity_row.addWidget(button)
        identity_row.addStretch(1)
        self.workflow_status_label = QLabel("状态：未打开项目")
        self.workflow_status_label.setObjectName("workflowStatus")
        self.workflow_status_label.setMinimumWidth(180)
        identity_row.addWidget(self.workflow_status_label)
        outer.addLayout(identity_row)

        self.create_action = QPushButton("新建项目")
        self.create_action.setToolTip("创建一个新的勘探项目。")
        self.create_action.clicked.connect(self.choose_create_project)
        self.open_action = QPushButton("打开项目")
        self.open_action.setToolTip("打开已有 MyGPR 项目目录。")
        self.open_action.clicked.connect(self.choose_project)
        self.import_action = QPushButton("快速打开文件")
        self.import_action.setToolTip("先临时打开数据，检查无误后再归档为正式项目。")
        self.import_action.clicked.connect(self.choose_loose_path)
        self.import_folder_action = QPushButton("快速打开文件夹")
        self.import_folder_action.setToolTip("快速打开 A-scan 文件夹并创建临时检查工程。")
        self.import_folder_action.clicked.connect(self.choose_loose_folder)
        self.add_line_action = QPushButton("导入测线")
        self.add_line_action.setToolTip("向当前正式项目导入一条新测线。")
        add_line_menu = QMenu(self.add_line_action)
        add_line_file = QAction("导入数据文件", self)
        add_line_file.triggered.connect(self.choose_import_line)
        add_line_folder = QAction("导入 A-scan 文件夹", self)
        add_line_folder.triggered.connect(self.choose_import_line_folder)
        add_line_menu.addActions([add_line_file, add_line_folder])
        self.add_line_action.setMenu(add_line_menu)
        self.formalize_action = QPushButton("归档为正式项目")
        self.formalize_action.setToolTip("复制原始数据并建立正式项目归档，归档后可进入测线处理。")
        self.formalize_action.clicked.connect(self.choose_formal_destination)
        self.sidecar_action = QPushButton("匹配辅助文件")
        self.sidecar_action.setToolTip("为当前测线匹配 RTK、IMU、高度计或逐道时间戳等辅助文件。")
        self.sidecar_action.clicked.connect(self.choose_sidecar)
        self.qc_action = QPushButton("运行质控")
        self.qc_action.setToolTip("检查数据结构、辅助文件时序和处理前条件。")
        self.qc_action.clicked.connect(self.run_selected_line_qc_async)
        self.ack_warning_action = QPushButton("确认警告")
        self.ack_warning_action.setToolTip("对需要人工复核的检查项记录现场说明。")
        self.ack_warning_action.clicked.connect(self.acknowledge_selected_warning)
        self.legacy_action = QPushButton("打开完整处理")
        self.legacy_action.setToolTip("在完整处理窗口中继续细化当前测线；结果只有点击“保存处理结果”才写入当前项目。")
        self.legacy_action.clicked.connect(self.open_selected_in_legacy)
        self.save_legacy_action = QPushButton("保存处理结果")
        self.save_legacy_action.setToolTip("把完整处理窗口中的当前结果保存为项目处理结果。")
        self.save_legacy_action.clicked.connect(self.save_legacy_result)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        action_row.addWidget(self._action_section_label("工程"))
        for widget in (
            self.create_action,
            self.open_action,
            self.import_action,
            self.import_folder_action,
            self.add_line_action,
            self.formalize_action,
        ):
            widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            action_row.addWidget(widget)
        action_row.addSpacing(10)
        action_row.addWidget(self._action_section_label("测线"))
        for widget in (
            self.sidecar_action,
            self.qc_action,
            self.ack_warning_action,
            self.legacy_action,
            self.save_legacy_action,
        ):
            widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            action_row.addWidget(widget)
        action_row.addStretch(1)
        outer.addLayout(action_row)
        self.workspace_buttons["data_management"].setChecked(True)
        return bar


    @staticmethod
    def _action_section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("actionSectionLabel")
        label.setMinimumWidth(54)
        return label

    def _build_inspector(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("contextInspector")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        title = QLabel("当前信息")
        title.setObjectName("panelTitle")
        self.inspector_title = QLabel("未选择对象")
        self.inspector_title.setObjectName("inspectorTitle")
        self.inspector_body = QLabel("选择左侧项目资源，查看测线、检查结果或处理结果的详细信息。")
        self.inspector_body.setWordWrap(True)
        self.inspector_body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(title)
        layout.addWidget(self.inspector_title)
        layout.addWidget(self.inspector_body)
        layout.addStretch(1)
        return panel

    def _build_future_workspace(self, label: str) -> QWidget:
        page = QWidget()
        page.setObjectName("futureWorkspace")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(48, 48, 48, 48)
        title = QLabel(label)
        title.setObjectName("futureTitle")
        body = QLabel("该页面会按勘探项目流程逐步接入更多能力。")
        body.setWordWrap(True)
        body.setMaximumWidth(620)
        layout.addWidget(title)
        layout.addWidget(body)
        layout.addStretch(1)
        return page

    @staticmethod
    def _make_log_table(headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        return table

    def switch_workspace(self, key: str) -> None:
        if key not in WORKSPACES:
            raise KeyError(key)
        if key not in self.workspace_pages:
            raise KeyError(key)
        self.active_workspace = key
        for name, button in self.workspace_buttons.items():
            button.setChecked(name == key)
        self.workspace_stack.setCurrentWidget(self.workspace_pages[key])
        if key == "processing_lab":
            self._open_selected_in_processing_lab()
        elif key == "interpretation":
            self._open_selected_in_interpretation()
        elif key == "spatial" and self.project is not None:
            self.spatial_synthesis.open_project(self.project)
        elif key == "delivery" and self.project is not None:
            self.delivery_page.open_project(self.project)
        if self.project is not None:
            self._append_log("页面", "已切换", WORKSPACES[key])
        self.statusBar().showMessage(f"当前页面：{WORKSPACES[key]}")

    def _open_selected_in_processing_lab(self) -> None:
        if self.project is None or self.selected_line_id is None:
            return
        if self.project.manifest.temporary:
            self.statusBar().showMessage("临时项目仅允许浏览和检查；归档为正式项目后可处理。")
            return
        report = QcService(self.project).run_line_qc(self.selected_line_id)
        self._sync_actions(report)
        try:
            assert_command_allowed(self._command_state(report), "processing")
        except PermissionError as exc:
            self.statusBar().showMessage(str(exc))
            return
        if (
            self.processing_lab.session is None
            or self.processing_lab.session.project is not self.project
            or self.processing_lab.session.line_id != self.selected_line_id
        ):
            self.processing_lab.open_line(self.project, self.selected_line_id)

    def _open_selected_in_interpretation(self) -> None:
        if self.project is None or self.selected_line_id is None:
            return
        if self.project.manifest.temporary:
            self.statusBar().showMessage("临时项目不能保存正式目标标注。")
            return
        if (
            self.interpretation_workbench.project is not self.project
            or self.interpretation_workbench.line_id != self.selected_line_id
        ):
            self.interpretation_workbench.open_line(self.project, self.selected_line_id)

    def _restore_global_layout(self) -> None:
        layout = WorkspaceSessionService.load_global_layout_for("workbench")
        self._pending_global_layout = layout or None
        self._apply_global_layout(layout)

    def _apply_global_layout(self, layout: dict[str, Any] | None) -> None:
        layout = layout or {}
        horizontal = layout.get("horizontal")
        vertical = layout.get("vertical")
        if isinstance(horizontal, list) and len(horizontal) == 3:
            self.main_splitter.setSizes([int(value) for value in horizontal])
        if isinstance(vertical, list) and len(vertical) == 2:
            self.vertical_splitter.setSizes([int(value) for value in vertical])

    def _save_global_layout(self) -> None:
        WorkspaceSessionService.save_global_layout_for(
            "workbench",
            {
                "horizontal": self.main_splitter.sizes(),
                "vertical": self.vertical_splitter.sizes(),
            },
        )

    def choose_project(self) -> None:
        root = QFileDialog.getExistingDirectory(self, "选择 MyGPR 项目目录")
        if root:
            self.open_project(root)

    def choose_create_project(self) -> None:
        root = QFileDialog.getExistingDirectory(self, "选择新项目目录")
        if not root:
            return
        name, accepted = QInputDialog.getText(self, "新建 MyGPR 项目", "项目名称", text=Path(root).name)
        if accepted and name.strip():
            self.create_project(root, name=name.strip())

    def create_project(self, root: str | Path, *, name: str) -> None:
        self._close_project()
        self.project = ProjectService.create(root, name=name, temporary=False)
        self.session = WorkspaceSessionService(self.project)
        self._temporary_root = None
        self._refresh_project_ui()
        self._sync_actions()
        self.statusBar().showMessage(f"已创建项目：{name}")

    def choose_loose_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "快速打开 GPR 数据", "", supported_file_dialog_filter())
        if path:
            self.open_loose_path(path)

    def choose_loose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "快速打开 A-scan 文件夹")
        if path:
            self.open_loose_path(path)

    def choose_import_line(self) -> None:
        if self.project is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "导入测线数据", "", supported_file_dialog_filter())
        if path:
            self.import_line_async(path)

    def choose_import_line_folder(self) -> None:
        if self.project is None:
            return
        path = QFileDialog.getExistingDirectory(self, "导入 A-scan 测线文件夹")
        if path:
            self.import_line_async(path)

    def import_line_async(self, source: str | Path) -> None:
        if self.project is None:
            raise RuntimeError("未打开项目")
        self._set_task_controls_enabled(False)
        self._append_task("导入测线", "运行中", str(source))
        self._start_task(
            IngestService.import_into_project,
            self.project,
            source,
            verify_hashes=False,
            on_success=self._finish_import_line,
            task_name="导入测线",
        )

    def _finish_import_line(self, line: LineRecordV1) -> None:
        self.selected_line_id = line.line_id
        self._refresh_project_ui(reset_documents=False)
        self._select_tree_line(line.line_id)
        self._set_task_controls_enabled(True)
        self.statusBar().showMessage(f"测线已导入：{line.name}")
        if self.project is not None and not self.project.manifest.temporary:
            self._start_integrity_verification([line.line_id])

    def choose_sidecar(self) -> None:
        if self.project is None or self.selected_line_id is None:
            return
        labels = ["RTK", "IMU", "高度计", "逐道时间戳"]
        label, accepted = QInputDialog.getItem(self, "匹配辅助文件", "辅助文件类型", labels, 0, False)
        if not accepted:
            return
        kind = {"RTK": "rtk", "IMU": "imu", "高度计": "altimeter", "逐道时间戳": "trace_timestamps"}[label]
        path, _ = QFileDialog.getOpenFileName(self, f"选择 {label} 文件", "", "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        IngestService.assign_sidecar(self.project, self.selected_line_id, kind, path)
        self._refresh_project_ui(reset_documents=False)
        self._select_tree_line(self.selected_line_id)
        self.statusBar().showMessage(f"已更新 {label} 匹配；请重新运行质控。")
        if not self.project.manifest.temporary:
            self._start_integrity_verification([self.selected_line_id])

    def open_project(self, root: str | Path) -> None:
        self._close_project()
        self.project = ProjectService.open(root)
        self.session = WorkspaceSessionService(self.project)
        self._temporary_root = None
        self._refresh_project_ui()
        restored = self.session.load_project_session()
        self.selected_line_id = restored.get("selected_line_id")
        for document_id in restored.get("open_documents", []):
            if str(document_id).startswith("line:"):
                self.open_line_document(str(document_id).split(":", 1)[1])
            elif str(document_id).startswith("qc:"):
                self.run_line_qc(str(document_id).split(":", 1)[1])
        self.switch_workspace(restored.get("active_workspace", "data_management"))
        self._select_tree_line(self.selected_line_id)
        self._append_log("项目", "已打开", f"{self.project.manifest.name}，测线 {len(self.project.manifest.line_ids)} 条")
        if self.selected_line_id:
            try:
                self._append_log("当前测线", "已选中", self.project.get_line(self.selected_line_id).name)
            except Exception:
                pass
        self._sync_actions()

    def open_loose_path(self, path: str | Path) -> None:
        self._close_project()
        self.project = IngestService.open_temporary(path)
        self._temporary_root = self.project.root
        self.session = WorkspaceSessionService(self.project)
        self._refresh_project_ui()
        line = self.project.list_lines()[0]
        self.selected_line_id = line.line_id
        self.open_line_document(line.line_id)
        self._select_tree_line(line.line_id)
        self._sync_actions()
        self.statusBar().showMessage("已创建临时检查项目；正式处理前请归档为正式项目。")

    def choose_formal_destination(self) -> None:
        if self.project is None or not self.project.manifest.temporary:
            return
        root = QFileDialog.getExistingDirectory(self, "选择正式工程存储目录")
        if root:
            self.formalize_project_async(root, name=Path(root).name)

    def formalize_project_async(self, destination: str | Path, *, name: str) -> None:
        if self.project is None or not self.project.manifest.temporary:
            raise RuntimeError("当前不是临时项目")
        source = self.project
        self._set_task_controls_enabled(False)
        self._append_task("归档正式项目", "运行中", "复制只读原始数据")
        self._start_task(
            IngestService.formalize,
            source,
            destination,
            name=name,
            verify_hashes=False,
            on_success=lambda formal: self._finish_formalize(source, formal),
            task_name="归档正式项目",
        )

    def formalize_project(self, destination: str | Path, *, name: str) -> None:
        if self.project is None or not self.project.manifest.temporary:
            raise RuntimeError("当前不是临时项目")
        old = self.project
        formal = IngestService.formalize(old, destination, name=name)
        old.close()
        self.project = formal
        self.session = WorkspaceSessionService(formal)
        self._temporary_root = None
        self._refresh_project_ui()
        self._select_tree_line(self.selected_line_id)
        self._sync_actions()
        self.statusBar().showMessage("正式项目已建立，原始数据已复制并完成完整性验证。")

    def _start_integrity_verification(self, line_ids: list[str] | None = None) -> None:
        if self.project is None or self.project.manifest.temporary:
            return
        self._append_task("原始数据完整性", "运行中", "后台计算完整 SHA-256")
        self._start_task(
            IngestService.verify_project_integrity,
            self.project,
            line_ids,
            on_success=self._finish_integrity_verification,
            task_name="原始数据完整性",
        )

    def _finish_integrity_verification(self, line_ids: list[str]) -> None:
        self._refresh_project_ui(reset_documents=False)
        self._select_tree_line(self.selected_line_id)
        self.statusBar().showMessage(f"已完成 {len(line_ids)} 条测线的原始数据完整性验证。")

    def _finish_formalize(self, old: ProjectService, formal: ProjectService) -> None:
        old.close()
        self.project = formal
        self.session = WorkspaceSessionService(formal)
        self._temporary_root = None
        self._refresh_project_ui()
        self._select_tree_line(self.selected_line_id)
        self._set_task_controls_enabled(True)
        self._sync_actions()
        self.statusBar().showMessage("正式项目已建立，原始数据完整性正在后台验证。")
        self._start_integrity_verification()

    def _refresh_project_ui(self, *, reset_documents: bool = True) -> None:
        self.project_tree.clear()
        if reset_documents:
            self._clear_documents()
        if self.project is None:
            self.project_label.setText("未打开项目")
            self.line_label.setText("测线：--")
            self._update_workflow_status(state="empty")
            self._show_overview_document()
            return
        suffix = " · 临时检查" if self.project.manifest.temporary else ""
        self.project_label.setText(f"{self.project.manifest.name}{suffix}")
        self._update_workflow_status()
        root = QTreeWidgetItem(["测线", str(len(self.project.manifest.line_ids))])
        root.setData(0, Qt.ItemDataRole.UserRole, ("group", "lines"))
        for line in self.project.list_lines():
            item = QTreeWidgetItem([line.name, line_status_label(line.status)])
            item.setData(0, Qt.ItemDataRole.UserRole, ("line", line.line_id))
            root.addChild(item)
        self.project_tree.addTopLevelItem(root)
        results = self.project.list_processing_results()
        result_group = QTreeWidgetItem(["处理结果", str(len(results))])
        result_group.setData(0, Qt.ItemDataRole.UserRole, ("group", "results"))
        for result in results:
            label = str(result.name or result.result_id)
            child = QTreeWidgetItem([label, result.line_id])
            child.setToolTip(
                0,
                "\n".join(
                    [
                        f"处理结果：{label}",
                        f"结果 ID：{result.result_id}",
                        f"测线 ID：{result.line_id}",
                        "双击/选择后在文档区打开预览。",
                    ]
                ),
            )
            child.setData(
                0,
                Qt.ItemDataRole.UserRole,
                ("result", result.line_id, result.result_id),
            )
            result_group.addChild(child)
        self.project_tree.addTopLevelItem(result_group)
        for label, key in (("目标标注", "interpretations"), ("交付成果", "exports")):
            count = len([path for path in (self.project.root / key).rglob("*") if path.is_file()])
            item = QTreeWidgetItem([label, str(count)])
            item.setData(0, Qt.ItemDataRole.UserRole, ("group", key))
            self.project_tree.addTopLevelItem(item)
        root.setExpanded(True)
        self._show_overview_document()

    def _show_overview_document(self) -> None:
        existing = self._find_document("overview")
        if existing >= 0:
            return
        body = QWidget()
        body.setProperty("document_id", "overview")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(28, 28, 28, 28)
        title = QLabel("项目与测线")
        title.setObjectName("documentTitle")
        text = QLabel(
            "从项目资源树选择测线。快速打开散装数据会创建临时检查项目；"
            "转为正式项目后，原始数据进入只读归档并接受完整性校验。"
        )
        text.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(text)
        layout.addStretch(1)
        self.document_tabs.addTab(body, "项目概览")

    def open_line_document(self, line_id: str) -> None:
        if self.project is None:
            return
        line = self.project.get_line(line_id)
        document_id = f"line:{line_id}"
        existing = self._find_document(document_id)
        if existing >= 0:
            self.document_tabs.setCurrentIndex(existing)
            return
        body = QWidget()
        body.setProperty("document_id", document_id)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        figure = Figure(facecolor="#101820")
        canvas = FigureCanvas(figure)
        axis = figure.add_subplot(111)
        axis.set_facecolor("#101820")
        axis.tick_params(colors="#AFC2CF")
        axis.set_title(line.name, color="#EAF4F4")
        try:
            # Use the same project-aware loader as the processing page.
            # Airborne CSV files store per-sample rows with longitude/latitude/height
            # side columns; using the generic CSV matrix reader would display those
            # columns as false vertical bands.
            session = ProcessingSessionService.open_line(
                self.project,
                line_id,
                enforce_processing_gate=False,
            )
            data = np.asarray(session.original_data, dtype=np.float32)
            if data.ndim == 2 and data.size:
                axis.imshow(data, cmap="gray", aspect="auto", interpolation="nearest")
                axis.set_xlabel("道号", color="#AFC2CF")
                axis.set_ylabel("采样点", color="#AFC2CF")
            else:
                axis.text(0.5, 0.5, "无法生成二维预览", ha="center", color="#EAF4F4")
        except Exception as exc:
            axis.text(0.5, 0.5, f"预览失败\n{exc}", ha="center", color="#F3B66B")
        figure.tight_layout()
        layout.addWidget(canvas)
        self.document_tabs.addTab(body, f"测线 · {line.name}")
        self.document_tabs.setCurrentWidget(body)
        self.selected_line_id = line_id
        self._update_line_context(line)
        self._append_log("测线预览", "已打开", line.name)
        self._sync_actions()

    def run_selected_line_qc(self) -> QcReportV1:
        if self.selected_line_id is None:
            raise RuntimeError("未选择测线")
        return self.run_line_qc(self.selected_line_id)

    def run_selected_line_qc_async(self) -> None:
        if self.project is None or self.selected_line_id is None:
            return
        self._set_task_controls_enabled(False)
        self._append_task("入库质控", "运行中", self.selected_line_id)
        self._start_task(
            QcService(self.project).run_line_qc,
            self.selected_line_id,
            on_success=self._finish_qc,
            task_name="入库质控",
        )

    def _finish_qc(self, report: QcReportV1) -> None:
        self._show_qc_document(report)
        self._populate_warning_drawer(report)
        self._set_task_controls_enabled(True)
        self._sync_actions(report)

    def acknowledge_selected_warning(self) -> None:
        if self.project is None or self.selected_line_id is None:
            return
        row = self.warning_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "确认检查提示", "请先在底部“检查提示”中选择一条需要复核的记录。")
            return
        severity_item = self.warning_table.item(row, 0)
        code_item = self.warning_table.item(row, 1)
        if severity_item is None or code_item is None:
            QMessageBox.information(self, "确认警告", "请选择一条完整的检查记录。")
            return
        raw_severity = str(severity_item.data(Qt.ItemDataRole.UserRole) or severity_item.text()).lower()
        if raw_severity != "warning":
            QMessageBox.information(self, "确认警告", "只能对待复核级别检查项记录人工确认。")
            return
        note, accepted = QInputDialog.getText(self, "确认质控警告", "现场确认说明")
        if not accepted or not note.strip():
            return
        report = QcService(self.project).acknowledge_warning(
            self.selected_line_id, str(code_item.data(Qt.ItemDataRole.UserRole) or code_item.text()), note.strip()
        )
        self._show_qc_document(report)
        self._populate_warning_drawer(report)
        self._sync_actions(report)
        self.statusBar().showMessage("已记录质控警告人工确认。")

    def run_line_qc(self, line_id: str) -> QcReportV1:
        if self.project is None:
            raise RuntimeError("未打开项目")
        report = QcService(self.project).run_line_qc(line_id)
        self._show_qc_document(report)
        self._populate_warning_drawer(report)
        self._sync_actions(report)
        return report

    def _show_qc_document(self, report: QcReportV1) -> None:
        document_id = f"qc:{report.line_id}"
        existing = self._find_document(document_id)
        if existing >= 0:
            old = self.document_tabs.widget(existing)
            self.document_tabs.removeTab(existing)
            old.deleteLater()
        body = QWidget()
        body.setProperty("document_id", document_id)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(18, 18, 18, 18)
        if not report.can_process:
            title = QLabel("存在阻断问题")
            title.setObjectName("qcBlockTitle")
        elif report.requires_review:
            title = QLabel("需要现场复核")
            title.setObjectName("qcReviewTitle")
        else:
            title = QLabel("可进入测线处理")
            title.setObjectName("qcPassTitle")
        table = QTableWidget(len(report.items), 3)
        table.setHorizontalHeaderLabels(["等级", "检查内容", "说明"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for row, item in enumerate(report.items):
            severity_cell = QTableWidgetItem(severity_label(item.severity))
            severity_cell.setData(Qt.ItemDataRole.UserRole, item.severity)
            code_cell = QTableWidgetItem(qc_code_label(item.code))
            code_cell.setData(Qt.ItemDataRole.UserRole, item.code)
            table.setItem(row, 0, severity_cell)
            table.setItem(row, 1, code_cell)
            table.setItem(row, 2, QTableWidgetItem(item.message))
        layout.addWidget(title)
        layout.addWidget(table)
        self.document_tabs.addTab(body, f"质控 · {report.line_id}")
        self.document_tabs.setCurrentWidget(body)

    def _populate_warning_drawer(self, report: QcReportV1) -> None:
        self.warning_table.setRowCount(len(report.items))
        for row, item in enumerate(report.items):
            severity_cell = QTableWidgetItem(severity_label(item.severity))
            severity_cell.setData(Qt.ItemDataRole.UserRole, item.severity)
            code_cell = QTableWidgetItem(qc_code_label(item.code))
            code_cell.setData(Qt.ItemDataRole.UserRole, item.code)
            self.warning_table.setItem(row, 0, severity_cell)
            self.warning_table.setItem(row, 1, code_cell)
            self.warning_table.setItem(row, 2, QTableWidgetItem(item.message))

    def open_selected_in_legacy(self) -> None:
        if self.project is None or self.selected_line_id is None:
            return
        report = QcService(self.project).run_line_qc(self.selected_line_id)
        state = self._command_state(report)
        assert_command_allowed(state, "processing")
        if self.legacy_bridge is None:
            self.legacy_bridge = LegacyProcessingBridge(self.project, self)
            self.legacy_bridge.result_saved.connect(self._on_result_saved)
        self.legacy_bridge.open_line(self.selected_line_id, state=state)
        self._sync_actions(report)

    def save_legacy_result(self) -> None:
        if self.legacy_bridge is None:
            return
        result = self.legacy_bridge.save_current_result()
        self.statusBar().showMessage(f"已保存处理结果：{result.result_id}")

    def _on_result_saved(self, result: Any) -> None:
        self._append_log("处理结果", "已保存", getattr(result, "result_id", ""))
        self._refresh_project_ui(reset_documents=False)
        self._select_tree_line(self.selected_line_id)

    def _on_delivery_package_built(self, package_path: str) -> None:
        package = Path(package_path)
        self._populate_evidence_table_from_package(package)
        self._refresh_project_ui(reset_documents=False)
        self._append_log("交付成果", "已生成", str(package))
        self.bottom_tabs.setCurrentWidget(self.evidence_table)

    def _populate_evidence_table_from_package(self, package: Path) -> None:
        manifest_path = package / "manifest.json"
        rows: list[tuple[str, str, str]] = []

        def add_row(role: str, path: Path | str, status: str = "已索引") -> None:
            location = str(path)
            if self.project is not None:
                try:
                    absolute = Path(path)
                    if not absolute.is_absolute():
                        absolute = self.project.resolve_relative_path(str(path))
                    location = absolute.relative_to(self.project.root).as_posix()
                except Exception:
                    location = str(path)
            rows.append((role, location, status))

        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                rows.append((delivery_role_label("delivery_manifest"), str(manifest_path), f"读取失败：{exc}"))
            else:
                add_row(delivery_role_label("delivery_manifest"), manifest_path)
                add_row(delivery_role_label("delivery_report"), package / "report.md", "已生成")
                add_row(delivery_role_label("delivery_checksums"), package / "checksums.sha256", "已生成")
                spatial_path = manifest.get("spatial_synthesis")
                if spatial_path:
                    add_row(delivery_role_label("spatial_synthesis"), str(spatial_path))
                for item in manifest.get("evidence", []):
                    role = delivery_role_label(str(item.get("role") or "evidence"))
                    path = str(item.get("path") or item.get("data_path") or "")
                    status = "已索引" if path else "路径缺失"
                    add_row(role, path or "--", status)
        else:
            rows.append((delivery_role_label("delivery_manifest"), str(manifest_path), "未找到"))

        self.evidence_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self.evidence_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _command_state(self, report: QcReportV1 | None = None) -> str:
        if self.project is None:
            return "empty"
        if self.selected_line_id is None and report is None:
            return "no_line"
        if self.project.manifest.temporary:
            return "temporary_preview"
        if report is not None and not report.can_process:
            return "qc_blocked"
        if report is not None and report.requires_review:
            return "qc_review_required"
        return "formal_ready"

    def _sync_actions(self, report: QcReportV1 | None = None) -> None:
        has_project = self.project is not None
        has_line = has_project and self.selected_line_id is not None
        temporary = bool(has_project and self.project and self.project.manifest.temporary)
        self.formalize_action.setEnabled(temporary)
        self.add_line_action.setEnabled(has_project and not temporary)
        self.qc_action.setEnabled(has_line)
        self.ack_warning_action.setEnabled(has_line)
        self.sidecar_action.setEnabled(has_line)
        state = self._command_state(report)
        can_process = state == "formal_ready"
        self.legacy_action.setEnabled(has_line and can_process)
        self.save_legacy_action.setEnabled(bool(self.legacy_bridge and self.legacy_bridge.window))
        self._update_workflow_status(report=report, state=state)

    def _update_workflow_status(
        self,
        *,
        report: QcReportV1 | None = None,
        state: str | None = None,
    ) -> None:
        label = getattr(self, "workflow_status_label", None)
        if label is None:
            return
        state = state or self._command_state(report)
        text_by_state = {
            "empty": "状态：未打开项目",
            "no_line": "状态：请选择测线",
            "temporary_preview": "状态：临时检查 · 不可正式处理",
            "qc_blocked": "状态：质控阻断",
            "qc_review_required": "状态：待确认质控警告",
            "formal_ready": "状态：正式就绪 · 可处理",
        }
        tone_by_state = {
            "empty": "neutral",
            "no_line": "neutral",
            "temporary_preview": "warning",
            "qc_blocked": "danger",
            "qc_review_required": "warning",
            "formal_ready": "good",
        }
        label.setText(text_by_state.get(state, f"状态：{state}"))
        label.setProperty("tone", tone_by_state.get(state, "neutral"))
        try:
            label.style().unpolish(label)
            label.style().polish(label)
        except Exception:
            pass
        label.update()

    def _on_tree_selection(self) -> None:
        items = self.project_tree.selectedItems()
        if not items:
            return
        payload = items[0].data(0, Qt.ItemDataRole.UserRole)
        if isinstance(payload, tuple) and payload[0] == "line":
            self.selected_line_id = payload[1]
            self.open_line_document(payload[1])
            if self.active_workspace == "processing_lab":
                self._open_selected_in_processing_lab()
            elif self.active_workspace == "interpretation":
                self._open_selected_in_interpretation()
        elif isinstance(payload, tuple) and payload[0] == "result" and len(payload) >= 3:
            line_id = str(payload[1])
            result_id = str(payload[2])
            self.selected_line_id = line_id
            if self.active_workspace != "data_management":
                self.switch_workspace("data_management")
            self.open_result_document(line_id, result_id)

    def open_result_document(self, line_id: str, result_id: str) -> None:
        """Open a saved processing result from the project tree.

        The project-first UI exposes processing versions as first-class project
        resources.  Earlier builds counted ``results/*/result.json`` files in
        the resource tree but did not let users inspect the saved version from
        that tree.  This document view closes that loop with a read-only B-scan
        preview and a compact processing-chain summary.
        """
        if self.project is None:
            return
        document_id = f"result:{line_id}:{result_id}"
        existing = self._find_document(document_id)
        if existing >= 0:
            self.document_tabs.setCurrentIndex(existing)
            return
        try:
            payload = self.project.load_processing_result(result_id, line_id=line_id)
        except Exception as exc:
            QMessageBox.warning(self, "打开处理结果失败", str(exc))
            return
        record = payload["record"]
        data = np.asarray(payload.get("data"))
        body = QWidget()
        body.setProperty("document_id", document_id)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel(f"处理结果 · {record.name}")
        title.setObjectName("documentTitle")
        layout.addWidget(title)

        summary = QLabel(
            "\n".join(
                [
                    f"结果 ID：{record.result_id}",
                    f"测线 ID：{record.line_id}",
                    f"数据尺寸：{tuple(data.shape) if data.size else '--'}",
                    f"创建时间：{record.created_at or '--'}",
                ]
            )
        )
        summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(summary)

        figure = Figure(facecolor="#101820")
        canvas = FigureCanvas(figure)
        axis = figure.add_subplot(111)
        axis.set_facecolor("#101820")
        axis.tick_params(colors="#AFC2CF")
        axis.set_title(record.name, color="#EAF4F4")
        if data.ndim == 2 and data.size:
            axis.imshow(data, cmap="gray", aspect="auto", interpolation="nearest")
        else:
            axis.text(0.5, 0.5, "该处理结果没有可预览的二维数据", ha="center", color="#EAF4F4")
        figure.tight_layout()
        canvas.setMinimumHeight(260)
        layout.addWidget(canvas, 1)

        chain = list(record.processing_chain or [])
        table = QTableWidget(max(1, len(chain)), 3)
        table.setHorizontalHeaderLabels(["步骤", "方法", "参数/说明"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        if chain:
            for row, step in enumerate(chain):
                method = str(
                    step.get("method")
                    or step.get("method_key")
                    or step.get("name")
                    or step.get("stage")
                    or "未命名步骤"
                )
                params = step.get("params", step)
                detail = json.dumps(params, ensure_ascii=False, default=str)
                if len(detail) > 240:
                    detail = detail[:237] + "..."
                table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
                table.setItem(row, 1, QTableWidgetItem(method))
                table.setItem(row, 2, QTableWidgetItem(detail))
        else:
            table.setItem(0, 0, QTableWidgetItem("--"))
            table.setItem(0, 1, QTableWidgetItem("未记录"))
            table.setItem(0, 2, QTableWidgetItem("该处理结果未保存处理链。"))
        table.setMaximumHeight(190)
        layout.addWidget(table)

        self.document_tabs.addTab(body, f"处理结果 · {record.name}")
        self.document_tabs.setCurrentWidget(body)
        self._update_result_context(record)
        self._sync_actions()

    def _update_result_context(self, result: Any) -> None:
        self.line_label.setText(f"测线：{result.line_id}")
        self.inspector_title.setText(str(result.name or result.result_id))
        self.inspector_body.setText(
            "\n".join(
                [
                    "类型：处理结果",
                    f"结果 ID：{result.result_id}",
                    f"测线 ID：{result.line_id}",
                    f"处理步骤：{len(result.processing_chain or [])}",
                    f"数据文件：{result.data_path}",
                ]
            )
        )

    def _select_tree_line(self, line_id: str | None) -> None:
        if not line_id:
            return
        iterator = self.project_tree.invisibleRootItem()
        for top_index in range(iterator.childCount()):
            top = iterator.child(top_index)
            for child_index in range(top.childCount()):
                child = top.child(child_index)
                payload = child.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(payload, tuple) and payload == ("line", line_id):
                    self.project_tree.setCurrentItem(child)
                    return

    def _update_line_context(self, line: LineRecordV1) -> None:
        self.line_label.setText(f"测线：{line.name}")
        self.inspector_title.setText(line.name)
        self.inspector_body.setText(
            "\n".join(
                [
                    f"ID：{line.line_id}",
                    f"格式：{line.source_format}",
                    f"状态：{line_status_label(line.status)}",
                    f"原始文件数：{len(line.raw_files)}",
                    f"辅助文件：{', '.join(sidecar_label(k) for k in sorted(line.sidecars)) or '未发现'}",
                ]
            )
        )
        self._update_workflow_status()

    def _line_primary_path(self, line: LineRecordV1) -> Path:
        path = Path(line.raw_files[0].path)
        return path if path.is_absolute() else self.project.resolve_relative_path(path)  # type: ignore[union-attr]

    def _find_document(self, document_id: str) -> int:
        for index in range(self.document_tabs.count()):
            if self.document_tabs.widget(index).property("document_id") == document_id:
                return index
        return -1

    def _close_document(self, index: int) -> None:
        widget = self.document_tabs.widget(index)
        if widget.property("document_id") == "overview":
            return
        self.document_tabs.removeTab(index)
        widget.deleteLater()

    def _clear_documents(self) -> None:
        while self.document_tabs.count():
            widget = self.document_tabs.widget(0)
            self.document_tabs.removeTab(0)
            widget.deleteLater()

    def _append_log(self, event: str, status: str, detail: str) -> None:
        row = self.log_table.rowCount()
        self.log_table.insertRow(row)
        timestamp = utc_now().replace("T", " ").split("+")[0]
        for col, value in enumerate((timestamp, event, f"{status}: {detail}")):
            self.log_table.setItem(row, col, QTableWidgetItem(value))

    def _append_task(self, task: str, status: str, detail: str) -> None:
        row = self.task_table.rowCount()
        self.task_table.insertRow(row)
        for col, value in enumerate((task, status, detail)):
            self.task_table.setItem(row, col, QTableWidgetItem(value))

    def _set_task_controls_enabled(self, enabled: bool) -> None:
        self.import_action.setEnabled(enabled)
        self.import_folder_action.setEnabled(enabled)
        self.add_line_action.setEnabled(enabled and self.project is not None)
        self.create_action.setEnabled(enabled)
        self.open_action.setEnabled(enabled)
        if enabled:
            self._sync_actions()
        else:
            self.formalize_action.setEnabled(False)
            self.qc_action.setEnabled(False)
            self.ack_warning_action.setEnabled(False)
            self.sidecar_action.setEnabled(False)
            self.legacy_action.setEnabled(False)

    def _start_task(
        self,
        operation,
        *args,
        on_success,
        task_name: str,
        **kwargs,
    ) -> None:
        thread = QThread(self)
        worker = WorkbenchTaskWorker(operation, *args, **kwargs)
        thread.worker = worker  # type: ignore[attr-defined]
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_success)
        worker.finished.connect(lambda _value: self._task_finished(task_name, thread))
        worker.failed.connect(lambda message: self._task_failed(task_name, message, thread))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._task_threads.add(thread)
        thread.start()

    def _task_finished(self, task_name: str, thread: QThread) -> None:
        self._append_task(task_name, "完成", "")
        self._task_threads.discard(thread)

    def _task_failed(self, task_name: str, message: str, thread: QThread) -> None:
        self._append_task(task_name, "失败", message)
        self._task_threads.discard(thread)
        self._set_task_controls_enabled(True)
        QMessageBox.critical(self, f"{task_name}失败", message)

    def _save_session(self) -> None:
        if self.session is None:
            return
        documents = [
            str(self.document_tabs.widget(index).property("document_id"))
            for index in range(self.document_tabs.count())
            if self.document_tabs.widget(index).property("document_id") != "overview"
        ]
        self.session.save_project_session(
            open_documents=documents,
            selected_line_id=self.selected_line_id,
            active_workspace=self.active_workspace,
        )

    def _close_project(self) -> None:
        if self.legacy_bridge is not None:
            self.legacy_bridge.close()
            self.legacy_bridge = None
        if self.project is not None:
            self._save_session()
            self.project.close()
            self.project = None
            self.session = None
        self.selected_line_id = None

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        if self._pending_global_layout is not None:
            pending_layout = self._pending_global_layout
            self._apply_global_layout(pending_layout)
            QTimer.singleShot(
                0,
                lambda layout=pending_layout: self._apply_global_layout(layout),
            )
            self._pending_global_layout = None

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        processing_lab = getattr(self, "processing_lab", None)
        shutdown_processing = getattr(processing_lab, "shutdown_background_task", None)
        if callable(shutdown_processing):
            shutdown_processing()
        for thread in list(self._task_threads):
            thread.requestInterruption()
            thread.quit()
            if not thread.wait(3000):
                thread.terminate()
                thread.wait(1000)
        temporary_root = self._temporary_root
        self._save_global_layout()
        self._release_plot_resources()
        self._close_project()
        if temporary_root is not None:
            shutil.rmtree(temporary_root, ignore_errors=True)
        super().closeEvent(event)

    def _release_plot_resources(self) -> None:
        for canvas in self.findChildren(FigureCanvas):
            try:
                canvas.figure.clear()
            except Exception:
                pass
            try:
                canvas.close()
                canvas.deleteLater()
            except Exception:
                pass

    def _apply_workbench_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget#workbenchRoot { background: #E9EDF1; color: #1B2831; }
            QFrame#workbenchTopBar { background: #F8FAFB; border-bottom: 1px solid #C8D0D7; }
            QLabel#projectIdentity { font-size: 15px; font-weight: 700; color: #102A33; }
            QLabel#lineIdentity { color: #58717A; }
            QLabel#actionSectionLabel { color: #6B7E86; font-weight: 700; padding: 5px 8px; background: #EEF3F6; border: 1px solid #D7E0E5; border-radius: 3px; }
            QLabel#workflowStatus { padding: 5px 10px; border-radius: 11px; border: 1px solid #D7E0E5; background: #EEF3F6; color: #49636C; font-weight: 700; }
            QLabel#workflowStatus[tone="good"] { background: #E7F7ED; border-color: #BEE7CC; color: #147A4E; }
            QLabel#workflowStatus[tone="warning"] { background: #FFF4D8; border-color: #F4D08B; color: #9A5B00; }
            QLabel#workflowStatus[tone="danger"] { background: #FEECEC; border-color: #F7BABA; color: #B42318; }
            QPushButton { background: #F8FAFB; border: 1px solid #C7D0D7; border-radius: 3px; padding: 3px 6px; }
            QPushButton:hover { border-color: #168C91; color: #0B7478; }
            QPushButton:checked, QPushButton#workspaceButton:checked { background: #123F48; color: #F4FBFB; border-color: #123F48; }
            QPushButton:disabled { color: #9CA8AE; background: #E5E9EC; }
            QTreeWidget#projectTree, QFrame#contextInspector, QTabWidget#bottomDrawer { background: #F7F9FA; border: 0; }
            QTabWidget#dataDocumentTabs::pane { border: 0; background: #101820; }
            QTabWidget#dataDocumentTabs QWidget { background: #101820; color: #D9E7EA; }
            QTabWidget#dataDocumentTabs QTableWidget { background: #F7F9FA; color: #1B2831; gridline-color: #C7D0D7; }
            QTabWidget#dataDocumentTabs QTabBar::tab { background: #DDE3E7; color: #344A53; padding: 8px 14px; }
            QTabWidget#dataDocumentTabs QTabBar::tab:selected { background: #101820; color: #EAF4F4; }
            QLabel#panelTitle { color: #607782; font-size: 11px; font-weight: 700; }
            QLabel#inspectorTitle, QLabel#futureTitle { font-size: 20px; font-weight: 700; color: #123F48; }
            QTabWidget#dataDocumentTabs QLabel#documentTitle { font-size: 20px; font-weight: 700; color: #57C4C8; }
            QLabel#qcPassTitle { font-size: 18px; font-weight: 700; color: #087A63; }
            QLabel#qcReviewTitle { font-size: 18px; font-weight: 700; color: #9A6700; }
            QLabel#qcBlockTitle { font-size: 18px; font-weight: 700; color: #B42318; }
            QSplitter::handle { background: #CBD3D9; }
            """
        )


__all__ = ["MyGPRWorkbenchWindow", "WORKSPACES"]
