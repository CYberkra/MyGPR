#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Product-grade field workbench shell for MyGPR."""
from __future__ import annotations
import math
import os
import csv
from pathlib import Path
from typing import Callable
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import Qt, QSize, QPoint, QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QIcon, QPixmap, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QDoubleSpinBox,
    QInputDialog,
    QLineEdit,
    QFileDialog,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
try:  # keep the workbench usable even if the branding helper changes later
    from ui.app_branding import make_mygpr_brand_pixmap
except Exception:  # pragma: no cover - fallback for isolated smoke tests
    make_mygpr_brand_pixmap = None
from core.field_project_store import FieldLineRecord, FieldProjectStore
from core.field_project_operations import (
    FieldProjectOperationError,
    RecentProjectsStore,
    create_project,
    batch_import_line_data,
    backup_project_archive,
    export_line_manifest_csv,
    import_line_data,
    next_line_id,
    preview_import_source,
    import_trajectory_file,
    open_project,
    project_dialog_filter,
    update_project_metadata,
)
from core.gpr_data_model import GPRDataSet
from core.manual_processing_chain import ManualProcessingSession
from core.field_project_status import ProjectStatusSnapshot, build_project_status_snapshot
from core.source_file_registry import source_status_label_for_line
from core.project_events import ProjectEventType
from ui.field_linkage_controller import ProjectLinkageController
from core.field_processing_bridge import get_field_method_categories
from core.trajectory_model import TrajectoryModel
from ui.field_panels.field_ui_styles import (
    ACCENT,
    ACCENT_DARK,
    BORDER,
    CARD_BG,
    DEFAULT_1080P_SIZE,
    CAPTURE_1080P_SIZE,
    COMPACT_1080P_FIT_SIZE,
    MIN_WORKBENCH_SIZE,
    COMPACT_SCREEN_HEIGHT_THRESHOLD,
    COMPACT_SCREEN_WIDTH_THRESHOLD,
    PAGE_BG,
    SUBTEXT,
    TEXT,
)
from ui.field_panels.widgets import Card, MetricCard, PlotCard
from ui.field_panels.plots import draw_bscan, draw_line_strip, draw_map, synthetic_bscan
from ui.field_panels.home_page import HomePageMixin
from ui.field_panels.interpretation_page import InterpretationPageMixin
from ui.field_panels.preview_helpers import FieldPreviewMixin
from ui.field_panels.project_page import ProjectPageMixin
from ui.field_panels.processing_page import ProcessingPageMixin
from ui.field_panels.table_utils import FieldTableMixin
from ui.field_panels.spatial_page import SpatialPageMixin
from ui.field_panels.delivery_page import DeliveryPageMixin
WORKSPACES = {
    "data_management": "项目管理",
    "processing_lab": "测线处理",
    "interpretation": "目标定位",
    "spatial": "空间成果",
    "delivery": "成果报告",
}
class FieldWorkbenchWindow(HomePageMixin, FieldTableMixin, FieldPreviewMixin, ProjectPageMixin, ProcessingPageMixin, InterpretationPageMixin, SpatialPageMixin, DeliveryPageMixin, QMainWindow):
    """A polished, target-like MyGPR field workbench shell."""
    def __init__(self, version_text: str = "MyGPR 勘探定位工作台") -> None:
        super().__init__()
        self.version_text = version_text
        self.active_workspace = "data_management"
        self.workspace_pages: dict[str, QWidget] = {}
        self.workspace_buttons: dict[str, QPushButton] = {}
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.selected_line = "L03"
        self.sample_path: Path | None = None
        self.sample_row_count = 0
        self.processing_gain = 1.0
        self.processing_applied = False
        self.processing_session: ManualProcessingSession | None = None
        self.processing_categories = get_field_method_categories()
        self.selected_processing_category = "背景抑制" if "背景抑制" in self.processing_categories else next(iter(self.processing_categories), "")
        default_methods = self.processing_categories.get(self.selected_processing_category, [])
        default_method = "subtracting_average_2D"
        if default_method not in {m.method_id for m in default_methods}:
            default_method = default_methods[0].method_id if default_methods else "dewow"
        self.selected_processing_method_id = default_method
        self.processing_param_widgets: dict[str, QWidget] = {}
        self.processing_category_combo: QComboBox | None = None
        self.processing_method_combo: QComboBox | None = None
        self.processing_params_body: QWidget | None = None
        self.processing_params_body_layout: QVBoxLayout | None = None
        self.last_processing_manifest: dict | None = None
        self.last_processing_error: str = ""
        self.processing_last_failed = False
        self.processing_save_button: QPushButton | None = None
        self.processing_execute_button: QPushButton | None = None
        self.processing_undo_step_button: QPushButton | None = None
        self.processing_reset_button: QPushButton | None = None
        self.processing_compare_button: QPushButton | None = None
        self.processing_chain_status_label: QLabel | None = None
        self.processing_history_table = None
        self.current_target_index = 2
        self._line_status_message = "已加载示例项目，等待导入真实测线文件。"
        self._init_preview_state()
        self.active_gpr_dataset: GPRDataSet | None = None
        self.processed_gpr_dataset: GPRDataSet | None = None
        self.trajectory_model: TrajectoryModel | None = None
        self.project_store: FieldProjectStore | None = None
        self.project_manifest = None
        self.project_root: Path | None = None
        self.recent_projects = RecentProjectsStore()
        self.linkage_controller = ProjectLinkageController(self)
        self.project_selector_label: QLabel | None = None
        self.project_selector_combo: QComboBox | None = None
        self.project_selector_button: QToolButton | None = None
        self.recent_project_combo: QComboBox | None = None
        self._init_field_project_store()
        self._refresh_project_status_snapshot()
        self.setWindowTitle(self.version_text or "MyGPR 勘探定位工作台")
        if make_mygpr_brand_pixmap:
            self.setWindowIcon(QIcon(make_mygpr_brand_pixmap(64)))
        # 15.6 英寸 1080P 笔记本作为主视觉基准。
        # 启动时读取 availableGeometry，而不是完整屏幕尺寸；Windows 任务栏可见时
        # 会自动进入 compact mode，减少顶部栏、侧栏和图像区高度。
        self.screen_profile = self._detect_screen_profile()
        self.compact_mode = self._should_use_compact_mode()
        self.setMinimumSize(*MIN_WORKBENCH_SIZE)
        self.resize(*self._initial_window_size())
        self.statusBar().hide()
        self._setup_ui()
        self._apply_style()
        self.switch_workspace("data_management")
    def _detect_screen_profile(self) -> dict[str, int | float | str]:
        """Return the real Qt screen geometry used for laptop fit decisions.

        On Windows the available geometry excludes the taskbar.  In headless
        offscreen tests Qt often reports an artificial 800×800 screen; that is
        recorded for diagnostics but ignored for the initial desktop size.
        """
        app = QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return {
                "geometry_width": 0,
                "geometry_height": 0,
                "available_width": 0,
                "available_height": 0,
                "device_pixel_ratio": 1.0,
                "logical_dpi_x": 96.0,
                "logical_dpi_y": 96.0,
                "source": "no_screen",
            }
        geo = screen.geometry()
        available = screen.availableGeometry()
        return {
            "geometry_width": int(geo.width()),
            "geometry_height": int(geo.height()),
            "available_width": int(available.width()),
            "available_height": int(available.height()),
            "device_pixel_ratio": float(screen.devicePixelRatio()),
            "logical_dpi_x": float(screen.logicalDotsPerInchX()),
            "logical_dpi_y": float(screen.logicalDotsPerInchY()),
            "source": screen.name() or "primary_screen",
        }

    def _has_real_desktop_geometry(self) -> bool:
        width = int(self.screen_profile.get("available_width", 0) or 0)
        height = int(self.screen_profile.get("available_height", 0) or 0)
        return width >= MIN_WORKBENCH_SIZE[0] and height >= MIN_WORKBENCH_SIZE[1]

    def _should_use_compact_mode(self) -> bool:
        if not self._has_real_desktop_geometry():
            # Default to the notebook-safe layout in CI/offscreen and on
            # unknown desktops.  Larger monitors still have room for it.
            return True
        width = int(self.screen_profile.get("available_width", 0) or 0)
        height = int(self.screen_profile.get("available_height", 0) or 0)
        return width <= COMPACT_SCREEN_WIDTH_THRESHOLD or height <= COMPACT_SCREEN_HEIGHT_THRESHOLD

    def _initial_window_size(self) -> tuple[int, int]:
        target = COMPACT_1080P_FIT_SIZE if self.compact_mode else DEFAULT_1080P_SIZE
        if not self._has_real_desktop_geometry():
            return target
        available_width = int(self.screen_profile.get("available_width", 0) or 0)
        available_height = int(self.screen_profile.get("available_height", 0) or 0)
        if self.compact_mode:
            # On 15.6-inch 1080P notebooks with 125% Windows scaling,
            # availableGeometry is typically 1536×816.  The previous
            # min(target, available - 12) path opened the real app at
            # about 1524×804 while diagnostics captured 1536×816, hiding
            # tight right-panel regressions.  Compact mode should use the
            # full usable desktop area; larger desktops stay on the normal
            # target size below.
            width = max(MIN_WORKBENCH_SIZE[0], available_width)
            height = max(MIN_WORKBENCH_SIZE[1], available_height)
        else:
            width = min(target[0], max(MIN_WORKBENCH_SIZE[0], available_width - 12))
            height = min(target[1], max(MIN_WORKBENCH_SIZE[1], available_height - 12))
        return int(width), int(height)

    def _compact_value(self, normal: int, compact: int) -> int:
        return int(compact if getattr(self, "compact_mode", True) else normal)

    # Compatibility hooks used by existing smoke scripts and old callers.
    def open_loose_path(self, path: str | Path) -> None:
        self.sample_path = Path(path)
        self.sample_row_count = self._count_rows_safely(self.sample_path)
        if self.project_store is not None:
            try:
                self.project_store.import_line_file("L03", self.sample_path, name="过路口测线", copy_into_project=True)
                self._sync_project_lines_to_ui()
            except Exception:
                # Keep the legacy preview update below as a safe fallback.
                pass
        if self.sample_row_count:
            size_mb = self.sample_path.stat().st_size / (1024 * 1024)
            self._line_status_message = f"已导入 {self.sample_path.name}，{self.sample_row_count:,} 行，{size_mb:.2f} MB；项目清单已刷新。"
            # Use the actually loaded sidecar as the active field record.  The
            # other rows remain deterministic project-preview data, but L03 now
            # reflects a real file path and row count so screenshots can prove
            # the UI is not purely static.
            for line in self.line_records:
                if line["id"] == "L03":
                    line["status"] = "● 已导入"
                    line["quality"] = "★★★★☆"
                    line["updated"] = "刚刚"
                    line["source"] = str(self.sample_path)
                    line["rows"] = self.sample_row_count
                    break
        self._refresh_project_widgets()
        self.switch_workspace("data_management")
        self.statusBar().showMessage(f"已加载示例数据：{self.sample_path.name}")
    def choose_loose_path(self) -> None:  # compatibility hook for old toolbar callbacks
        self._action_import_line_dialog()
    def _init_preview_state(self) -> None:
        self.line_records = [
            {"id": "L01", "name": "经度道路主线", "length": 212.35, "quality": "★★★★★", "rtk": "● 固定解", "status": "● 已完成", "updated": "2025-05-24 10:32", "targets": 6},
            {"id": "L02", "name": "纬向道路辅线", "length": 184.62, "quality": "★★★★☆", "rtk": "● 固定解", "status": "● 已完成", "updated": "2025-05-24 09:58", "targets": 4},
            {"id": "L03", "name": "过路口测线", "length": 121.40, "quality": "★★★★☆", "rtk": "● 浮动解", "status": "◌ 处理中", "updated": "2025-05-24 10:28", "targets": 5},
            {"id": "L04", "name": "人行道测线", "length": 96.83, "quality": "★★★☆☆", "rtk": "● 固定解", "status": "● 未处理", "updated": "2025-05-24 08:41", "targets": 2},
            {"id": "L05", "name": "雨水管线疑似A", "length": 156.22, "quality": "★★★★☆", "rtk": "● 固定解", "status": "● 已完成", "updated": "2025-05-24 09:12", "targets": 4},
            {"id": "L06", "name": "雨水管线疑似B", "length": 143.78, "quality": "★★★☆☆", "rtk": "● 浮动解", "status": "● 未处理", "updated": "2025-05-23 17:55", "targets": 3},
            {"id": "L07", "name": "检查井区域", "length": 88.91, "quality": "★★★★☆", "rtk": "● 固定解", "status": "● 已完成", "updated": "2025-05-24 10:05", "targets": 2},
            {"id": "L08", "name": "横穿支路测线", "length": 73.54, "quality": "★★★☆☆", "rtk": "● 浮动解", "status": "● 未处理", "updated": "2025-05-23 16:42", "targets": 1},
        ]
        self.targets = [
            {"name": "T-01", "type": "疑似管线", "mileage": 18.62, "depth": 1.35, "confidence": "★★★★☆", "status": "已确认", "note": "疑似电缆管线，走向近似垂直", "color": "#25B26B", "width": 46, "height": 70},
            {"name": "T-02", "type": "疑似空洞", "mileage": 62.47, "depth": 2.02, "confidence": "★★★☆☆", "status": "待复核", "note": "振幅弱，双曲线特征不连续", "color": "#7C4DFF", "width": 45, "height": 58},
            {"name": "T-03", "type": "疑似排水管", "mileage": 96.83, "depth": 1.60, "confidence": "★★★★☆", "status": "已确认", "note": "双曲线征清晰，尺寸较大", "color": "#F04444", "width": 100, "height": 78},
            {"name": "T-04", "type": "疑似管线", "mileage": 142.18, "depth": 1.25, "confidence": "★★★☆☆", "status": "待复核", "note": "信号中等，建议开挖验证", "color": "#2B86F6", "width": 48, "height": 64},
            {"name": "T-05", "type": "疑似结构物", "mileage": 179.41, "depth": 1.90, "confidence": "★★★☆☆", "status": "待确认", "note": "可能为检查井或井室结构", "color": "#F5A623", "width": 38, "height": 50},
        ]
        self.line_table: QTableWidget | None = None
        self.line_status_label: QLabel | None = None
        self.processing_status_label: QLabel | None = None
        self.processing_task_label: QLabel | None = None
        self.processing_bscan_canvas: FigureCanvas | None = None
        self.processing_diff_canvas: FigureCanvas | None = None
        self.processing_log_label: QLabel | None = None
        self.processing_info_label: QLabel | None = None
        self.target_table: QTableWidget | None = None
        self.target_canvas: FigureCanvas | None = None
        self.target_log_label: QLabel | None = None
        self.target_field_labels: dict[str, QLabel] = {}
        self.target_preview_canvas: FigureCanvas | None = None
        self.target_source_combo: QComboBox | None = None
        self.current_target_source_id = f"{self.selected_line}_raw"
        self.project_status: ProjectStatusSnapshot = build_project_status_snapshot(None)
        self.home_metric_cards: dict[str, MetricCard] = {}
        self.project_metric_cards: dict[str, MetricCard] = {}
        self.project_task_tabs: QTabWidget | None = None
        self.project_tree_widget: QTreeWidget | None = None
        self.project_tree_widgets: list[QTreeWidget] = []
    def _init_field_project_store(self) -> None:
        """Open/create the persisted field-demo project for the product UI.
        Round 3 turns the preview workbench into a real project container.  The
        UI still uses deterministic sample visuals, but project metadata, line
        status, processing artifacts, targets and spatial CSV files are now
        backed by files under ``runtime_projects/field_demo_project``.
        """
        repo_root = Path(__file__).resolve().parents[1]
        sample_csv = repo_root / "sample_data" / "gui_sidecar_all_data_main.csv"
        try:
            self.project_store = FieldProjectStore.create_or_open_demo(repo_root, sample_csv=sample_csv)
            self.project_manifest = self.project_store.manifest
            self.project_root = self.project_store.root
            lines = self.project_store.list_lines()
            if lines:
                self.line_records = [line.to_ui_dict() for line in lines]
            self.active_gpr_dataset = self.project_store.load_gpr_dataset(self.selected_line)
            self.trajectory_model = self._load_line_trajectory_if_present(self.selected_line)
            persisted_targets = self.project_store.load_targets(self.selected_line)
            if persisted_targets:
                self.targets = persisted_targets
                self.current_target_index = min(self.current_target_index, len(self.targets) - 1)
            matrix_text = f"{self.active_gpr_dataset.sample_count}×{self.active_gpr_dataset.trace_count}" if self.active_gpr_dataset is not None else "--"
            trajectory_points = len(self.trajectory_model.points) if self.trajectory_model is not None else 0
            self._line_status_message = (
                f"项目已打开：{self.project_root}；GPR矩阵 {matrix_text}、RTK轨迹 {trajectory_points} 点已接入。"
            )
        except Exception as exc:  # keep screenshots/runtime available even if disk is read-only
            self.project_store = None
            self.project_manifest = None
            self.project_root = None
            self._line_status_message = f"项目存储初始化失败，已切回内存预览：{exc}"
    def _refresh_project_status_snapshot(self) -> None:
        self.project_status = build_project_status_snapshot(self.project_store)
    @staticmethod
    def _format_mb(value: float) -> tuple[str, str]:
        try:
            mb = float(value)
        except Exception:
            return "0", "MB"
        if mb >= 1024:
            return f"{mb / 1024:.1f}", "GB"
        return f"{mb:.1f}", "MB"
    def _metric_card(self, group: dict[str, MetricCard], key: str, icon: str, title: str, value: str, suffix: str = "", note: str = "") -> MetricCard:
        card = MetricCard(icon, title, value, suffix, note)
        group[key] = card
        return card
    def _update_metric_card(self, group: dict[str, MetricCard], key: str, value: str, suffix: str = "", note: str | None = None) -> None:
        card = group.get(key)
        if card is None:
            return
        card.value_label.setText(value)
        card.suffix_label.setText(suffix)
        if note is not None and card.note_label is not None:
            card.note_label.setText(note)
    def _update_metric_cards(self) -> None:
        self._refresh_project_status_snapshot()
        st = self.project_status
        raw_value, raw_suffix = self._format_mb(st.raw_size_mb)
        self._update_metric_card(self.home_metric_cards, "lines", str(st.line_count), "条", f"已导入 {st.imported_line_count} 条")
        self._update_metric_card(self.home_metric_cards, "processed", str(st.processed_line_count), "条", f"处理完成率 {st.processed_percent:.1f}%")
        self._update_metric_card(self.home_metric_cards, "targets", str(st.confirmed_target_count), "个", f"待复核 {st.pending_target_count} 个")
        self._update_metric_card(self.home_metric_cards, "spatial", str(st.spatial_point_count), "个", f"轨迹文件 {st.trajectory_file_count} 个")
        self._update_metric_card(self.home_metric_cards, "reports", st.report_status, "", f"交付文件 {st.report_file_count} 个")
        self._update_metric_card(self.project_metric_cards, "lines", str(st.line_count), "条")
        self._update_metric_card(self.project_metric_cards, "raw", raw_value, raw_suffix)
        self._update_metric_card(self.project_metric_cards, "trajectory", str(st.trajectory_file_count), "个")
        self._update_metric_card(self.project_metric_cards, "reports", st.report_status, "", f"交付文件 {st.report_file_count} 个")
        self._update_metric_card(self.project_metric_cards, "status", st.data_health_label, "", f"最后更新：{st.latest_update}")
        self._update_project_tree()
        self._update_project_task_tabs()
    def _project_tree_rows(self) -> tuple[list[FieldLineRecord], list[str], list[str], list[str]]:
        if self.project_store is None:
            return [], [], [], []
        lines = self.project_store.list_lines()
        processed: list[str] = []
        targets: list[str] = []
        spatial: list[str] = []
        for line in lines:
            if line.processed_result:
                processed.append(f"{line.line_id}_处理结果")
            if line.target_count or self.project_store.targets_path(line.line_id).exists():
                targets.append(f"{line.line_id}_目标标注")
            spatial_path = self.project_store.root / "spatial" / f"{line.line_id}_targets_xy.csv"
            if spatial_path.exists():
                spatial.append(f"{line.line_id}_空间成果")
        return lines, processed, targets, spatial
    def _update_project_tree(self) -> None:
        trees = [tree for tree in getattr(self, "project_tree_widgets", []) if tree is not None]
        if self.project_tree_widget is not None and self.project_tree_widget not in trees:
            trees.append(self.project_tree_widget)
        if not trees:
            return
        for tree in trees:
            self._populate_project_tree(tree)

    def _populate_project_tree(self, tree: QTreeWidget) -> None:
        tree.clear()
        tree.blockSignals(True)
        current_processing_item: QTreeWidgetItem | None = None
        lines, processed, targets, spatial = self._project_tree_rows()
        line_root = QTreeWidgetItem(["⌘  测线"])
        for line in lines:
            ok = bool(line.gpr_dataset_path or line.raw_path)
            status = "已导入" if ok else "未导入"
            item = QTreeWidgetItem([f"☑  {line.line_id}   {line.name}   {status}"])
            item.setData(0, Qt.ItemDataRole.UserRole, line.line_id)
            item.setData(0, Qt.ItemDataRole.UserRole + 3, "line")
            item.setToolTip(0, f"{line.line_id}｜{line.name}｜{status}")
            item.setForeground(0, QColor("#16A05D" if ok else "#9AA7B4"))
            line_root.addChild(item)
        if not lines:
            line_root.addChild(QTreeWidgetItem(["□  暂无测线"]))
        result_root = QTreeWidgetItem(["▱  处理结果"])
        result_child_count = 0
        line_name_lookup = {str(line.line_id): str(line.name) for line in lines}
        current_line_id = str(getattr(self, "selected_line", ""))
        session = getattr(self, "processing_session", None)
        has_current_chain = bool(getattr(self, "active_gpr_dataset", None) is not None and current_line_id)
        if has_current_chain:
            current_name = line_name_lookup.get(current_line_id, "当前测线")
            chain_header = QTreeWidgetItem([f"▾  {current_line_id} {current_name} 当前处理链"])
            chain_header.setData(0, Qt.ItemDataRole.UserRole, current_line_id)
            chain_header.setData(0, Qt.ItemDataRole.UserRole + 1, "processing_lab")
            chain_header.setData(0, Qt.ItemDataRole.UserRole + 3, "processing_chain")
            chain_header.setToolTip(0, "当前测线处理链；右键可重置或打开测线处理页")
            result_root.addChild(chain_header)
            result_child_count += 1

            raw_marker = "●" if session is None or int(getattr(session, "current_step_index", 0)) == 0 else "○"
            raw_item = QTreeWidgetItem([f"{raw_marker}  00 原始 B-scan"])
            raw_item.setData(0, Qt.ItemDataRole.UserRole, current_line_id)
            raw_item.setData(0, Qt.ItemDataRole.UserRole + 1, "processing_lab")
            raw_item.setData(0, Qt.ItemDataRole.UserRole + 2, 0)
            raw_item.setData(0, Qt.ItemDataRole.UserRole + 3, "processing_step")
            raw_item.setToolTip(0, "Step 00｜原始 B-scan｜右键可回到原始或删除后续步骤")
            if session is None or int(getattr(session, "current_step_index", 0)) == 0:
                raw_item.setForeground(0, QColor(ACCENT_DARK))
                current_processing_item = raw_item
            result_root.addChild(raw_item)
            result_child_count += 1

            if session is not None:
                for step in session.steps:
                    marker = "●" if int(getattr(session, "current_step_index", 0)) == step.index else "○"
                    child = QTreeWidgetItem([f"{marker}  {step.index:02d} {step.method_name}"])
                    child.setData(0, Qt.ItemDataRole.UserRole, current_line_id)
                    child.setData(0, Qt.ItemDataRole.UserRole + 1, "processing_lab")
                    child.setData(0, Qt.ItemDataRole.UserRole + 2, step.index)
                    child.setData(0, Qt.ItemDataRole.UserRole + 3, "processing_step")
                    child.setToolTip(0, f"Step {step.index:02d}｜{step.method_name}｜参数：{step.params or '默认'}｜状态：{step.status_text}｜右键可回到此步、删除后续或对比")
                    if int(getattr(session, "current_step_index", 0)) == step.index:
                        child.setForeground(0, QColor(ACCENT_DARK))
                        current_processing_item = child
                    result_root.addChild(child)
                    result_child_count += 1
        for name in processed:
            line_id = name.split("_", 1)[0] if "_" in name else ""
            if line_id and line_id == current_line_id and has_current_chain:
                continue
            child = QTreeWidgetItem([f"□  {name}"])
            if "_" in name:
                child.setData(0, Qt.ItemDataRole.UserRole, line_id)
                child.setData(0, Qt.ItemDataRole.UserRole + 1, "processing_lab")
                child.setData(0, Qt.ItemDataRole.UserRole + 3, "processed_result")
            result_root.addChild(child)
            result_child_count += 1
        if result_child_count == 0:
            result_root.addChild(QTreeWidgetItem(["暂无处理结果"]))
        target_root = QTreeWidgetItem(["◎  目标标注"])
        for name in targets or ["暂无目标标注"]:
            child = QTreeWidgetItem([f"□  {name}"])
            if "_" in name and not name.startswith("暂无"):
                child.setData(0, Qt.ItemDataRole.UserRole, name.split("_", 1)[0])
                child.setData(0, Qt.ItemDataRole.UserRole + 1, "interpretation")
                child.setData(0, Qt.ItemDataRole.UserRole + 3, "target")
            target_root.addChild(child)
        delivery_root = QTreeWidgetItem(["▣  交付成果"])
        delivery_names = spatial + (["报告文档"] if self.project_status.report_file_count else [])
        for name in delivery_names or ["暂无交付成果"]:
            child = QTreeWidgetItem([f"□  {name}"])
            if name == "报告文档":
                child.setData(0, Qt.ItemDataRole.UserRole + 1, "delivery")
                child.setData(0, Qt.ItemDataRole.UserRole + 3, "delivery")
            elif "_" in name and not name.startswith("暂无"):
                child.setData(0, Qt.ItemDataRole.UserRole, name.split("_", 1)[0])
                child.setData(0, Qt.ItemDataRole.UserRole + 1, "spatial")
                child.setData(0, Qt.ItemDataRole.UserRole + 3, "spatial")
            delivery_root.addChild(child)
        for root_item in [line_root, result_root, target_root, delivery_root]:
            tree.addTopLevelItem(root_item)
            root_item.setExpanded(True)
        tree.blockSignals(False)
        tree.expandAll()
        if current_processing_item is not None:
            tree.setCurrentItem(current_processing_item)
            current_processing_item.setSelected(True)
            tree.scrollToItem(current_processing_item, QAbstractItemView.ScrollHint.PositionAtCenter)
        tree.resizeColumnToContents(0)
        tree.viewport().update()
        tree.repaint()
    def _select_line_from_project_tree(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        line_id = item.data(0, Qt.ItemDataRole.UserRole)
        target_workspace = item.data(0, Qt.ItemDataRole.UserRole + 1)
        step_index = item.data(0, Qt.ItemDataRole.UserRole + 2)
        if line_id:
            for idx, line in enumerate(self.line_records):
                if line.get("id") == line_id:
                    self._select_line_from_table(idx, 0)
                    break
        if target_workspace:
            self.switch_workspace(str(target_workspace))
        if step_index is not None and hasattr(self, "_select_processing_tree_step"):
            try:
                self._select_processing_tree_step(int(step_index))
            except Exception:
                pass

    def _show_project_tree_context_menu(self, pos: QPoint) -> None:
        sender = self.sender()
        tree = sender if isinstance(sender, QTreeWidget) else self.project_tree_widget
        if tree is None:
            return
        item = tree.itemAt(pos)
        if item is None:
            return
        line_id = item.data(0, Qt.ItemDataRole.UserRole)
        item_kind = item.data(0, Qt.ItemDataRole.UserRole + 3) or "line"
        step_index = item.data(0, Qt.ItemDataRole.UserRole + 2)
        if not line_id and item_kind not in {"delivery"}:
            return
        if line_id:
            self._select_line_from_project_tree(item, 0)
        menu = QMenu(tree)

        def add_action(label: str, callback: Callable[[], None] | None = None, *, destructive: bool = False, enabled: bool = True) -> QAction:
            action = QAction(label, menu)
            action.setEnabled(enabled)
            if destructive:
                action.setObjectName("destructiveMenuAction")
            if callback is not None:
                action.triggered.connect(callback)
            menu.addAction(action)
            return action

        if item_kind == "processing_step" and step_index is not None:
            idx = int(step_index)
            add_action(f"回看 Step {idx:02d} 结果", lambda idx=idx: self._action_select_processing_step(idx))
            add_action("从此步继续处理", lambda idx=idx: self._action_select_processing_step(idx))
            add_action("与原始 / 差异图对比", lambda idx=idx: self._action_compare_processing_step(idx), enabled=idx > 0)
            menu.addSeparator()
            add_action("删除此步之后的全部步骤", lambda idx=idx: self._action_truncate_processing_chain_after(idx), destructive=True)
            add_action("重置到原始 B-scan", self._reset_processing_chain, destructive=True)
            menu.addSeparator()
            add_action("复制步骤摘要到剪贴板", lambda idx=idx: self._action_copy_processing_step_summary(idx))
        elif item_kind == "processing_chain":
            add_action("打开测线处理页", lambda: self.switch_workspace("processing_lab"))
            add_action("展开当前处理链", self._update_project_tree)
            add_action("重置到原始 B-scan", self._reset_processing_chain, destructive=True)
        elif item_kind in {"processed_result", "target", "spatial", "delivery"}:
            workspace = item.data(0, Qt.ItemDataRole.UserRole + 1)
            add_action("打开对应页面", lambda workspace=workspace: self.switch_workspace(str(workspace or "processing_lab")))
            add_action("复制节点名称", lambda text=item.text(0): QApplication.clipboard().setText(text))
        else:
            actions: list[tuple[str, Callable[[], None] | None, bool]] = [
                ("打开测线处理", lambda: self.switch_workspace("processing_lab"), False),
                ("定位到测线", lambda: self.switch_workspace("spatial"), False),
                ("查看质检详情", self._action_show_quality_detail_dialog, False),
                ("修正 B-scan 方向", self._action_fix_bscan_orientation, False),
                ("检查源文件", self._action_check_source_files, False),
                ("重新定位源文件", self._action_relocate_current_source, False),
                ("打开源文件目录", self._action_open_current_source_dir, False),
                ("导出测线信息", self._action_export_line_manifest, False),
                ("", None, False),
                ("删除测线…", self._action_delete_current_line, True),
            ]
            for label, callback, destructive in actions:
                if not label:
                    menu.addSeparator()
                    continue
                add_action(label, callback, destructive=destructive)
        menu.exec(tree.viewport().mapToGlobal(pos))

    def _action_select_processing_step(self, step_index: int) -> None:
        if hasattr(self, "_select_processing_tree_step"):
            self._select_processing_tree_step(int(step_index))

    def _action_compare_processing_step(self, step_index: int) -> None:
        self._action_select_processing_step(int(step_index))
        if hasattr(self, "_open_processing_compare_viewer"):
            self._open_processing_compare_viewer()

    def _action_truncate_processing_chain_after(self, step_index: int) -> None:
        session = getattr(self, "processing_session", None)
        if session is None:
            return
        changed = session.truncate_after_step(int(step_index))
        self.processed_gpr_dataset = session.current_dataset if session.current_step_index else None
        self.processing_last_failed = False
        self.last_processing_error = ""
        self.last_processing_manifest = session.steps[session.current_step_index - 1].manifest if session.current_step_index > 0 else None
        self._refresh_processing_preview()
        self._refresh_project_widgets()
        if getattr(self, "processing_log_label", None) is not None:
            if changed:
                self.processing_log_label.setText(f"✂  已保留到 Step {int(step_index):02d}，后续步骤已从项目树移除。")
            else:
                self.processing_log_label.setText(f"ℹ  Step {int(step_index):02d} 已经是最后一步，无需删除。")

    def _action_copy_processing_step_summary(self, step_index: int) -> None:
        session = getattr(self, "processing_session", None)
        if session is None:
            text = f"{self.selected_line} Step 00 原始 B-scan"
        elif int(step_index) <= 0:
            text = f"{self.selected_line} Step 00 原始 B-scan"
        elif int(step_index) <= len(session.steps):
            step = session.steps[int(step_index) - 1]
            text = f"{self.selected_line} Step {step.index:02d} {step.method_name} | 参数: {step.params or '默认'} | 状态: {step.status_text}"
        else:
            text = f"{self.selected_line} Step {int(step_index):02d}"
        QApplication.clipboard().setText(text)
        if getattr(self, "processing_log_label", None) is not None:
            self.processing_log_label.setText("📋  已复制当前步骤摘要。")

    def _update_project_task_tabs(self) -> None:
        tabs = self.project_task_tabs
        if tabs is None:
            return
        st = self.project_status
        for idx in range(tabs.count()):
            widget = tabs.widget(idx)
            if not isinstance(widget, QTableWidget):
                continue
            tab_name = tabs.tabText(idx)
            if tab_name == "任务":
                widget.setColumnCount(7)
                widget.setHorizontalHeaderLabels(["任务名称", "类型", "状态", "进度", "开始时间", "结束时间", "操作"])
                self._fill_table(widget, st.task_rows)
            elif tab_name == "检查提示":
                widget.setColumnCount(3)
                widget.setHorizontalHeaderLabels(["检查内容", "说明", "状态"])
                rows = [(title, desc, count) for _icon, title, desc, count in st.attention_items]
                self._fill_table(widget, rows)
            elif tab_name == "交付文件":
                widget.setColumnCount(6)
                widget.setHorizontalHeaderLabels(["文件名称", "类型", "大小", "更新时间", "状态", "操作"])
                self._fill_table(widget, st.delivery_rows)
            elif tab_name == "日志":
                widget.setColumnCount(4)
                widget.setHorizontalHeaderLabels(["类型", "事件", "说明", "时间"])
                self._fill_table(widget, st.activity_rows)
    def _project_selector_text(self) -> str:
        name = getattr(self.project_manifest, "name", "未打开项目")
        return f"当前项目：  {name}"

    def _refresh_project_selector_combo(self) -> None:
        """Refresh the top current-project switcher.

        The public method name is kept for backward compatibility with older
        tests and page mixins.  v0.9.19 renders the selector as a compact
        drop-down tool button instead of a permanently visible recent-project
        combo, so project creation/open/settings/backup/delete live in the
        header where users naturally look for project-level actions.
        """

        button = getattr(self, "project_selector_button", None)
        current_name = getattr(self.project_manifest, "name", "未打开项目")
        current_path = str(self.project_root or "")

        if button is not None:
            display_name = current_name if len(str(current_name)) <= 26 else f"{str(current_name)[:23]}…"
            button.setText(f"当前项目：  {display_name}  ▾")
            button.setToolTip(str(self.project_root or "未打开项目"))
            menu = QMenu(button)

            current_action = QAction(f"当前：{current_name}", menu)
            current_action.setEnabled(False)
            menu.addAction(current_action)
            menu.addSeparator()

            switch_header = QAction("切换项目", menu)
            switch_header.setEnabled(False)
            menu.addAction(switch_header)
            recent = self.recent_projects.load()
            added_recent = False
            for record in recent:
                if record.path and record.path != current_path:
                    action = QAction(record.name, menu)
                    action.setToolTip(record.path)
                    action.triggered.connect(lambda _=False, p=record.path: self._open_project_path_from_selector(p))
                    menu.addAction(action)
                    added_recent = True
            if not added_recent:
                empty_action = QAction("暂无其他最近项目", menu)
                empty_action.setEnabled(False)
                menu.addAction(empty_action)
            menu.addSeparator()

            for label, callback in [
                ("新建项目", self._action_new_project_dialog),
                ("打开项目", self._action_open_project_dialog),
                ("项目设置", self._action_project_settings_dialog),
                ("项目备份", self._action_backup_project),
            ]:
                action = QAction(label, menu)
                action.triggered.connect(callback)
                menu.addAction(action)
            menu.addSeparator()
            delete_action = QAction("删除项目…", menu)
            delete_action.triggered.connect(self._action_delete_current_project)
            menu.addAction(delete_action)
            button.setMenu(menu)

        combo = self.project_selector_combo
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(f"当前项目：{current_name}", current_path)
        recent = self.recent_projects.load()
        if recent:
            combo.insertSeparator(combo.count())
            for record in recent:
                if record.path and record.path != current_path:
                    combo.addItem(f"切换：{record.name}", record.path)
        combo.blockSignals(False)

    def _open_project_path_from_selector(self, project_path: str) -> None:
        if not project_path or str(project_path) == str(self.project_root or ""):
            return
        try:
            store = open_project(project_path, recent_store=self.recent_projects)
            self._set_active_project_store(store, status_message=f"已切换项目：{store.root}")
            self._post_project_operation_refresh(switch_to=self.active_workspace, kind="project_open")
        except Exception as exc:
            self._show_operation_error("切换项目", exc)

    def _on_project_selector_activated(self, index: int) -> None:
        if self.project_selector_combo is None or index < 0:
            return
        project_path = self.project_selector_combo.itemData(index)
        self._open_project_path_from_selector(str(project_path or ""))
    def _active_project_name(self) -> str:
        return getattr(self.project_manifest, "name", "未打开项目")
    def _set_active_project_store(self, store: FieldProjectStore, *, status_message: str | None = None) -> None:
        """Install an opened/created project into the field workbench state."""
        self.project_store = store
        self.project_manifest = store.manifest
        self.project_root = store.root
        lines = store.list_lines()
        self.line_records = [line.to_ui_dict() for line in lines]
        if lines:
            if self.selected_line not in {line.line_id for line in lines}:
                self.selected_line = lines[0].line_id
            try:
                self.active_gpr_dataset = store.load_gpr_dataset(self.selected_line)
            except Exception:
                self.active_gpr_dataset = None
            try:
                self.trajectory_model = self._load_line_trajectory_if_present(self.selected_line)
            except Exception:
                self.trajectory_model = None
            loaded_targets = store.load_targets(self.selected_line)
            self.targets = loaded_targets
            self.current_target_index = min(self.current_target_index, max(len(self.targets) - 1, 0))
        else:
            self.selected_line = "L01"
            self.active_gpr_dataset = None
            self.trajectory_model = None
            self.targets = []
            self.current_target_index = 0
        self.current_target_source_id = f"{self.selected_line}_raw"
        self._line_status_message = status_message or f"项目已打开：{store.root}"
        self._refresh_project_selector_combo()
        if getattr(self, "linkage_controller", None) is not None:
            self.linkage_controller.emit(ProjectEventType.PROJECT_OPENED, line_id=self.selected_line, reason="项目已打开", refresh=False)
    def _post_project_operation_refresh(self, *, switch_to: str = "data_management", kind: str = "") -> None:
        """Refresh UI after a project operation.

        Parameters
        ----------
        switch_to : str
            Workspace key to switch to after refresh (default ``"data_management"``).
        kind : str
            Operation category that determines refresh strategy:
            ``"project_open"`` / ``"processing_lab"``     → full rebuild
            ``"spatial"``                                 → _refresh_spatial_page
            ``"project"`` / ``"target"`` / ``"delivery"`` → lightweight page refresh
            ``""`` (default)                              → lightweight project refresh
        """
        self._sync_project_lines_to_ui()
        self._refresh_project_selector_combo()
        self._refresh_recent_projects_combo()
        target_workspace = switch_to or self.active_workspace

        # Project open/switch/create → full rebuild (overrides switch_to routing)
        if kind == "project_open":
            self._rebuild_workspace_pages()
            self._refresh_project_widgets()
            self._refresh_processing_preview()
            self._refresh_target_source_options()
            self._refresh_target_widgets()
            if target_workspace:
                self.switch_workspace(target_workspace)
            return

        # Spatial page: lightweight refresh (existing path)
        if kind == "spatial" or switch_to == "spatial":
            self._refresh_spatial_page()
            self._refresh_project_widgets()
            self._update_metric_cards()
            self.switch_workspace("spatial")
            return

        # Processing lab: full rebuild (creates new widgets on each step)
        if kind == "processing_lab" or switch_to == "processing_lab":
            self._rebuild_workspace_pages()
            self._refresh_project_widgets()
            self._refresh_processing_preview()
            self._refresh_target_source_options()
            self._refresh_target_widgets()
            if target_workspace:
                self.switch_workspace(target_workspace)
            return

        # Lightweight page-specific refresh (project/target/delivery)
        try:
            if kind == "project":
                self._refresh_project_page()
            elif kind == "target":
                self._refresh_target_page()
            elif kind == "delivery":
                self._refresh_delivery_page()
            else:
                # No kind specified → treat as project-level operation
                self._refresh_project_page()
        except Exception:
            self._rebuild_workspace_pages()

        self._refresh_project_widgets()
        self._refresh_processing_preview()
        self._refresh_target_source_options()
        self._refresh_target_widgets()
        if target_workspace:
            self.switch_workspace(target_workspace)
    def _refresh_spatial_page(self) -> None:
        """Lightweight spatial page refresh: redraw canvases only.

        Call sites (current, as of v0.9.24):
          - self._post_project_operation_refresh()  (early-return spatial path, line ~799)
        """
        try:
            self._draw_project_spatial_map(self.spatial_map_canvas)
            self._draw_current_elevation_profile(self.spatial_elevation_canvas)
            self._draw_project_line_correlation(self.spatial_correlation_canvas)
        except Exception:
            pass  # fallback to full rebuild

    def _refresh_project_page(self) -> None:
        """Lightweight project page refresh: redraw B-scan and strip previews.

        Does NOT destroy/recreate workspace pages.  Falls back to full rebuild
        on failure.

        Call sites (current, as of v0.9.24):
          - self._post_project_operation_refresh()  for kind="project"
        """
        try:
            page = self.workspace_pages.get("data_management")
            if page is not None:
                map_canvas = page.findChild(FigureCanvas, "projectSummaryMapCanvas")
                if map_canvas is not None:
                    self._draw_current_line_strip(map_canvas)
                bscan_canvas = page.findChild(FigureCanvas, "projectQuickPreviewBscanCanvas")
                if bscan_canvas is not None:
                    self._draw_current_line_bscan(bscan_canvas, title="")
                strip_canvas = page.findChild(FigureCanvas, "projectQuickPreviewMapCanvas")
                if strip_canvas is not None:
                    self._draw_current_line_strip(strip_canvas)
        except Exception:
            self._rebuild_workspace_pages()

    def _refresh_target_page(self) -> None:
        """Lightweight target page refresh: redraw target B-scan and table.

        Does NOT destroy/recreate workspace pages.  Falls back to full rebuild
        on failure.

        Call sites (current, as of v0.9.24):
          - self._post_project_operation_refresh()  for kind="target"
        """
        try:
            self._refresh_target_widgets()
        except Exception:
            self._rebuild_workspace_pages()

    def _refresh_delivery_page(self) -> None:
        """Lightweight delivery page refresh: redraw delivery previews.

        Does NOT destroy/recreate workspace pages.  Falls back to full rebuild
        on failure.

        Call sites (current, as of v0.9.24):
          - self._post_project_operation_refresh()  for kind="delivery"
        """
        try:
            page = self.workspace_pages.get("delivery")
            if page is not None:
                thumb_canvas = page.findChild(FigureCanvas, "deliveryReportThumbCanvas")
                if thumb_canvas is not None:
                    self._draw_current_line_bscan(thumb_canvas, title="")
                figure_canvas = page.findChild(FigureCanvas, "deliveryReportFigureThumbCanvas")
                if figure_canvas is not None:
                    self._draw_current_line_bscan(figure_canvas, title="")
        except Exception:
            self._rebuild_workspace_pages()

    def _rebuild_workspace_pages(self) -> None:
        """Full rebuild: destroys all workspace pages and recreates them from scratch.

        Call sites (current, as of v0.9.24):
          - self._post_project_operation_refresh()  (line ~810, non-spatial paths)
          - field_panels/project_page.py:534        (after delete-project)
        """
        if not hasattr(self, "stack") or self.stack is None:
            return
        while self.stack.count():
            old = self.stack.widget(0)
            self.stack.removeWidget(old)
            old.deleteLater()
        self.workspace_pages.clear()
        self.line_table = None
        self.project_tree_widget = None
        self.project_task_tabs = None
        self.project_tree_widgets = []
        self.target_table = None
        self.target_canvas = None
        self.target_preview_canvas = None
        for key in WORKSPACES:
            page = self._build_workspace_page(key)
            self.workspace_pages[key] = page
            self.stack.addWidget(page)

    def _show_operation_error(self, title: str, exc: Exception) -> None:
        self._line_status_message = f"{title}失败：{exc}"
        self._refresh_project_widgets()
        QMessageBox.warning(self, title, str(exc))





    def _sync_project_lines_to_ui(self) -> None:
        if self.project_store is None:
            self.line_records = []
            self.active_gpr_dataset = None
            self.trajectory_model = None
            self._refresh_project_status_snapshot()
            return
        lines = self.project_store.list_lines()
        self.line_records = [line.to_ui_dict() for line in lines]
        valid_ids = {line.line_id for line in lines}
        if not lines:
            self.selected_line = "L01"
            self.active_gpr_dataset = None
            self.trajectory_model = None
            self.targets = []
            self.current_target_index = 0
        elif self.selected_line not in valid_ids:
            self.selected_line = lines[0].line_id
            try:
                self.active_gpr_dataset = self.project_store.load_gpr_dataset(self.selected_line)
            except Exception:
                self.active_gpr_dataset = None
            try:
                self.trajectory_model = self._load_line_trajectory_if_present(self.selected_line)
            except Exception:
                self.trajectory_model = None
        self.current_target_source_id = f"{self.selected_line}_raw"
        self._refresh_project_status_snapshot()
    def _save_targets_to_project(self, *, emit_event: bool = True, reason: str = "目标标注已变化") -> None:
        if self.project_store is None:
            return
        self.project_store.save_targets(self.selected_line, self.targets)
        self._sync_project_lines_to_ui()
        if emit_event and getattr(self, "linkage_controller", None) is not None:
            self.linkage_controller.emit(ProjectEventType.TARGETS_CHANGED, line_id=self.selected_line, reason=f"{self.selected_line} {reason}", refresh=False)
            self._refresh_project_status_snapshot()
    def _count_rows_safely(self, path: Path) -> int:
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
                reader = csv.reader(fh)
                return max(sum(1 for _ in reader) - 1, 0)
        except Exception:
            return 0
    def _line_source_status_label(self, line_id: str) -> str:
        if self.project_store is None:
            return "未记录"
        try:
            return source_status_label_for_line(self.project_store, line_id)
        except Exception:
            return "未记录"

    def _line_rows(self) -> list[tuple]:
        if getattr(self, "compact_mode", True):
            return [
                (
                    f"{line['id']} {line['name']}",
                    self._line_source_status_label(str(line["id"])),
                    f"{line['length']:.1f}",
                    line["quality"],
                    str(line["rtk"]).replace("● ", ""),
                    str(line["status"]).replace("● ", ""),
                    str(line["updated"])[5:],
                )
                for line in self.line_records
            ]
        return [
            (
                f"{line['id']}   {line['name']}",
                self._line_source_status_label(str(line["id"])),
                f"{line['length']:.2f}",
                line["quality"],
                line["rtk"],
                line["status"],
                line["updated"],
                "⊙  ↗  …",
            )
            for line in self.line_records
        ]
    def _target_rows(self) -> list[tuple]:
        return [
            (
                "□",
                f"●  {target['name']}",
                target["type"],
                f"{float(target['mileage']):.2f}",
                f"{float(target['depth']):.2f}",
                target["confidence"],
                target["status"],
                target["note"],
            )
            for target in self.targets
        ]
    def _refresh_project_widgets(self) -> None:
        self._update_metric_cards()
        self._update_project_tree()
        self._update_project_task_tabs()
        if self.line_table is not None:
            self._fill_table(self.line_table, self._line_rows(), highlight_row=self._selected_line_row())
        if self.line_status_label is not None:
            self.line_status_label.setText(self._line_status_message)
    def _selected_line_row(self) -> int:
        for i, line in enumerate(self.line_records):
            if line["id"] == self.selected_line:
                return i
        return 0

    def _selected_line_record(self) -> dict:
        """Return the current line preview record used by compact overview cards."""
        for line in self.line_records:
            if line.get("id") == self.selected_line:
                return line
        return self.line_records[0] if self.line_records else {"id": "--", "name": "暂无测线", "length": 0.0, "rtk": "--", "status": "--"}

    def _load_line_trajectory_if_present(self, line_id: str) -> TrajectoryModel | None:
        if self.project_store is None:
            return None
        try:
            line = self.project_store.get_line(line_id)
        except Exception:
            return None
        if not getattr(line, "trajectory_path", ""):
            return None
        try:
            return self.project_store.load_trajectory(line_id)
        except Exception:
            return None

    def _clear_line_dependent_processing_state(self) -> None:
        """Clear cached processing/annotation-source state when switching lines."""
        self.processed_gpr_dataset = None
        self.processing_session = None
        self.processing_applied = False
        self.last_processing_manifest = None
        self.last_processing_error = ""
        self.processing_last_failed = False
        self.current_target_source_id = f"{self.selected_line}_raw"

    def _select_line_from_table(self, row: int, _column: int = 0) -> None:
        if row < 0 or row >= len(self.line_records):
            return
        old_line = getattr(self, "selected_line", "")
        self.selected_line = self.line_records[row]["id"]
        if self.selected_line != old_line:
            self._clear_line_dependent_processing_state()
        if self.project_store is not None:
            try:
                self.active_gpr_dataset = self.project_store.load_gpr_dataset(self.selected_line)
            except Exception:
                self.active_gpr_dataset = None
            try:
                self.trajectory_model = self._load_line_trajectory_if_present(self.selected_line)
            except Exception:
                self.trajectory_model = None
            try:
                loaded_targets = self.project_store.load_targets(self.selected_line)
                self.targets = loaded_targets
                self.current_target_index = min(self.current_target_index, max(len(self.targets) - 1, 0))
            except Exception:
                self.targets = []
                self.current_target_index = 0
        self._line_status_message = f"已选中 {self.selected_line} {self.line_records[row]['name']}，长度 {self.line_records[row]['length']:.2f} m。"
        self.current_target_source_id = f"{self.selected_line}_raw"
        self._refresh_project_widgets()
        self._refresh_processing_preview()
        self._refresh_target_source_options()
        self._refresh_target_widgets()
        if getattr(self, "linkage_controller", None) is not None:
            self.linkage_controller.emit(ProjectEventType.LINE_SELECTED, line_id=self.selected_line, reason="当前测线已切换", refresh=False)
    def _setup_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())
        self.stack = QStackedWidget()
        self.stack.setObjectName("pageStack")
        for key in WORKSPACES:
            page = self._build_workspace_page(key)
            self.workspace_pages[key] = page
            self.stack.addWidget(page)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(root)
    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(self._compact_value(70, 60))
        outer = QVBoxLayout(header)
        outer.setContentsMargins(6, 0, 6, 0)
        outer.setSpacing(0)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(self._compact_value(8, 5))
        logo = QLabel()
        logo.setObjectName("logo")
        logo.setFixedSize(self._compact_value(24, 22), self._compact_value(24, 22))
        if make_mygpr_brand_pixmap:
            pix = make_mygpr_brand_pixmap(self._compact_value(24, 22))
            logo.setPixmap(pix)
        else:
            logo.setText("M")
            logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(logo)
        title = QLabel(self.version_text or "MyGPR 勘探定位工作台")
        title.setObjectName("appTitle")
        top.addWidget(title)
        top.addSpacing(8)
        self.project_selector_button = QToolButton()
        self.project_selector_button.setObjectName("projectSelectorButton")
        self.project_selector_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.project_selector_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.project_selector_button.setMinimumWidth(self._compact_value(260, 220))
        top.addWidget(self.project_selector_button)
        self._refresh_project_selector_combo()
        top.addSpacing(6)
        status_pills = [("项目正常", "●"), ("RTK 已连接", "≋"), ("设备已连接", "⌘")]
        if self.compact_mode:
            status_pills = status_pills[:2]
        for text, icon in status_pills:
            pill = QLabel(f"{icon}  {text}")
            pill.setObjectName("statusPill")
            top.addWidget(pill)
        top.addStretch(1)
        user = QLabel("♙  操作员  ▾")
        user.setObjectName("userPill")
        top.addWidget(user)
        for mark in ("—", "□", "×"):
            ctrl = QLabel(mark)
            ctrl.setObjectName("windowCtrl")
            ctrl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ctrl.setFixedSize(22, 22)
            top.addWidget(ctrl)
        outer.addLayout(top, 1)
        nav = QHBoxLayout()
        nav.setContentsMargins(0, 0, 0, 4)
        nav.setSpacing(5)
        for key, text in WORKSPACES.items():
            btn = self._make_nav_button(text, key)
            btn.setMinimumWidth(self._compact_value(108, 94))
            nav.addWidget(btn)
            if key != "delivery":
                sep = QFrame()
                sep.setObjectName("navSeparator")
                sep.setFixedWidth(1)
                nav.addWidget(sep)
        nav.addStretch(1)
        outer.addLayout(nav)
        return header
    def _make_nav_button(self, text: str, key: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("navButton")
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda _=False, k=key: self.switch_workspace(k))
        self.workspace_buttons[key] = btn
        self.nav_group.addButton(btn)
        return btn
    def switch_workspace(self, key: str) -> None:
        if key not in self.workspace_pages:
            key = "data_management"
        self.active_workspace = key
        self.stack.setCurrentWidget(self.workspace_pages[key])
        for name, btn in self.workspace_buttons.items():
            btn.setChecked(name == key)
        if key == "processing_lab":
            self._refresh_processing_preview()
        elif key == "interpretation":
            self._refresh_target_widgets()
        self.statusBar().showMessage(f"当前页面：{WORKSPACES.get(key, '项目总览')}")
    def _build_workspace_page(self, key: str) -> QWidget:
        page = QWidget()
        page.setObjectName("workspacePage")
        layout = QHBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(5)
        sidebar = self._build_project_tree()
        sidebar.setFixedWidth(self._compact_value(244, 220))
        layout.addWidget(sidebar)
        content = QWidget()
        content.setObjectName("workspaceContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        if key == "data_management":
            content_layout.addWidget(self._page_project_management(), 1)
        elif key == "processing_lab":
            content_layout.addWidget(self._page_processing(), 1)
        elif key == "interpretation":
            content_layout.addWidget(self._page_interpretation(), 1)
        elif key == "spatial":
            content_layout.addWidget(self._page_spatial(), 1)
        elif key == "delivery":
            content_layout.addWidget(self._page_delivery(), 1)
        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        content_scroll.setObjectName("scrollArea")
        content_scroll.setWidget(content)
        layout.addWidget(content_scroll, 1)
        return page
    def _build_project_tree(self) -> Card:
        card = Card()
        card.setObjectName("sideCard")
        card.layout.setContentsMargins(8, 6, 8, 6)
        title_row = QHBoxLayout()
        title = QLabel("项目树")
        title.setObjectName("sideTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(QLabel("⌃"))
        card.layout.addLayout(title_row)
        tree = QTreeWidget()
        tree.setObjectName("projectTree")
        tree.setHeaderHidden(True)
        tree.setIndentation(14)
        tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        tree.setAnimated(True)
        tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.project_tree_widget = tree
        self.project_tree_widgets.append(tree)
        tree.itemClicked.connect(self._select_line_from_project_tree)
        tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        tree.customContextMenuRequested.connect(self._show_project_tree_context_menu)
        card.layout.addWidget(tree, 1)
        self._update_project_tree()
        card.layout.addSpacing(4)
        storage_title = QLabel("项目存储")
        storage_title.setObjectName("sideTitle")
        card.layout.addWidget(storage_title)
        bar = QFrame()
        bar.setObjectName("storageBar")
        inner = QFrame(bar)
        inner.setObjectName("storageBarInner")
        inner.setFixedWidth(66)
        inner.setFixedHeight(7)
        bar.setFixedHeight(8)
        card.layout.addWidget(bar)
        st = self.project_status
        usage_value, usage_suffix = self._format_mb(st.storage_usage_mb)
        usage = QLabel(f"{usage_value} {usage_suffix} / 项目目录")
        usage.setObjectName("sideUsage")
        card.layout.addWidget(usage)
        return card
    def _make_scroll_body(self) -> tuple[QScrollArea, QWidget, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setObjectName("scrollArea")
        body = QWidget()
        body.setObjectName("scrollBody")
        body.setMinimumWidth(0)
        body.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(body)
        return scroll, body, layout
    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget#root {{ background: {PAGE_BG}; color: {TEXT}; font-size: 11px; }}
            QLabel {{ background: transparent; border: none; }}
            QFrame#header {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #FFFFFF, stop:1 #F8FBFD); border-bottom: 1px solid #DAE3EB; }}
            QLabel#appTitle {{ font-size: 16px; font-weight: 800; color: #102033; }}
            QLabel#projectSelector {{ font-size: 12px; color: #223246; font-weight: 650; padding: 2px 6px; }}
            QToolButton#projectSelectorButton {{ background: #F6FAFD; border: 1px solid #D6E2EA; border-radius: 6px; color: #223246; font-size: 12px; font-weight: 750; padding: 3px 10px; min-height: 24px; text-align: left; }}
            QToolButton#projectSelectorButton:hover {{ background: #EEF8FB; border-color: {ACCENT}; color: {ACCENT_DARK}; }}
            QToolButton#projectSelectorButton::menu-indicator {{ image: none; width: 0px; }}
            QMenu {{ background: #FFFFFF; border: 1px solid #D8E3EC; border-radius: 7px; padding: 5px; color: #1A2B3E; }}
            QMenu::item {{ padding: 6px 28px 6px 24px; border-radius: 5px; }}
            QMenu::item:selected {{ background: #EAF8FB; color: {ACCENT_DARK}; }}
            QMenu::item:disabled {{ color: #7A8B9C; }}
            QMenu::separator {{ height: 1px; background: #E4ECF3; margin: 5px 4px; }}
            QLabel#statusPill {{ color: #1C3448; padding: 2px 7px; border-left: 1px solid #D7E0E8; }}
            QLabel#userPill {{ color: #1C3448; padding: 3px 8px; }}
            QLabel#windowCtrl {{ color: #102033; font-size: 14px; }}
            QPushButton#navButton {{ background: transparent; border: 0; border-radius: 6px; padding: 5px 8px; font-size: 13px; font-weight: 700; color: #111827; }}
            QPushButton#navButton:hover {{ background: #EAF4F7; color: {ACCENT_DARK}; }}
            QPushButton#navButton:checked {{ background: {ACCENT}; color: white; }}
            QFrame#navSeparator {{ background: #DCE5EC; margin-top: 8px; margin-bottom: 8px; }}
            QStackedWidget#pageStack, QWidget#workspacePage, QWidget#workspaceContent, QScrollArea#scrollArea, QWidget#scrollBody {{ background: {PAGE_BG}; border: 0; }}
            QScrollArea {{ border: 0; }}
            QScrollBar:vertical {{ width: 10px; background: #EEF3F7; }}
            QScrollBar::handle:vertical {{ background: #C8D5E1; border-radius: 4px; }}
            QFrame#card, QFrame#metricCard, QFrame#sideCard {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 3px; }}
            QWidget#cardTitleBar {{ background: transparent; border: 0; }}
            QLabel#cardTitle, QLabel#sideTitle {{ font-size: 11px; font-weight: 700; color: #1A2D42; }}
            QLabel#pageTitle {{ font-size: 16px; font-weight: 800; color: #102033; }}
            QLabel#pageSubtitle {{ color: #53687A; font-weight: 500; }}
            QLabel#metricIcon {{ background: #EAF6FF; border-radius: 7px; color: #1775FF; font-size: 16px; font-weight: 800; }}
            QLabel#metricTitle {{ color: #53687A; font-weight: 600; font-size: 10px; }}
            QLabel#metricValue {{ font-size: 15px; font-weight: 900; color: #101827; }}
            QLabel#metricSuffix {{ color: #102033; padding-top: 4px; }}
            QLabel#metricNote {{ color: #53687A; font-size: 10px; }}
            QFrame#metricCard {{ min-height: 46px; }}
            QFrame#sideCard {{ background: #FFFFFF; }}
            QTreeWidget#projectTree {{ background: transparent; border: 0; outline: 0; color: #1C2B3A; }}
            QTreeWidget#projectTree::item {{ height: 22px; border-radius: 6px; padding: 1px 5px; }}
            QTreeWidget#projectTree::item:hover {{ background: #EEF7FA; }}
            QTreeWidget#projectTree::item:selected {{ background: #D8F3F6; color: #0B4C5D; }}
            QFrame#storageBar {{ background: #D8E0E8; border-radius: 4px; }}
            QFrame#storageBarInner {{ background: {ACCENT}; border-radius: 4px; }}
            QLabel#sideUsage {{ color: #53687A; padding-top: 4px; }}
            QPushButton, QToolButton {{ background: #FFFFFF; border: 1px solid #D4DEE8; border-radius: 5px; padding: 3px 7px; color: #1A2B3E; font-weight: 600; font-size: 11px; }}
            QPushButton:hover, QToolButton:hover {{ border-color: {ACCENT}; color: {ACCENT_DARK}; background: #F4FBFD; }}
            QPushButton#primaryButton {{ background: {ACCENT}; color: white; border: 1px solid {ACCENT_DARK}; font-weight: 700; }}
            QPushButton#primaryButton:hover {{ background: {ACCENT_DARK}; color: white; }}
            QPushButton#primaryButton:pressed {{ background: #0B5ED7; color: white; }}
            QPushButton#smallButton {{ padding: 3px 6px; font-size: 11px; font-weight: 600; }}
            QPushButton#smallButton:hover {{ background-color: #E8EDF2; }}
            QPushButton#actionTileButton {{ min-height: 36px; padding: 3px 4px; font-size: 11px; }}
            QTableWidget#dataTable {{ background: white; alternate-background-color: #F8FBFD; border: 1px solid #E1E8EF; border-radius: 4px; gridline-color: transparent; selection-background-color: #DDF4F7; selection-color: #0B4151; color: #243447; outline: 0; }}
            QTableWidget#dataTable::item {{ padding: 3px 4px; color: #243447; border-bottom: 1px solid #EDF2F6; }}
            QTableWidget#dataTable::item:selected {{ background: #E1F6F8; color: #0B4151; }}
            QHeaderView::section {{ background: #F7FAFC; color: #506278; font-weight: 700; font-size: 11px; border: 0; border-bottom: 1px solid #DDE7F0; padding: 4px 4px; }}
            QTabWidget#innerTabs::pane {{ border: 0; }}
            QTabWidget#innerTabs QTabBar::tab {{ background: transparent; color: #53687A; padding: 6px 16px; border-bottom: 2px solid transparent; font-weight: 700; }}
            QTabWidget#innerTabs QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
            QLabel#sectionTitle {{ font-size: 13px; font-weight: 800; color: #14253A; }}
            QLabel#keyLabel {{ color: #53687A; min-width: 68px; font-weight: 600; font-size: 11px; }}
            QLabel#valueLabel {{ color: #17283B; font-weight: 600; font-size: 11px; }}
            QLabel#detailTitle {{ color: #102033; font-size: 13px; font-weight: 900; }}
            QLabel#detailSubtitle {{ color: #5B6F82; font-size: 11px; font-weight: 600; }}
            QFrame#lineDetailPanel, QFrame#targetHeroPanel, QFrame#reportStatusStrip {{ background: #F7FBFD; border: 1px solid #E2EAF2; border-radius: 4px; }}
            QFrame#miniStatBox {{ background: #FFFFFF; border: 1px solid #E2EAF2; border-radius: 4px; min-width: 72px; }}
            QLabel#miniStatValue {{ color: #0E2A3D; font-size: 13px; font-weight: 900; }}
            QLabel#miniStatLabel {{ color: #6A7E90; font-size: 10px; font-weight: 700; }}
            QFrame#actionTile {{ border: 1px solid #E0E8EF; border-radius: 4px; background: #FFFFFF; min-height: 76px; }}
            QFrame#actionTile:hover {{ background: #F5FCFD; border-color: {ACCENT}; }}
            QLabel#actionIcon {{ color: {ACCENT}; font-size: 23px; font-weight: 900; }}
            QLabel#actionLabel {{ color: #1B2D42; font-weight: 700; }}
            QFrame#flowStep {{ background: #FBFDFF; border: 1px solid #DDE7F0; border-radius: 4px; }}
            QLabel#flowBadge {{ background: #2B86F6; color: white; border-radius: 14px; font-weight: 900; }}
            QLabel#flowIcon {{ color: {ACCENT}; font-size: 19px; }}
            QLabel#flowTitle {{ font-weight: 900; color: #102033; }}
            QLabel#flowDesc {{ color: #53687A; font-size: 11px; }}
            QLabel#flowStatus {{ color: #1E9A5A; font-weight: 800; font-size: 12px; }}
            QLabel#flowArrow {{ color: #25364A; font-size: 24px; font-weight: 900; }}
            QFrame#moduleTile {{ background: #FBFDFF; border: 1px solid #DDE7F0; border-radius: 4px; }}
            QLabel#moduleThumb {{ background: #DDF4F7; color: {ACCENT}; border-radius: 6px; font-size: 18px; min-height: 32px; }}
            QLabel#moduleTitle {{ color: {ACCENT}; font-weight: 900; }}
            QLabel#moduleDesc {{ color: #53687A; font-size: 11px; }}
            QLabel#linkLabel {{ color: #1479B8; font-weight: 800; }}
            QLabel#attentionIcon {{ font-size: 17px; min-width: 22px; }}
            QLabel#attentionTitle, QLabel#activityTitle {{ color: #102033; font-weight: 900; font-size: 12px; }}
            QLabel#attentionDesc, QLabel#activityDesc {{ color: #53687A; font-size: 12px; }}
            QLabel#statusStrip {{ padding: 2px 0; border-left: 3px solid #0D6EFD; padding-left: 8px; }}
            QLabel#attentionCount {{ color: #D92D20; font-weight: 900; }}
            QFrame#activityTile {{ background: #FBFDFF; border: 1px solid #E4ECF3; border-radius: 4px; }}
            QLabel#activityIcon {{ color: {ACCENT}; font-size: 13px; font-weight: 900; }}
            QLabel#timeLabel {{ color: #6B7D90; font-size: 11px; }}
            QFrame#paramGroup {{ background: #FBFDFF; border: 1px solid #E0E8EF; border-radius: 3px; padding: 3px; }}
            QFrame#paramGroup QLabel {{ color: #243447; font-weight: 600; font-size: 11px; }}
            QCheckBox {{ font-weight: 800; color: #102033; }}
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: white; border: 1px solid #D4DEE8; border-radius: 5px; padding: 5px 8px; color: #102033; selection-background-color: #DDF4F7; selection-color: #0B4151; }}
            QSlider::groove:horizontal {{ height: 5px; background: #D7E2EA; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: white; border: 2px solid {ACCENT}; width: 12px; height: 12px; margin: -5px 0; border-radius: 7px; }}
            QLabel#fieldBox {{ background: #FFFFFF; border: 1px solid #D6E0EA; border-radius: 5px; padding: 5px 7px; color: #1B2D42; }}
            QTextEdit#noteBox {{ background: white; border: 1px solid #D6E0EA; border-radius: 6px; padding: 8px; }}
            QLabel#smallInfo {{ color: #31445A; font-size: 11px; padding: 1px 0; }}
            QLabel#staleNotice {{ background: #FFF8E6; border: 1px solid #F6D58A; border-radius: 6px; color: #8A5A00; font-weight: 750; padding: 6px 9px; }}
            QFrame#reportPage {{ background: #FFFFFF; border: 1px solid #DDE6EF; border-radius: 4px; }}
            QLabel#reportTitle {{ font-size: 16px; font-weight: 900; color: #111827; }}
            QLabel#tocLine {{ color: #1B2D42; font-size: 12px; padding: 2px; }}
            """
        )
__all__ = ["FieldWorkbenchWindow", "WORKSPACES"]
