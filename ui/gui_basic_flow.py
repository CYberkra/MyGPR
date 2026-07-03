#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI基础流程页面 - 包含快速开始、方法选择、参数设置等基础UI"""

import numpy as np

from PyQt6.QtCore import QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPen
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QTextEdit,
    QLineEdit,
    QMenu,
    QScrollArea,
    QFrame,
    QSizePolicy,
)
from qfluentwidgets import (
    PushButton,
    FluentIcon,
    Theme,
    isDarkTheme,
    themeColor,
)

from core.methods_registry import (
    PROCESSING_METHODS,
    get_method_category,
    get_method_display_name,
    get_method_category_label,
    get_public_method_keys,
)
from core.preset_profiles import STOLT_MIGRATION_PRESETS


class SplitActionButton(QWidget):
    """单体式分裂按钮：左侧主点击，右侧箭头弹菜单。"""

    clicked = pyqtSignal()

    def __init__(self, text: str, icon, parent=None):
        super().__init__(parent)
        self._text = text
        self._icon = icon
        self._menu = None
        self._drop_width = 22
        self._hover_part = None
        self._pressed_part = None
        self._visual_state = "normal"
        self.setMinimumHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def setText(self, text: str):
        self._text = str(text or "")
        self.update()

    def text(self) -> str:
        return self._text

    def setVisualState(self, state: str):
        state = str(state or "normal").lower()
        if state not in {"normal", "busy", "success", "error", "dirty"}:
            state = "normal"
        if self._visual_state == state:
            return
        self._visual_state = state
        self.setProperty("applyState", state)
        self.update()

    def setMenu(self, menu):
        self._menu = menu

    def sizeHint(self):
        return QSize(132, 30)

    def _main_rect(self) -> QRect:
        return QRect(0, 0, max(0, self.width() - self._drop_width), self.height())

    def _drop_rect(self) -> QRect:
        return QRect(
            max(0, self.width() - self._drop_width), 0, self._drop_width, self.height()
        )

    def _hit_part(self, pos):
        if self._drop_rect().contains(pos):
            return "drop"
        if self._main_rect().contains(pos):
            return "main"
        return None

    def _background_color(self) -> QColor:
        state = getattr(self, "_visual_state", "normal")
        state_colors = {
            "busy": "#2563EB",
            "success": "#16A34A",
            "error": "#DC2626",
            "dirty": "#F59E0B",
        }
        base = QColor(state_colors.get(state, themeColor()))
        if not self.isEnabled():
            base.setAlpha(90)
            return base
        if self._pressed_part is not None:
            return base.darker(112)
        if self._hover_part is not None:
            return base.lighter(108)
        return base

    def _separator_color(self) -> QColor:
        return QColor(255, 255, 255, 70 if isDarkTheme() else 95)

    def _show_menu(self):
        if self._menu is None:
            return
        anchor = self.mapToGlobal(self.rect().bottomLeft())
        self._menu.popup(anchor)

    def enterEvent(self, event):
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event):
        self._hover_part = None
        self._pressed_part = None
        self._visual_state = "normal"
        self.update()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        self._hover_part = self._hit_part(event.pos())
        self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self._pressed_part = self._hit_part(event.pos())
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        released_part = self._hit_part(event.pos())
        pressed_part = self._pressed_part
        self._pressed_part = None
        self._hover_part = released_part
        self.update()
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.isEnabled()
            and pressed_part is not None
            and pressed_part == released_part
        ):
            if released_part == "drop":
                self._show_menu()
            elif released_part == "main":
                self.clicked.emit()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg = self._background_color()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(self.rect(), 8, 8)

        separator_x = self.width() - self._drop_width
        pen = QPen(self._separator_color())
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(separator_x, 7, separator_x, self.height() - 7)

        if self._hover_part == "drop" and self.isEnabled():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 22 if isDarkTheme() else 28))
            painter.drawRoundedRect(self._drop_rect(), 8, 8)

        icon_size = 17
        left_rect = self._main_rect()

        font = QFont(self.font())
        font.setPixelSize(14)
        painter.setFont(font)

        # 让第一行分裂按钮的图标和文字在主按钮区域内居中，和下面两行按钮对齐。
        gap = 8
        inner_margin = 10
        font_metrics = painter.fontMetrics()
        max_text_width = max(0, left_rect.width() - inner_margin * 2 - icon_size - gap)
        display_text = font_metrics.elidedText(
            self._text,
            Qt.TextElideMode.ElideRight,
            max_text_width,
        )
        text_width = font_metrics.horizontalAdvance(display_text) + 2
        content_width = icon_size + gap + text_width

        # Keep text stable in compact drawers; avoid pushing content under the
        # drop-down area on non-fullscreen windows.
        visual_offset = 0
        content_left = left_rect.left() + max(
            inner_margin,
            (left_rect.width() - content_width) // 2 + visual_offset,
        )
        max_left = left_rect.right() - inner_margin - content_width + 1
        content_left = min(content_left, max_left)

        icon_rect = QRect(
            content_left,
            (self.height() - icon_size) // 2,
            icon_size,
            icon_size,
        )
        if hasattr(self._icon, "render"):
            self._icon.render(painter, icon_rect, theme=Theme.DARK)
        else:
            QIcon(self._icon).paint(painter, icon_rect)

        painter.setPen(
            QColor("white") if self.isEnabled() else QColor(255, 255, 255, 150)
        )
        text_rect = QRect(
            icon_rect.x() + icon_rect.width() + gap,
            0,
            text_width,
            self.height(),
        )
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            display_text,
        )

        arrow_rect = self._drop_rect()
        arrow_size = 12
        FluentIcon.CHEVRON_DOWN_MED.render(
            painter,
            QRect(
                arrow_rect.center().x() - arrow_size // 2,
                arrow_rect.center().y() - arrow_size // 2,
                arrow_size,
                arrow_size,
            ),
            theme=Theme.DARK,
        )


class BasicFlowPage(QWidget):
    """基础流程页面 - 快速开始、方法选择、参数设置"""

    parametersChanged = pyqtSignal()

    BASIC_COMMON_PARAM_NAMES = {
        "motion_compensation_height": [
            "reference_height_mode",
            "manual_height",
            "compensate_amplitude",
            "compensate_time_shift",
            "wave_speed_m_per_ns",
        ],
        "motion_compensation_v2": [
            "height_reference_mode",
            "manual_height_m",
            "height_source",
            "compensate_time_shift",
            "compensate_amplitude",
            "resample_spacing_m",
        ],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.param_vars = {}
        self._method_param_overrides = {}
        self._apply_source_mode = "manual"
        self._apply_source_hint_text = "应用来源：当前参数"
        self._auto_tune_result_available = False
        self._basic_ultra_mode = False
        self._params_dirty = False
        self._apply_state = "idle"
        self.BASIC_PARAM_LIMIT = 4  # compatibility only; daily panel now shows full params by default
        self.btn_stolt_apply = None
        self.stolt_preset_combo = None
        self.stolt_auto_adapt_var = None
        self.motion_v2_trace_metadata_status_label = None
        self.motion_v2_apc_status_label = None
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("basicFlowRoot")
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setObjectName("DailyProcessingScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Reserve a small visual gutter so the vertical scrollbar does not cover
        # the right edge of parameter cards or the form editor.
        scroll.setViewportMargins(0, 0, 8, 0)
        outer_layout.addWidget(scroll)

        content = QWidget()
        content.setObjectName("DailyProcessingScrollContent")
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 20, 7)
        layout.setSpacing(7)

        basic_heading = QLabel("日常处理")
        basic_heading.setProperty("class", "sectionTitle")
        basic_heading.setMaximumHeight(28)
        layout.addWidget(basic_heading)

        # 核心操作区域
        action_box = QGroupBox("核心操作")
        action_box.setProperty("cardStyle", "modern")
        action_box.setObjectName("basicActionCard")
        action_layout = QVBoxLayout(action_box)
        action_layout.setContentsMargins(8, 12, 8, 7)
        action_layout.setSpacing(5)

        self.btn_import = SplitActionButton("导入数据", FluentIcon.FOLDER, self)
        self.btn_import.setObjectName("basicImportButton")
        self.btn_import.setToolTip(
            "点击主区域导入常见 GPR / CSV 数据，点击右侧箭头查看其它导入方式"
        )

        self.import_menu = QMenu(self)
        self.action_import_csv = QAction("导入 GPR / CSV 文件", self)
        self.action_import_folder = QAction("导入 A-scan 文件夹", self)
        self.action_import_out = QAction("导入 .out 数据文件", self)
        self.import_menu.addAction(self.action_import_csv)
        self.import_menu.addAction(self.action_import_folder)
        self.import_menu.addAction(self.action_import_out)

        self.btn_import.setMenu(self.import_menu)

        self.btn_apply = SplitActionButton("应用方法", FluentIcon.PLAY_SOLID, self)
        self.btn_apply.setObjectName("basicApplyButton")
        self.btn_apply.setToolTip(
            "点击主区域按当前默认来源执行，点击右侧箭头切换默认应用来源"
        )

        self.apply_menu = QMenu(self)
        self.action_apply_manual = QAction("使用当前参数（默认）", self)
        self.action_apply_auto_tuned = QAction("使用推荐参数", self)
        self.apply_menu.addAction(self.action_apply_manual)
        self.apply_menu.addAction(self.action_apply_auto_tuned)
        self.btn_apply.setMenu(self.apply_menu)
        self.action_apply_manual.triggered.connect(
            lambda: self.set_apply_source_mode("manual")
        )
        self.action_apply_auto_tuned.triggered.connect(
            lambda: self.set_apply_source_mode("auto_tune")
        )
        self.set_auto_tune_result_available(False)

        self.btn_quick = PushButton(FluentIcon.SYNC, "默认流程")
        self.btn_quick.setObjectName("basicQuickButton")
        self.btn_quick.setProperty("class", "basicGhostBtn")
        self.btn_quick.setMinimumHeight(28)
        self.btn_quick.setToolTip(
            "自动执行推荐处理流程：零时矫正 → 低频漂移抑制 → 频域滤波 → UAV 运动补偿 → 背景/F-K → 去噪 → SEC 增益；参数来源跟随“应用方法”的当前选项"
        )

        self.btn_cancel = PushButton(FluentIcon.CLOSE, "取消处理")
        self.btn_cancel.setObjectName("btnCancel")
        self.btn_cancel.setProperty("class", "basicGhostBtn")
        self.btn_cancel.setMinimumHeight(28)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("取消当前正在进行的处理任务")

        self.btn_undo = PushButton(FluentIcon.RETURN, "撤销")
        self.btn_undo.setProperty("class", "basicGhostBtn")
        self.btn_undo.setMinimumHeight(28)
        self.btn_undo.setToolTip("撤销上一步操作，恢复到之前的状态（最多保存10步历史）")

        self.btn_reset = PushButton(FluentIcon.ROTATE, "重置原始")
        self.btn_reset.setProperty("class", "basicGhostBtn")
        self.btn_reset.setMinimumHeight(28)
        self.btn_reset.setToolTip("重置为原始导入的数据状态")

        row_first = QWidget()
        row_first_l = QHBoxLayout(row_first)
        row_first_l.setContentsMargins(0, 0, 0, 0)
        row_first_l.setSpacing(5)

        row_first_l.addWidget(self.btn_import)
        row_first_l.addWidget(self.btn_apply)
        row_first_l.setStretch(0, 1)
        row_first_l.setStretch(1, 1)
        action_layout.addWidget(row_first)

        row_second = QWidget()
        row_second_l = QHBoxLayout(row_second)
        row_second_l.setContentsMargins(0, 0, 0, 0)
        row_second_l.setSpacing(5)
        row_second_l.addWidget(self.btn_quick)
        row_second_l.addWidget(self.btn_cancel)
        row_second_l.setStretch(0, 1)
        row_second_l.setStretch(1, 1)
        action_layout.addWidget(row_second)

        row_third = QWidget()
        row_third_l = QHBoxLayout(row_third)
        row_third_l.setContentsMargins(0, 0, 0, 0)
        row_third_l.setSpacing(5)
        row_third_l.addWidget(self.btn_undo)
        row_third_l.addWidget(self.btn_reset)
        row_third_l.setStretch(0, 1)
        row_third_l.setStretch(1, 1)
        action_layout.addWidget(row_third)

        self.apply_feedback_label = QLabel("就绪：按当前参数执行。")
        self.apply_feedback_label.setObjectName("ApplyFeedbackLabel")
        self.apply_feedback_label.setWordWrap(True)
        self.apply_feedback_label.setMaximumHeight(24)
        self.apply_feedback_label.setProperty("tone", "neutral")
        action_layout.addWidget(self.apply_feedback_label)

        layout.addWidget(action_box)

        # 处理阶段：用于筛选真实处理方法。真实执行历史由主图下方链路条记录。
        layout.addWidget(self._build_processing_stage_filter())

        # 当前步骤参数
        method_box = QGroupBox("当前步骤参数")
        method_box.setProperty("cardStyle", "modern")
        method_box.setObjectName("basicMethodCard")
        method_box.setToolTip("选择当前处理步骤对应的方法，并配置关键参数")
        method_layout = QVBoxLayout(method_box)
        method_layout.setContentsMargins(10, 15, 10, 10)
        method_layout.setSpacing(7)

        self.method_combo = QComboBox()
        self.method_combo.setObjectName("methodCombo")
        self.method_combo.setMinimumHeight(31)
        self.method_combo.setToolTip("选择GPR数据处理方法")
        self.all_method_keys = get_public_method_keys()
        self.method_keys = []
        self._active_method_stage = "all"
        self._rebuild_method_combo("all")
        method_layout.addWidget(self.method_combo)

        self.method_inspector_header = QFrame()
        self.method_inspector_header.setObjectName("MethodInspectorHeader")
        inspector_header_layout = QVBoxLayout(self.method_inspector_header)
        inspector_header_layout.setContentsMargins(10, 8, 10, 8)
        inspector_header_layout.setSpacing(3)
        self.method_category_tag = QLabel("类别")
        self.method_category_tag.setObjectName("MethodCategoryTag")
        self.method_name_label = QLabel("当前方法")
        self.method_name_label.setObjectName("MethodNameLabel")
        self.method_name_label.setWordWrap(True)
        inspector_header_layout.addWidget(self.method_category_tag)
        inspector_header_layout.addWidget(self.method_name_label)
        method_layout.addWidget(self.method_inspector_header)

        self.param_dirty_label = QLabel("参数未修改")
        self.param_dirty_label.setObjectName("ParamDirtyLabel")
        self.param_dirty_label.setProperty("tone", "clean")
        self.param_dirty_label.setWordWrap(True)
        method_layout.addWidget(self.param_dirty_label)

        # Basic and advanced parameters are displayed together by default.
        # The old collapsible toggle is intentionally removed to keep all
        # method parameters immediately visible in the daily workflow.
        self.show_advanced_params_var = None

        self.param_container = QWidget()
        self.param_container.setObjectName("InspectorParamPanel")
        self.param_layout = QFormLayout(self.param_container)
        self.param_layout.setContentsMargins(7, 7, 7, 7)
        self.param_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.param_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.param_layout.setHorizontalSpacing(12)
        self.param_layout.setVerticalSpacing(6)
        method_layout.addWidget(self.param_container)

        self.param_hint_label = QLabel("")
        self.param_hint_label.setWordWrap(True)
        self.param_hint_label.setProperty("class", "hintText")
        method_layout.addWidget(self.param_hint_label)

        layout.addWidget(method_box)

        data_view_box = QGroupBox("当前状态与运行反馈")
        data_view_box.setProperty("cardStyle", "modern")
        data_view_box.setObjectName("basicStatusCard")
        data_view_box.setToolTip("显示当前加载数据、当前方法和执行反馈")
        data_view_layout = QVBoxLayout(data_view_box)
        data_view_layout.setContentsMargins(10, 15, 10, 10)
        data_view_layout.setSpacing(6)

        self.data_brief = QLabel("未加载数据")
        self.data_brief.setProperty("class", "statusChip")
        self.data_brief.setWordWrap(True)
        self.data_brief.setMinimumHeight(28)
        self.data_brief.setToolTip("当前数据状态：显示数据矩阵尺寸和所选方法")
        data_view_layout.addWidget(self.data_brief)

        status_hint = QLabel("导入概况、当前方法和最近一次执行反馈。")
        status_hint.setWordWrap(True)
        status_hint.setProperty("class", "hintText")
        status_hint.setMaximumHeight(24)
        data_view_layout.addWidget(status_hint)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMinimumHeight(84)
        self.info.setMaximumHeight(110)
        self.info.setObjectName("basicInfoLog")
        self.info.setPlaceholderText("导入后展示：数据概况 / 当前方法 / 执行反馈")
        self.info.setToolTip("处理和操作日志显示区域")
        data_view_layout.addWidget(self.info)
        layout.addWidget(data_view_box)

        layout.addStretch(1)

        # 初始化参数渲染
        self._render_params(self.method_keys[0])

    METHOD_STAGE_DEFS = [
        ("all", "全部"),
        ("correction", "校正/QC"),
        ("filter", "滤波"),
        ("suppress", "抑制"),
        ("denoise", "去噪"),
        ("gain", "增益"),
        ("image", "成像/属性"),
    ]

    METHOD_STAGE_CATEGORIES = {
        "correction": {
            "time_correction",
            "drift_correction",
            "motion_compensation",
            "quality_control",
        },
        "filter": {"filtering"},
        "suppress": {
            "background_suppression",
            "clutter_suppression",
            "artifact_suppression",
        },
        "denoise": {"denoising"},
        "gain": {"gain"},
        "image": {"migration", "depth_conversion", "attribute_analysis"},
    }

    def _build_processing_stage_filter(self) -> QGroupBox:
        """Build actionable stage filter for the manual processing page.

        The previous static flow card only explained Raw/校正/抑制/增益.  The
        real processing history is already shown by the bottom lineage bar, so
        this compact control now filters the method list by processing stage.
        """
        box = QGroupBox("处理阶段")
        box.setProperty("cardStyle", "modern")
        box.setObjectName("basicStageFilterCard")
        box.setToolTip("筛选下方算法列表；真实处理链路显示在主图下方。")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)

        self._stage_filter_buttons = {}
        stage_rows = [self.METHOD_STAGE_DEFS[:3], self.METHOD_STAGE_DEFS[3:]]
        for row_defs in stage_rows:
            row = QWidget()
            row_l = QHBoxLayout(row)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(4)
            for stage_key, stage_label in row_defs:
                btn = PushButton(stage_label)
                btn.setObjectName("StageFilterButton")
                btn.setProperty("class", "stageFilterBtn")
                btn.setCheckable(True)
                btn.setMinimumHeight(26)
                btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                btn.setToolTip(f"只显示{stage_label}阶段的处理方法")
                btn.clicked.connect(lambda _checked=False, key=stage_key: self.set_method_stage_filter(key))
                row_l.addWidget(btn, 1)
                self._stage_filter_buttons[stage_key] = btn
            layout.addWidget(row)

        self.stage_filter_hint = QLabel("按阶段筛选算法；显示设置请到“显示”页，真实处理历史见底部链路。")
        self.stage_filter_hint.setObjectName("StageFilterHint")
        self.stage_filter_hint.setProperty("class", "hintText")
        self.stage_filter_hint.setWordWrap(True)
        self.stage_filter_hint.setMaximumHeight(32)
        layout.addWidget(self.stage_filter_hint)
        self._refresh_stage_filter_buttons("all")
        return box

    def _method_stage_from_category(self, category: str) -> str:
        category = str(category or "")
        for stage_key, categories in self.METHOD_STAGE_CATEGORIES.items():
            if category in categories:
                return stage_key
        return "all"

    def _method_stage_for_key(self, method_key: str) -> str:
        return self._method_stage_from_category(get_method_category(method_key))

    def _method_keys_for_stage(self, stage_key: str) -> list[str]:
        stage_key = str(stage_key or "all")
        all_keys = list(getattr(self, "all_method_keys", []) or get_public_method_keys())
        if stage_key == "all":
            return all_keys
        categories = self.METHOD_STAGE_CATEGORIES.get(stage_key, set())
        return [key for key in all_keys if get_method_category(key) in categories]

    def _format_method_combo_label(self, method_key: str) -> str:
        return f"[{get_method_category_label(method_key)}] {get_method_display_name(method_key)}"

    def _rebuild_method_combo(self, stage_key: str = "all", preserve_key: str | None = None) -> None:
        if not hasattr(self, "method_combo") or self.method_combo is None:
            return
        preserve_key = preserve_key or (self.get_current_method_key() if getattr(self, "method_keys", None) else None)
        self._active_method_stage = str(stage_key or "all")
        keys = self._method_keys_for_stage(self._active_method_stage) or list(getattr(self, "all_method_keys", []) or get_public_method_keys())
        self.method_keys = keys
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        self.method_combo.addItems([self._format_method_combo_label(k) for k in self.method_keys])
        if preserve_key in self.method_keys:
            self.method_combo.setCurrentIndex(self.method_keys.index(preserve_key))
        elif self.method_keys:
            self.method_combo.setCurrentIndex(0)
        self.method_combo.blockSignals(False)
        self._refresh_stage_filter_buttons(self._active_method_stage)
        if self.method_keys and hasattr(self, "param_layout"):
            key = self.method_keys[self.method_combo.currentIndex()]
            self._render_params(key)

    def set_method_stage_filter(self, stage_key: str) -> None:
        """Filter visible processing methods by stage without changing data."""
        stage_key = str(stage_key or "all")
        if stage_key not in dict(self.METHOD_STAGE_DEFS):
            stage_key = "all"
        previous_key = self.get_current_method_key()
        self._rebuild_method_combo(stage_key, previous_key)
        if getattr(self, "stage_filter_hint", None) is not None:
            label = dict(self.METHOD_STAGE_DEFS).get(stage_key, "全部")
            count = len(getattr(self, "method_keys", []) or [])
            self.stage_filter_hint.setText(f"当前阶段：{label}，共 {count} 个可用方法。真实处理历史显示在底部链路。")

    def _refresh_stage_filter_buttons(self, active_stage: str) -> None:
        for stage_key, btn in getattr(self, "_stage_filter_buttons", {}).items():
            btn.setChecked(stage_key == active_stage)
            btn.setProperty("active", stage_key == active_stage)
            try:
                btn.style().unpolish(btn); btn.style().polish(btn); btn.update()
            except Exception:
                pass

    def _flow_stage_from_category(self, category: str, method_key: str | None = None) -> str:
        stage = self._method_stage_from_category(get_method_category(method_key)) if method_key else "all"
        if stage in {"correction", "filter"}:
            return "correct"
        if stage == "suppress":
            return "suppress"
        if stage in {"gain", "denoise", "image"}:
            return "enhance"
        return "raw"

    def _update_processing_flow_stepper(self, category: str, method_key: str | None = None) -> None:
        # Legacy compatibility hook. The old static stepper was removed; stage
        # filtering and the bottom lineage bar now carry the workflow context.
        return

    def render_method_params(self, method_key: str, overrides: dict | None = None):
        """公开方法：渲染指定方法的参数输入。"""
        self._render_params(method_key, overrides)

    def get_current_params(self) -> dict:
        """公开方法：读取当前基础页参数。"""
        return self._get_params()

    def set_method_overrides(self, method_key: str, params: dict | None = None):
        """公开方法：更新方法参数覆盖。"""
        self._method_param_overrides[method_key] = dict(params or {})


    def _refresh_method_inspector_header(self, method_key: str) -> None:
        """Update the compact Inspector-style method summary."""
        category = get_method_category_label(method_key)
        name = get_method_display_name(method_key)
        if getattr(self, "method_category_tag", None) is not None:
            self.method_category_tag.setText(category)
        if getattr(self, "method_name_label", None) is not None:
            self.method_name_label.setText(name)
        self._update_processing_flow_stepper(category, method_key)

    def _render_params(self, method_key: str, overrides: dict | None = None):
        """渲染方法参数输入框"""
        while self.param_layout.rowCount():
            self.param_layout.removeRow(0)
        self.param_vars = {}
        self.btn_stolt_apply = None
        self.stolt_preset_combo = None
        self.stolt_auto_adapt_var = None
        self.motion_v2_trace_metadata_status_label = None
        self.motion_v2_apc_status_label = None

        self._refresh_method_inspector_header(method_key)
        all_params = PROCESSING_METHODS[method_key].get("params", [])
        params = all_params
        category_label = get_method_category_label(method_key)
        if overrides is not None:
            self._method_param_overrides[method_key] = dict(overrides)
        active_overrides = self._method_param_overrides.get(method_key, {})

        if self._basic_ultra_mode:
            common_names = self.BASIC_COMMON_PARAM_NAMES.get(method_key)
            if common_names:
                params_by_name = {param.get("name"): param for param in all_params}
                params = [params_by_name[name] for name in common_names if name in params_by_name]
            else:
                params = all_params[: self.BASIC_PARAM_LIMIT]
            hidden_count = max(0, len(all_params) - len(params))
            if method_key == "motion_compensation_v2":
                self.param_hint_label.setText(
                    f"类别：{category_label}。V2 会自动读取导入或传感器同步后的逐道信息；"
                    "这里只配置高度来源、参考高度、开关和等距重采样。"
                    "展开高级参数（原高级设置能力）可编辑安全阈值与 APC offset。"
                )
            elif hidden_count > 0:
                self.param_hint_label.setText(
                    f"类别：{category_label}。已精简：仅展示前 {len(params)} 个常用参数，另外 {hidden_count} 个参数请到'高级设置'调整。"
                )
            else:
                self.param_hint_label.setText(f"类别：{category_label}。")
        else:
            if method_key == "motion_compensation_v2":
                self.param_hint_label.setText(
                    f"类别：{category_label}。显示完整参数。APC offset 是设备安装几何标定参数，"
                    "不是动态飞行传感器数据；resample_spacing_m=0 表示自动使用中位道间距；"
                    "V2 会在可用时自动读取逐道信息。"
                )
            else:
                self.param_hint_label.setText(
                    f"类别：{category_label}。显示完整参数，参数默认值保持注册表定义。"
                )

        # Stolt迁移特殊预设
        if method_key == "stolt_migration":
            from PyQt6.QtWidgets import QCheckBox

            stolt_preset_row = QWidget()
            stolt_preset_layout = QHBoxLayout(stolt_preset_row)
            stolt_preset_layout.setContentsMargins(0, 0, 0, 0)
            stolt_preset_layout.setSpacing(6)

            self.stolt_preset_combo = QComboBox()
            for preset_key, preset in STOLT_MIGRATION_PRESETS.items():
                self.stolt_preset_combo.addItem(preset["label"], preset_key)
            self.stolt_preset_combo.setToolTip(
                "选择Stolt迁移的预设配置：平衡或聚焦优先"
            )

            self.stolt_auto_adapt_var = QCheckBox("应用时自适应")
            self.stolt_auto_adapt_var.setChecked(True)
            self.stolt_auto_adapt_var.setToolTip("根据数据特征自动选择最佳预设配置")

            self.btn_stolt_apply = PushButton(FluentIcon.SETTING, "应用Stolt推荐")
            self.btn_stolt_apply.setToolTip("应用选中的Stolt预设参数")

            stolt_preset_layout.addWidget(self.stolt_preset_combo)
            stolt_preset_layout.addWidget(self.stolt_auto_adapt_var)
            stolt_preset_layout.addWidget(self.btn_stolt_apply)
            self.param_layout.addRow(QLabel("Stolt快速预设"), stolt_preset_row)
            self.stolt_preset_combo.currentIndexChanged.connect(lambda _idx: self._mark_params_dirty())
            self.stolt_auto_adapt_var.toggled.connect(lambda _checked: self._mark_params_dirty())

        if not params:
            self.param_layout.addRow(QLabel("(无参数)"))
            self._refresh_apply_menu_state()
            self.mark_params_applied("当前方法无可调参数。")
            return

        for p in params:
            value = active_overrides.get(p["name"], p.get("default", ""))
            edit = self._create_param_editor(p, value)
            label = QLabel(p["label"])
            label.setObjectName("ParamFieldLabel")
            label.setWordWrap(True)
            self.param_layout.addRow(label, edit)
            self.param_vars[p["name"]] = (edit, p)
            self._wire_param_dirty_signal(edit)

        if method_key == "motion_compensation_v2":
            self._add_motion_v2_status_rows(all_params, active_overrides)

        self._wire_motion_param_dependencies(method_key)
        self._refresh_apply_menu_state()
        self.mark_params_applied("参数已载入；修改后需重新应用。")

    def _wire_param_dirty_signal(self, widget) -> None:
        """Connect editor changes to the dirty-parameter indicator."""
        try:
            if isinstance(widget, QLineEdit):
                widget.textEdited.connect(lambda _text: self._mark_params_dirty())
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(lambda _idx: self._mark_params_dirty())
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(lambda _checked: self._mark_params_dirty())
        except Exception:
            return

    def _set_param_dirty_visual(self, dirty: bool, text: str | None = None) -> None:
        self._params_dirty = bool(dirty)
        label = getattr(self, "param_dirty_label", None)
        if label is not None:
            if dirty:
                label.setText(text or "参数已修改，尚未应用")
                label.setProperty("tone", "dirty")
            else:
                label.setText(text or "参数未修改")
                label.setProperty("tone", "clean")
            try:
                label.style().unpolish(label); label.style().polish(label); label.update()
            except Exception:
                pass

    def _mark_params_dirty(self) -> None:
        self._set_param_dirty_visual(True, "参数已修改，尚未应用")
        self.set_apply_button_state("dirty", "参数已修改，点击“应用方法”更新 B-scan。")
        try:
            self.parametersChanged.emit()
        except Exception:
            pass

    def mark_params_applied(self, message: str | None = None) -> None:
        self._set_param_dirty_visual(False, "参数未修改")
        self.set_apply_button_state("idle", message or "就绪：按当前参数执行。")

    def set_apply_button_state(self, state: str = "idle", message: str | None = None) -> None:
        """Show clear apply-button feedback without changing processing semantics."""
        state = str(state or "idle").lower()
        mapping = {
            "idle": ("应用方法", "normal", "neutral", "就绪：按当前参数执行。"),
            "dirty": ("应用方法", "dirty", "warning", "参数已修改，尚未应用。"),
            "busy": ("正在处理…", "busy", "info", "正在执行，请等待后台任务完成。"),
            "success": ("已应用", "success", "success", "已应用到当前 B-scan。"),
            "error": ("执行失败", "error", "danger", "方法执行失败，请查看全局日志。"),
        }
        text, visual, tone, default_msg = mapping.get(state, mapping["idle"])
        self._apply_state = state
        try:
            self.btn_apply.setText(text)
            self.btn_apply.setVisualState(visual)
        except Exception:
            pass
        feedback = getattr(self, "apply_feedback_label", None)
        if feedback is not None:
            feedback.setText(message or default_msg)
            feedback.setProperty("tone", tone)
            try:
                feedback.style().unpolish(feedback); feedback.style().polish(feedback); feedback.update()
            except Exception:
                pass

    def _on_show_advanced_params_toggled(self, checked: bool) -> None:
        """Switch the daily panel between common and full parameter sets."""
        current_key = self.get_current_method_key()
        if not current_key:
            return
        self._update_current_method_overrides()
        self._basic_ultra_mode = not bool(checked)
        self._render_params(current_key)

    def _add_motion_v2_status_rows(self, all_params: list[dict], active_overrides: dict) -> None:
        """Add read-only trace metadata/APC status rows for the V2 basic panel."""
        self.motion_v2_trace_metadata_status_label = QLabel(
            self._motion_v2_trace_metadata_status_text()
        )
        self.motion_v2_trace_metadata_status_label.setWordWrap(True)
        self.motion_v2_trace_metadata_status_label.setProperty("class", "hintText")
        self.motion_v2_trace_metadata_status_label.setToolTip(
            "这些逐道字段来自导入和传感器同步结果，只显示状态，不允许在此手动填写数组。"
        )
        self.param_layout.addRow(
            QLabel("逐道信息状态"),
            self.motion_v2_trace_metadata_status_label,
        )

        self.motion_v2_apc_status_label = QLabel(
            self._motion_v2_apc_status_text(all_params, active_overrides)
        )
        self.motion_v2_apc_status_label.setWordWrap(True)
        self.motion_v2_apc_status_label.setProperty("class", "hintText")
        self.motion_v2_apc_status_label.setToolTip(
            "APC offset 是设备安装几何标定，不是逐道飞行传感器数组。"
        )
        self.param_layout.addRow(
            QLabel("APC 配置状态"),
            self.motion_v2_apc_status_label,
        )

    def _motion_v2_trace_metadata_status_text(self) -> str:
        metadata = self._current_trace_metadata()
        header = self._current_header_info()
        required_groups = [
            (
                "height_agl_m",
                ("height_agl_m",),
                "缺 height_agl_m：高度时移 / 幅值归一可能跳过或 fallback。",
            ),
            (
                "trace_distance_m",
                ("trace_distance_m",),
                "缺 trace_distance_m：等距重采样会跳过。",
            ),
            (
                "local_x_m / local_y_m",
                ("local_x_m", "local_y_m"),
                "缺 local_x_m / local_y_m：三维轨迹和 APC 足迹显示会降级。",
            ),
            (
                "roll_deg / pitch_deg / yaw_deg",
                ("roll_deg", "pitch_deg", "yaw_deg"),
                "缺 roll/pitch/yaw：姿态 footprint 修正会跳过。",
            ),
            (
                "trace_timestamp_s",
                ("trace_timestamp_s",),
                "缺 trace_timestamp_s：传感器同步质量无法复核。",
            ),
        ]
        ok: list[str] = []
        warnings: list[str] = []
        for label, keys, missing_text in required_groups:
            if metadata is not None and all(self._metadata_field_present(metadata, key) for key in keys):
                ok.append(label)
            else:
                warnings.append(missing_text)

        has_time_window = False
        if metadata is not None and self._metadata_field_present(metadata, "time_window_ns"):
            has_time_window = True
        if header:
            has_time_window = has_time_window or any(
                key in header and header.get(key) not in (None, "", 0)
                for key in ("time_window_ns", "total_time_ns", "Time windows (ns)")
            )
        if has_time_window:
            ok.append("time_window_ns")
        else:
            warnings.append("缺 time_window_ns：高度 time-shift 会跳过。")

        if metadata is None:
            return "未检测到逐道信息；请在导入时同步 RTK、IMU 或高度计辅助文件。 " + " ".join(warnings)
        return f"已检测：{', '.join(ok) if ok else '无'}。 " + " ".join(warnings)

    def _motion_v2_apc_status_text(self, all_params: list[dict], active_overrides: dict) -> str:
        params_by_name = {param.get("name"): param for param in all_params}
        values = []
        for key in ("apc_offset_x_m", "apc_offset_y_m", "apc_offset_z_m"):
            default = params_by_name.get(key, {}).get("default", 0.0)
            values.append(float(active_overrides.get(key, default) or 0.0))
        if any(abs(value) > 1.0e-12 for value in values):
            return (
                "已配置 APC offset 覆盖："
                f"X={values[0]:.4g} m, Y={values[1]:.4g} m, Z={values[2]:.4g} m。"
                "这些是设备安装几何参数。"
            )
        return (
            "未配置设备 APC 标定，当前按 0 处理；"
            "如设备安装方式固定，应在高级设置或设备配置中标定一次。"
        )

    def _current_trace_metadata(self):
        parent = self.parent_window
        shared = getattr(parent, "shared_data", None)
        if shared is not None:
            metadata = getattr(shared, "current_trace_metadata", None) or getattr(
                shared, "original_trace_metadata", None
            )
            if metadata is not None:
                return metadata
        return getattr(parent, "trace_metadata", None)

    def _current_header_info(self) -> dict:
        parent = self.parent_window
        shared = getattr(parent, "shared_data", None)
        if shared is not None:
            header = getattr(shared, "header_info", None) or getattr(
                shared, "original_header_info", None
            )
            if isinstance(header, dict):
                return header
        header = getattr(parent, "header_info", None)
        return header if isinstance(header, dict) else {}

    @staticmethod
    def _metadata_field_present(metadata: dict, key: str) -> bool:
        if key not in metadata:
            return False
        value = metadata.get(key)
        if value is None:
            return False
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            return bool(arr.size and np.isfinite(arr.astype(float, copy=False)).any())
        return bool(arr.size)

    def _create_param_editor(self, meta: dict, value):
        """Create a parameter editor matching the registry parameter type."""
        param_type = meta.get("type")
        if param_type == "choice":
            edit = QComboBox()
            choices = list(meta.get("choices", []))
            for choice in choices:
                edit.addItem(str(choice), choice)
            value_text = str(value)
            idx = edit.findText(value_text)
            if idx < 0 and value_text:
                edit.addItem(value_text, value)
                idx = edit.findText(value_text)
            edit.setCurrentIndex(max(idx, 0))
        elif param_type == "bool":
            edit = QCheckBox("启用")
            edit.setChecked(bool(value))
        else:
            edit = QLineEdit(str(value))

        edit.setMinimumWidth(180)
        edit.setMinimumHeight(32)
        tooltip = str(meta.get("tooltip") or "").strip()
        range_text = f"参数范围: {meta.get('min', '无下限')} ~ {meta.get('max', '无上限')}"
        edit.setToolTip(f"{tooltip}\n{range_text}" if tooltip else range_text)
        return edit

    def _wire_motion_param_dependencies(self, method_key: str) -> None:
        """Disable manual height unless the reference-height mode is manual."""
        mode_name = None
        manual_name = None
        if method_key == "motion_compensation_height":
            mode_name = "reference_height_mode"
            manual_name = "manual_height"
        elif method_key == "motion_compensation_v2":
            mode_name = "height_reference_mode"
            manual_name = "manual_height_m"
        if not mode_name or not manual_name:
            return
        mode_entry = self.param_vars.get(mode_name)
        manual_entry = self.param_vars.get(manual_name)
        if not mode_entry or not manual_entry:
            return
        mode_widget, _ = mode_entry
        manual_widget, _ = manual_entry

        def update_manual_enabled():
            mode_value = self._read_param_widget_value(mode_widget, {"type": "choice"})
            manual_widget.setEnabled(str(mode_value) == "manual")

        if isinstance(mode_widget, QComboBox):
            mode_widget.currentTextChanged.connect(lambda _text: update_manual_enabled())
        update_manual_enabled()

    def _read_param_widget_value(self, widget, meta: dict):
        """Read a parameter editor value without assuming everything is QLineEdit."""
        if isinstance(widget, QComboBox):
            data = widget.currentData()
            return widget.currentText() if data is None else data
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QLineEdit):
            return widget.text().strip()
        return ""

    def _get_params(self):
        """获取当前参数值"""
        params = {}
        for name, (edit, meta) in self.param_vars.items():
            label = meta.get("label", name)
            raw = self._read_param_widget_value(edit, meta)
            if raw == "":
                default_val = meta.get("default", "")
                if default_val in (None, ""):
                    raise ValueError(f"参数'{label}'为空且无默认值")
                raw = str(default_val)

            try:
                if meta["type"] == "int":
                    val = int(float(raw))
                elif meta["type"] == "float":
                    val = float(raw)
                elif meta["type"] == "bool":
                    if isinstance(raw, bool):
                        val = raw
                    else:
                        lowered = str(raw).lower()
                        if lowered in {"true", "1", "yes", "on"}:
                            val = True
                        elif lowered in {"false", "0", "no", "off"}:
                            val = False
                        else:
                            raise ValueError
                elif meta["type"] == "choice":
                    choices = meta.get("choices", [])
                    val = raw
                    if choices and val not in choices:
                        raise ValueError
                else:
                    val = raw
            except ValueError:
                raise ValueError(f"参数'{label}'类型错误：输入值={raw!r}")

            min_v = meta.get("min")
            max_v = meta.get("max")
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if min_v is not None and val < min_v:
                    raise ValueError(f"参数'{label}'={val} 低于最小值 {min_v}")
                if max_v is not None and val > max_v:
                    raise ValueError(f"参数'{label}'={val} 高于最大值 {max_v}")

            params[name] = val
        return params

    def _update_current_method_overrides(self):
        """更新当前方法的参数覆盖"""
        if not hasattr(self, "param_vars"):
            return
        idx = self.method_combo.currentIndex()
        if idx < 0:
            return
        try:
            params = self._get_params()
        except ValueError:
            return
        self._method_param_overrides[self.method_keys[idx]] = params

    def get_current_method_key(self):
        """获取当前选中的方法key"""
        if not hasattr(self, "method_combo") or self.method_combo is None:
            return None
        idx = self.method_combo.currentIndex()
        if idx < 0:
            return None
        return self.method_keys[idx]

    def set_apply_source_hint(self, text: str):
        """设置应用来源提示。"""
        self._apply_source_hint_text = str(text or "")
        self._refresh_apply_menu_state()

    def get_apply_source_mode(self) -> str:
        """获取当前默认应用来源。"""
        return str(self._apply_source_mode or "manual")

    def set_apply_source_mode(self, mode: str):
        """设置点击“应用方法”主按钮时的默认来源。"""
        requested = str(mode or "manual")
        if requested not in {"manual", "auto_tune"}:
            requested = "manual"

        current_key = self.get_current_method_key()
        method_info = PROCESSING_METHODS.get(current_key, {}) if current_key else {}
        supports_auto_tune = bool(method_info.get("auto_tune_enabled"))
        if requested == "auto_tune" and not supports_auto_tune:
            requested = "manual"

        self._apply_source_mode = requested
        self._refresh_apply_menu_state()

    def set_auto_tune_result_available(
        self, available: bool, profiles: dict | None = None
    ):
        """根据 auto-tune 结果刷新应用菜单可用性。"""
        profiles = profiles or {}
        self._auto_tune_result_available = bool(available and profiles)
        if available:
            self.set_apply_source_hint("已生成推荐参数，可切换默认应用来源。")
        else:
            self.set_apply_source_hint("当前未生成推荐参数。")

    def _refresh_apply_menu_state(self):
        """刷新应用方法菜单与按钮提示。"""
        current_key = self.get_current_method_key()
        method_info = PROCESSING_METHODS.get(current_key, {}) if current_key else {}
        supports_auto_tune = bool(method_info.get("auto_tune_enabled"))

        if not supports_auto_tune and self._apply_source_mode == "auto_tune":
            self._apply_source_mode = "manual"

        manual_text = "使用当前参数"
        auto_text = "使用推荐参数"
        if self._apply_source_mode == "manual":
            manual_text += "（默认）"
        else:
            auto_text += "（默认）"

        self.action_apply_manual.setText(manual_text)
        self.action_apply_auto_tuned.setText(auto_text)
        self.action_apply_auto_tuned.setEnabled(bool(supports_auto_tune))

        if self._apply_source_mode == "auto_tune" and supports_auto_tune:
            hint = self._apply_source_hint_text or (
                "已生成推荐参数，可直接应用。"
                if self._auto_tune_result_available
                else "当前还没有可用候选结果。"
            )
            self.btn_apply.setToolTip(
                "点击主区域使用推荐参数执行；若当前方法尚无候选结果，会提示先前往“参数推荐”页生成。\n"
                + hint
            )
        else:
            hint = self._apply_source_hint_text or "将按当前参数执行。"
            self.btn_apply.setToolTip(
                "点击主区域按当前参数执行，点击右侧箭头切换默认应用来源。\n" + hint
            )

    def set_method_by_key(self, key: str):
        """通过key设置当前方法。若当前阶段过滤隐藏了该方法，则自动切换到对应阶段。"""
        all_keys = list(getattr(self, "all_method_keys", []) or get_public_method_keys())
        if key not in all_keys:
            return
        if key not in self.method_keys:
            self._rebuild_method_combo(self._method_stage_for_key(key), key)
        idx = self.method_keys.index(key)
        previous_idx = self.method_combo.currentIndex()
        self.method_combo.setCurrentIndex(idx)
        if idx == previous_idx or not self._parent_handles_method_change():
            self._render_params(key)

    def apply_method_params(self, method_key: str, params: dict | None = None):
        """切换到指定方法并应用参数覆盖。"""
        all_keys = list(getattr(self, "all_method_keys", []) or get_public_method_keys())
        if method_key not in all_keys:
            return
        if method_key not in self.method_keys:
            self._rebuild_method_combo(self._method_stage_for_key(method_key), method_key)
        idx = self.method_keys.index(method_key)
        if params is not None:
            self._method_param_overrides[method_key] = dict(params)
        previous_idx = self.method_combo.currentIndex()
        self.method_combo.setCurrentIndex(idx)
        if idx == previous_idx or not self._parent_handles_method_change():
            self._render_params(method_key)

    def _parent_handles_method_change(self) -> bool:
        """Return whether changing the combo will be rendered by the main window."""
        return callable(getattr(self.parent_window, "_on_method_change", None))
