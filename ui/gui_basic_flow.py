#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI基础流程页面 - 包含快速开始、方法选择、参数设置等基础UI"""

from PyQt6.QtCore import QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPen
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QComboBox,
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
        self._drop_width = 24
        self._hover_part = None
        self._pressed_part = None
        self.setMinimumHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def setMenu(self, menu):
        self._menu = menu

    def sizeHint(self):
        return QSize(180, 34)

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
        base = QColor(themeColor())
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

        # 分裂按钮右侧有单独下拉区，单纯按主区居中会显得整体偏左；
        # 这里补偿半个下拉区宽度，让视觉中心更接近下面普通按钮。
        visual_offset = self._drop_width // 2
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.param_vars = {}
        self._method_param_overrides = {}
        self._apply_source_mode = "manual"
        self._apply_source_hint_text = "应用来源：当前参数"
        self._auto_tune_result_available = False
        self._basic_ultra_mode = True
        self.BASIC_PARAM_LIMIT = 4
        self.btn_stolt_apply = None
        self.stolt_preset_combo = None
        self.stolt_auto_adapt_var = None
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("basicFlowRoot")
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer_layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        basic_heading = QLabel("日常处理")
        basic_heading.setProperty("class", "sectionTitle")
        layout.addWidget(basic_heading)

        self.quick_path_hint = QLabel(
            "围绕单条测线的日常处理操作。推荐顺序：导入数据 → 选择方法 → 应用方法或执行默认流程。"
        )
        self.quick_path_hint.setWordWrap(True)
        self.quick_path_hint.setProperty("class", "hintText")
        layout.addWidget(self.quick_path_hint)

        # 核心操作区域
        action_box = QGroupBox("核心操作")
        action_box.setObjectName("basicActionCard")
        action_layout = QVBoxLayout(action_box)
        action_layout.setContentsMargins(10, 14, 10, 10)
        action_layout.setSpacing(8)

        self.btn_import = SplitActionButton("导入数据", FluentIcon.FOLDER, self)
        self.btn_import.setToolTip(
            "点击主区域默认导入 CSV，点击右侧箭头查看其它导入方式"
        )

        self.import_menu = QMenu(self)
        self.action_import_csv = QAction("导入 CSV", self)
        self.action_import_folder = QAction("导入 A-scan 文件夹", self)
        self.action_import_out = QAction("导入 gprMax .out", self)
        self.import_menu.addAction(self.action_import_csv)
        self.import_menu.addAction(self.action_import_folder)
        self.import_menu.addAction(self.action_import_out)

        self.btn_import.setMenu(self.import_menu)

        self.btn_apply = SplitActionButton("应用方法", FluentIcon.PLAY_SOLID, self)
        self.btn_apply.setToolTip(
            "点击主区域按当前默认来源执行，点击右侧箭头切换默认应用来源"
        )

        self.apply_menu = QMenu(self)
        self.action_apply_manual = QAction("使用当前参数（默认）", self)
        self.action_apply_auto_tuned = QAction("使用自动调参参数", self)
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
        self.btn_quick.setProperty("class", "basicGhostBtn")
        self.btn_quick.setMinimumHeight(34)
        self.btn_quick.setToolTip(
            "自动执行默认流程：零时矫正 → 低频漂移抑制 → 背景抑制 → AGC增益 → 尖锐杂波抑制；参数来源跟随“应用方法”的当前选项"
        )

        self.btn_cancel = PushButton(FluentIcon.CLOSE, "取消处理")
        self.btn_cancel.setObjectName("btnCancel")
        self.btn_cancel.setProperty("class", "basicGhostBtn")
        self.btn_cancel.setMinimumHeight(34)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("取消当前正在进行的处理任务")

        self.btn_undo = PushButton(FluentIcon.RETURN, "撤销")
        self.btn_undo.setProperty("class", "basicGhostBtn")
        self.btn_undo.setMinimumHeight(34)
        self.btn_undo.setToolTip("撤销上一步操作，恢复到之前的状态（最多保存10步历史）")

        self.btn_reset = PushButton(FluentIcon.ROTATE, "重置原始")
        self.btn_reset.setProperty("class", "basicGhostBtn")
        self.btn_reset.setMinimumHeight(34)
        self.btn_reset.setToolTip("重置为原始导入的数据状态")

        row_first = QWidget()
        row_first_l = QHBoxLayout(row_first)
        row_first_l.setContentsMargins(0, 0, 0, 0)
        row_first_l.setSpacing(8)

        row_first_l.addWidget(self.btn_import)
        row_first_l.addWidget(self.btn_apply)
        row_first_l.setStretch(0, 1)
        row_first_l.setStretch(1, 1)
        action_layout.addWidget(row_first)

        row_second = QWidget()
        row_second_l = QHBoxLayout(row_second)
        row_second_l.setContentsMargins(0, 0, 0, 0)
        row_second_l.setSpacing(8)
        row_second_l.addWidget(self.btn_quick)
        row_second_l.addWidget(self.btn_cancel)
        row_second_l.setStretch(0, 1)
        row_second_l.setStretch(1, 1)
        action_layout.addWidget(row_second)

        row_third = QWidget()
        row_third_l = QHBoxLayout(row_third)
        row_third_l.setContentsMargins(0, 0, 0, 0)
        row_third_l.setSpacing(8)
        row_third_l.addWidget(self.btn_undo)
        row_third_l.addWidget(self.btn_reset)
        row_third_l.setStretch(0, 1)
        row_third_l.setStretch(1, 1)
        action_layout.addWidget(row_third)

        self.basic_save_hint = QLabel(
            "结果图可在图像工具栏点击 保存 按钮保存；处理后的数据会自动同步到工作台。"
        )
        self.basic_save_hint.setWordWrap(True)
        self.basic_save_hint.setProperty("class", "hintText")
        action_layout.addWidget(self.basic_save_hint)
        layout.addWidget(action_box)

        # 方法与参数
        method_box = QGroupBox("方法与常用参数")
        method_box.setObjectName("basicMethodCard")
        method_box.setToolTip("选择处理方法并配置参数")
        method_layout = QVBoxLayout(method_box)
        method_layout.setContentsMargins(10, 14, 10, 10)
        method_layout.setSpacing(8)

        self.method_combo = QComboBox()
        self.method_combo.setMinimumHeight(34)
        self.method_combo.setToolTip("选择GPR数据处理方法")
        self.method_keys = get_public_method_keys()
        self.method_combo.addItems(
            [
                f"[{get_method_category_label(k)}] {get_method_display_name(k)}"
                for k in self.method_keys
            ]
        )
        method_layout.addWidget(self.method_combo)

        self.param_container = QWidget()
        self.param_layout = QFormLayout(self.param_container)
        self.param_layout.setContentsMargins(4, 4, 4, 4)
        self.param_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.param_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.param_layout.setHorizontalSpacing(12)
        self.param_layout.setVerticalSpacing(8)
        method_layout.addWidget(self.param_container)

        self.param_hint_label = QLabel("")
        self.param_hint_label.setWordWrap(True)
        self.param_hint_label.setProperty("class", "hintText")
        method_layout.addWidget(self.param_hint_label)

        layout.addWidget(method_box)

        data_view_box = QGroupBox("当前状态与运行反馈")
        data_view_box.setObjectName("basicStatusCard")
        data_view_box.setToolTip("显示当前加载数据、当前方法和执行反馈")
        data_view_layout = QVBoxLayout(data_view_box)
        data_view_layout.setContentsMargins(10, 14, 10, 10)
        data_view_layout.setSpacing(8)

        self.data_brief = QLabel("未加载数据")
        self.data_brief.setProperty("class", "statusChip")
        self.data_brief.setToolTip("当前数据状态：显示数据矩阵尺寸和所选方法")
        data_view_layout.addWidget(self.data_brief)

        status_hint = QLabel("这里集中显示导入概况、当前方法和最近一次执行反馈。")
        status_hint.setWordWrap(True)
        status_hint.setProperty("class", "hintText")
        data_view_layout.addWidget(status_hint)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMinimumHeight(120)
        self.info.setMaximumHeight(160)
        self.info.setObjectName("basicInfoLog")
        self.info.setPlaceholderText("导入后展示：数据概况 / 当前方法 / 执行反馈")
        self.info.setToolTip("处理和操作日志显示区域")
        data_view_layout.addWidget(self.info)
        layout.addWidget(data_view_box)

        layout.addStretch(1)

        # 初始化参数渲染
        self._render_params(self.method_keys[0])

    def render_method_params(self, method_key: str, overrides: dict | None = None):
        """公开方法：渲染指定方法的参数输入。"""
        self._render_params(method_key, overrides)

    def get_current_params(self) -> dict:
        """公开方法：读取当前基础页参数。"""
        return self._get_params()

    def set_method_overrides(self, method_key: str, params: dict | None = None):
        """公开方法：更新方法参数覆盖。"""
        self._method_param_overrides[method_key] = dict(params or {})

    def _render_params(self, method_key: str, overrides: dict | None = None):
        """渲染方法参数输入框"""
        while self.param_layout.rowCount():
            self.param_layout.removeRow(0)
        self.param_vars = {}
        self.btn_stolt_apply = None
        self.stolt_preset_combo = None
        self.stolt_auto_adapt_var = None

        all_params = PROCESSING_METHODS[method_key].get("params", [])
        params = all_params
        category_label = get_method_category_label(method_key)
        if overrides is not None:
            self._method_param_overrides[method_key] = dict(overrides)
        active_overrides = self._method_param_overrides.get(method_key, {})

        if self._basic_ultra_mode:
            params = all_params[: self.BASIC_PARAM_LIMIT]
            hidden_count = max(0, len(all_params) - len(params))
            if hidden_count > 0:
                self.param_hint_label.setText(
                    f"类别：{category_label}。已精简：仅展示前 {len(params)} 个常用参数，另外 {hidden_count} 个参数请到'高级设置'调整。"
                )
            else:
                self.param_hint_label.setText(f"类别：{category_label}。")
        else:
            self.param_hint_label.setText(f"类别：{category_label}。")

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
                "选择Stolt迁移的预设配置：速度优先/平衡/聚焦优先"
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

        if not params:
            self.param_layout.addRow(QLabel("(无参数)"))
            self._refresh_apply_menu_state()
            return

        for p in params:
            value = active_overrides.get(p["name"], p.get("default", ""))
            edit = QLineEdit(str(value))
            edit.setMinimumWidth(180)
            edit.setMinimumHeight(32)
            edit.setToolTip(
                f"参数范围: {p.get('min', '无下限')} ~ {p.get('max', '无上限')}"
            )
            label = QLabel(p["label"])
            label.setWordWrap(True)
            self.param_layout.addRow(label, edit)
            self.param_vars[p["name"]] = (edit, p)

        self._refresh_apply_menu_state()

    def _get_params(self):
        """获取当前参数值"""
        params = {}
        for name, (edit, meta) in self.param_vars.items():
            label = meta.get("label", name)
            raw = edit.text().strip()
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
                    lowered = raw.lower()
                    if lowered in {"true", "1", "yes", "on"}:
                        val = True
                    elif lowered in {"false", "0", "no", "off"}:
                        val = False
                    else:
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
            self.set_apply_source_hint("已生成自动调参结果，可切换默认应用来源。")
        else:
            self.set_apply_source_hint("当前未生成自动调参结果。")

    def _refresh_apply_menu_state(self):
        """刷新应用方法菜单与按钮提示。"""
        current_key = self.get_current_method_key()
        method_info = PROCESSING_METHODS.get(current_key, {}) if current_key else {}
        supports_auto_tune = bool(method_info.get("auto_tune_enabled"))

        if not supports_auto_tune and self._apply_source_mode == "auto_tune":
            self._apply_source_mode = "manual"

        manual_text = "使用当前参数"
        auto_text = "使用自动调参参数"
        if self._apply_source_mode == "manual":
            manual_text += "（默认）"
        else:
            auto_text += "（默认）"

        self.action_apply_manual.setText(manual_text)
        self.action_apply_auto_tuned.setText(auto_text)
        self.action_apply_auto_tuned.setEnabled(bool(supports_auto_tune))

        if self._apply_source_mode == "auto_tune" and supports_auto_tune:
            hint = self._apply_source_hint_text or (
                "已生成自动调参结果，可直接应用。"
                if self._auto_tune_result_available
                else "当前还没有可用候选结果。"
            )
            self.btn_apply.setToolTip(
                "点击主区域使用自动调参参数执行；若当前方法尚无候选结果，会提示先前往“调参与实验”页分析。\n"
                + hint
            )
        else:
            hint = self._apply_source_hint_text or "将按当前参数执行。"
            self.btn_apply.setToolTip(
                "点击主区域按当前参数执行，点击右侧箭头切换默认应用来源。\n" + hint
            )

    def set_method_by_key(self, key: str):
        """通过key设置当前方法"""
        if key in self.method_keys:
            idx = self.method_keys.index(key)
            self.method_combo.setCurrentIndex(idx)
            self._render_params(key)

    def apply_method_params(self, method_key: str, params: dict | None = None):
        """切换到指定方法并应用参数覆盖。"""
        if method_key not in self.method_keys:
            return
        idx = self.method_keys.index(method_key)
        self.method_combo.setCurrentIndex(idx)
        self._render_params(method_key, params)
