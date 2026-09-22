"""SettingsPage — 系统设置页（SPEC §6.4）。

纯展示 + 发信号：不直接写 SettingsManager，由主窗口接线持久化。
公共接口：
- load_settings(dict)：回放设置到控件（blockSignals，不触发 theme_changed）
- settings() -> dict：当前控件值（键与 ui.settings_manager.DEFAULT_SETTINGS 对齐：
  theme / default_dielectric / max_workers / project_root；
  default_colormap / preview_max_samples 因消费端硬编码已移除，避免"设置了没反应"）
- set_theme_text(str)：主窗口主题切换后回写主题 ComboBox（blockSignals）

信号：theme_changed(str)（'浅色主题' / '深色主题'）。
"""

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel, CheckBox, ComboBox, DoubleSpinBox,
    LineEdit, PushButton, ScrollArea, SpinBox,
)

from ui import constants, file_dialogs
from ui.page_scaffold import (make_card, make_form_row, make_hint,
                              style_transparent_scroll, wrap_centered)

_FALLBACK_VERSION = '0.9.38'
_AUTHOR = '邸建豪 袁林 詹萍'
_COPYRIGHT = '© 2025 MyGPR 保留所有权利'


def _levels_or_default(data: dict) -> tuple[float, float]:
    """从设置字典取色阶百分位；缺失/非法回落 2 / 98（与视图默认一致）。"""
    try:
        low = float(data.get('bscan_p_low', 2.0))
        high = float(data.get('bscan_p_high', 98.0))
    except (TypeError, ValueError):
        return 2.0, 98.0
    if not 0.0 <= low < high <= 100.0:
        return 2.0, 98.0
    return low, high


def _read_version() -> str:
    """版本读仓库根 VERSION 文件，缺失/异常回退 0.9.38（SPEC §6.4）。"""
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        with open(os.path.join(root, 'VERSION'), 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return text or _FALLBACK_VERSION
    except OSError:
        return _FALLBACK_VERSION


class SettingsPage(ScrollArea):
    """系统设置：通用设置 / B-Scan 视图 / 处理设置 / 存储 / 关于。"""

    theme_changed = pyqtSignal(str)
    # B-Scan 视图设置变化（比例/轴单位/色阶）；主窗口据此统一下发并写盘
    bscan_view_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        style_transparent_scroll(self)
        # 回放设置期间抑制 bscan_view_changed（否则「读设置→写设置」回环）
        self._loading_settings = False

        container = QWidget(self)
        container.setStyleSheet('background-color: transparent;')
        root = QVBoxLayout(container)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)

        root.addWidget(self._build_general_card(container))
        root.addWidget(self._build_bscan_card(container))
        root.addWidget(self._build_processing_card(container))
        root.addWidget(self._build_storage_card(container))
        root.addWidget(self._build_about_card(container))
        root.addStretch(1)

        self.setWidget(wrap_centered(container, constants.FORM_COLUMN_MAX_WIDTH))

    # ============================================================ 卡片构建
    def _build_general_card(self, parent):
        """卡片1"通用设置"：主题 / 默认介电常数 / 默认颜色映射 / 预览下采样上限。"""
        card, layout = make_card('通用设置')

        # 界面主题
        self._theme_combo = ComboBox(card)
        self._theme_combo.addItems([constants.THEME_LIGHT, constants.THEME_DARK])
        self._theme_combo.setCurrentText(constants.THEME_LIGHT)
        self._theme_combo.setMinimumWidth(150)
        self._theme_combo.currentTextChanged.connect(self.theme_changed)
        layout.addLayout(make_form_row('界面主题:', self._theme_combo,
                                       parent=card))

        # 默认介电常数
        self._dielectric_spin = DoubleSpinBox(card)
        self._dielectric_spin.setRange(1.0, 81.0)
        self._dielectric_spin.setDecimals(2)
        self._dielectric_spin.setSingleStep(0.5)
        self._dielectric_spin.setValue(constants.DEFAULT_DIELECTRIC)
        self._dielectric_spin.setMinimumWidth(120)
        layout.addLayout(make_form_row('默认介电常数:', self._dielectric_spin,
                                       parent=card))

        # 自动预下载底图（空间页加载轨迹后自动下载瓦片）
        prefetch_row = QHBoxLayout()
        self._prefetch_check = CheckBox('自动预下载测线区域底图', card)
        self._prefetch_check.setToolTip('空间页加载轨迹后自动下载当地底图瓦片；关掉可避免自动联网。')
        prefetch_row.addWidget(self._prefetch_check)
        prefetch_row.addStretch(1)
        layout.addLayout(prefetch_row)

        # 默认颜色映射（九项，默认 seismic）—— 消费端（bscan_view）目前硬编码，暂不接线
        # 预览下采样上限（300-4000，默认 900）—— 消费端（project_controller）目前硬编码，暂不接线
        # 这两个设置项已从 UI 移除，避免「设置了没反应」伤信任；待消费端统一接入后再恢复。
        return card

    def _build_bscan_card(self, parent):
        """卡片2"B-Scan 视图"：比例 / 横纵轴单位 / 色阶（即时生效并跨会话记住）。

        这几项在 BScanView 工具条与右键菜单里都能改，此处只是把「习惯」集中
        可查；改动即时应用到本会话所有 B-Scan（发 ``bscan_view_changed``，
        由主窗口下发），不必重启。
        """
        card, layout = make_card('B-Scan 视图')

        self._bscan_aspect_combo = ComboBox(card)
        for label, key in (('拉伸铺满（推荐）', 'free'),
                           ('数据盒正方形', 'square'),
                           ('数据格 1:1', 'cell')):
            self._bscan_aspect_combo.addItem(label, userData=key)
        self._bscan_aspect_combo.setMinimumWidth(180)
        self._bscan_aspect_combo.currentIndexChanged.connect(
            self._emit_bscan_changed)
        layout.addLayout(make_form_row('显示比例:', self._bscan_aspect_combo,
                                       parent=card))

        self._bscan_x_axis_combo = ComboBox(card)
        self._bscan_x_axis_combo.addItem('道号', userData='trace')
        self._bscan_x_axis_combo.addItem('距离 (m)', userData='distance')
        self._bscan_x_axis_combo.setMinimumWidth(180)
        self._bscan_x_axis_combo.currentIndexChanged.connect(
            self._emit_bscan_changed)
        layout.addLayout(make_form_row('横轴单位:', self._bscan_x_axis_combo,
                                       parent=card))

        self._bscan_y_axis_combo = ComboBox(card)
        self._bscan_y_axis_combo.addItem('采样轴（时间/深度）', userData='sample')
        self._bscan_y_axis_combo.addItem('海拔 (m)', userData='elevation')
        self._bscan_y_axis_combo.setMinimumWidth(180)
        self._bscan_y_axis_combo.setToolTip(
            '海拔需要测线带逐道地面高程与介电常数；数据不具备时该视图会自动'
            '回落为采样轴（工具条上的「海拔」钮也处于置灰态）。')
        self._bscan_y_axis_combo.currentIndexChanged.connect(
            self._emit_bscan_changed)
        layout.addLayout(make_form_row('纵轴单位:', self._bscan_y_axis_combo,
                                       parent=card))

        self._bscan_p_low_spin = DoubleSpinBox(card)
        self._bscan_p_high_spin = DoubleSpinBox(card)
        for spin, value in ((self._bscan_p_low_spin, 2.0),
                            (self._bscan_p_high_spin, 98.0)):
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setValue(value)
            spin.setMinimumWidth(110)
            spin.valueChanged.connect(self._emit_bscan_changed)
        layout.addLayout(make_form_row(
            '色阶低/高百分位:', self._bscan_p_low_spin, self._bscan_p_high_spin,
            parent=card, trailing_stretch=False))
        layout.addWidget(make_hint(
            '色阶只影响显示的明暗对比，不改动数据；B-Scan 上右键「色阶设置…」'
            '可只改单个视图。', parent=card))
        return card

    def _emit_bscan_changed(self) -> None:
        """任一 B-Scan 视图设置变化 → 通知主窗口下发到本会话所有视图。

        写盘由主窗口统一做（共享 SettingsManager 唯一写者），页面只发信号。
        """
        if self._loading_settings:
            return
        self.bscan_view_changed.emit()

    def _build_processing_card(self, parent):
        """卡片2"处理设置"：并行工作线程数 SpinBox(1-8, 默认 2)（重启生效提示）。"""
        card, layout = make_card('处理设置')
        self._workers_spin = SpinBox(card)
        self._workers_spin.setRange(1, 8)
        self._workers_spin.setValue(constants.MAX_WORKERS)
        self._workers_spin.setMinimumWidth(120)
        hint = make_hint('（重启后生效）', parent=card)
        layout.addLayout(make_form_row('并行工作线程数:', self._workers_spin,
                                       hint, parent=card))
        return card

    def _build_storage_card(self, parent):
        """卡片3"存储"：默认项目根目录 LineEdit+浏览（默认 ~/Documents/MyGPRProjects）。"""
        card, layout = make_card('存储')
        self._root_edit = LineEdit(card)
        self._root_edit.setText(constants.DEFAULT_PROJECT_ROOT)
        self._root_edit.setMinimumWidth(300)
        browse_btn = PushButton('浏览', card)
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_project_root)
        layout.addLayout(make_form_row('默认项目根目录:', self._root_edit,
                                       browse_btn, parent=card,
                                       trailing_stretch=False))
        return card

    def _build_about_card(self, parent):
        """卡片4"关于"：版本（VERSION 文件，回退 0.9.38）/ 作者 / 版权。"""
        card, layout = make_card('关于')
        for label_text, value_text in (
                ('版本:', _read_version()),
                ('作者:', _AUTHOR),
                ('版权:', _COPYRIGHT)):
            value = BodyLabel(value_text, card)
            value.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addLayout(make_form_row(label_text, value, parent=card,
                                           trailing_stretch=False))
        return card

    # ============================================================ 公共接口
    def load_settings(self, data: dict) -> None:
        """回放设置到控件（blockSignals，不触发 theme_changed / bscan 变更）。"""
        data = dict(data or {})
        widgets = (self._theme_combo, self._dielectric_spin,
                   self._workers_spin, self._root_edit, self._prefetch_check,
                   self._bscan_aspect_combo, self._bscan_x_axis_combo,
                   self._bscan_y_axis_combo, self._bscan_p_low_spin,
                   self._bscan_p_high_spin)
        self._loading_settings = True
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self._theme_combo.setCurrentText(
                str(data.get('theme', constants.THEME_LIGHT)))
            self._dielectric_spin.setValue(float(data.get(
                'default_dielectric', constants.DEFAULT_DIELECTRIC)))
            self._workers_spin.setValue(
                int(data.get('max_workers', constants.MAX_WORKERS)))
            self._root_edit.setText(
                str(data.get('project_root', constants.DEFAULT_PROJECT_ROOT)))
            self._prefetch_check.setChecked(bool(data.get(
                'auto_prefetch_basemap', True)))
            self._select_by_data(self._bscan_aspect_combo,
                                 data.get('bscan_aspect_mode'), 'free')
            self._select_by_data(self._bscan_x_axis_combo,
                                 data.get('bscan_x_axis'), 'trace')
            self._select_by_data(self._bscan_y_axis_combo,
                                 data.get('bscan_y_axis'), 'sample')
            low, high = _levels_or_default(data)
            self._bscan_p_low_spin.setValue(low)
            self._bscan_p_high_spin.setValue(high)
        finally:
            for widget in widgets:
                widget.blockSignals(False)
            self._loading_settings = False

    @staticmethod
    def _select_by_data(combo, value, fallback: str) -> None:
        """按 userData 选中；未知值回落默认项（坏设置不该让下拉框空白）。"""
        wanted = str(value or fallback)
        index = combo.findData(wanted)
        combo.setCurrentIndex(index if index >= 0 else combo.findData(fallback))

    def bscan_view_settings(self) -> dict:
        """B-Scan 视图设置（主窗口下发 + 写盘用；键与 DEFAULT_SETTINGS 对齐）。"""
        return {
            'bscan_aspect_mode': str(self._bscan_aspect_combo.currentData()),
            'bscan_x_axis': str(self._bscan_x_axis_combo.currentData()),
            'bscan_y_axis': str(self._bscan_y_axis_combo.currentData()),
            'bscan_p_low': float(self._bscan_p_low_spin.value()),
            'bscan_p_high': float(self._bscan_p_high_spin.value()),
        }

    def settings(self) -> dict:
        """当前控件值（键与 DEFAULT_SETTINGS 对齐）。"""
        values = {
            'theme': self._theme_combo.currentText(),
            'default_dielectric': float(self._dielectric_spin.value()),
            'max_workers': int(self._workers_spin.value()),
            'project_root': self._root_edit.text().strip()
                            or constants.DEFAULT_PROJECT_ROOT,
            'auto_prefetch_basemap': bool(self._prefetch_check.isChecked()),
        }
        values.update(self.bscan_view_settings())
        return values

    def set_theme_text(self, text: str) -> None:
        """主窗口主题切换后回写主题 ComboBox（blockSignals 防循环）。"""
        self._theme_combo.blockSignals(True)
        try:
            self._theme_combo.setCurrentText(str(text))
        finally:
            self._theme_combo.blockSignals(False)

    # ============================================================ 内部
    def _browse_project_root(self) -> None:
        path = file_dialogs.getExistingDirectory(
            self, '选择默认项目根目录', self._root_edit.text().strip()
            or constants.DEFAULT_PROJECT_ROOT)
        if path:
            self._root_edit.setText(path)


__all__ = ['SettingsPage']
