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
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    BodyLabel, CaptionLabel, CardWidget, CheckBox, ComboBox, DoubleSpinBox,
    LineEdit, PushButton, ScrollArea, SpinBox, SubtitleLabel,
)

from ui import constants

_FALLBACK_VERSION = '0.9.38'
_AUTHOR = '邸建豪 袁林 詹萍'
_COPYRIGHT = '© 2025 MyGPR 保留所有权利'


def _read_version() -> str:
    """版本读仓库根 VERSION 文件，缺失/异常回退 0.9.36（SPEC §6.4）。"""
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        with open(os.path.join(root, 'VERSION'), 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return text or _FALLBACK_VERSION
    except OSError:
        return _FALLBACK_VERSION


# ------------------------------------------------------------ 小工厂（卡片范式逐字 SPEC §1）
def _create_card(title: str) -> tuple:
    """卡片范式：CardWidget + QVBoxLayout(spacing=10, margins=15)，首行 SubtitleLabel 10pt Bold。"""
    card = CardWidget()
    layout = QVBoxLayout(card)
    layout.setContentsMargins(*constants.CARD_MARGINS)
    layout.setSpacing(constants.CARD_SPACING)
    header = SubtitleLabel(title, card)
    header.setFont(QFont(constants.FONT_FAMILY, 10, QFont.Weight.Bold))
    layout.addWidget(header)
    return card, layout


class SettingsPage(ScrollArea):
    """系统设置：通用设置 / 处理设置 / 存储 / 关于。"""

    theme_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(
            'QScrollArea { background-color: transparent; border: none; }')

        container = QWidget(self)
        container.setStyleSheet('background-color: transparent;')
        root = QVBoxLayout(container)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)

        # 页面标题：SubtitleLabel 微软雅黑 12pt Bold 居中（SPEC §1）
        title = SubtitleLabel('系统设置', container)
        title.setFont(QFont(constants.FONT_FAMILY, 12, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        root.addWidget(self._build_general_card(container))
        root.addWidget(self._build_processing_card(container))
        root.addWidget(self._build_storage_card(container))
        root.addWidget(self._build_about_card(container))
        root.addStretch(1)

        self.setWidget(container)

    # ============================================================ 卡片构建
    def _build_general_card(self, parent) -> CardWidget:
        """卡片1"通用设置"：主题 / 默认介电常数 / 默认颜色映射 / 预览下采样上限。"""
        card, layout = _create_card('通用设置')

        # 界面主题
        theme_row = QHBoxLayout()
        theme_label = CaptionLabel('界面主题:', card)
        theme_label.setMinimumWidth(100)
        self._theme_combo = ComboBox(card)
        self._theme_combo.addItems([constants.THEME_LIGHT, constants.THEME_DARK])
        self._theme_combo.setCurrentText(constants.THEME_LIGHT)
        self._theme_combo.setMinimumWidth(150)
        self._theme_combo.currentTextChanged.connect(self.theme_changed)
        theme_row.addWidget(theme_label)
        theme_row.addWidget(self._theme_combo)
        theme_row.addStretch(1)
        layout.addLayout(theme_row)

        # 默认介电常数
        diel_row = QHBoxLayout()
        diel_label = CaptionLabel('默认介电常数:', card)
        diel_label.setMinimumWidth(100)
        self._dielectric_spin = DoubleSpinBox(card)
        self._dielectric_spin.setRange(1.0, 81.0)
        self._dielectric_spin.setDecimals(2)
        self._dielectric_spin.setSingleStep(0.5)
        self._dielectric_spin.setValue(constants.DEFAULT_DIELECTRIC)
        self._dielectric_spin.setMinimumWidth(120)
        diel_row.addWidget(diel_label)
        diel_row.addWidget(self._dielectric_spin)
        diel_row.addStretch(1)
        layout.addLayout(diel_row)

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

    def _build_processing_card(self, parent) -> CardWidget:
        """卡片2"处理设置"：并行工作线程数 SpinBox(1-8, 默认 2)（重启生效提示）。"""
        card, layout = _create_card('处理设置')
        row = QHBoxLayout()
        label = CaptionLabel('并行工作线程数:', card)
        label.setMinimumWidth(100)
        self._workers_spin = SpinBox(card)
        self._workers_spin.setRange(1, 8)
        self._workers_spin.setValue(constants.MAX_WORKERS)
        self._workers_spin.setMinimumWidth(120)
        hint = CaptionLabel('（重启后生效）', card)
        hint.setStyleSheet('color: %s; font-size: 11px;' % constants.COLOR_DISABLED)
        row.addWidget(label)
        row.addWidget(self._workers_spin)
        row.addWidget(hint)
        row.addStretch(1)
        layout.addLayout(row)
        return card

    def _build_storage_card(self, parent) -> CardWidget:
        """卡片3"存储"：默认项目根目录 LineEdit+浏览（默认 ~/Documents/MyGPRProjects）。"""
        card, layout = _create_card('存储')
        row = QHBoxLayout()
        label = CaptionLabel('默认项目根目录:', card)
        label.setMinimumWidth(100)
        self._root_edit = LineEdit(card)
        self._root_edit.setText(constants.DEFAULT_PROJECT_ROOT)
        self._root_edit.setMinimumWidth(300)
        browse_btn = PushButton('浏览', card)
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_project_root)
        row.addWidget(label)
        row.addWidget(self._root_edit, 1)
        row.addWidget(browse_btn)
        layout.addLayout(row)
        return card

    def _build_about_card(self, parent) -> CardWidget:
        """卡片4"关于"：版本（VERSION 文件，回退 0.9.36）/ 作者 / 版权。"""
        card, layout = _create_card('关于')
        for label_text, value_text in (
                ('版本:', _read_version()),
                ('作者:', _AUTHOR),
                ('版权:', _COPYRIGHT)):
            row = QHBoxLayout()
            label = CaptionLabel(label_text, card)
            label.setMinimumWidth(100)
            value = BodyLabel(value_text, card)
            value.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            row.addWidget(label)
            row.addWidget(value, 1)
            layout.addLayout(row)
        return card

    # ============================================================ 公共接口
    def load_settings(self, data: dict) -> None:
        """回放设置到控件（blockSignals，不触发 theme_changed）。未知键忽略。"""
        data = dict(data or {})
        widgets = (self._theme_combo, self._dielectric_spin,
                   self._workers_spin, self._root_edit, self._prefetch_check)
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
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def settings(self) -> dict:
        """当前控件值（键与 DEFAULT_SETTINGS 对齐）。"""
        return {
            'theme': self._theme_combo.currentText(),
            'default_dielectric': float(self._dielectric_spin.value()),
            'max_workers': int(self._workers_spin.value()),
            'project_root': self._root_edit.text().strip()
                            or constants.DEFAULT_PROJECT_ROOT,
            'auto_prefetch_basemap': bool(self._prefetch_check.isChecked()),
        }

    def set_theme_text(self, text: str) -> None:
        """主窗口主题切换后回写主题 ComboBox（blockSignals 防循环）。"""
        self._theme_combo.blockSignals(True)
        try:
            self._theme_combo.setCurrentText(str(text))
        finally:
            self._theme_combo.blockSignals(False)

    # ============================================================ 内部
    def _browse_project_root(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, '选择默认项目根目录', self._root_edit.text().strip()
            or constants.DEFAULT_PROJECT_ROOT)
        if path:
            self._root_edit.setText(path)


__all__ = ['SettingsPage']
