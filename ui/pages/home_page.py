"""HomePage — 主页工作台（SPEC §6.2）。

纯展示 + 发信号：不直接调 controller/backend。
公共接口（供主窗口接线喂数据）：
- set_current_project(summary|None)：刷新"当前项目"卡片
- set_preview_bundle(bundle|None)：刷新"数据预览"卡片（PreviewBundle 鸭子类型）
- mini_jobs() -> MiniJobList：内嵌最近任务列表访问器（JobBridge 信号由主窗口接入）

信号：new_project_requested / open_project_requested /
import_line_requested / goto_page(str)。
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel, CaptionLabel, CardWidget, ComboBox, PrimaryPushButton,
    PushButton, ScrollArea, SubtitleLabel, TitleLabel,
)

from ui import constants
from ui.widgets import BScanView, MiniJobList


# ------------------------------------------------------------ 小工厂（卡片范式逐字 SPEC §1）
def _create_separator() -> QFrame:
    """分隔线工厂：QFrame.HLine + Sunken + 'color: #e0e0e0;'。"""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet('color: rgba(128, 128, 128, 90);')
    return line


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


def _make_badge(text: str, fg: str, bg: str) -> QLabel:
    """徽章 QSS 逐字 SPEC §1。"""
    badge = QLabel(text)
    badge.setStyleSheet(
        'QLabel { padding: 2px 10px; border-radius: 10px; font-size: 12px; '
        'font-weight: bold; color: %s; background-color: %s; }' % (fg, bg))
    return badge


class HomePage(ScrollArea):
    """主页：当前项目 / 快速操作 / 数据预览 / 最近任务。"""

    new_project_requested = pyqtSignal()
    open_project_requested = pyqtSignal()
    import_line_requested = pyqtSignal()
    goto_page = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._summary = None

        self.setWidgetResizable(True)
        self.setStyleSheet(
            'QScrollArea { background-color: transparent; border: none; }')

        container = QWidget(self)
        container.setStyleSheet('background-color: transparent;')
        root = QVBoxLayout(container)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)

        # 页面大标题：TitleLabel 微软雅黑 11pt Bold 居中（SPEC §1）
        title = TitleLabel('MyGPR 探地雷达数据处理工作台', container)
        title.setFont(QFont(constants.FONT_FAMILY, 11, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        root.addWidget(self._build_project_card(container))
        root.addWidget(self._build_actions_card(container))
        root.addWidget(self._build_preview_card(container), 1)
        root.addWidget(self._build_jobs_card(container))
        root.addStretch(1)

        self.setWidget(container)
        self.set_current_project(None)

    # ============================================================ 卡片构建
    def _build_project_card(self, parent) -> CardWidget:
        """卡片1"当前项目"：未打开显示提示；打开后显示 名称/路径/测线数/存储后端/状态徽章。"""
        card, layout = _create_card('当前项目')

        # 空态
        self._empty_widget = QWidget(card)
        empty_layout = QVBoxLayout(self._empty_widget)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.setSpacing(6)
        empty_label = BodyLabel('尚未打开项目', self._empty_widget)
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = CaptionLabel('请通过下方「快速操作」新建或打开项目', self._empty_widget)
        hint.setStyleSheet('color: %s; font-size: 11px;' % constants.COLOR_DISABLED)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_label)
        empty_layout.addWidget(hint)
        layout.addWidget(self._empty_widget)

        # 有项目态
        self._info_widget = QWidget(card)
        info_layout = QVBoxLayout(self._info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(constants.CARD_SPACING)
        self._proj_name = self._add_info_row(info_layout, '名称:')
        self._proj_path = self._add_info_row(info_layout, '路径:')
        self._proj_lines = self._add_info_row(info_layout, '测线数:')
        self._proj_backend = self._add_info_row(info_layout, '存储后端:')
        status_row = QHBoxLayout()
        status_label = CaptionLabel('状态:', self._info_widget)
        status_label.setMinimumWidth(100)
        self._proj_badge = _make_badge('--', '#9ca3af', '#f3f4f6')
        status_row.addWidget(status_label)
        status_row.addWidget(self._proj_badge)
        status_row.addStretch(1)
        info_layout.addLayout(status_row)
        layout.addWidget(self._info_widget)
        return card

    def _add_info_row(self, layout: QVBoxLayout, label_text: str) -> BodyLabel:
        row = QHBoxLayout()
        label = CaptionLabel(label_text)
        label.setMinimumWidth(100)
        value = BodyLabel('--')
        value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(label)
        row.addWidget(value, 1)
        layout.addLayout(row)
        return value

    def _build_actions_card(self, parent) -> CardWidget:
        """卡片2"快速操作"：按钮行，发信号给主窗口跳导航。"""
        card, layout = _create_card('快速操作')
        row = QHBoxLayout()
        row.setSpacing(constants.CARD_SPACING)

        self.new_btn = PrimaryPushButton('新建项目', card)
        self.open_btn = PushButton('打开项目', card)
        self.import_btn = PushButton('导入测线', card)
        self.processing_btn = PushButton('打开处理台', card)

        self.new_btn.clicked.connect(self.new_project_requested)
        self.open_btn.clicked.connect(self.open_project_requested)
        self.import_btn.clicked.connect(self.import_line_requested)
        self.processing_btn.clicked.connect(
            lambda: self.goto_page.emit('processingInterface'))

        for btn in (self.new_btn, self.open_btn, self.import_btn,
                    self.processing_btn):
            row.addWidget(btn)
        row.addStretch(1)
        layout.addLayout(row)
        return card

    def _build_preview_card(self, parent) -> CardWidget:
        """卡片3"数据预览"：BScanView(min 300px) + colormap ComboBox（九项，默认 seismic）。"""
        card, layout = _create_card('数据预览')
        self._bscan = BScanView(card)
        self._bscan.setMinimumHeight(300)
        layout.addWidget(self._bscan, 1)

        row = QHBoxLayout()
        label = CaptionLabel('B-Scan颜色映射:', card)
        label.setMinimumWidth(100)
        self._cmap_combo = ComboBox(card)
        self._cmap_combo.addItems(constants.COLORMAPS)
        self._cmap_combo.setCurrentText(constants.DEFAULT_COLORMAP)
        self._cmap_combo.setMinimumWidth(120)
        self._cmap_combo.currentTextChanged.connect(self._bscan.set_colormap)
        row.addWidget(label)
        row.addWidget(self._cmap_combo)
        row.addStretch(1)
        layout.addLayout(row)
        return card

    def _build_jobs_card(self, parent) -> CardWidget:
        """卡片4"最近任务"：内嵌 MiniJobList。"""
        card, layout = _create_card('最近任务')
        self._mini_jobs = MiniJobList(card)
        self._mini_jobs.setMinimumHeight(150)
        layout.addWidget(self._mini_jobs)
        return card

    # ============================================================ 公共接口（主窗口喂数据）
    def set_current_project(self, summary) -> None:
        """ProjectSummary（鸭子类型）或 None。"""
        self._summary = summary
        has_project = summary is not None
        self._empty_widget.setVisible(not has_project)
        self._info_widget.setVisible(has_project)
        if not has_project:
            return
        self._proj_name.setText(str(getattr(summary, 'name', '') or '--'))
        self._proj_path.setText(str(getattr(summary, 'root_path', '') or '--'))
        self._proj_lines.setText(str(getattr(summary, 'line_count', 0)))
        self._proj_backend.setText(
            str(getattr(summary, 'storage_backend', '') or '--'))
        read_only = bool(getattr(summary, 'read_only', False))
        status = str(getattr(summary, 'status', '') or '').strip()
        badge_text = status or ('只读' if read_only else '已打开')
        if read_only:
            fg, bg = constants.COLOR_WARNING, '#fffbeb'
        else:
            fg, bg = constants.COLOR_SUCCESS, '#f0fdf4'
        self._proj_badge.setText(badge_text)
        self._proj_badge.setStyleSheet(
            'QLabel { padding: 2px 10px; border-radius: 10px; font-size: 12px; '
            'font-weight: bold; color: %s; background-color: %s; }' % (fg, bg))

    def set_preview_bundle(self, bundle) -> None:
        """PreviewBundle（鸭子类型）或 None（清空）。"""
        if bundle is None:
            self._bscan.clear()
        else:
            self._bscan.set_bundle(bundle)

    def mini_jobs(self) -> MiniJobList:
        """内嵌 MiniJobList 访问器（JobBridge 信号由主窗口接入）。"""
        return self._mini_jobs

    def colormap(self) -> str:
        """当前颜色映射名（供主窗口持久化）。"""
        return self._cmap_combo.currentText()

    def set_colormap(self, name: str) -> None:
        """设置颜色映射（ComboBox 与 BScanView 同步）。"""
        self._cmap_combo.setCurrentText(name)
        self._bscan.set_colormap(name)


__all__ = ['HomePage']
