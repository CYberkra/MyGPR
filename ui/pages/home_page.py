"""HomePage — 主页工作台（SPEC §6.2）。

纯展示 + 发信号：不直接调 controller/backend。
公共接口（供主窗口接线喂数据）：
- set_current_project(summary|None)：刷新"当前项目"卡片
- set_preview_bundle(bundle|None)：刷新"数据预览"卡片（PreviewBundle 鸭子类型）
- mini_jobs() -> MiniJobList：内嵌最近任务列表访问器（JobBridge 信号由主窗口接入）

信号：new_project_requested / open_project_requested /
import_line_requested / goto_page(str)。

布局（v0.9.38 重设计）：
- 左栏固定 ~400px：当前项目（含快速操作按钮组）+ 最近任务
- 右栏 stretch：数据预览（B-Scan 默认近似方形显示，符合雷达剖面习惯）
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel, CaptionLabel, CardWidget, ComboBox, PrimaryPushButton,
    PushButton, ScrollArea, SubtitleLabel, TitleLabel,
)
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.widgets import BScanView, MiniJobList, make_separator


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

        # 主体两栏：左栏（项目+任务）/ 右栏（预览）
        body = QHBoxLayout()
        body.setSpacing(constants.PAGE_SPACING)
        root.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(constants.PAGE_SPACING)
        left.addWidget(self._build_project_card(container))
        left.addWidget(self._build_jobs_card(container), 1)
        left_widget = QWidget(container)
        left_widget.setLayout(left)
        left_widget.setFixedWidth(400)
        left_widget.setStyleSheet('background-color: transparent;')
        body.addWidget(left_widget, 0)

        body.addWidget(self._build_preview_card(container), 1)

        self.setWidget(container)
        self.set_current_project(None)

    # ============================================================ 卡片构建
    def _build_project_card(self, parent) -> CardWidget:
        """"当前项目"卡：项目信息 + 快速操作按钮（合并为一张卡，减少纵向堆叠）。"""
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

        # 快速操作（2×2 网格，窄栏下不拥挤）
        layout.addWidget(make_separator())
        actions_title = CaptionLabel('快速操作', card)
        actions_title.setStyleSheet('font-size: 11px; font-weight: bold;')
        layout.addWidget(actions_title)

        self.new_btn = PrimaryPushButton('新建项目', card, FIF.ADD)
        self.open_btn = PushButton('打开项目', card, FIF.FOLDER)
        self.import_btn = PushButton('导入测线', card, FIF.DOWNLOAD)
        self.processing_btn = PushButton('打开处理台', card, FIF.DEVELOPER_TOOLS)

        self.new_btn.clicked.connect(self.new_project_requested)
        self.open_btn.clicked.connect(self.open_project_requested)
        self.import_btn.clicked.connect(self.import_line_requested)
        self.processing_btn.clicked.connect(
            lambda: self.goto_page.emit('processingInterface'))

        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(constants.CARD_SPACING)
        btn_row1.addWidget(self.new_btn, 1)
        btn_row1.addWidget(self.open_btn, 1)
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(constants.CARD_SPACING)
        btn_row2.addWidget(self.import_btn, 1)
        btn_row2.addWidget(self.processing_btn, 1)
        layout.addLayout(btn_row1)
        layout.addLayout(btn_row2)
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

    def _build_preview_card(self, parent) -> CardWidget:
        """"数据预览"卡：BScanView（默认近似方形）+ 色标 ComboBox。"""
        card, layout = _create_card('数据预览')
        self._bscan = BScanView(card)
        self._bscan.setMinimumHeight(320)
        layout.addWidget(self._bscan, 1)

        row = QHBoxLayout()
        label = CaptionLabel('B-Scan颜色映射:', card)
        label.setMinimumWidth(100)
        self._cmap_combo = ComboBox(card)
        self._cmap_combo.addItems(constants.COLORMAPS)
        self._cmap_combo.setCurrentText(constants.DEFAULT_COLORMAP)
        self._cmap_combo.setMinimumWidth(120)
        self._cmap_combo.currentTextChanged.connect(self._bscan.set_colormap)
        # 反向同步：右键菜单改色标 → ComboBox 跟随
        self._bscan.sig_colormap_changed.connect(self._cmap_combo.setCurrentText)
        row.addWidget(label)
        row.addWidget(self._cmap_combo)
        row.addStretch(1)
        layout.addLayout(row)
        return card

    def _build_jobs_card(self, parent) -> CardWidget:
        """"最近任务"卡：内嵌 MiniJobList。"""
        card, layout = _create_card('最近任务')
        self._mini_jobs = MiniJobList(card)
        self._mini_jobs.setMinimumHeight(120)
        layout.addWidget(self._mini_jobs, 1)
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
