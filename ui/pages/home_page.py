# -*- coding: utf-8 -*-
"""HomePage — 主页工作台（SPEC §6.2）。

纯展示 + 发信号：不直接调 controller/backend。
公共接口（供主窗口接线喂数据）：
- set_current_project(summary|None)：刷新"当前项目"卡片
- set_preview_bundle(bundle|None)：刷新"数据预览"卡片（PreviewBundle 鸭子类型）
- mini_jobs() -> MiniJobList：内嵌最近任务列表访问器（JobBridge 信号由主窗口接入）

信号：new_project_requested / open_project_requested /
import_line_requested / goto_page(str)。

布局（v0.9.38 重设计）：
- 左栏固定 SIDE_TOOL_WIDTH（320px，布局统一轮档位）：当前项目（含快速操作按钮组）+ 最近任务
- 右栏 stretch：数据预览（B-Scan 默认近似方形显示，符合雷达剖面习惯）
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel, CaptionLabel, PrimaryPushButton,
    PushButton, ScrollArea,
)
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.page_scaffold import make_card, make_hint, style_transparent_scroll
from ui.theme_helpers import badge_colors, badge_qss_pair, make_badge
from ui.widgets import BScanView, MiniJobList, make_separator


class HomePage(ScrollArea):
    """主页：当前项目 / 快速操作 / 数据预览 / 最近任务。"""

    new_project_requested = pyqtSignal()
    open_project_requested = pyqtSignal()
    import_line_requested = pyqtSignal()
    goto_page = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._summary = None

        style_transparent_scroll(self)

        container = QWidget(self)
        container.setStyleSheet('background-color: transparent;')
        root = QVBoxLayout(container)
        root.setContentsMargins(*constants.PAGE_MARGINS)
        root.setSpacing(constants.PAGE_SPACING)

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
        left_widget.setFixedWidth(constants.SIDE_TOOL_WIDTH)
        left_widget.setStyleSheet('background-color: transparent;')
        body.addWidget(left_widget, 0)

        body.addWidget(self._build_preview_card(container), 1)

        self.setWidget(container)
        self.set_current_project(None)

    # ============================================================ 卡片构建
    def _build_project_card(self, parent):
        """"当前项目"卡：项目信息 + 快速操作按钮（合并为一张卡，减少纵向堆叠）。"""
        card, layout = make_card('当前项目')

        # 空态
        self._empty_widget = QWidget(card)
        empty_layout = QVBoxLayout(self._empty_widget)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.setSpacing(6)
        empty_label = BodyLabel('尚未打开项目', self._empty_widget)
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = make_hint('请通过下方「快速操作」新建或打开项目',
                         parent=self._empty_widget)
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
        self._proj_badge = make_badge('--', 'neutral')
        status_row = QHBoxLayout()
        status_label = CaptionLabel('状态:', self._info_widget)
        status_label.setMinimumWidth(constants.FORM_LABEL_MIN_WIDTH)
        status_row.addWidget(status_label)
        status_row.addWidget(self._proj_badge)
        status_row.addStretch(1)
        info_layout.addLayout(status_row)
        layout.addWidget(self._info_widget)

        # 快速操作（2×2 网格，窄栏下不拥挤）
        layout.addWidget(make_separator())
        actions_title = CaptionLabel('快速操作', card)
        actions_title.setStyleSheet(
            f'font-size: {constants.FONT_SIZE_SECONDARY}pt; '
            'font-weight: bold;')
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
        label.setMinimumWidth(constants.FORM_LABEL_MIN_WIDTH)
        value = BodyLabel('--')
        value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(label)
        row.addWidget(value, 1)
        layout.addLayout(row)
        return value

    def _build_preview_card(self, parent):
        """"数据预览"卡：BScanView（默认近似方形）。

        色标行已退役（2026-09-23 审计）：主页 combo 是游离于持久化体系外
        的第三份状态源（不写盘、启动恢复后显示 stale）。色标切换走 B-Scan
        右键色标子菜单（写盘+同步设置页），全局入口在设置页 B-Scan 卡。
        """
        card, layout = make_card('数据预览')
        self._bscan = BScanView(card)
        self._bscan.setMinimumHeight(constants.PREVIEW_MIN_HEIGHT)
        layout.addWidget(self._bscan, 1)
        return card

    def _build_jobs_card(self, parent):
        """"最近任务"卡：内嵌 MiniJobList。"""
        card, layout = make_card('最近任务')
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
        self._proj_badge.setText(badge_text)
        self._proj_badge.setStyleSheet(
            badge_qss_pair(*badge_colors('warning' if read_only else 'success')))

    def set_preview_bundle(self, bundle) -> None:
        """PreviewBundle（鸭子类型）或 None（清空）。"""
        if bundle is None:
            self._bscan.clear()
        else:
            self._bscan.set_bundle(bundle)

    def mini_jobs(self) -> MiniJobList:
        """内嵌 MiniJobList 访问器（JobBridge 信号由主窗口接入）。"""
        return self._mini_jobs


__all__ = ['HomePage']
