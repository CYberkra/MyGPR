# -*- coding: utf-8 -*-
"""OutputPanel — 底部通栏输出面板（IDE 风格：日志/任务横向页签）。

布局::

    ┌ 页签[日志|任务] │ 工具区(随页签切换) … │ 收/展钮 ┐  ← 头部栏常驻
    │ 内容区 QStackedWidget：日志视图 | MiniJobList      │  ← 可收展
    └──────────────────────────────────────────────────┘

交互（VS Code 面板语义，横向化）：
- 点未激活页签 → 切换并展开内容区；
- 点已激活页签 → 收起（只留头部栏），再点展开；
- 面板高度由主窗口竖向 QSplitter 拖拽分配（收展瞬时切换，与切页
  零动画一致）；
- 状态持久化：``output_panel_active_tab`` / ``output_panel_open``。

页签用 SlimSegment（共享瘦版分段控件，见 segment_tabs.py）；文本 tab
切换保持即时切换，不用 OpacityAniStackedWidget（逐帧重栅格化卡顿、
回切闪白，同旧 LogPanel 结论）。

日志视图：结构化存储（deque(maxlen=5000)，(level, stamp, text)），级别
用关键词规则在写入时解析（与旧 LogPanel 的 _LEVEL_RULES 同规则），文字色
随主题查表；自动滚动在用户上翻离开底部时自动暂停（toggle 钮同步），滚回
底部自动恢复。

任务 tab：MiniJobList（仅活动任务；完整历史见任务中心导航页）。
"""

from collections import deque
from datetime import datetime

from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout, QSizePolicy, QStackedWidget, QTextEdit, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    CaptionLabel, TransparentToggleToolButton, TransparentToolButton,
)
from qfluentwidgets import FluentIcon as FIF

from ui import constants, file_dialogs
from ui.theme_helpers import log_panel_qss, status_color

from .job_widgets import MiniJobList
from .segment_tabs import SlimSegment
from .separators import make_h_separator

# 级别关键词（沿用旧 LogPanel 规则；级别名即 theme_helpers.status_color
# 的语义键——文字色随主题查表，浅底深底都有对比度）
_LEVEL_RULES = (
    (('ERROR', '失败', '错误'), 'error'),
    (('WARNING', '警告'), 'warning'),
    (('SUCCESS', '成功', '完成'), 'success'),
    (('INFO',), 'info'),
)

_SETTINGS_KEY_TAB = 'output_panel_active_tab'
_SETTINGS_KEY_OPEN = 'output_panel_open'

_SCROLL_BOTTOM_SLOP = 4   # 距底部 ≤4px 视为"在底部"（高分屏取整误差）


def _parse_level(msg: str) -> str:
    """按关键词规则解析日志级别；无命中返回 'default'（仅"全部"过滤可见）。"""
    upper = msg.upper()
    for keywords, level in _LEVEL_RULES:
        for kw in keywords:
            if kw in msg or kw in upper:
                return level
    return 'default'


class OutputPanel(QWidget):
    """底部输出面板：常驻头部栏（页签 + 工具区 + 收展钮）+ 可收展内容区。"""

    cancel_job_requested = pyqtSignal(str)
    # 开合态变化（主窗口驱动竖向 QSplitter 调整面板高度）
    sig_open_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = None
        self._current_tab = 'log'
        self._open = True

        self._entries = deque(maxlen=5000)   # (level, stamp, text)
        self._auto_scroll = True
        self._programmatic_scroll = False

        self._build_ui()
        self._activate('log')

    # ============================================================ UI 构建
    def _build_ui(self) -> None:
        # ---------------- 头部栏：页签 + 工具区 + 收展钮
        self._tabs = SlimSegment(self)
        self._tabs.addItem('log', '日志',
                           onClick=lambda: self._on_tab_clicked('log'))
        self._tabs.addItem('jobs', '任务',
                           onClick=lambda: self._on_tab_clicked('jobs'))
        self._tabs.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

        # 日志工具条（左对齐紧凑排布：三个图标钮，余量甩到尾部）
        log_bar = QWidget(self)
        log_layout = QHBoxLayout(log_bar)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(6)
        self._auto_scroll_btn = TransparentToggleToolButton(FIF.DOWN, log_bar)
        self._auto_scroll_btn.setChecked(True)
        self._auto_scroll_btn.setToolTip('自动滚动到底部')
        self._auto_scroll_btn.toggled.connect(self._on_auto_scroll_toggled)
        log_layout.addWidget(self._auto_scroll_btn)

        clear_btn = TransparentToolButton(FIF.DELETE, log_bar)
        clear_btn.setToolTip('清空日志')
        clear_btn.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_btn)
        export_btn = TransparentToolButton(FIF.SAVE, log_bar)
        export_btn.setToolTip('导出日志到文件')
        export_btn.clicked.connect(self._export_log)
        log_layout.addWidget(export_btn)
        log_layout.addStretch(1)

        # 任务工具条（说明性占位：MiniJobList 只显示活动任务）
        jobs_bar = QWidget(self)
        jobs_layout = QHBoxLayout(jobs_bar)
        jobs_layout.setContentsMargins(0, 0, 0, 0)
        hint = CaptionLabel('仅显示进行中的任务，完整历史见「任务中心」页', jobs_bar)
        jobs_layout.addWidget(hint)
        jobs_layout.addStretch(1)

        self._tool_stacked = QStackedWidget(self)
        self._tool_stacked.addWidget(log_bar)
        self._tool_stacked.addWidget(jobs_bar)

        self._fold_btn = TransparentToolButton(FIF.DOWN, self)
        self._fold_btn.setIconSize(QSize(14, 14))
        self._fold_btn.setToolTip('收起面板')
        self._fold_btn.clicked.connect(lambda: self._set_open(not self._open))

        header = QWidget(self)
        header.setFixedHeight(constants.OUTPUT_PANEL_HEADER_HEIGHT)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 6, 0)
        header_layout.setSpacing(10)
        header_layout.addWidget(self._tabs)
        header_layout.addSpacing(6)
        header_layout.addWidget(self._tool_stacked, 1)
        header_layout.addWidget(self._fold_btn)

        # ---------------- 内容区：日志 / 任务
        self._log_edit = QTextEdit(self)
        self._log_edit.setReadOnly(True)
        self._log_edit.setStyleSheet(log_panel_qss('light'))  # 主题在启动时立即覆盖
        self._log_edit.document().setMaximumBlockCount(5000)
        self._log_edit.verticalScrollBar().valueChanged.connect(self._on_scroll)

        self._mini_jobs = MiniJobList(self)
        self._mini_jobs.cancel_requested.connect(self.cancel_job_requested)

        self._content = QStackedWidget(self)
        self._content.addWidget(self._log_edit)
        self._content.addWidget(self._mini_jobs)
        # 高度交给主窗口竖向 QSplitter 拖拽分配。min 必须为 0：本面板的
        # minimumSizeHint 随收/展显隐变化（37 ↔ 97+）会让 QSplitter 在
        # invalidate 重算时按 stretch 把显式分配洗掉（下格 stretch=0 被
        # 压回 min，表现为展开后只剩一条、拖出的高度丢失）；min 恒定后
        # 重算的 clamp 无操作，分配得以保持。
        self._content.setMinimumHeight(0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(make_h_separator())
        layout.addWidget(header)
        layout.addWidget(self._content)

    # ============================================================ 对外接口（与旧 LogPanel 对齐）
    def set_settings_manager(self, settings) -> None:
        """注入共享 SettingsManager 并恢复上次页签与开合态（无动画）。"""
        self._settings = settings
        if settings is None:
            return
        tab = settings.get(_SETTINGS_KEY_TAB)
        if tab in ('log', 'jobs'):
            self._activate(tab)
        open_ = settings.get(_SETTINGS_KEY_OPEN)
        if open_ is not None:
            self._set_open(bool(open_), animate=False)

    def append_log(self, msg: str) -> None:
        """解析级别 → 存结构化条目 → 增量渲染 + 自动滚到底。

        自动滚动暂停时保持用户阅读位置：QTextEdit.append 自身会带动视口
        滚动，整个「追加 + 复位/滚底」都包在程序化滚动护栏里——否则 append
        引发的 valueChanged 会被 _on_scroll 误判为用户滚回底部而恢复自动滚动。
        """
        stamp = datetime.now().strftime('[%H:%M:%S]')
        entry = (_parse_level(msg), stamp, str(msg))
        self._entries.append(entry)
        scrollbar = self._log_edit.verticalScrollBar()
        old_value = scrollbar.value()
        self._programmatic_scroll = True
        try:
            self._log_edit.append(self._render_entry(entry))
            scrollbar.setValue(scrollbar.maximum() if self._auto_scroll
                               else old_value)
        finally:
            self._programmatic_scroll = False

    def clear_log(self) -> None:
        """清空存储与视图（"清空"按钮）。"""
        self._entries.clear()
        self._log_edit.clear()

    def mini_jobs(self) -> MiniJobList:
        return self._mini_jobs

    def toggle_panel(self) -> None:
        """Ctrl+J：收/展切换。"""
        self._set_open(not self._open)

    def set_open(self, open_: bool) -> None:
        """按目标态开合（主窗口收起态拖拽联动用，不动高度分配）。"""
        self._set_open(bool(open_))

    def minimumSizeHint(self):   # noqa: N802（Qt 虚函数命名）
        """min hint 恒定 = 头部栏（收/展不变化）。

        QStackedWidget 的 min hint 会透传当前页（QTextEdit ~80px），随
        显隐变化让 QSplitter 在 invalidate 重算时按 stretch 洗掉用户
        拖出的高度（下格 stretch=0 被压回 min）。恒定后重算 clamp 无
        操作；显式 setMinimumHeight 压不住 minimumSizeHint() 虚函数链，
        必须覆写本方法。
        """
        return QSize(0, constants.OUTPUT_PANEL_HEADER_HEIGHT + 2)

    def apply_theme(self, dark: bool) -> None:
        """主题换肤：日志底/字色与级别着色一起跟随（全宽面板下深底大色块过重）。"""
        self._log_edit.setStyleSheet(log_panel_qss('dark' if dark else 'light'))
        self._refilter()  # 级别文字色随主题重染

    # ============================================================ 日志渲染
    def _render_entry(self, entry) -> str:
        level, stamp, text = entry
        line = '%s %s' % (stamp, self._escape(text))
        if level in ('error', 'warning', 'success', 'info'):
            return '<span style="color:%s;">%s</span>' % (
                status_color(level), line)
        return line

    def _refilter(self) -> None:
        """主题切换后从 deque 全量重渲染（级别文字色随主题重染）。"""
        self._log_edit.setHtml('<br>'.join(
            self._render_entry(e) for e in self._entries))
        self._scroll_to_bottom(force=True)

    # ============================================================ 自动滚动
    def _scroll_to_bottom(self, force: bool = False) -> None:
        if not (self._auto_scroll or force):
            return
        scrollbar = self._log_edit.verticalScrollBar()
        self._programmatic_scroll = True
        try:
            scrollbar.setValue(scrollbar.maximum())
        finally:
            self._programmatic_scroll = False

    def _on_scroll(self, value: int) -> None:
        """用户上翻离开底部 → 暂停自动滚动；滚回底部 → 恢复。"""
        if self._programmatic_scroll:
            return
        scrollbar = self._log_edit.verticalScrollBar()
        at_bottom = value >= scrollbar.maximum() - _SCROLL_BOTTOM_SLOP
        if at_bottom != self._auto_scroll:
            self._auto_scroll = at_bottom
            self._auto_scroll_btn.blockSignals(True)
            self._auto_scroll_btn.setChecked(at_bottom)
            self._auto_scroll_btn.blockSignals(False)

    def _on_auto_scroll_toggled(self, checked: bool) -> None:
        self._auto_scroll = bool(checked)
        if checked:
            self._scroll_to_bottom(force=True)

    # ============================================================ 页签交互
    def _on_tab_clicked(self, route_key: str) -> None:
        """VS Code 行为：点未激活页签切换并展开；点已激活页签收/展。"""
        if route_key == self._current_tab:
            self._set_open(not self._open)
            return
        self._activate(route_key)
        if not self._open:
            self._set_open(True)

    def _activate(self, route_key: str) -> None:
        self._current_tab = route_key
        index = 0 if route_key == 'log' else 1
        self._tabs.setCurrentItem(route_key)
        self._content.setCurrentIndex(index)
        self._tool_stacked.setCurrentIndex(index)
        if self._settings is not None:
            self._settings.set(_SETTINGS_KEY_TAB, route_key)
            self._settings.save()

    def is_open(self) -> bool:
        return self._open

    def _set_open(self, open_: bool, animate: bool = True) -> None:
        """展开/收起内容区；头部栏常驻。

        面板高度由主窗口竖向 QSplitter 分配（用户可拖拽），收展本身瞬时
        切换（与切页零动画一致）；开合态经 ``sig_open_toggled`` 通知主窗口
        调整 splitter 尺寸。``animate`` 仅保留签名兼容（旧动画已退役）。
        """
        open_ = bool(open_)
        if self._open == open_:
            return
        self._open = open_
        if self._settings is not None:
            self._settings.set(_SETTINGS_KEY_OPEN, self._open)
            self._settings.save()
        self._fold_btn.setIcon(FIF.DOWN.icon() if open_ else FIF.UP.icon())
        self._fold_btn.setToolTip('收起面板' if open_ else '展开面板')
        self._content.setVisible(open_)
        self.sig_open_toggled.emit(open_)

    # ============================================================ 其他
    def _export_log(self) -> None:
        """全部日志（不过滤）保存为 .txt 纯文本。"""
        path, _selected = file_dialogs.getSaveFileName(
            self, '导出日志',
            'mygpr_log_%s.txt' % datetime.now().strftime('%Y%m%d_%H%M%S'),
            '文本文件 (*.txt)')
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                for _level, stamp, text in self._entries:
                    fh.write('%s %s\n' % (stamp, text))
        except OSError as exc:
            self.append_log('WARNING 日志导出失败: %s' % exc)

    @staticmethod
    def _escape(text: str) -> str:
        return (text.replace('&', '&amp;').replace('<', '&lt;')
                .replace('>', '&gt;'))
