# -*- coding: utf-8 -*-
"""左侧常驻测线树面板（DockPanel 子类：项目 → 分组 → 测线）。

职责边界：
- 数据唯一来源仍是 ``ProjectController``（经 ProjectChain 扇出，本面板
  不发起任何后端调用）；
- 叶子点击 → ``line_selected(str)`` → 复用 ``ProjectChain.on_line_selected``
  现有链路（含切线作废成果预览代数），与项目页测线表同语义；
- ``set_current_line`` 是同步入口（``_syncing`` 守卫防回环）；
- **分组行与项目根不可选**（无 ItemIsSelectable）——分组行若能触发预览
  会推进预览代数造成串台，是成果预览代际竞态的同族风险；
- busy 只禁叶子点击，收/展开始终可用（长任务中更该允许让出空间）。

壳（头/细条/动画）全部来自 ``DockPanel`` 基类；本类只保留测线树自己的
三件事：树内容构建、按页展开态记忆（SettingsManager 持久化）、
细条指示文字 = 当前线 ID。
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QDialog, QTreeWidgetItem
from qfluentwidgets import BodyLabel, MessageBox, TreeWidget
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.line_tree import group_lines, group_stats
from ui.theme_helpers import status_color
from ui.widgets.context_menus import add_action, make_menu
from ui.widgets.dock_panel import DockPanel

_ROLE_LINE_ID = Qt.ItemDataRole.UserRole

_HINT_QSS = f'color: #888888; font-size: {constants.FONT_SIZE_BODY}px;'


def _status_key(status: str) -> str:
    """processing_status → status_color 语义键（与 field_project_status
    的"完成"判定同规则）：已完成=success / 已导入=info / 其余=disabled。"""
    if '完成' in status:
        return 'success'
    if '导入' in status:
        return 'info'
    return 'disabled'


def _status_icon(status: str) -> QIcon:
    """12×12 状态色圆点（随主题查表；主题切换经 apply_theme 重建刷新）。"""
    pixmap = QPixmap(12, 12)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(status_color(_status_key(status))))
    painter.drawEllipse(2, 2, 8, 8)
    painter.end()
    return QIcon(pixmap)

# 各页面默认收起态（True=收起细条）；键缺省按收起处理
_DEFAULT_PAGE_COLLAPSED = {
    'projectInterface': False,   # 项目页=资产管理，展开
    'homeInterface': True,
    'settingsInterface': True,
}
_SETTINGS_KEY = 'line_tree_page_states'


class LineTreePanel(DockPanel):
    """项目 → 分组 → 测线 的常驻导航树（可收成细条）。"""

    line_selected = pyqtSignal(str)
    line_process_requested = pyqtSignal(str)
    line_delete_requested = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__('测线树', constants.LINE_TREE_PANEL_WIDTH, parent)
        self._lines: list = []
        self._line_id_by_item: dict = {}
        self._syncing = False
        self._busy = False
        self._current_line_id = ''
        self._current_page = ''
        self._page_states: dict = dict(_DEFAULT_PAGE_COLLAPSED)
        self._settings = None

        # ---------------- 主体：项目名 + 树 + 空态
        self._project_label = BodyLabel('未打开项目')
        self._project_label.setStyleSheet(_HINT_QSS)
        self.body_layout().addWidget(self._project_label)

        self._tree = TreeWidget(self._expanded_view)
        self._tree.setHeaderHidden(True)
        self._tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self.body_layout().addWidget(self._tree, 1)

        self._empty_label = BodyLabel('尚未导入测线')
        self._empty_label.setStyleSheet(_HINT_QSS)
        self.body_layout().addWidget(self._empty_label)
        self._tree.hide()
        self._empty_label.hide()

    # ------------------------------------------------ 页面协议（链路喂数据）
    def set_settings_manager(self, settings) -> None:
        """注入共享 SettingsManager（与页面同一约定：共享实例是唯一写者）。"""
        self._settings = settings
        saved = settings.get(_SETTINGS_KEY) if settings else None
        if isinstance(saved, dict) and saved:
            states = dict(_DEFAULT_PAGE_COLLAPSED)
            states.update({str(k): bool(v) for k, v in saved.items()})
            self._page_states = states

    def set_project_info(self, summary) -> None:
        """项目上下文切换；None = 无项目（整面板显式空态并隐藏）。"""
        name = str(getattr(summary, 'name', '') or '') if summary else ''
        self._project_label.setText(name or '未打开项目')
        self.setVisible(bool(summary))
        if not summary:
            self._lines = []
            self._current_line_id = ''
            self._rebuild()

    def set_lines(self, lines: list) -> None:
        """重建树；保留当前选中（展开态新建组默认全开）。"""
        self._lines = list(lines or [])
        self._rebuild()

    def set_current_line(self, line_id: str) -> None:
        """同步高亮（外部换线 → 树，不回发信号）；细条文字同步更新。"""
        self._current_line_id = str(line_id or '')
        self._syncing = True
        try:
            self._select_leaf(self._current_line_id)
        finally:
            self._syncing = False
        self._update_strip_text()

    def set_busy(self, busy: bool) -> None:
        """项目控制器 busy → 禁叶子点击；收/展开仍可用。"""
        self._busy = bool(busy)
        self._tree.setEnabled(not self._busy)

    # ------------------------------------------------ 页面记忆（主窗口调）
    def apply_page(self, object_name: str) -> None:
        """切页时按该页记忆的展开态切换（瞬时，不做动画）。"""
        page = str(object_name or '')
        if page:
            self._current_page = page
        self.set_collapsed(bool(self._page_states.get(page, True)),
                           animate=False)

    # ------------------------------------------------------------ DockPanel 钩子
    def strip_text(self) -> str:
        return self._current_line_id

    def _on_toggle_clicked(self) -> None:
        # 手动切换：立即按当前页记忆并持久化
        self._remember_current(not self._collapsed)
        self.toggle()

    def _remember_current(self, collapsed: bool) -> None:
        page = self._current_page or 'projectInterface'
        self._page_states[page] = bool(collapsed)
        if self._settings is not None:
            self._settings.set(_SETTINGS_KEY, dict(self._page_states))
            self._settings.save()

    def _on_view_state_changed(self) -> None:
        if self._collapsed:
            return
        if not self._lines:
            self._tree.hide()
            has_project = self._project_label.text() != '未打开项目'
            self._empty_label.setVisible(has_project)
            return
        self._empty_label.hide()
        self._tree.show()

    # ------------------------------------------------------------ 内部
    def _rebuild(self) -> None:
        self._tree.clear()
        self._line_id_by_item.clear()

        if not self._lines:
            self._apply_view_state()
            return

        for key, bucket in group_lines(self._lines):
            if not key:
                # 平铺：测线直接做顶层节点
                for line in bucket:
                    self._tree.addTopLevelItem(self._make_leaf(line))
                continue
            group_item = self._make_group(key, bucket)
            self._tree.addTopLevelItem(group_item)
            for line in bucket:
                group_item.addChild(self._make_leaf(line))

        # 新建组默认展开（一期不保留折叠记忆，避免重建时组全收起）
        for item in self._top_items():
            item.setExpanded(True)
        self._select_leaf(self._current_line_id)
        self._apply_view_state()

    def _top_items(self) -> list:
        return [self._tree.topLevelItem(i)
                for i in range(self._tree.topLevelItemCount())]

    def _make_group(self, key: str, bucket: list) -> QTreeWidgetItem:
        node = QTreeWidgetItem([f'{key}   ({group_stats(bucket)})'])
        node.setFlags(Qt.ItemFlag.ItemIsEnabled)  # 不可选，仅展开
        return node

    def _make_leaf(self, line) -> QTreeWidgetItem:
        line_id = str(getattr(line, 'line_id', '') or '')
        status = str(getattr(line, 'processing_status', '') or '未处理')
        name = str(getattr(line, 'name', '') or '')
        length = float(getattr(line, 'length_m', 0.0) or 0.0)
        updated = str(getattr(line, 'updated_at', '') or '')[:10]
        text = line_id if name in ('', line_id) else f'{line_id} · {name}'
        node = QTreeWidgetItem([text])
        node.setIcon(0, _status_icon(status))
        tip_lines = [f'状态：{status}']
        if length > 0:
            tip_lines.append(f'长度：{length:.1f} m')
        if updated:
            tip_lines.append(f'更新：{updated}')
        node.setToolTip(0, '\n'.join(tip_lines))
        node.setData(0, _ROLE_LINE_ID, line_id)
        self._line_id_by_item[line_id] = node
        return node

    def _select_leaf(self, line_id: str) -> None:
        node = self._line_id_by_item.get(str(line_id or ''))
        if node is not None:
            self._tree.setCurrentItem(node)
        elif self._tree.currentItem() is not None:
            self._tree.setCurrentItem(None)

    def _on_item_clicked(self, item, _column: int) -> None:
        if self._syncing or self._busy:
            return
        line_id = item.data(0, _ROLE_LINE_ID)
        if line_id:
            self._current_line_id = str(line_id)
            self.line_selected.emit(str(line_id))

    # ------------------------------------------------------------ 右键菜单
    def _on_context_menu(self, pos) -> None:
        """叶子右键：先选中该线（与项目页表格右键即选中同语义），再弹菜单。

        分组行 / 空白处 / busy 中不出菜单。
        """
        if self._busy:
            return
        item = self._tree.itemAt(pos)
        if item is None:
            return
        line_id = item.data(0, _ROLE_LINE_ID)
        if not line_id:
            return
        line_id = str(line_id)
        self._tree.setCurrentItem(item)
        if line_id != self._current_line_id:
            self._current_line_id = line_id
            self.line_selected.emit(line_id)
        self._update_strip_text()

        menu = make_menu(parent=self._tree)
        add_action(menu, FIF.DEVELOPER_TOOLS, '处理该测线',
                   lambda: self.line_process_requested.emit(line_id))
        add_action(menu, FIF.DELETE, '删除测线…',
                   lambda: self._confirm_delete(line_id))
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制测线号',
                   lambda: QApplication.clipboard().setText(line_id))
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _confirm_delete(self, line_id: str) -> None:
        box = MessageBox(
            '确认删除测线？',
            f'将删除测线 {line_id}（数据会移入项目 .trash 回收站，可恢复）。',
            self.window() or self,
        )
        box.yesButton.setText('删除')
        box.cancelButton.setText('取消')
        if box.exec() == QDialog.DialogCode.Accepted:
            self.line_delete_requested.emit([line_id])

    # ------------------------------------------------------------ 主题
    def apply_theme(self, _dark: bool) -> None:
        """主题切换 → 重建树刷新状态圆点色（主窗口 findChildren 统一调度）。"""
        self._rebuild()
