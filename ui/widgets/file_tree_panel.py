# -*- coding: utf-8 -*-
"""左侧常驻文件树面板（DockPanel 子类，顶部分段：测线｜成果｜文件）。

职责边界：
- 数据唯一来源仍是各 controller（经 ProjectChain 扇出，本面板不发起任何
  后端调用）；
- 测线/成果视图的节点模型由 ``ui.file_tree`` 纯函数装配（Provider 层），
  本面板只渲染 TreeNode、发信号；文件视图由 ``ProjectFilesView``
  （QFileSystemModel）自管，随项目根切换；
- 测线叶子点击 → ``line_selected(str)`` → 复用 ``ProjectChain.on_line_selected``
  现有链路（含切线作废成果预览代数），与项目页测线表同语义；
  双击 → ``line_process_requested``（选中并跳处理页，与项目页双击同语义）；
- 成果叶子点击 → ``artifact_focus_requested(line_id, artifact_id)`` →
  换线（如需）+ 跳处理页选中预览；
- 空间成果/项目报告叶子点击 → ``delivery_focus_requested(kind)`` → 跳成果页；
- ``set_current_line`` 是同步入口（``_syncing`` 守卫防回环）；
- **分组行不可选**（无 ItemIsSelectable）——分组行若能触发预览
  会推进预览代数造成串台，是成果预览代际竞态的同族风险；
- busy 只禁叶子点击，收/展开始终可用（长任务中更该允许让出空间）。

壳（头/细条/动画）全部来自 ``DockPanel`` 基类；本类只保留文件树自己的
三件事：视图切换与内容构建、按页×按视图展开态记忆（SettingsManager
持久化）、细条指示文字 = 当前线 ID。
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QDesktopServices, QIcon, QKeySequence, QPainter, QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication, QDialog, QHeaderView, QTreeWidgetItem,
)
from qfluentwidgets import BodyLabel, MessageBox, TreeWidget
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.file_tree import (
    TreeNode, build_artifacts_model, build_files_model, build_tree_model,
)
from ui.theme_helpers import status_color
from ui.widgets.context_menus import add_action, make_menu
from ui.widgets.dock_panel import DockPanel
from ui.widgets.segment_tabs import SlimSegment

_ROLE_PAYLOAD = Qt.ItemDataRole.UserRole        # line → line_id；artifact → artifact_id；spatial → result_id；report → package_dir
_ROLE_KIND = Qt.ItemDataRole.UserRole + 1       # TreeNode.kind
_ROLE_AUX = Qt.ItemDataRole.UserRole + 2        # artifact → 所属 line_id

_SUFFIX_BRUSH = QBrush(QColor('#8a8a8a'))       # 行尾角标灰

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

# 分段视图 routeKey（树顶 SlimSegment：测线｜成果｜文件）
_VIEWS = ('lines', 'artifacts', 'files')
_DEFAULT_VIEW = 'lines'
# 各页面默认收起态（True=收起细条）；键缺省按收起处理
_DEFAULT_PAGE_COLLAPSED = {
    'projectInterface': False,   # 项目页=资产管理，展开
    'homeInterface': True,
    'settingsInterface': True,
}
_SETTINGS_KEY = 'file_tree_page_states'
# 旧版设置键：仅作读取回退（老用户的按页记忆不丢），写入只写新键
_LEGACY_SETTINGS_KEY = 'line_tree_page_states'
_VIEW_SETTINGS_KEY = 'file_tree_current_view'
# 各视图空态文案
_EMPTY_TEXT = {
    'lines': '尚未导入测线',
    'artifacts': '尚无成果',
    'files': '打开项目后在此浏览项目文件',
}


class FileTreePanel(DockPanel):
    """测线 + 空间成果 + 项目报告 的常驻导航树（可收成细条）。"""

    line_selected = pyqtSignal(str)
    line_process_requested = pyqtSignal(str)
    line_delete_requested = pyqtSignal(list)
    # 成果叶子点击 → 换线（如需）+ 跳处理页选中预览（line_id, artifact_id）
    artifact_focus_requested = pyqtSignal(str, str)
    # 空间成果/项目报告叶子点击 → 跳成果页（参数为 kind：'spatial'/'report'）
    delivery_focus_requested = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__('文件树', constants.FILE_TREE_PANEL_WIDTH, parent)
        self._lines: list = []
        self._artifacts: list = []
        self._artifacts_sig: list = []   # set_artifacts 去重签名
        self._spatial_results: list = []
        self._reports: list = []
        self._line_id_by_item: dict = {}
        self._syncing = False
        self._busy = False
        self._current_line_id = ''
        self._current_page = ''
        self._current_view = _DEFAULT_VIEW
        self._page_states: dict = {}   # {page: {view: collapsed}}
        self._settings = None

        # ---------------- 主体：项目名 + 分段 + 树 + 空态
        self._project_label = BodyLabel('未打开项目')
        self._project_label.setStyleSheet(_HINT_QSS)
        self.body_layout().addWidget(self._project_label)

        # 视图分段（与顶部页签/输出面板同款 SlimSegment 药丸）
        self._view_segment = SlimSegment(self._expanded_view)
        self._view_segment.addItem('lines', '测线')
        self._view_segment.addItem('artifacts', '成果')
        self._view_segment.addItem('files', '文件')
        self._view_segment.setCurrentItem(_DEFAULT_VIEW)
        self._view_segment.currentItemChanged.connect(
            self._on_view_segment_changed)
        self.body_layout().addWidget(self._view_segment)

        self._tree = TreeWidget(self._expanded_view)
        self._tree.setHeaderHidden(True)
        self._tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        # 第二列：行尾角标（灰字右对齐，ResizeToContents 紧贴右缘）
        self._tree.setColumnCount(2)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.itemExpanded.connect(self._on_item_expanded)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        # F5 刷新文件视图（无 FS 实时监视，手动刷新是补偿路径）
        self._refresh_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Refresh), self._tree,
            context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._refresh_shortcut.activated.connect(self._refresh_current_view)
        self.body_layout().addWidget(self._tree, 1)
        self._project_root = ''

        self._empty_label = BodyLabel(_EMPTY_TEXT[_DEFAULT_VIEW])
        self._empty_label.setStyleSheet(_HINT_QSS)
        self.body_layout().addWidget(self._empty_label)
        self._tree.hide()
        self._empty_label.hide()

    # ------------------------------------------------ 页面协议（链路喂数据）
    def set_settings_manager(self, settings) -> None:
        """注入共享 SettingsManager（与页面同一约定：共享实例是唯一写者）。

        二维记忆格式：``{page: {view: collapsed}}``；读取兼容一维旧格式
        ``{page: collapsed}``（广播到三个视图）与旧版设置键。
        """
        self._settings = settings
        saved = settings.get(_SETTINGS_KEY) if settings else None
        if not isinstance(saved, dict) or not saved:
            # 向后兼容：新键无值时回读旧版 'line_tree_page_states'
            saved = settings.get(_LEGACY_SETTINGS_KEY) if settings else None
        if isinstance(saved, dict) and saved:
            states: dict = {}
            for page, val in saved.items():
                if isinstance(val, dict):
                    states[str(page)] = {str(k): bool(v)
                                         for k, v in val.items()}
                else:  # 一维旧格式：广播到全部视图
                    states[str(page)] = {v: bool(val) for v in _VIEWS}
            self._page_states = states
        view = settings.get(_VIEW_SETTINGS_KEY) if settings else None
        if view in _VIEWS and view != self._current_view:
            self._set_view(str(view), remember=False)

    def set_project_info(self, summary) -> None:
        """项目上下文切换；None = 无项目（面板常驻，显示"未打开项目"空态）。"""
        name = str(getattr(summary, 'name', '') or '') if summary else ''
        self._project_label.setText(name or '未打开项目')
        self._project_root = str(getattr(summary, 'root_path', '') or '') \
            if summary else ''
        if not summary:
            self._lines = []
            self._artifacts = []
            self._artifacts_sig = []   # 重置去重签名：重开同项目也要重建
            self._spatial_results = []
            self._reports = []
            self._current_line_id = ''
        self._rebuild()

    def set_lines(self, lines: list) -> None:
        """重建树；保留当前选中（展开态新建组默认全开）。"""
        self._lines = list(lines or [])
        self._rebuild()

    def set_artifacts(self, artifacts: list) -> None:
        """全项目处理成果列表（ProjectController.all_artifacts_updated 扇出）。

        按 artifact_id 签名去重：换线触发的全量刷新内容不变时不重建树。
        """
        artifacts = list(artifacts or [])
        sig = [str(getattr(a, 'artifact_id', '') or '') for a in artifacts]
        if sig == getattr(self, '_artifacts_sig', None):
            return
        self._artifacts_sig = sig
        self._artifacts = artifacts
        self._rebuild()

    def set_spatial_results(self, results: list) -> None:
        """空间成果列表（DeliveryController.spatial_results_updated 扇出）。"""
        self._spatial_results = list(results or [])
        self._rebuild()

    def set_reports(self, packages: list) -> None:
        """项目报告列表（DeliveryController.report_list_updated 扇出）。"""
        self._reports = list(packages or [])
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
        """切页时按该页×当前视图记忆的展开态切换（瞬时，不做动画）。"""
        page = str(object_name or '')
        if page:
            self._current_page = page
        self.set_collapsed(self._collapsed_for(page, self._current_view),
                           animate=False)

    def toggle_panel(self) -> None:
        """窗口级入口（页签条右端按钮 / Ctrl+B）：与头部收起钮同一路径，
        按页×视图记忆并持久化。"""
        self._on_toggle_clicked()

    # ------------------------------------------------------------ DockPanel 钩子
    def strip_text(self) -> str:
        return self._current_line_id

    def _on_toggle_clicked(self) -> None:
        # 手动切换：立即按当前页×当前视图记忆并持久化
        self._remember_current(not self._collapsed)
        self.toggle()

    def _collapsed_for(self, page: str, view: str) -> bool:
        """（页, 视图）→ 记忆的收起态；兼容一维旧值与缺省页。"""
        default = _DEFAULT_PAGE_COLLAPSED.get(page, True)
        views = self._page_states.get(page)
        if isinstance(views, dict):
            return bool(views.get(view, default))
        if isinstance(views, bool):  # 旧格式残值
            return views
        return bool(default)

    def _remember_current(self, collapsed: bool) -> None:
        page = self._current_page or 'projectInterface'
        views = self._page_states.setdefault(page, {})
        if not isinstance(views, dict):
            views = {}
            self._page_states[page] = views
        views[self._current_view] = bool(collapsed)
        if self._settings is not None:
            self._settings.set(_SETTINGS_KEY,
                               {p: dict(v) for p, v in self._page_states.items()
                                if isinstance(v, dict)})
            self._settings.save()

    # ------------------------------------------------------------ 视图分段
    def _on_view_segment_changed(self, route_key: str) -> None:
        self._set_view(str(route_key), remember=True)

    def _set_view(self, view: str, remember: bool) -> None:
        """切换 测线/成果/文件 视图：重建内容 + 恢复该（页, 视图）的收起态。"""
        if view not in _VIEWS or view == self._current_view:
            return
        self._current_view = view
        if self._view_segment.currentItem() is not None and \
                self._view_segment.currentRouteKey() != view:
            self._view_segment.setCurrentItem(view)
        if remember and self._settings is not None:
            self._settings.set(_VIEW_SETTINGS_KEY, view)
            self._settings.save()
        self._empty_label.setText(_EMPTY_TEXT[view])
        self._rebuild()
        # 视图切换瞬时恢复该视图的收起态（与切页同语义）
        if self._current_page:
            self.set_collapsed(
                self._collapsed_for(self._current_page, view), animate=False)

    def _on_view_state_changed(self) -> None:
        if self._collapsed:
            return
        if self._current_view == 'lines':
            has_content = bool(self._lines)
        elif self._current_view == 'artifacts':
            has_content = bool(
                self._artifacts or self._spatial_results or self._reports)
        else:  # files：有项目根即显示浏览树
            has_content = bool(self._project_root)
        if not has_content:
            self._tree.hide()
            has_project = bool(self._project_root)
            self._empty_label.setVisible(has_project)
            return
        self._empty_label.hide()
        self._tree.show()

    # ------------------------------------------------------------ 内部
    def _rebuild(self) -> None:
        self._tree.clear()
        self._line_id_by_item.clear()

        if self._current_view == 'lines':
            model = build_tree_model(self._lines)
        elif self._current_view == 'artifacts':
            model = build_artifacts_model(
                self._artifacts, self._spatial_results, self._reports)
        else:  # files：项目根单层扫描，目录展开时懒加载子层
            model = build_files_model(self._project_root)
        if not model:
            self._apply_view_state()
            return

        for node in model:
            self._add_node(None, node)

        # 分组行默认展开（文件视图的目录不自动展开——子层走懒加载）
        for item in self._top_items():
            if item.data(0, _ROLE_KIND) == 'group':
                item.setExpanded(True)
        self._select_leaf(self._current_line_id)
        self._apply_view_state()

    def _top_items(self) -> list:
        return [self._tree.topLevelItem(i)
                for i in range(self._tree.topLevelItemCount())]

    def _add_node(self, parent_item, node: TreeNode) -> None:
        item = QTreeWidgetItem()
        item.setText(0, node.text)
        if node.suffix:
            item.setText(1, node.suffix)
            item.setForeground(1, _SUFFIX_BRUSH)
            item.setTextAlignment(
                1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if node.tooltip:
            item.setToolTip(0, node.tooltip)
        item.setData(0, _ROLE_KIND, node.kind)
        if node.kind == 'group':
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # 不可选，仅展开
        else:
            item.setData(0, _ROLE_PAYLOAD, node.payload)
            if node.aux:
                item.setData(0, _ROLE_AUX, node.aux)
            if node.kind == 'line':
                item.setIcon(0, _status_icon(node.icon))
                self._line_id_by_item[node.payload] = item
            elif node.kind == 'spatial':
                item.setIcon(0, FIF.GLOBE.icon())
            elif node.kind == 'report':
                item.setIcon(0, FIF.DOCUMENT.icon())
            elif node.kind == 'dir':
                item.setIcon(0, FIF.FOLDER.icon())
            elif node.kind == 'file':
                item.setIcon(0, FIF.DOCUMENT.icon())
        if parent_item is None:
            self._tree.addTopLevelItem(item)
        else:
            parent_item.addChild(item)
        for child in node.children:
            self._add_node(item, child)
        # 目录无 children = 子层未加载：放占位行让展开箭头出现
        if node.kind == 'dir' and not node.children:
            placeholder = QTreeWidgetItem()
            placeholder.setData(0, _ROLE_KIND, 'placeholder')
            placeholder.setText(0, '…')
            item.addChild(placeholder)

    def _on_item_expanded(self, item) -> None:
        """目录首次展开 → 就地扫描子层并替换占位行（懒加载）。"""
        if item.data(0, _ROLE_KIND) != 'dir':
            return
        if item.childCount() != 1 or \
                item.child(0).data(0, _ROLE_KIND) != 'placeholder':
            return
        item.removeChild(item.child(0))
        for node in build_files_model(item.data(0, _ROLE_PAYLOAD)):
            self._add_node(item, node)

    def _select_leaf(self, line_id: str) -> None:
        node = self._line_id_by_item.get(str(line_id or ''))
        if node is not None:
            self._tree.setCurrentItem(node)
        elif self._tree.currentItem() is not None:
            self._tree.setCurrentItem(None)

    def _on_item_clicked(self, item, _column: int) -> None:
        if self._syncing or self._busy:
            return
        kind = item.data(0, _ROLE_KIND)
        payload = item.data(0, _ROLE_PAYLOAD)
        if kind == 'line' and payload:
            self._current_line_id = str(payload)
            self.line_selected.emit(str(payload))
        elif kind == 'artifact' and payload:
            self.artifact_focus_requested.emit(
                str(item.data(0, _ROLE_AUX) or ''), str(payload))
        elif kind in ('spatial', 'report'):
            self.delivery_focus_requested.emit(str(kind))

    def _on_item_double_clicked(self, item, _column: int) -> None:
        """测线双击 = 选中并跳处理页；文件双击 → 系统默认程序打开
        （目录双击保留默认展开/收起）。"""
        if self._busy:
            return
        kind = item.data(0, _ROLE_KIND)
        if kind == 'line':
            payload = str(item.data(0, _ROLE_PAYLOAD) or '')
            if payload:
                self.line_process_requested.emit(payload)
        elif kind == 'file':
            path = str(item.data(0, _ROLE_PAYLOAD) or '')
            if path:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    # ------------------------------------------------------------ 刷新
    def _refresh_current_view(self) -> None:
        """F5：文件视图整体重扫（测线/成果数据由 controller 扇出，无需本地刷）。"""
        if self._busy or self._current_view != 'files':
            return
        self._rebuild()

    def _refresh_dir(self, item) -> None:
        """就地重扫目录子层（替换现有子项，保持展开）。"""
        item.takeChildren()
        for node in build_files_model(item.data(0, _ROLE_PAYLOAD)):
            self._add_node(item, node)
        item.setExpanded(True)

    # ------------------------------------------------------------ 右键菜单
    def _on_context_menu(self, pos) -> None:
        """测线叶子右键：先选中该线（与项目页表格右键即选中同语义），再弹菜单。

        分组行 / 空白处 / busy 中不出菜单；目录/文件走系统级菜单。
        """
        if self._busy:
            return
        item = self._tree.itemAt(pos)
        if item is None:
            # 文件视图空白处：整体重扫（F5 的菜单等价路径）
            if self._current_view == 'files' and self._project_root:
                menu = make_menu(parent=self._tree)
                add_action(menu, FIF.SYNC, '刷新', self._refresh_current_view)
                menu.exec(self._tree.viewport().mapToGlobal(pos))
            return
        kind = item.data(0, _ROLE_KIND)
        if kind in ('dir', 'file'):
            self._on_file_context_menu(item, pos)
            return
        # 仅测线叶子出菜单；分组行/成果/报告叶子的交互走单击
        if kind != 'line':
            return
        line_id = item.data(0, _ROLE_PAYLOAD)
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

    def _on_file_context_menu(self, item, pos) -> None:
        """目录/文件右键：目录多一项「刷新此目录」（就地重扫）；打开 /
        在资源管理器中显示 / 复制路径（只读浏览，文件管理交给系统，
        不绕过后端事务与回收站机制）。"""
        path = str(item.data(0, _ROLE_PAYLOAD) or '')
        if not path:
            return
        menu = make_menu(parent=self._tree)
        if os.path.isdir(path):
            add_action(menu, FIF.SYNC, '刷新此目录',
                       lambda: self._refresh_dir(item))
            menu.addSeparator()
        add_action(menu, FIF.DOCUMENT, '打开',
                   lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(path)))
        add_action(menu, FIF.FOLDER, '在资源管理器中显示',
                   lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(
                       path if os.path.isdir(path) else os.path.dirname(path))))
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制路径',
                   lambda: QApplication.clipboard().setText(path))
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
