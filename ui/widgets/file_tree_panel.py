# -*- coding: utf-8 -*-
"""左侧常驻文件树面板（DockPanel 子类，单棵真树：测线 / 成果 / 文件 三分类）。

职责边界：
- 数据唯一来源仍是各 controller（经 ProjectChain 扇出，本面板不发起任何
  后端调用）；
- 整棵树由 ``ui.file_tree.build_project_model`` 纯函数装配（Provider 层）：
  测线/成果/文件是三个一级分类节点（不可选），各自挂原有子模型——测线按
  日期分组、成果按测线分组、文件为项目根目录树（目录展开时逐层懒加载）；
- 空分类折叠且 ``(0)`` 计数自明（无独立空态文案）；有内容的分类默认展开；
- 测线叶子点击 → ``line_selected(str)`` → 复用 ``ProjectChain.on_line_selected``
  现有链路（含切线作废成果预览代数），与项目页测线表同语义；
- 成果叶子点击 → ``artifact_focus_requested(line_id, artifact_id)`` →
  换线（如需）+ 跳处理页选中预览；
- 空间成果/项目报告叶子点击 → ``delivery_focus_requested(kind)`` → 跳成果页；
- ``set_current_line`` 是同步入口（``_syncing`` 守卫防回环）；
- **分类行/分组行不可选**（无 ItemIsSelectable）——分组行若能触发预览
  会推进预览代数造成串台，是成果预览代际竞态的同族风险；
- busy 只禁叶子点击，收/展开始终可用（长任务中更该允许让出空间）。

壳（头/细条/动画）全部来自 ``DockPanel`` 基类；本类只保留文件树自己的
三件事：内容构建、按页展开态记忆（SettingsManager 持久化，一维
``{page: collapsed}``，读取兼容旧二维格式）、细条指示文字 = 当前线 ID。
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDesktopServices, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QDialog, QHeaderView, QSizePolicy, QTreeWidgetItem, QWidget,
)
from qfluentwidgets import BodyLabel, MessageBox, TreeWidget
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import isDarkTheme

from ui import constants
from ui.file_tree import (
    TreeNode, build_files_model, build_project_model,
)
from ui.theme_helpers import hint_qss, status_color
from ui.widgets.context_menus import add_action, make_menu
from ui.widgets.dock_panel import DockPanel

_ROLE_PAYLOAD = Qt.ItemDataRole.UserRole        # line → line_id；artifact → artifact_id；spatial → result_id；report → package_dir
_ROLE_KIND = Qt.ItemDataRole.UserRole + 1       # TreeNode.kind
_ROLE_AUX = Qt.ItemDataRole.UserRole + 2        # artifact → 所属 line_id

_SUFFIX_BRUSH = QBrush(QColor('#8a8a8a'))       # 行尾角标灰


def suffix_column_width(view_width: int, widest_suffix_px: int) -> int:
    """角标列（第二列）宽度：内容宽 + 留白，且不超过视口宽的 _SUFFIX_MAX_RATIO。

    抽成纯函数以便无布局单测（offscreen 下把面板 show 出来拿真实布局会引入
    库级 access violation，见 tests/test_file_tree.py 的 fixture 注释）。
    名称列是 Stretch、吃掉全部余量，所以角标列越窄名称列越宽——深层节点
    （分组→测线→成果）才不会退化成只显示两三个字。
    """
    if view_width <= 0 or widest_suffix_px <= 0:
        return 0
    return max(0, min(widest_suffix_px + _SUFFIX_PAD,
                      int(view_width * _SUFFIX_MAX_RATIO)))


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

# 按页收起态记忆（True=收起细条）；键缺省按收起处理。
# 新格式一维 ``{page: collapsed}``；旧版二维 ``{page: {view: collapsed}}``
# 仅作读取回退（取测线视图的值——主视图语义最接近原意图）。
_DEFAULT_PAGE_COLLAPSED = {
    'projectInterface': False,   # 项目页=资产管理，展开
    'homeInterface': True,
    'settingsInterface': True,
}
_SETTINGS_KEY = 'file_tree_page_states'
# 旧版设置键：仅作读取回退（老用户的按页记忆不丢），写入只写新键
_LEGACY_SETTINGS_KEY = 'line_tree_page_states'

# 角标列（第二列）宽度策略 —— 见 :meth:`FileTreePanel._apply_suffix_width`。
# 实测（offscreen，面板 232px / 视口 215px）：Qt6 的 QHeaderView 默认
# stretchLastSection=True，即便第二列设为 ResizeToContents 也会被拉伸到与
# 名称列平分（108 / 107）；再叠加每级 20px 缩进 + 状态圆点图标，深层节点
# 的实际文字绘制区只剩约两个汉字宽——即线上"文件树只能显示两个字"。
# 因此：关掉末列拉伸，角标列按内容宽定宽并封顶，名称列 Stretch 吃余量。
_SUFFIX_MAX_RATIO = 0.38   # 角标列最多占视口宽的比例（名称列始终拿大头）
_SUFFIX_PAD = 8            # 角标文字两侧留白


class FileTreePanel(DockPanel):
    """测线 / 成果 / 文件 三分类常驻导航树（可收成细条）。"""

    line_selected = pyqtSignal(str)
    line_process_requested = pyqtSignal(str)
    line_delete_requested = pyqtSignal(list)
    # 成果叶子点击 → 换线（如需）+ 跳处理页选中预览（line_id, artifact_id）
    artifact_focus_requested = pyqtSignal(str, str)
    # 成果叶子右键「删除成果」→ 复用项目页删除链路（后代闭包级联确认 +
    # 回收站语义），面板自身不弹确认框
    artifact_delete_requested = pyqtSignal(str, str)
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
        self._has_project = False
        self._page_states: dict = {}   # {page: collapsed}
        self._settings = None

        # ---------------- 主体：项目名 + 三分类树
        self._project_label = BodyLabel('未打开项目')
        # 垂直 Fixed：树隐藏（无项目）时 label 若被 QVBoxLayout 拉满
        # 会垂直居中悬在面板中部，必须钉在顶部
        self._project_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._project_label.setStyleSheet(hint_qss('secondary'))
        self.body_layout().addWidget(self._project_label)

        self._tree = TreeWidget(self._expanded_view)
        self._tree.setHeaderHidden(True)
        # 中间省略：测线/成果名往往是「前缀稳定 + 尾部分辨」（如
        # L09_processed_20260802_160259_972694），右侧省略会把唯一有分辨力的
        # 尾部切掉；同宽下中间省略保留头尾，可读性更好。
        self._tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        # 缩进 20 → 14：三层树（分组→测线→成果）下每层省 6px，深层节点能多
        # 显示约一个字。qfluentwidgets 的分支点击热区按 level*indentation+20
        # 计算（level 0 恒为 20..30），不受影响。
        self._tree.setIndentation(14)
        # 第二列：行尾角标（灰字右对齐，定宽 + 封顶，宽度由 _apply_suffix_width
        # 按内容算；不用 ResizeToContents——末列拉伸会把它顶到与名称列平分）
        self._tree.setColumnCount(2)
        header = self._tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._suffixes: set = set()   # 当前树用到的角标文本（去重，定宽时量宽）
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.itemExpanded.connect(self._on_item_expanded)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self.body_layout().addWidget(self._tree, 1)
        # 透明占位：QVBoxLayout 中隐藏的 stretch 项会让剩余空间把 Fixed 项
        # 挤到垂直居中（Qt6 实测）。树隐藏（无项目）时 filler 补位吃掉余量，
        # 项目名 label 才能钉在顶部。
        self._filler = QWidget(self._expanded_view)
        self._filler.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._filler.hide()
        self.body_layout().addWidget(self._filler)
        self._project_root = ''

        self._tree.hide()
        # 构造完成后刷一次主题（基类构造期不能调 apply_theme——那时本类
        # 的 _project_label 尚未创建）。
        self.apply_theme(isDarkTheme())

    # ------------------------------------------------ 页面协议（链路喂数据）
    def set_settings_manager(self, settings) -> None:
        """注入共享 SettingsManager（与页面同一约定：共享实例是唯一写者）。

        一维记忆格式 ``{page: collapsed}``；读取兼容旧版二维格式
        ``{page: {view: collapsed}}``（取测线视图值）与旧版设置键。
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
                    # 旧二维格式：测线是主视图，其收起态最接近原意图
                    if 'lines' in val:
                        states[str(page)] = bool(val['lines'])
                    elif val:
                        states[str(page)] = bool(next(iter(val.values())))
                else:
                    states[str(page)] = bool(val)
            self._page_states = states

    def set_project_info(self, summary) -> None:
        """项目上下文切换；None = 无项目（面板常驻，显示"未打开项目"空态）。"""
        name = str(getattr(summary, 'name', '') or '') if summary else ''
        self._project_label.setText(name or '未打开项目')
        self._has_project = summary is not None
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
        """切页时按该页记忆的展开态切换（瞬时，不做动画）。"""
        page = str(object_name or '')
        if page:
            self._current_page = page
        self.set_collapsed(self._collapsed_for(page), animate=False)

    def toggle_panel(self) -> None:
        """窗口级入口（页签条右端按钮 / Ctrl+B）：与头部收起钮同一路径，
        按页记忆并持久化。"""
        self._on_toggle_clicked()

    # ------------------------------------------------------------ DockPanel 钩子
    def strip_text(self) -> str:
        return self._current_line_id

    def _on_toggle_clicked(self) -> None:
        # 手动切换：立即按当前页记忆并持久化
        self._remember_current(not self._collapsed)
        self.toggle()

    def _collapsed_for(self, page: str) -> bool:
        """页 → 记忆的收起态；兼容旧二维残值与缺省页。"""
        default = _DEFAULT_PAGE_COLLAPSED.get(page, True)
        val = self._page_states.get(page)
        if isinstance(val, bool):
            return val
        if isinstance(val, dict):  # 旧二维残值：主视图值最接近原意图
            if 'lines' in val:
                return bool(val['lines'])
            if val:
                return bool(next(iter(val.values())))
        return bool(default)

    def _remember_current(self, collapsed: bool) -> None:
        page = self._current_page or 'projectInterface'
        self._page_states[page] = bool(collapsed)
        if self._settings is not None:
            self._settings.set(_SETTINGS_KEY,
                               {p: v for p, v in self._page_states.items()
                                if isinstance(v, bool)})
            self._settings.save()

    # ------------------------------------------------------------ 内部
    def _on_view_state_changed(self) -> None:
        """展开态下树常驻显示（有项目时）：空分类折叠 + (0) 计数自明，
        不再需要「尚无成果」之类的独立空态文案。"""
        if self._collapsed:
            return
        self._tree.setVisible(self._has_project)
        self._filler.setVisible(not self._has_project)

    def _rebuild(self) -> None:
        self._tree.clear()
        self._line_id_by_item.clear()
        self._suffixes.clear()

        # 单棵三分类树（测线/成果/文件）；文件子层由面板懒加载
        model = build_project_model(
            self._lines, self._artifacts, self._spatial_results,
            self._reports, self._project_root)
        if not model:
            self._apply_view_state()
            return

        for node in model:
            self._add_node(None, node)

        # 分类/分组行有内容才默认展开（文件视图的目录不自动展开——
        # 子层走懒加载；空分类折叠，(0) 计数自明）。分组行现在位于
        # 分类下一层，必须递归展开，否则测线日期分组默认折叠（回归）。
        def _expand_groups(item) -> None:
            if item.data(0, _ROLE_KIND) in ('category', 'group') \
                    and item.childCount() > 0:
                item.setExpanded(True)
            for i in range(item.childCount()):
                _expand_groups(item.child(i))

        for item in self._top_items():
            _expand_groups(item)
        self._apply_suffix_width()
        self._select_leaf(self._current_line_id)
        self._apply_view_state()

    def _apply_suffix_width(self) -> None:
        """角标列定宽：内容宽 + 留白，且不超过视口宽的 _SUFFIX_MAX_RATIO。

        名称列（col0）为 Stretch，自动吃掉余量——这是"显示不全"修复的
        另一半：只关末列拉伸还不够，长角标（如 16 字符时间戳）按内容定宽
        会反向把名称列压到 20px，必须封顶。
        """
        viewport_w = self._tree.viewport().width()
        if viewport_w <= 0:      # 尚未布局（构造期/隐藏态）
            return
        # 去重后量宽：不用 sizeHintForColumn（后者含缩进开销，实测把 192px
        # 的角标算成 238px），也不在建树时量（那时字体可能还没定型）
        fm = self._tree.fontMetrics()
        widest = max((fm.horizontalAdvance(s) for s in self._suffixes),
                     default=0)
        self._tree.setColumnWidth(1, suffix_column_width(viewport_w, widest))

    def resizeEvent(self, event) -> None:  # Qt 虚函数命名（CamelCase）
        """面板宽度变化 → 角标列封顶值随之变化，需重算（名称列同步得余量）。"""
        super().resizeEvent(event)
        self._apply_suffix_width()

    def showEvent(self, event) -> None:  # Qt 虚函数命名（CamelCase）
        """首次显示时字体度量才定型（构造期量得 31px vs 显示后 72px），
        此时按真实度量重算——否则角标列会按未定型字体永久偏窄。"""
        super().showEvent(event)
        self._apply_suffix_width()

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
            # 量宽推迟到 _apply_suffix_width：此处可能早于首次布局，字体尚未
            # 定型（实测构造期量得 31px、显示后 72px），此刻算会永久偏窄。
            self._suffixes.add(node.suffix)
        if node.tooltip:
            item.setToolTip(0, node.tooltip)
        item.setData(0, _ROLE_KIND, node.kind)
        if node.kind in ('group', 'category'):
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
        self._apply_suffix_width()   # 懒加载出的角标可能比原有的更宽

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
        """文件双击 → 系统默认程序打开（目录双击保留默认展开/收起）。"""
        if self._busy:
            return
        if item.data(0, _ROLE_KIND) == 'file':
            path = str(item.data(0, _ROLE_PAYLOAD) or '')
            if path:
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    # ------------------------------------------------------------ 右键菜单
    def _on_context_menu(self, pos) -> None:
        """测线叶子右键：先选中该线（与项目页表格右键即选中同语义），再弹菜单。

        分组行 / 空白处 / busy 中不出菜单；目录/文件走系统级菜单。
        """
        if self._busy:
            return
        item = self._tree.itemAt(pos)
        if item is None:
            return
        kind = item.data(0, _ROLE_KIND)
        if kind in ('dir', 'file'):
            self._on_file_context_menu(item, pos)
            return
        if kind == 'artifact':
            self._on_artifact_context_menu(item, pos)
            return
        # 仅测线/成果叶子出菜单；分组行/报告叶子的交互走单击
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

    def _on_artifact_context_menu(self, item, pos) -> None:
        """成果叶子右键：添加到显示 / 删除成果 / 复制成果号。

        「添加到显示」与单击同语义（artifact_focus_requested：换线如需 +
        跳处理页选中并载入 B-Scan 对比显示）；「删除」发信号复用项目页
        删除链路（异步查后代闭包 → 级联确认框 → 回收站），面板自身不弹
        确认框、不做预检。
        """
        artifact_id = str(item.data(0, _ROLE_PAYLOAD) or '')
        line_id = str(item.data(0, _ROLE_AUX) or '')
        if not artifact_id:
            return
        self._tree.setCurrentItem(item)
        menu = make_menu(parent=self._tree)
        add_action(menu, FIF.VIEW, '添加到显示',
                   lambda: self.artifact_focus_requested.emit(
                       line_id, artifact_id))
        add_action(menu, FIF.DELETE, '删除成果…',
                   lambda: self.artifact_delete_requested.emit(
                       line_id, artifact_id))
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制成果号',
                   lambda: QApplication.clipboard().setText(artifact_id))
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _on_file_context_menu(self, item, pos) -> None:
        """目录/文件右键：打开 / 在资源管理器中显示 / 复制路径（只读浏览，
        文件管理交给系统，不绕过后端事务与回收站机制）。"""
        path = str(item.data(0, _ROLE_PAYLOAD) or '')
        if not path:
            return
        menu = make_menu(parent=self._tree)
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
    def apply_theme(self, dark: bool) -> None:
        """主题切换 → 重刷 hint 文字色 + 重建树刷新状态圆点色。

        主窗口 findChildren 统一调度。hint label 的颜色经
        :func:`hint_qss` 随主题查表——历史实现用模块级常量写死 ``#888888``，
        深色底上对比度约 3.5:1 不达 WCAG AA，且模块常量不参与重刷。
        """
        super().apply_theme(dark)
        self._project_label.setStyleSheet(hint_qss('secondary'))
        self._rebuild()
