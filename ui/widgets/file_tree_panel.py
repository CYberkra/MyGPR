# -*- coding: utf-8 -*-
"""Project navigation: line sources, spatial results, deliveries and attachments.

Controllers supply cached metadata; this panel never calls the backend. Raw and
saved artifacts activate through explicit signals. Tree selection, the marked
processing input and a running task's frozen source are separate identities.
Opening a context menu or viewing metadata does not activate its target. Running
tasks protect their source ancestry while other results remain viewable.

DockPanel owns the shell. Page collapse preferences, stable-key expansion
snapshots, lazy directory loading and bounded suffix width remain local here.
"""
from __future__ import annotations

import json
import os

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QPoint, QSize, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDesktopServices, QFont, QFontMetrics, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QHeaderView,
    QSizePolicy, QTreeWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import BodyLabel, MessageBox, PlainTextEdit, TransparentToolButton, TreeWidget
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import Theme, isDarkTheme, setCustomStyleSheet

from ui import constants
from ui.icons import WorkbenchIcon, result_delete_icon
from ui.design_tokens import color
from ui.file_tree import (
    TreeNode, artifact_direct_parent, artifact_parameters, artifact_step,
    build_files_model, build_project_model,
)
from ui.theme_helpers import control_palette, hint_qss, status_color
from ui.widgets.context_menus import add_action, make_menu
from ui.widgets.dock_panel import DockPanel
from ui.widgets.fluent_menu_keyboard import FluentMenuKeyboard

_ROLE_PAYLOAD = Qt.ItemDataRole.UserRole        # line → line_id；artifact → artifact_id；spatial → result_id；report → package_dir
_ROLE_KIND = Qt.ItemDataRole.UserRole + 1       # TreeNode.kind
_ROLE_AUX = Qt.ItemDataRole.UserRole + 2        # artifact → 所属 line_id
_ROLE_KEY = Qt.ItemDataRole.UserRole + 3        # TreeNode.key（重建快照展开态用）
_ROLE_SUFFIX = Qt.ItemDataRole.UserRole + 4     # 原始角标；输入标记不覆盖它

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
    """Project source and delivery navigation (collapsible to a narrow strip)."""

    line_selected = pyqtSignal(str)
    line_process_requested = pyqtSignal(str)
    line_delete_requested = pyqtSignal(list)
    line_rename_requested = pyqtSignal(str)
    # 成果叶子点击 → 换线（如需）+ 跳处理页选中预览（line_id, artifact_id）
    artifact_focus_requested = pyqtSignal(str, str)
    raw_focus_requested = pyqtSignal(str)
    artifact_reconfigure_requested = pyqtSignal(str, str)
    source_info_requested = pyqtSignal(str, str)
    # 成果叶子右键「删除成果」→ 复用项目页删除链路（后代闭包级联确认 +
    # 回收站语义），面板自身不弹确认框
    artifact_delete_requested = pyqtSignal(str, str)
    processing_package_requested = pyqtSignal(str, str)  # line_id, artifact_id (empty = whole line)
    # 空间成果/项目报告叶子点击 → 跳成果页（参数为 kind：'spatial'/'report'）
    delivery_focus_requested = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__('项目资源', constants.SIDE_TOOL_WIDTH, parent)
        self._lines: list = []
        self.rename_allowed = lambda:False
        self._rename_shortcut = QShortcut(QKeySequence('F2'), self,
            context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._rename_shortcut.activated.connect(self._rename_current_line)
        self._artifacts: list = []
        self._artifacts_sig: list = []   # set_artifacts 去重签名
        self._spatial_results: list = []
        self._reports: list = []
        self._line_id_by_item: dict = {}
        self._syncing = False
        self._busy = False
        self._current_line_id = ''
        self._active_source: tuple[str, str] | None = None
        self._processing_input: tuple[str, str] | None = None
        self._project_identity = ''
        self._current_page = ''
        self._has_project = False
        self._page_states: dict = {}   # {page: collapsed}
        self._settings = None
        self._context_menu = None
        self._menu_generation = 0
        self.installEventFilter(self)

        # ---------------- 主体：项目名 + 项目资源树
        self._project_label = BodyLabel('未打开项目')
        # 垂直 Fixed：树隐藏（无项目）时 label 若被 QVBoxLayout 拉满
        # 会垂直居中悬在面板中部，必须钉在顶部
        self._project_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._project_label.setStyleSheet(hint_qss('secondary'))
        self.body_layout().addWidget(self._project_label)

        self._tree = TreeWidget(self._expanded_view)
        self._tree.setObjectName('projectResourceTree')
        self._tree.setHeaderHidden(True)
        self._tree.setIconSize(QSize(*constants.TOOL_BTN_ICON))
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
        self._menu_shortcuts = []
        for sequence in ('Shift+F10', 'Menu'):
            shortcut = QShortcut(QKeySequence(sequence), self._tree)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self._on_keyboard_context)
            self._menu_shortcuts.append(shortcut)
        self._more_btn = TransparentToolButton(FIF.MORE, self)
        self._more_btn.setFixedSize(24, 24)
        self._more_btn.setIconSize(QSize(*constants.TOOL_BTN_ICON))
        self._more_btn.setToolTip('项目资源操作（Shift+F10）')
        self._more_btn.setAccessibleName('项目资源操作')
        self._more_btn.clicked.connect(self._on_keyboard_context)
        self._head_layout.insertWidget(self._head_layout.count() - 1, self._more_btn)
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
        identity = str(getattr(summary, 'project_id', '') or self._project_root or name)
        changed = identity != self._project_identity
        if changed or summary is None:
            self._active_source = None
            self._processing_input = None
        self._project_identity = identity
        if changed or summary is None:
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

    def set_active_source(self, line_id: str, artifact_id: str = '') -> None:
        """Mark the actual processing input independently of tree selection."""
        previous = self._active_source
        self._active_source = (str(line_id), str(artifact_id or '')) if line_id else None
        self._refresh_input_marker()
        if previous != self._active_source and self._active_source is not None:
            line, aid = self._active_source
            key = f'artifact:{aid}' if aid else f'raw:{line}'
            for item in self._items():
                if item.data(0, _ROLE_KEY) == key:
                    parent = item.parent()
                    while parent is not None:
                        parent.setExpanded(True)
                        parent = parent.parent()
                    break

    def set_processing_input(self, line_id: str, artifact_id: str = '', running: bool = False) -> None:
        """Protect a running task's source ancestry; viewing remains available."""
        previous = self._processing_input
        self._processing_input = (str(line_id), str(artifact_id or '')) if running and line_id else None
        self._refresh_processing_placeholder(reveal=previous != self._processing_input)

    def _refresh_processing_placeholder(self, *, reveal: bool = False) -> None:
        for item in list(self._items()):
            if item.data(0, _ROLE_KIND) == 'task' and item.parent() is not None:
                item.parent().takeChild(item.parent().indexOfChild(item))
        if self._processing_input is None:
            return
        line_id, artifact_id = self._processing_input
        source_key = f'artifact:{artifact_id}' if artifact_id else f'raw:{line_id}'
        source = next((item for item in self._items()
                       if item.data(0, _ROLE_KEY) == source_key), None)
        if source is None:
            return
        self._add_node(source, TreeNode(
            key=f'task:{line_id}:{artifact_id}', kind='task', text='处理中…',
            tooltip='任务使用此冻结输入；只有成功保存的输出才可打开或继续处理。',
        ))
        if reveal:
            while source is not None:
                source.setExpanded(True)
                source = source.parent()

    def _items(self):
        work = list(reversed(self._top_items()))
        while work:
            item = work.pop()
            yield item
            work.extend(item.child(index) for index in reversed(range(item.childCount())))

    def _refresh_input_marker(self) -> None:
        self._suffixes.clear()
        for item in self._items():
            kind = item.data(0, _ROLE_KIND)
            identity = None
            if kind == 'raw':
                identity = (str(item.data(0, _ROLE_PAYLOAD) or ''), '')
            elif kind == 'artifact':
                identity = (str(item.data(0, _ROLE_AUX) or ''),
                            str(item.data(0, _ROLE_PAYLOAD) or ''))
            active = identity is not None and identity == self._active_source
            suffix = str(item.data(0, _ROLE_SUFFIX) or '')
            text = ' '.join(part for part in (suffix, '输入' if active else '') if part)
            item.setText(1, text)
            item.setForeground(1, QBrush(QColor(color('primary' if active else 'text_muted', self._dark))))
            item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            font = item.font(0)
            font.setWeight(QFont.Weight.DemiBold if active or kind in ('category', 'line')
                           else QFont.Weight.Normal)
            item.setFont(0, font)
            if kind in ('raw', 'artifact'):
                item.setForeground(0, QBrush(QColor(color('primary' if active else
                    ('text_secondary' if kind == 'raw' else 'text_primary'), self._dark))))
            if text:
                self._suffixes.add(text)
        self._apply_suffix_width()

    def _artifact(self, line_id: str, artifact_id: str):
        return next((artifact for artifact in self._artifacts
                     if str(getattr(artifact, 'artifact_id', '')) == artifact_id
                     and str(getattr(artifact, 'line_id', '')) == line_id), None)

    def _delete_blocked(self, line_id: str, artifact_id: str | None = None) -> bool:
        if self._processing_input is None or self._processing_input[0] != line_id:
            return False
        if artifact_id is None:
            return True
        current = self._processing_input[1]
        seen = set()
        while current and current not in seen:
            if current == artifact_id:
                return True
            seen.add(current)
            artifact = self._artifact(line_id, current)
            if artifact is None:
                break
            direct = artifact_direct_parent(artifact)
            current = (direct if direct is not None else
                       str(getattr(artifact, 'parent_artifact_id', '') or ''))
        return False

    def _can_reconfigure(self, line_id: str, artifact_id: str) -> bool:
        artifact = self._artifact(line_id, artifact_id)
        if artifact is None or not artifact_step(artifact):
            return False
        parent = artifact_direct_parent(artifact)
        if parent is None:
            return False
        params = artifact_parameters(artifact)
        try:
            index = int(params.get('run_step_index', 0))
        except (TypeError, ValueError):
            return False
        if index == 1:
            return not parent or self._artifact(line_id, parent) is not None
        predecessor = self._artifact(line_id, parent)
        if predecessor is None:
            return False
        previous = artifact_parameters(predecessor)
        return (previous.get('lineage_version') == 2 and bool(params.get('run_group_id'))
                and previous.get('run_group_id') == params.get('run_group_id')
                and previous.get('run_step_index') == index - 1)

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
        # 收起细条不显示竖排文字：左坞仅此一面板，线号/面板名无辨识价值
        return ''

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

    def _expanded_keys(self) -> set:
        """Expanded identities, including directories whose lazy scan was empty."""
        return {item.data(0, _ROLE_KEY) for item in self._items()
                if item.isExpanded() and item.data(0, _ROLE_KEY)
                and (item.childCount() > 0 or item.data(0, _ROLE_KIND) == 'dir')}

    def _empty_category_keys(self) -> set:
        """当前为空的分类/分组 key 集合：这类节点首次长出内容时应默认
        展开（「新内容到达」事件，不能因快照里没有它而被压住折叠）。"""
        return {item.data(0, _ROLE_KEY) for item in self._items()
                if item.data(0, _ROLE_KIND) in ('category', 'group', 'line', 'raw', 'artifact')
                and item.childCount() == 0 and item.data(0, _ROLE_KEY)}

    def _rebuild(self) -> None:
        self._close_context_menu()
        self._menu_generation += 1
        # 快照展开态：数据刷新（打开工程/处理完成回灌成果/主题切换重建）
        # 不应冲掉用户手动折叠的分类。树为空 = 首次构建，走默认展开规则。
        snapshot = None
        existing_keys: set = set()
        empty_keys: set = set()
        if self._tree.topLevelItemCount():
            snapshot = self._expanded_keys()
            empty_keys = self._empty_category_keys()
            existing_keys = {item.data(0, _ROLE_KEY) for item in self._items()}
        self._tree.clear()
        self._line_id_by_item.clear()
        self._suffixes.clear()

        # 测线下挂原始输入与来源树；文件子层由面板懒加载
        model = build_project_model(
            self._lines, self._artifacts, self._spatial_results,
            self._reports, self._project_root)
        if not model:
            self._apply_view_state()
            return

        for node in model:
            self._add_node(None, node)

        # 展开策略：无快照（首次构建）→ 分类/分组有内容就默认展开，
        # 来源与附件保持折叠，让真实处理分支优先；用户展开后照常恢复。
        # 空分类折叠（(0) 计数自明），目录不自动展开（子层走懒加载）。
        # 有快照 → 按快照恢复；空分类首次长出内容（key ∈ empty_keys）
        # 仍默认展开。目录恢复展开会触发懒加载替换占位行，新子节点
        # 继续入栈，深层展开态也能恢复（worklist 而非递归的原因）。
        work = list(self._top_items())
        while work:
            item = work.pop()
            kind = item.data(0, _ROLE_KIND)
            key = item.data(0, _ROLE_KEY)
            if item.childCount() > 0 and not item.isExpanded():
                if kind in ('category', 'group', 'line', 'raw', 'artifact'):
                    if key == 'category:来源与附件':
                        item.setExpanded(bool(snapshot and key in snapshot))
                    elif snapshot is None or key in snapshot \
                            or key in empty_keys or key not in existing_keys:
                        item.setExpanded(True)
                elif kind == 'dir' and snapshot and key in snapshot:
                    item.setExpanded(True)
            work.extend(item.child(i)
                        for i in range(item.childCount()))
        self._refresh_processing_placeholder()
        self._refresh_input_marker()
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
        font = QFont(self._tree.font())
        font.setPointSize(constants.FONT_SIZE_SECONDARY)
        fm = QFontMetrics(font)
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
        work = [(parent_item, node)]
        while work:
            parent, current = work.pop()
            item = self._add_single_node(parent, current)
            work.extend((item, child) for child in reversed(current.children))

    def _add_single_node(self, parent_item, node: TreeNode):
        item = QTreeWidgetItem()
        item.setText(0, node.text)
        item.setSizeHint(0, QSize(0, constants.BTN_HEIGHT))
        font = self._tree.font()
        font.setPointSize(constants.FONT_SIZE_SECONDARY if node.kind in ('category', 'task')
                          else constants.FONT_SIZE_BODY)
        font.setWeight(QFont.Weight.DemiBold if node.kind in ('category', 'line')
                       else QFont.Weight.Normal)
        item.setFont(0, font)
        suffix_font = QFont(font)
        suffix_font.setPointSize(constants.FONT_SIZE_SECONDARY)
        suffix_font.setWeight(QFont.Weight.Normal)
        item.setFont(1, suffix_font)
        suffix_brush = QBrush(QColor(color('text_muted', self._dark)))
        item.setForeground(0, QBrush(QColor(color(
            'warning' if node.key.startswith('unknown:') else
            'text_secondary' if node.kind in ('category', 'raw', 'dir', 'file')
            else 'text_primary', self._dark))))
        if node.suffix:
            item.setText(1, node.suffix)
            item.setForeground(1, suffix_brush)
            item.setTextAlignment(
                1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            # 量宽推迟到 _apply_suffix_width：此处可能早于首次布局，字体尚未
            # 定型（实测构造期量得 31px、显示后 72px），此刻算会永久偏窄。
            self._suffixes.add(node.suffix)
        # The visible name may elide in a narrow dock; the tooltip retains it.
        item.setToolTip(0, '\n'.join(part for part in (node.text, node.tooltip) if part))
        item.setToolTip(1, node.suffix)
        item.setData(0, _ROLE_KEY, node.key)
        item.setData(0, _ROLE_KIND, node.kind)
        item.setData(0, _ROLE_SUFFIX, node.suffix)
        if node.kind in ('group', 'category', 'task'):
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # 不可选，仅展开
            if node.kind == 'task':
                item.setIcon(0, FIF.SYNC.icon())
                item.setForeground(0, suffix_brush)
            elif node.kind == 'category':
                icon = FIF.GLOBE if node.text == '空间成果' else FIF.FOLDER
                item.setIcon(0, icon.icon(color=color('text_muted', self._dark)))
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
            elif node.kind == 'raw':
                item.setIcon(0, WorkbenchIcon.WAVEFORM_FILE.icon(
                    Theme.DARK if self._dark else Theme.LIGHT))
            elif node.kind == 'artifact':
                item.setIcon(0, WorkbenchIcon.RESULT_CHART.icon(
                    Theme.DARK if self._dark else Theme.LIGHT))
        if parent_item is None:
            self._tree.addTopLevelItem(item)
        else:
            parent_item.addChild(item)
        # 目录无 children = 子层未加载：放占位行让展开箭头出现
        if node.kind == 'dir' and not node.children:
            placeholder = QTreeWidgetItem()
            placeholder.setData(0, _ROLE_KIND, 'placeholder')
            placeholder.setText(0, '…')
            item.addChild(placeholder)
        return item

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
        elif kind == 'raw' and payload:
            self.raw_focus_requested.emit(str(payload))
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
    def _close_context_menu(self):
        menu, self._context_menu = self._context_menu, None
        if menu is not None and not sip.isdeleted(menu):
            menu.close()

    def _forget_context_menu(self, identity):
        if self._context_menu is not None and id(self._context_menu) == identity:
            self._context_menu = None

    def _on_keyboard_context(self):
        item = self._tree.currentItem()
        position = self._tree.visualItemRect(item).center() if item is not None else QPoint(-1, -1)
        self._on_context_menu(position)

    def _add_menu_action(self, menu, icon, text, callback, **kwargs):
        scope = (self._project_identity, self._menu_generation)
        def guarded():
            if scope == (self._project_identity, self._menu_generation):
                callback()
        return add_action(menu, icon, text, guarded, **kwargs)

    def _show_context_menu(self, menu, position):
        self._close_context_menu()
        self._context_menu = menu
        identity = id(menu)
        menu.closedSignal.connect(lambda: self._forget_context_menu(identity))
        menu.destroyed.connect(lambda _obj=None: self._forget_context_menu(identity))
        menu._tree_keyboard = FluentMenuKeyboard(menu)
        menu.exec(self._tree.viewport().mapToGlobal(position))
        menu._tree_keyboard.focus_menu(menu)

    def eventFilter(self, watched, event):
        if watched is self and event.type() == QEvent.Type.Hide:
            self._close_context_menu()
        return super().eventFilter(watched, event)

    def _on_context_menu(self, pos) -> None:
        """Capture the hit identity; opening a menu never changes processing input."""
        if self._busy:
            return
        item = self._tree.itemAt(pos)
        if item is None:
            self._on_navigation_context_menu(None, pos)
            return
        kind = item.data(0, _ROLE_KIND)
        if kind in ('category', 'group'):
            self._on_navigation_context_menu(item, pos)
            return
        if kind in ('dir', 'file'):
            self._on_file_context_menu(item, pos)
            return
        if kind == 'artifact':
            self._on_artifact_context_menu(item, pos)
            return
        if kind == 'raw':
            self._on_raw_context_menu(item, pos)
            return
        # 仅测线/成果叶子出菜单；分组行/报告叶子的交互走单击
        if kind != 'line':
            return
        line_id = item.data(0, _ROLE_PAYLOAD)
        if not line_id:
            return
        line_id = str(line_id)
        self._tree.setCurrentItem(item)
        blocked = self._delete_blocked(line_id)
        menu = make_menu(parent=self._tree)
        self._add_menu_action(menu, FIF.DEVELOPER_TOOLS, '处理该测线',
                   lambda: self.line_process_requested.emit(line_id), enabled=not blocked)
        self._add_menu_action(menu,FIF.EDIT,'重命名测线…',
            lambda:self.line_rename_requested.emit(line_id) if self.rename_allowed() else None,
            enabled=self.rename_allowed())
        has_results = any(str(getattr(a, 'line_id', '')) == line_id
                          and str(getattr(a, 'status', 'committed')) in ('committed', 'completed', 'success')
                          for a in self._artifacts)
        self._add_menu_action(menu, FIF.SAVE, '打包处理结果…',
                   lambda: self.processing_package_requested.emit(line_id, ''),
                   enabled=has_results and not blocked)
        action = self._add_menu_action(menu, FIF.DELETE, '删除测线…',
                            lambda: self._confirm_delete(line_id), enabled=not blocked)
        if blocked and action is not None:
            action.setToolTip('该测线的处理任务正在运行，请先等待结束或取消任务。')
        menu.addSeparator()
        self._add_menu_action(menu, FIF.COPY, '复制测线号',
                   lambda: QApplication.clipboard().setText(line_id))
        self._show_context_menu(menu, pos)

    def _on_artifact_context_menu(self, item, pos) -> None:
        """Capture one saved result for open, reconfigure, info or dependency delete."""
        artifact_id = str(item.data(0, _ROLE_PAYLOAD) or '')
        line_id = str(item.data(0, _ROLE_AUX) or '')
        if not artifact_id:
            return
        self._tree.setCurrentItem(item)
        menu = make_menu(parent=self._tree)
        self._add_menu_action(menu, FIF.VIEW, '打开结果标签',
                   lambda: self.artifact_focus_requested.emit(line_id, artifact_id))
        self._add_menu_action(menu, FIF.DEVELOPER_TOOLS, '从此结果继续处理',
                   lambda: self.artifact_focus_requested.emit(
                       line_id, artifact_id))
        can_reconfigure = self._can_reconfigure(line_id, artifact_id)
        action = self._add_menu_action(menu, FIF.EDIT, '修改此步参数并另建分支',
                            lambda: self.artifact_reconfigure_requested.emit(line_id, artifact_id),
                            enabled=can_reconfigure)
        if not can_reconfigure and action is not None:
            action.setToolTip('缺少可靠的直接来源或前置步骤未保存，不能修改单步；可按完整运行配置重新处理。')
        self._add_menu_action(menu, FIF.INFO, '参数与来源',
                   lambda: self._show_source_info(line_id, artifact_id))
        blocked = self._delete_blocked(line_id, artifact_id)
        self._add_menu_action(menu, FIF.SAVE, '打包到此步的结果…',
                   lambda: self.processing_package_requested.emit(line_id, artifact_id),
                   enabled=not blocked)
        action = self._add_menu_action(menu, result_delete_icon(), '删除此结果及依赖成果…',
                   lambda: self._request_artifact_delete(line_id, artifact_id), enabled=not blocked)
        if blocked and action is not None:
            action.setToolTip('运行中的任务依赖此结果，请先等待结束或取消任务。')
        menu.addSeparator()
        self._add_menu_action(menu, FIF.COPY, '复制成果号',
                   lambda: QApplication.clipboard().setText(artifact_id))
        self._show_context_menu(menu, pos)

    def _on_raw_context_menu(self, item, pos) -> None:
        line_id = str(item.data(0, _ROLE_PAYLOAD) or '')
        if not line_id:
            return
        self._tree.setCurrentItem(item)
        menu = make_menu(parent=self._tree)
        self._add_menu_action(menu, FIF.VIEW, '打开原始输入',
                   lambda: self.raw_focus_requested.emit(line_id))
        self._add_menu_action(menu, FIF.INFO, '参数与来源', lambda: self._show_source_info(line_id, ''))
        self._show_context_menu(menu, pos)

    def _request_artifact_delete(self, line_id: str, artifact_id: str) -> None:
        if not self._busy and not self._delete_blocked(line_id, artifact_id):
            self.artifact_delete_requested.emit(line_id, artifact_id)

    def _on_navigation_context_menu(self, item, pos) -> None:
        menu = make_menu(parent=self._tree)
        if item is not None:
            self._add_menu_action(menu, FIF.FOLDER, '展开此项', lambda: item.setExpanded(True))
            self._add_menu_action(menu, FIF.FOLDER, '折叠此项', lambda: item.setExpanded(False))
        else:
            self._add_menu_action(menu, FIF.FOLDER, '展开测线与成果', self._expand_data_nodes)
            self._add_menu_action(menu, FIF.FOLDER, '全部折叠', self._tree.collapseAll)
        self._add_menu_action(menu, FIF.SYNC, '刷新目录视图', self._rebuild)
        self._show_context_menu(menu, pos)

    def _expand_data_nodes(self) -> None:
        for item in self._items():
            if item.data(0, _ROLE_KIND) in ('category', 'group', 'line', 'raw', 'artifact'):
                item.setExpanded(True)

    def _show_source_info(self, line_id: str, artifact_id: str) -> None:
        artifact = self._artifact(line_id, artifact_id) if artifact_id else None
        if artifact_id and artifact is None:
            return
        self.source_info_requested.emit(line_id, artifact_id)
        details = [f'测线：{line_id}', f'数据：{artifact_id or "原始输入"}']
        if artifact is not None:
            params = artifact_parameters(artifact)
            parent = artifact_direct_parent(artifact)
            details.extend([
                f'名称：{getattr(artifact, "name", "")}',
                f'直接输入：{"来源待核对" if parent is None else parent or "原始输入"}',
                f'形状：{getattr(artifact, "shape", ())}',
                f'数据类型：{getattr(artifact, "dtype", "") or "未记录"}',
                f'输出 SHA-256：{getattr(artifact, "sha256", "") or "未记录"}',
                '', json.dumps(params, ensure_ascii=False, indent=2, default=str),
            ])
        else:
            line = next((line for line in self._lines if str(getattr(line, 'line_id', '')) == line_id), None)
            details.append(f'源文件：{getattr(line, "raw_path", "") or "项目原始数据"}')
        dialog = QDialog(self.window() or self)
        dialog.setWindowTitle('参数与来源')
        dialog.resize(680, 440)
        colors = control_palette(isDarkTheme())
        dialog.setStyleSheet(f'QDialog {{ background: {colors["panel_bg"]}; color: {colors["text"]}; }}')
        layout = QVBoxLayout(dialog)
        content = PlainTextEdit(dialog)
        content.setReadOnly(True)
        content.setPlainText('\n'.join(details))
        layout.addWidget(content)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()
        dialog.deleteLater()

    def _on_file_context_menu(self, item, pos) -> None:
        """目录/文件右键：打开 / 在资源管理器中显示 / 复制路径（只读浏览，
        文件管理交给系统，不绕过后端事务与回收站机制）。"""
        path = str(item.data(0, _ROLE_PAYLOAD) or '')
        if not path:
            return
        menu = make_menu(parent=self._tree)
        self._add_menu_action(menu, FIF.DOCUMENT, '打开',
                   lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(path)))
        self._add_menu_action(menu, FIF.FOLDER, '在资源管理器中显示',
                   lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(
                       path if os.path.isdir(path) else os.path.dirname(path))))
        menu.addSeparator()
        self._add_menu_action(menu, FIF.COPY, '复制路径',
                   lambda: QApplication.clipboard().setText(path))
        self._show_context_menu(menu, pos)

    def _confirm_delete(self, line_id: str) -> None:
        if self._busy or self._delete_blocked(line_id):
            return
        box = MessageBox(
            '确认删除测线？',
            f'将删除测线 {line_id}（数据会移入项目 .trash 回收站，可恢复）。',
            self.window() or self,
        )
        box.yesButton.setText('删除')
        box.cancelButton.setText('取消')
        if box.exec() == QDialog.DialogCode.Accepted and not self._delete_blocked(line_id):
            self.line_delete_requested.emit([line_id])

    def _rename_current_line(self):
        item=self._tree.currentItem()
        if (item is not None and len(self._tree.selectedItems())==1 and self.rename_allowed()
                and item.data(0,_ROLE_KIND)=='line'):
            self.line_rename_requested.emit(str(item.data(0,_ROLE_PAYLOAD)))

    # ------------------------------------------------------------ 主题
    def apply_theme(self, dark: bool) -> None:
        """主题切换 → 重刷 hint 文字色 + 重建树刷新状态圆点色。

        主窗口 findChildren 统一调度。hint label 的颜色经
        :func:`hint_qss` 随主题查表——历史实现用模块级常量写死 ``#888888``，
        深色底上对比度约 3.5:1 不达 WCAG AA，且模块常量不参与重刷。
        """
        super().apply_theme(dark)
        self._title_label.setStyleSheet(
            f'font-size: {constants.FONT_SIZE_BODY}pt; font-weight: 600; '
            f'color: {color("text_primary", dark)};')
        self._project_label.setStyleSheet(
            f'font-size: {constants.FONT_SIZE_SECONDARY}pt; '
            f'color: {color("text_muted", dark)}; '
            f'padding: {constants.SPACE_1}px 0;')
        tree_qss = (
            f'QTreeWidget#projectResourceTree {{ border: none; background: transparent; }}'
            f'QTreeWidget#projectResourceTree::item {{ height: {constants.BTN_HEIGHT}px; '
            f'padding-top: 0px; padding-bottom: 0px; margin-top: 0px; margin-bottom: 0px; }}')
        setCustomStyleSheet(self._tree, tree_qss, tree_qss)
        self._rebuild()
