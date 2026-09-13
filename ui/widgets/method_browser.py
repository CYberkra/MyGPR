"""MethodBrowser — 处理方法分类浏览控件（SPEC §5.4）。

UI：顶部搜索 LineEdit（占位"搜索方法…"，实时过滤）→ 分类 QTreeWidget
（一级 = category_label，二级 = display_name + 标签徽章）。
徽章：白字彩底、随主题刷新——推荐 = 运行时 themeColor()（经 apply_theme
重建，避免构建时快照过期）/ 备选 = disabled 灰 / 实验 = warning 琥珀，
QSS 用 SPEC §1 徽章模板。tooltip 显示 method_id 与参数数。
双击发 sig_add_requested；单击选中发 sig_method_selected。
右键菜单（RoundMenu）：添加到处理链（等同双击）/ 复制方法名。
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget, QLabel)
from qfluentwidgets import LineEdit, themeColor
from qfluentwidgets import FluentIcon as FIF

from ui.theme_helpers import status_color
from ui.widgets.context_menus import add_action, make_menu

_BADGE_QSS = ('QLabel { padding: 2px 10px; border-radius: 10px; '
              'font-size: 12px; font-weight: bold; '
              'color: #ffffff; background-color: %s; }')


def _tag_badge_bg(tag: str) -> str:
    """标签徽章底色（随主题）：推荐 = 当前主题色（构建后经 apply_theme 刷新）。"""
    if tag == '推荐':
        return themeColor().name()
    if tag == '实验':
        return status_color('warning')
    return status_color('disabled')   # 备选 / 未知标签


def _make_badge(tag: str) -> QLabel:
    badge = QLabel(tag)
    badge.setStyleSheet(_BADGE_QSS % _tag_badge_bg(tag))
    return badge


class MethodBrowser(QWidget):
    """方法库浏览器。"""

    sig_method_selected = pyqtSignal(str)   # method_id
    sig_add_requested = pyqtSignal(str)     # 双击或"添加"按钮

    def __init__(self, parent=None):
        super().__init__(parent)
        self._methods = []
        self._badges = []   # (badge QLabel, tag)：主题切换时重刷底色

        self._search = LineEdit(self)
        self._search.setPlaceholderText('搜索方法…')
        self._search.textChanged.connect(self._apply_filter)

        self._tree = QTreeWidget(self)
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(1)
        self._tree.currentItemChanged.connect(self._on_current_changed)
        self._tree.itemDoubleClicked.connect(self._on_double_clicked)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._search)
        layout.addWidget(self._tree, 1)

    # ------------------------------------------------------------- 数据
    def set_methods(self, methods) -> None:
        """methods: [{method_id,name,display_name,category,category_label,
                     tags(list[str]),parameter_schema(list[dict]),...}]"""
        self._methods = [dict(m) for m in (methods or [])]
        self._rebuild_tree()

    def _rebuild_tree(self):
        self._tree.clear()
        groups = {}
        order = []
        for m in self._methods:
            cat = m.get('category_label') or m.get('category') or '未分类'
            if cat not in groups:
                groups[cat] = []
                order.append(cat)
            groups[cat].append(m)

        for cat in order:
            top = QTreeWidgetItem([cat])
            top.setFlags(top.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._tree.addTopLevelItem(top)
            for m in groups[cat]:
                child = QTreeWidgetItem([''])
                child.setData(0, Qt.ItemDataRole.UserRole,
                              m.get('method_id', ''))
                child.setData(0, Qt.ItemDataRole.UserRole + 1,
                              m.get('display_name') or m.get('name', ''))
                n_params = len(m.get('parameter_schema') or [])
                child.setToolTip(0, '方法ID: %s\n参数数: %d'
                                    % (m.get('method_id', ''), n_params))
                top.addChild(child)

                row_widget = QWidget(self._tree)
                row = QHBoxLayout(row_widget)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(6)
                name_label = QLabel(m.get('display_name') or m.get('name', ''),
                                    row_widget)
                row.addWidget(name_label, 1)
                for tag in (m.get('tags') or []):
                    badge = _make_badge(str(tag))
                    self._badges.append((badge, str(tag)))
                    row.addWidget(badge)
                self._tree.setItemWidget(child, 0, row_widget)
            top.setExpanded(True)

        self._apply_filter(self._search.text())

    # ------------------------------------------------------------- 过滤
    def _apply_filter(self, text):
        needle = (text or '').strip().lower()
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            visible_children = 0
            for j in range(top.childCount()):
                child = top.child(j)
                name = (child.data(0, Qt.ItemDataRole.UserRole + 1) or '')
                mid = (child.data(0, Qt.ItemDataRole.UserRole) or '')
                hit = (not needle or needle in name.lower()
                       or needle in mid.lower())
                child.setHidden(not hit)
                visible_children += int(hit)
            top.setHidden(visible_children == 0)

    def apply_theme(self, dark: bool) -> None:
        """主题切换：重建徽章底色——「推荐」的 themeColor() 是构建时快照，
        主窗口主题切换遍历（findChildren + apply_theme）会调到本方法。"""
        for badge, tag in self._badges:
            badge.setStyleSheet(_BADGE_QSS % _tag_badge_bg(tag))

    # ------------------------------------------------------------- 信号
    def _method_id_of(self, item):
        if item is None or item.parent() is None:
            return None
        return item.data(0, Qt.ItemDataRole.UserRole)

    def _on_current_changed(self, current, _previous):
        mid = self._method_id_of(current)
        if mid:
            self.sig_method_selected.emit(mid)

    def _on_double_clicked(self, item, _column):
        mid = self._method_id_of(item)
        if mid:
            self.sig_add_requested.emit(mid)

    def _on_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        mid = self._method_id_of(item)
        if not mid:
            return
        self._tree.setCurrentItem(item)
        menu = make_menu(self)
        add_action(menu, FIF.ADD, '添加到处理链',
                   lambda: self.sig_add_requested.emit(mid))
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制方法名',
                   lambda: QApplication.clipboard().setText(mid))
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def current_method_id(self):
        """当前选中方法 id（无则 None），供"添加所选方法"按钮使用。"""
        return self._method_id_of(self._tree.currentItem())
