#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""左侧方法导航树 - 支持分组、搜索"""

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
)
from qfluentwidgets import PushButton

from core.methods_registry import (
    PROCESSING_METHODS,
    get_method_category_label,
    get_method_display_name,
    get_public_methods_grouped_by_category,
)


# 方法分类定义
METHOD_TREE_DATA = {
    "data": {
        "icon": "",
        "name": "数据管理",
        "children": [
            {"id": "_import_csv", "name": "数据导入"},
            {"id": "_data_info", "name": "数据信息"},
        ],
    },
    "processing": {
        "icon": "",
        "name": "单步处理",
        "children": [],
    },
    "workflow": {
        "icon": "",
        "name": "流程/模板",
        "children": [
            {"id": "_quick_preview", "name": "快速预览"},
            {"id": "_robust_imaging", "name": "稳健成像"},
            {"id": "_high_focus", "name": "高聚焦"},
        ],
    },
    "favorites": {
        "icon": "",
        "name": "收藏/最近使用",
        "children": [],  # 动态填充
    },
}


class MethodBrowserTree(QWidget):
    """左侧方法导航树"""

    # 信号：方法被选中 (method_id)
    method_selected = pyqtSignal(str)
    # 信号：特殊操作 (_import_csv, _data_info, _quick_preview, etc.)
    action_triggered = pyqtSignal(str)
    # 信号：模板执行 (template_name)
    template_execute_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.recent_methods = []
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("workbenchMethodPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 搜索框（更紧凑）
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 搜索方法...")
        self.search_box.setFixedHeight(36)
        self.search_box.setObjectName("workbenchSearchBox")
        self.search_box.textChanged.connect(self._on_search)
        layout.addWidget(self.search_box)

        # 树形控件（更紧凑）
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setAnimated(True)
        self.tree.setIndentation(18)
        self.tree.setUniformRowHeights(False)
        self.tree.setObjectName("workbenchMethodTree")
        self.tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.tree)

        # 初始化树
        self._build_tree()

    def _build_tree(self):
        """构建方法树"""
        self.tree.clear()

        for cat_id, cat_info in METHOD_TREE_DATA.items():
            # 创建分类节点
            cat_item = QTreeWidgetItem()
            title = (
                cat_info["name"]
                if not cat_info["icon"]
                else f"{cat_info['icon']} {cat_info['name']}"
            )
            cat_item.setText(0, title)
            cat_item.setData(0, Qt.ItemDataRole.UserRole, f"cat:{cat_id}")
            cat_item.setExpanded(True)
            cat_font = QFont()
            cat_font.setBold(True)
            cat_font.setPointSize(11)
            cat_item.setFont(0, cat_font)
            cat_item.setSizeHint(0, QSize(0, 34))

            if cat_id == "processing":
                for category, keys in get_public_methods_grouped_by_category():
                    group_item = QTreeWidgetItem()
                    group_item.setText(0, get_method_category_label(keys[0]))
                    group_item.setData(0, Qt.ItemDataRole.UserRole, f"group:{category}")
                    group_font = QFont()
                    group_font.setBold(True)
                    group_font.setPointSize(10)
                    group_item.setFont(0, group_font)
                    group_item.setExpanded(True)
                    group_item.setSizeHint(0, QSize(0, 30))

                    for method_key in keys:
                        child_item = QTreeWidgetItem()
                        child_item.setText(0, get_method_display_name(method_key))
                        child_item.setData(0, Qt.ItemDataRole.UserRole, method_key)
                        child_font = QFont()
                        child_font.setPointSize(11)
                        child_item.setFont(0, child_font)
                        child_item.setSizeHint(0, QSize(0, 30))
                        group_item.addChild(child_item)

                    cat_item.addChild(group_item)
            else:
                children = self._resolve_category_children(cat_id, cat_info)
                for method in children:
                    child_item = QTreeWidgetItem()
                    child_item.setText(0, method["name"])
                    child_item.setData(0, Qt.ItemDataRole.UserRole, method["id"])
                    child_font = QFont()
                    child_font.setPointSize(11)
                    child_item.setFont(0, child_font)
                    child_item.setSizeHint(0, QSize(0, 30))

                    cat_item.addChild(child_item)

            self.tree.addTopLevelItem(cat_item)

    def _resolve_category_children(self, cat_id: str, cat_info: dict) -> list[dict]:
        """根据分类节点解析子项列表。"""
        return list(cat_info.get("children", []))

    def _on_item_clicked(self, item, column):
        """点击项目"""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and not data.startswith("cat:"):
            method_id = data

            # 添加到最近使用
            if not method_id.startswith("_") and not method_id.startswith("template:"):
                if method_id not in self.recent_methods:
                    self.recent_methods.insert(0, method_id)
                    if len(self.recent_methods) > 5:
                        self.recent_methods.pop()
                    self._update_recent_section()

            # 发射信号
            if method_id.startswith("template:"):
                # 模板执行
                template_name = method_id[9:]  # 去掉 "template:" 前缀
                self.template_execute_requested.emit(template_name)
            elif method_id.startswith("_"):
                self.action_triggered.emit(method_id)
            else:
                self.method_selected.emit(method_id)

    def _on_search(self, text):
        """搜索过滤"""
        text = text.lower().strip()
        for i in range(self.tree.topLevelItemCount()):
            cat_item = self.tree.topLevelItem(i)
            cat_visible = False

            for j in range(cat_item.childCount()):
                child = cat_item.child(j)
                child_visible = self._filter_item_recursive(child, text)
                if child_visible:
                    cat_visible = True

            cat_item.setHidden(not cat_visible)

    def _filter_item_recursive(self, item: QTreeWidgetItem, text: str) -> bool:
        """递归过滤树项目。"""
        if item.childCount() == 0:
            visible = text in item.text(0).lower()
            item.setHidden(not visible)
            return visible

        visible = False
        for idx in range(item.childCount()):
            if self._filter_item_recursive(item.child(idx), text):
                visible = True
        item.setHidden(not visible)
        return visible

    def _update_recent_section(self):
        """更新最近使用部分"""
        # 找到收藏节点
        fav_item = None
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data == "cat:favorites":
                fav_item = item
                break

        if not fav_item:
            return

        # 清空现有子项
        fav_item.takeChildren()

        # 添加最近使用
        for method_id in self.recent_methods:
            if method_id in PROCESSING_METHODS:
                child_item = QTreeWidgetItem()
                child_item.setText(0, get_method_display_name(method_id))
                child_item.setData(0, Qt.ItemDataRole.UserRole, method_id)
                child_item.setSizeHint(0, QSize(0, 30))
                fav_item.addChild(child_item)

    def select_method(self, method_id: str):
        """通过代码选择方法"""
        # 在树中找到并选中该方法
        for i in range(self.tree.topLevelItemCount()):
            cat_item = self.tree.topLevelItem(i)
            for j in range(cat_item.childCount()):
                child = cat_item.child(j)
                if child.childCount() == 0:
                    if child.data(0, Qt.ItemDataRole.UserRole) == method_id:
                        cat_item.setExpanded(True)
                        self.tree.setCurrentItem(child)
                        self._on_item_clicked(child, 0)
                        return
                else:
                    for k in range(child.childCount()):
                        grand = child.child(k)
                        if grand.data(0, Qt.ItemDataRole.UserRole) == method_id:
                            cat_item.setExpanded(True)
                            child.setExpanded(True)
                            self.tree.setCurrentItem(grand)
                            self._on_item_clicked(grand, 0)
                            return

    def update_workflow_templates(self, templates: list):
        """更新流程模板列表"""
        # 找到流程节点
        workflow_item = None
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data == "cat:workflow":
                workflow_item = item
                break

        if not workflow_item:
            return

        # 清空现有子项
        workflow_item.takeChildren()

        # 添加预设流程
        preset_workflows = [
            {"id": "_quick_preview", "name": "快速预览"},
            {"id": "_robust_imaging", "name": "稳健成像"},
            {"id": "_high_focus", "name": "高聚焦"},
        ]

        for workflow in preset_workflows:
            child_item = QTreeWidgetItem()
            child_item.setText(0, workflow["name"])
            child_item.setData(0, Qt.ItemDataRole.UserRole, workflow["id"])
            child_item.setSizeHint(0, QSize(0, 30))
            workflow_item.addChild(child_item)

        # 添加自定义模板
        for template in templates:
            name = template.get("name", "未命名")
            method_count = template.get("method_count", 0)
            child_item = QTreeWidgetItem()
            child_item.setText(0, f"{name} ({method_count}步)")
            child_item.setData(0, Qt.ItemDataRole.UserRole, f"template:{name}")
            child_item.setSizeHint(0, QSize(0, 30))
            workflow_item.addChild(child_item)
