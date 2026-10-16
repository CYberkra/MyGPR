# -*- coding: utf-8 -*-
"""统一右键菜单封装（qfluentwidgets RoundMenu）。

全项目右键菜单统一走这里：Fluent 风格 RoundMenu（主题自动跟随），
避免 pyqtgraph 原生英文菜单混风格。图形视图接入前需先
``vb.setMenuEnabled(False)`` 关闭 ViewBox 原生菜单。
"""
from __future__ import annotations

from qfluentwidgets import Action, RoundMenu
from qfluentwidgets import FluentIcon as FIF


_LIVE_MENUS: dict[int, RoundMenu] = {}


class SafeRoundMenu(RoundMenu):
    """Keep the menu alive until its synchronous action dispatch returns.

    Fluent closes before triggering an action. A modal action runs a nested Qt
    event loop, so deleting on close can destroy the list currently dispatching
    the click. Disposal must wait for the entire click handler to unwind.
    """

    def __init__(self, title='', parent=None):
        self._dispatch_depth = 0
        self._dispose_requested = False
        super().__init__(title, parent)

    def _dispose_when_idle(self):
        self._dispose_requested = True
        if not self._dispatch_depth:
            self.deleteLater()

    @staticmethod
    def _dispatch(menu, callback, item):
        root = menu
        while root.parentMenu is not None:
            root = root.parentMenu
        if not isinstance(root, SafeRoundMenu):
            return callback(item)
        root._dispatch_depth += 1
        try:
            return callback(item)
        finally:
            root._dispatch_depth -= 1
            if not root._dispatch_depth and root._dispose_requested:
                root.deleteLater()

    def _onItemClicked(self, item):
        return self._dispatch(self, super()._onItemClicked, item)

    def _createSubMenuItem(self, menu):
        result = super()._createSubMenuItem(menu)
        # Existing callers also construct plain RoundMenu submenus. Guard their
        # pointer and keyboard entry points using the same root lifetime.
        pending = [menu]
        while pending:
            child = pending.pop()
            pending.extend(child._subMenus)
            if isinstance(child, SafeRoundMenu) or getattr(child, '_mygpr_dispatch_guard', False):
                continue
            original = child._onItemClicked
            child.view.itemClicked.disconnect(original)
            def dispatch(item, child=child, original=original):
                return self._dispatch(child, original, item)
            child._onItemClicked = dispatch
            child._mygpr_dispatch_guard = True
            child.view.itemClicked.connect(dispatch)
        return result


def make_menu(parent=None) -> RoundMenu:
    """新建 RoundMenu（主题由 qfluentwidgets 自动跟随）。

    RoundMenu 签名是 (title='', parent=None)，parent 必须关键字传。
    """
    menu = SafeRoundMenu(parent=parent)
    # RoundMenu 的 Python 信号/子菜单形成环；GC 先回收包装对象、Qt 随后
    # 删除父对象时，会在 qfluentwidgets 的 destroyed 回调中原生崩溃。
    # 保留到 Qt 确认销毁；回调只捕获键，避免再捕获正在析构的菜单。
    key = id(menu)
    _LIVE_MENUS[key] = menu
    menu.destroyed.connect(lambda _obj=None, key=key: _LIVE_MENUS.pop(key, None))
    # exec() 非阻塞；菜单关闭后才安排删除，避免每次右键积累隐藏菜单。
    menu.closedSignal.connect(menu._dispose_when_idle)
    return menu


def add_action(menu: RoundMenu, icon, text: str, slot, *,
               enabled: bool = True, shortcut: str = '') -> Action:
    """添加普通菜单项；icon 为 FluentIcon 或 None。"""
    action = Action(icon, text) if icon is not None else Action(text)
    action.setEnabled(bool(enabled))
    if shortcut:
        action.setShortcut(shortcut)
    action.triggered.connect(slot)
    menu.addAction(action)
    return action


def add_checkable_submenu(menu: RoundMenu, title: str, items,
                          current: str, on_chosen) -> RoundMenu:
    """添加单选风格子菜单：items 为名称列表，current 打勾。

    on_chosen(name) 在点选时回调。
    """
    submenu = SafeRoundMenu(title, menu)
    for name in items:
        action = Action(str(name))
        action.setCheckable(True)
        action.setChecked(str(name) == str(current))
        action.triggered.connect(
            lambda _checked=False, n=str(name): on_chosen(n))
        submenu.addAction(action)
    menu.addMenu(submenu)
    return submenu


__all__ = ['FIF', 'Action', 'RoundMenu', 'make_menu', 'add_action',
           'add_checkable_submenu']
