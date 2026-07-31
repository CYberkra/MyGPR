# -*- coding: utf-8 -*-
"""统一右键菜单封装（qfluentwidgets RoundMenu）。

全项目右键菜单统一走这里：Fluent 风格 RoundMenu（主题自动跟随），
避免 pyqtgraph 原生英文菜单混风格。图形视图接入前需先
``vb.setMenuEnabled(False)`` 关闭 ViewBox 原生菜单。
"""
from __future__ import annotations

from qfluentwidgets import Action, RoundMenu
from qfluentwidgets import FluentIcon as FIF


def make_menu(parent=None) -> RoundMenu:
    """新建 RoundMenu（主题由 qfluentwidgets 自动跟随）。

    RoundMenu 签名是 (title='', parent=None)，parent 必须关键字传。
    """
    return RoundMenu(parent=parent)


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
    submenu = RoundMenu(title, menu)
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
