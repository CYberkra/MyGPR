# -*- coding: utf-8 -*-
"""ui.file_dialogs — 强制 Qt 风格文件/目录对话框（禁用原生 IFileDialog）。

为什么存在：PyQt6-Fluent-Widgets 的无边框窗口（frameless + Win32 钩子）
与 Windows 原生 IFileDialog 组合时，对话框打开瞬间会触发
``OSError returned a result with an exception set`` 甚至无声退出
（用户实测：主页"新建项目/打开项目"必现；Linux offscreen CI 亦有
同族段错误）。根因是原生对话框挂在 Win32 owner 钩子上与 qfw 的
消息钩子互相干扰。

方案：QFileDialog 全静态方法的影子包装，统一注入
``DontUseNativeDialog``——Qt 内部实现走 QWidget 绘制，与无边框窗口
完全兼容。所有 UI 层文件对话框一律 import 本模块，不直接用
QFileDialog。
"""
from __future__ import annotations

from PyQt6.QtWidgets import QFileDialog, QWidget

_OPTIONS = QFileDialog.Option.DontUseNativeDialog


def getExistingDirectory(parent: QWidget | None, caption: str,
                         directory: str) -> str:
    """选择目录（非原生）。返回所选路径，取消返回空串。"""
    return QFileDialog.getExistingDirectory(
        parent, caption, directory, options=_OPTIONS)


def getOpenFileName(parent: QWidget | None, caption: str, directory: str,
                    file_filter: str = '') -> str:
    """选择单个文件（非原生）。返回 (路径, 过滤器)，取消路径为空串。"""
    path, selected = QFileDialog.getOpenFileName(
        parent, caption, directory, file_filter, options=_OPTIONS)
    return path, selected


def getSaveFileName(parent: QWidget | None, caption: str, directory: str,
                    file_filter: str = '') -> str:
    """保存文件（非原生）。返回 (路径, 过滤器)，取消路径为空串。"""
    path, selected = QFileDialog.getSaveFileName(
        parent, caption, directory, file_filter, options=_OPTIONS)
    return path, selected
