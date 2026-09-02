# -*- coding: utf-8 -*-
"""ui.file_dialogs — 文件/目录对话框统一入口。

所有 UI 层文件对话框一律 import 本模块，不直接用 QFileDialog，
便于全局策略（路径规范化等）只改一处。

历史说明：曾在此强制 ``DontUseNativeDialog``，因为当时误诊原生
IFileDialog 为 "<class 'OSError'> returned a result with an exception
set" 弹窗的元凶；根因实为 ``core.storage_primitives._pid_alive`` 在
Windows 上用 ``os.kill(pid, 0)`` 判活（死 PID 抛 SystemError 逃逸），
修复后原生对话框与无边框窗口并无冲突，已恢复默认原生样式（更美观）。

本模块同时规范化返回路径：Qt 对话框在 Windows 上返回正斜杠路径，
这里统一转回反斜杠，避免下游 Windows API / 显示层混用两种分隔符。
"""
from __future__ import annotations

import sys
from pathlib import PureWindowsPath

from PyQt6.QtWidgets import QFileDialog, QWidget


def _normalize(path: str) -> str:
    """Windows 上把 Qt 返回的 'C:/a/b' 规范为 'C:\\a\\b'；其他平台原样。"""
    if not path:
        return path
    if sys.platform == 'win32':
        return str(PureWindowsPath(path))
    return path


def getExistingDirectory(parent: QWidget | None, caption: str,
                         directory: str) -> str:
    """选择目录。返回所选路径，取消返回空串。"""
    return _normalize(QFileDialog.getExistingDirectory(
        parent, caption, directory))


def getOpenFileName(parent: QWidget | None, caption: str, directory: str,
                    file_filter: str = '') -> tuple[str, str]:
    """选择单个文件。返回 (路径, 过滤器)，取消路径为空串。"""
    path, selected = QFileDialog.getOpenFileName(
        parent, caption, directory, file_filter)
    return _normalize(path), selected


def getSaveFileName(parent: QWidget | None, caption: str, directory: str,
                    file_filter: str = '') -> tuple[str, str]:
    """保存文件。返回 (路径, 过滤器)，取消路径为空串。"""
    path, selected = QFileDialog.getSaveFileName(
        parent, caption, directory, file_filter)
    return _normalize(path), selected
