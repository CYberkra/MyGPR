# -*- coding: utf-8 -*-
"""ui.file_dialogs — 文件/目录对话框统一入口。

所有 UI 层文件对话框一律 import 本模块，不直接用 QFileDialog，
便于全局策略（路径规范化等）只改一处。

原生 vs 自绘策略（分两类，勿一刀切）：

- **目录选择**：用 Qt 自绘（``DontUseNativeDialog``）。原生
  ``IFileDialog`` 与无边框 ``FluentWindow`` 在 Win11 上有关闭时序冲突：
  shell 对话框销毁时 Windows 令无边框父窗口重走 DWM 合成，表现为
  「整窗界面消失再出现」的瞬间闪烁（用户报告"打开项目后像重启一下"）。
  自绘对话框是普通子窗口，关闭不触发该路径。
- **文件选择**（打开/保存）：保留原生——它是用户熟悉的系统界面，
  且文件对话框关闭时主窗口通常已被新数据接管，闪烁不明显。

历史说明：曾在此强制 ``DontUseNativeDialog``，因为当时误诊原生
IFileDialog 为 "<class 'OSError'> returned a result with an exception
set" 弹窗的元凶；根因实为 ``core.storage_primitives._pid_alive`` 在
Windows 上用 ``os.kill(pid, 0)`` 判活（死 PID 抛 SystemError 逃逸），
修复后原生对话框与无边框窗口的该冲突已解除，但不代表无重合成闪烁。

本模块同时规范化返回路径：Qt 对话框在 Windows 上返回正斜杠路径，
这里统一转回反斜杠，避免下游 Windows API / 显示层混用两种分隔符。
"""
from __future__ import annotations

import sys
from pathlib import PureWindowsPath

from PyQt6.QtWidgets import QFileDialog, QWidget

# 目录选择走 Qt 自绘，规避无边框窗口的 DWM 重合成闪烁（见模块 docstring）。
# 注意 options 默认值是 ShowDirsOnly 而非 0，必须显式并回，否则目录框可选中文件。
_DIR_OPTIONS = (QFileDialog.Option.DontUseNativeDialog
                | QFileDialog.Option.ShowDirsOnly)


def _normalize(path: str) -> str:
    """Windows 上把 Qt 返回的 'C:/a/b' 规范为 'C:\\a\\b'；其他平台原样。"""
    if not path:
        return path
    if sys.platform == 'win32':
        return str(PureWindowsPath(path))
    return path


def getExistingDirectory(parent: QWidget | None, caption: str,
                         directory: str) -> str:
    """选择目录。返回所选路径，取消返回空串。

    用自绘对话框（``DontUseNativeDialog``）：原生 IFileDialog 关闭会让
    无边框主窗口闪一下（见模块 docstring）。
    """
    return _normalize(QFileDialog.getExistingDirectory(
        parent, caption, directory, _DIR_OPTIONS))


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
