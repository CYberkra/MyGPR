# -*- coding: utf-8 -*-
"""主题辅助与 qfluentwidgets 猴补丁（复刻 style_spec §2.9，模块导入即生效）。

- ``apply_theme()``：全局深浅主题切换 + pyqtgraph 同步换肤；
- ``log_panel_qss()``：日志面板三套配色 QSS（style_spec §1.2/§2.5 精确值）；
- ComboBoxMenu 猴补丁：修复下拉菜单「透明边框」——
  ``hBoxLayout.setContentsMargins(0,0,0,0)`` 去边距、
  ``view.setGraphicsEffect(None)`` 去阴影、
  给 ``#comboListWidget`` 加实色 1px 边框（深色 ``rgb(100,100,100)`` / 浅色 ``rgb(200,200,200)``）。
"""
import pyqtgraph as pg
from qfluentwidgets import Theme, isDarkTheme, setTheme
from qfluentwidgets.components.widgets.combo_box import ComboBoxMenu

from ui import constants

_combo_patch_applied = False


def patch_combo_box_menu() -> None:
    """ComboBoxMenu 透明边框修复（幂等，模块导入即调用一次）。"""
    global _combo_patch_applied
    if _combo_patch_applied:
        return
    _combo_patch_applied = True

    _original_init = ComboBoxMenu.__init__

    def _patched_init(self, parent=None):
        _original_init(self, parent)
        # 去边距（避免透明缝）
        self.hBoxLayout.setContentsMargins(0, 0, 0, 0)
        # 去阴影（阴影区域在无边距时会透出底色）
        self.view.setGraphicsEffect(None)
        # 实色 1px 边框（深色 rgb(100,100,100) / 浅色 rgb(200,200,200)）
        border = 'rgb(100,100,100)' if isDarkTheme() else 'rgb(200,200,200)'
        self.view.setStyleSheet(
            f'#comboListWidget {{ border: 1px solid {border}; }}')

    ComboBoxMenu.__init__ = _patched_init


def apply_theme(theme: str) -> None:
    """应用主题：setTheme + pyqtgraph 背景同步（'k'/'w'）。

    :param theme: ``'浅色主题'`` / ``'深色主题'``（也接受 ``Theme`` 枚举）。
    """
    if isinstance(theme, Theme):
        dark = theme == Theme.DARK
    else:
        dark = str(theme) == constants.THEME_DARK
    setTheme(Theme.DARK if dark else Theme.LIGHT)
    pg.setConfigOption('background', 'k' if dark else 'w')
    pg.setConfigOption('foreground', 'w' if dark else 'k')


def log_panel_qss(variant: str = 'terminal') -> str:
    """日志面板 QSS（style_spec §2.5 逐字值）。

    :param variant: ``'terminal'`` 初始深底 / ``'dark'`` 深色主题 / ``'light'`` 浅色主题。
    """
    bg, fg, border = {
        'terminal': constants.LOG_QSS_TERMINAL,   # #2b2b2b / #e0e0e0 / #404040
        'dark': constants.LOG_QSS_DARK,           # #1e1e1e / #e0e0e0 / #333
        'light': constants.LOG_QSS_LIGHT,         # #f5f5f5 / #333 / #ddd
    }[variant]
    return (
        'QTextEdit {'
        f' background-color: {bg};'
        f' color: {fg};'
        f' border: 1px solid {border};'
        ' border-radius: 4px;'
        ' padding: 5px;'
        " font-family: 'Consolas', 'Courier New', monospace;"
        ' font-size: 11px;'
        ' }'
    )


# 模块导入即生效（style_spec §2.9 / §7.4-9）
patch_combo_box_menu()
