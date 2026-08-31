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
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QStyleFactory
from qfluentwidgets import Theme, isDarkTheme, setTheme
from qfluentwidgets.components.widgets.combo_box import ComboBoxMenu

from ui import constants

_combo_patch_applied = False
_light_palette = None
_dark_palette = None


def _get_light_palette() -> QPalette:
    """浅色 palette：取 Fusion 标准 palette（与系统深浅模式无关，恒为浅色）。

    注意：``standardPalette()`` 的 resolveMask 为 0，Qt6 会把它视作
    "默认 palette"，在系统深色模式下自动替换成系统深色 palette。
    这里逐角色显式 ``setColor`` 填上 resolve 位，将其标记为应用自定义
    palette，防止被系统深色模式覆盖。
    """
    global _light_palette
    if _light_palette is None:
        style = QStyleFactory.create('fusion')
        p = QPalette(style.standardPalette())
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.WindowText,
                     QPalette.ColorRole.Base, QPalette.ColorRole.AlternateBase,
                     QPalette.ColorRole.Text, QPalette.ColorRole.Button,
                     QPalette.ColorRole.ButtonText, QPalette.ColorRole.ToolTipBase,
                     QPalette.ColorRole.ToolTipText,
                     QPalette.ColorRole.PlaceholderText,
                     QPalette.ColorRole.Highlight,
                     QPalette.ColorRole.HighlightedText, QPalette.ColorRole.Link):
            p.setColor(role, p.color(role))
        _light_palette = p
    return _light_palette


def _get_dark_palette() -> QPalette:
    """深色 palette：手工构造，保证系统浅色模式下深色主题依然正确。

    Qt6 在 Windows 深色系统模式下会自动给应用套深色 palette，导致浅色
    应用主题里原生控件（QTableWidget/QListWidget/QSpinBox 等）仍然深色；
    反之系统浅色模式 + 应用深色主题时原生控件又是浅色。显式 setPalette
    让原生控件始终跟随应用主题，与系统模式解耦。
    """
    global _dark_palette
    if _dark_palette is None:
        p = QPalette()
        window = QColor(45, 45, 45)
        base = QColor(30, 30, 30)
        text = QColor(230, 230, 230)
        disabled = QColor(127, 127, 127)
        highlight = QColor(0, 150, 136)
        p.setColor(QPalette.ColorRole.Window, window)
        p.setColor(QPalette.ColorRole.WindowText, text)
        p.setColor(QPalette.ColorRole.Base, base)
        p.setColor(QPalette.ColorRole.AlternateBase, window)
        p.setColor(QPalette.ColorRole.Text, text)
        p.setColor(QPalette.ColorRole.Button, window)
        p.setColor(QPalette.ColorRole.ButtonText, text)
        p.setColor(QPalette.ColorRole.ToolTipBase, base)
        p.setColor(QPalette.ColorRole.ToolTipText, text)
        p.setColor(QPalette.ColorRole.PlaceholderText, disabled)
        p.setColor(QPalette.ColorRole.Highlight, highlight)
        p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        p.setColor(QPalette.ColorRole.Link, highlight)
        p.setColor(QPalette.ColorGroup.Disabled,
                   QPalette.ColorRole.WindowText, disabled)
        p.setColor(QPalette.ColorGroup.Disabled,
                   QPalette.ColorRole.Text, disabled)
        p.setColor(QPalette.ColorGroup.Disabled,
                   QPalette.ColorRole.ButtonText, disabled)
        _dark_palette = p
    return _dark_palette


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


def native_views_qss(dark: bool) -> str:
    """原生控件（表格/列表/树/表头）随主题换肤的应用级 QSS。

    之所以用 QSS 而不是 palette：QApplication.setPalette 对已创建的
    item view 传播不可靠（实测深→浅往返后 QTableWidget 残留深色），
    而应用级 QSS 立即对所有现存及未来匹配的控件生效。
    qfluentwidgets 自有控件带控件级 QSS，同优先级下控件级优先，不受影响。
    """
    if dark:
        base, text, border = '#1e1e1e', '#e6e6e6', '#3c3c3c'
        header_bg, grid = '#2d2d2d', '#3c3c3c'
    else:
        base, text, border = '#ffffff', '#1a1a1a', '#d9d9d9'
        header_bg, grid = '#f5f5f5', '#e5e5e5'
    return (
        'QTableWidget, QTableView, QListWidget, QListView, QTreeWidget,'
        ' QTreeView {'
        f' background-color: {base}; color: {text};'
        f' border: 1px solid {border}; gridline-color: {grid};'
        ' selection-background-color: #009688; selection-color: #ffffff;'
        ' }'
        'QHeaderView::section {'
        f' background-color: {header_bg}; color: {text};'
        ' border: none;'
        f' border-right: 1px solid {grid}; border-bottom: 1px solid {grid};'
        ' padding: 4px;'
        ' }'
        'QTableCornerButton::section {'
        f' background-color: {header_bg}; border: none;'
        f' border-right: 1px solid {grid}; border-bottom: 1px solid {grid};'
        ' }'
    )


def apply_theme(theme: str) -> None:
    """应用主题：setTheme + pyqtgraph 背景同步（'k'/'w'）+ palette + 原生控件 QSS。

    :param theme: ``'浅色主题'`` / ``'深色主题'``（也接受 ``Theme`` 枚举）。
    """
    if isinstance(theme, Theme):
        dark = theme == Theme.DARK
    else:
        dark = str(theme) == constants.THEME_DARK
    setTheme(Theme.DARK if dark else Theme.LIGHT)
    pg.setConfigOption('background', 'k' if dark else 'w')
    pg.setConfigOption('foreground', 'w' if dark else 'k')
    # 原生控件（表格/列表/下拉框等）显式跟随应用主题，不受系统深浅模式影响
    app = QApplication.instance()
    if app is not None:
        app.setPalette(_get_dark_palette() if dark else _get_light_palette())
        app.setStyleSheet(native_views_qss(dark))


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
