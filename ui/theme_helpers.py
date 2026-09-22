# -*- coding: utf-8 -*-
"""主题辅助与 qfluentwidgets 猴补丁（复刻 style_spec §2.9，模块导入即生效）。

- ``apply_theme()``：全局深浅主题切换 + pyqtgraph 同步换肤；
- ``log_panel_qss()``：日志面板三套配色 QSS（style_spec §1.2/§2.5 精确值）；
- ComboBoxMenu 猴补丁：修复下拉菜单「透明边框」——
  ``hBoxLayout.setContentsMargins(0,0,0,0)`` 去边距、
  ``view.setGraphicsEffect(None)`` 去阴影、
  给 ``#comboListWidget`` 加实色 1px 边框（深色 ``rgb(100,100,100)`` / 浅色 ``rgb(200,200,200)``）。
"""
import gc
from typing import TYPE_CHECKING

import pyqtgraph as pg
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication, QStyleFactory
from qfluentwidgets import Theme, isDarkTheme, setTheme
from qfluentwidgets.components.widgets.combo_box import ComboBoxMenu

from ui import constants

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QLabel   # 仅 make_badge 返回注解用

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
        highlight = QColor(constants.ACCENT_SOLID)
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


def control_palette(dark: bool) -> dict:
    """主题感知控件调色板（按钮/工具条/覆盖层/表格底色单源）。

    收敛自 bscan_view 维护的第二套深浅调色板（surface/border/hover/
    button_bg/button_text + 地图浮动覆盖层面板色），并把
    :func:`native_views_qss` 的表格配色并入同一来源。键：

    - 图表/工具条：``plot_bg``/``plot_fg``/``surface``/``border``/
      ``hover``/``button_bg``/``button_text``；
    - 浮动覆盖层（地图图例/缩放面板/坐标读出）：``panel_bg``/
      ``panel_border``/``text``；
    - 原生 item view（QTableWidget 等 QSS）：``table_base``/``table_text``/
      ``table_border``/``table_header_bg``/``table_grid``/``selection``；
    - 顶部页签条（主窗口）：``nav_line``/``nav_track``。
    """
    if dark:
        return {
            'plot_bg': 'k', 'plot_fg': 'w',
            'surface': '#000000', 'border': '#5a5a5a',
            'hover': '#3d3d3d', 'button_bg': '#2d2d2d',
            'button_text': '#f0f0f0',
            'panel_bg': 'rgba(32,32,32,200)',
            'panel_border': 'rgba(255,255,255,45)',
            'text': '#f0f0f0',
            'table_base': '#1e1e1e', 'table_text': '#e6e6e6',
            'table_border': '#3c3c3c', 'table_header_bg': '#2d2d2d',
            'table_grid': '#3c3c3c', 'selection': constants.ACCENT_SOLID,
            'nav_line': 'rgba(255, 255, 255, 0.10)',
            'nav_track': 'rgba(255, 255, 255, 0.06)',
        }
    return {
        'plot_bg': 'w', 'plot_fg': 'k',
        'surface': '#ffffff', 'border': '#d9d9d9',
        'hover': '#f0f0f0', 'button_bg': '#ffffff',
        'button_text': '#202020',
        'panel_bg': 'rgba(255,255,255,220)',
        'panel_border': 'rgba(0,0,0,45)',
        'text': '#202020',
        'table_base': '#ffffff', 'table_text': '#1a1a1a',
        'table_border': '#d9d9d9', 'table_header_bg': '#f5f5f5',
        'table_grid': '#e5e5e5', 'selection': constants.ACCENT_SOLID,
        'nav_line': 'rgba(0, 0, 0, 0.07)',
        'nav_track': 'rgba(0, 0, 0, 0.05)',
    }


def status_color(key: str) -> str:
    """语义状态文字色（随主题查表）：success/warning/error/info/secondary/disabled。

    新代码置文字色一律走本函数，勿直接用 constants.COLOR_* 裸常量
    （深色主题下固定浅灰/浅色值对比度不足或刺眼）。hint/次要说明用
    ``'secondary'``（WCAG AA 达标），``'disabled'`` 仅限真禁用控件。
    """
    light, dark = constants.STATUS_COLORS[key]
    return dark if isDarkTheme() else light


def accent_color() -> str:
    """强调色（随主题）：选中态文字/线条/链接用；实色块底用 ACCENT_SOLID。"""
    light, dark = constants.ACCENT
    return dark if isDarkTheme() else light


def hint_qss(key: str = 'secondary') -> str:
    """辅助说明文字 QSS 单源：语义色 + SECONDARY 字号（随主题查表）。

    供 :func:`ui.page_scaffold.make_hint` 与运行期动态改色的 hint 使用；
    主题切换后须重设（HintLabel.apply_theme 已自动处理）。
    """
    return (f'color: {status_color(key)}; '
            f'font-size: {constants.FONT_SIZE_SECONDARY}pt;')


def ui_font(size_pt: int, weight: 'QFont.Weight | None' = None) -> QFont:
    """带回退链的 UI 字体工厂（P2-5）。

    QFont 单值构造只认 ``FONT_FAMILY`` 首选族，Windows 缺该族时直接落到
    Qt 自动替换；显式 ``setFamilies`` 让 Qt 按 token 栈逐族匹配。新代码
    一律经本工厂创建字体，勿再手写 ``QFont(constants.FONT_FAMILY, ...)``。
    """
    font = QFont(constants.FONT_FAMILY, size_pt)
    font.setFamilies(list(constants.FONT_FAMILY_STACK))
    if weight is not None:
        font.setWeight(weight)
    return font


def font_families_qss() -> str:
    """QSS ``font-family`` 值单源：token 栈加引号 + ``sans-serif`` 关键字兜底。

    CSS 关键字不加引号；族名含空格必须带引号。dock_panel 等手写 QSS
    的场景一律引用本函数，勿再写死单族。
    """
    quoted = ', '.join(f'"{name}"' for name in constants.FONT_FAMILY_STACK)
    return f'{quoted}, sans-serif'


def badge_colors(key: str) -> tuple:
    """药丸徽章配色 ``(文字色, 底色)``（随主题查表）。

    key 为语义键（success/warning/error/info/neutral）或 RTK 定位状态
    中文（经 constants.BADGE_STATUS_KEYS 映射）。浅色 = 彩字淡底；
    深色 = 白字彩底（淡底在深底上呈刺眼白块，参照任务中心徽章做法）。
    """
    key = constants.BADGE_STATUS_KEYS.get(key, key)
    light, dark = constants.BADGE_COLOR_SETS[key]
    return dark if isDarkTheme() else light


# 药丸徽章 QSS 模板（单源，两种变体）：
# - BADGE_QSS：白字 + 实色彩底（任务中心/方法浏览器；单 %s 占位底色，
#   ui.motion.animate_badge_color 依赖该单占位做正则回填，勿改双占位）；
# - BADGE_QSS_PAIR：彩字 + 淡彩底（随 badge_colors 色对，浅色主题用）。
# 字号统一 FONT_SIZE_SECONDARY（9pt = 原 12px 等值）。
_BADGE_QSS_BODY = ('QLabel { padding: 2px 10px; border-radius: 10px; '
                   f'font-size: {constants.FONT_SIZE_SECONDARY}pt; '
                   'font-weight: bold; ')
BADGE_QSS = _BADGE_QSS_BODY + 'color: #ffffff; background-color: %s; }'
BADGE_QSS_PAIR = _BADGE_QSS_BODY + 'color: %s; background-color: %s; }'


def badge_qss(bg: str) -> str:
    """药丸徽章样式（白字 + 指定实色底，单占位模板）。"""
    return BADGE_QSS % bg


def badge_qss_pair(text_color: str, bg: str) -> str:
    """药丸徽章样式（彩字淡底变体，配 badge_colors() 色对）。"""
    return BADGE_QSS_PAIR % (text_color, bg)


def make_badge(text: str, color_key: str) -> 'QLabel':
    """状态药丸徽章（浅色彩字淡底 / 深色白字彩底，随主题查表）。"""
    from PyQt6.QtWidgets import QLabel
    badge = QLabel(text)
    badge.setStyleSheet(badge_qss_pair(*badge_colors(color_key)))
    return badge


def native_views_qss(dark: bool) -> str:
    """原生控件（表格/列表/树/表头）随主题换肤的应用级 QSS。

    之所以用 QSS 而不是 palette：QApplication.setPalette 对已创建的
    item view 传播不可靠（实测深→浅往返后 QTableWidget 残留深色），
    而应用级 QSS 立即对所有现存及未来匹配的控件生效。
    qfluentwidgets 自有控件带控件级 QSS，同优先级下控件级优先，不受影响。
    配色单源：control_palette(dark)['table_*']。
    """
    palette = control_palette(dark)
    base = palette['table_base']
    text = palette['table_text']
    border = palette['table_border']
    header_bg = palette['table_header_bg']
    grid = palette['table_grid']
    selection = palette['selection']
    return (
        'QTableWidget, QTableView, QListWidget, QListView, QTreeWidget,'
        ' QTreeView {'
        f' background-color: {base}; color: {text};'
        f' border: 1px solid {border}; gridline-color: {grid};'
        f' selection-background-color: {selection}; selection-color: #ffffff;'
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


# 已应用主题状态（None = 尚未应用）。启动链 main() 在窗口树为空时先
# apply_theme 一次（palette/QSS 成本极低），MyGPRMainWindow._init_state
# 回放同主题时经此状态短路——旧行为重复 setTheme + app.setStyleSheet 会
# 对已建好的几百个控件 re-polish（实测 ~1.1s，是启动耗时大头之一）。
_applied_dark: bool | None = None


def apply_theme(theme: str) -> None:
    """应用主题：setTheme + pyqtgraph 背景同步（'k'/'w'）+ palette + 原生控件 QSS。

    幂等：目标主题与已应用主题一致、**且 qfluentwidgets 当前实际主题也一致**
    时直接返回（重复的全局样式重算对大控件树是秒级开销）。需要强制重放时
    先置 ``theme_helpers._applied_dark = None``。

    为什么短路要同时校验 ``isDarkTheme()``：``_applied_dark`` 只记录「本函数
    上次把主题设成了什么」，无法反映外部对主题的改动。第三方代码（或历史
    脚本）直接调 ``qfluentwidgets.setTheme()`` 后，``_applied_dark`` 会与实际
    状态脱节；此时若只信 ``_applied_dark``，本函数会误判为「已经是目标主题」
    而跳过，导致 palette / pyqtgraph 背景 / 原生控件 QSS 全部停留在旧主题。
    这是一种静默失效——界面看起来"应用了主题"，实际三层基础设施没跟上。
    """
    global _applied_dark
    if isinstance(theme, Theme):
        dark = theme == Theme.DARK
    else:
        dark = str(theme) == constants.THEME_DARK
    if _applied_dark is not None and dark == _applied_dark \
            and isDarkTheme() == dark:
        return
    # 先收一次垃圾：qfluentwidgets 的 updateStyleSheet 用
    # `list(styleSheetManager.items())` 遍历内部 WeakKeyDictionary，而
    # `items()` 是直通该字典的迭代器视图——若在展开它的过程中某个已销毁
    # 控件的弱引用键恰好被 GC 回收，就会抛
    #   RuntimeError: dictionary changed size during iteration
    # （CPython 3.13 上更易命中，CI gui-linux-offscreen 全量跑时偶发，
    #  见 tests/test_main_window_titlebar.py 的 setup 报错）。
    # 这里主动 gc.collect() 把"已死未回收"的键提前清掉，让库的遍历过程
    # 不再因 GC 抖动而改字典；本函数本就是秒级重算路径，一次 collect 的
    # 开销可忽略。
    gc.collect()
    setTheme(Theme.DARK if dark else Theme.LIGHT)
    pg.setConfigOption('background', 'k' if dark else 'w')
    pg.setConfigOption('foreground', 'w' if dark else 'k')
    # 抗锯齿（pyqtgraph 默认关闭）：曲线与文字边缘明显更平滑。该 hint 作用于
    # GraphicsView 的矢量绘制，B-Scan 等 ImageItem 走图像绘制路径不受影响，
    # 但多 GB 真实数据下的实际帧率仍需在 Windows 目标机验收。
    pg.setConfigOption('antialias', True)
    # 原生控件（表格/列表/下拉框等）显式跟随应用主题，不受系统深浅模式影响
    app = QApplication.instance()
    if app is not None:
        app.setPalette(_get_dark_palette() if dark else _get_light_palette())
        app.setStyleSheet(native_views_qss(dark))
    _applied_dark = dark


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
        f' font-size: {constants.FONT_SIZE_SECONDARY}pt;'
        ' }'
    )


# 模块导入即生效（style_spec §2.9 / §7.4-9）
patch_combo_box_menu()
