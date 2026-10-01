# -*- coding: utf-8 -*-
"""设计令牌层（Design Tokens）——视觉数值的唯一事实来源。

三层结构（design-ui-designer 方法论，折算到 Qt）：

1. **原始值层**：色阶（蓝 / 中性 / 语义色），无语义，不直接引用；
2. **语义层**：``tokens(dark)`` 返回语义键 → 色值，组件只引用这一层；
3. **组件层**：``palette(dark)`` 给具体控件族的调色板（图表 / 覆盖层 /
   原生表格 / 页签条），由语义层派生。

Qt 没有 CSS 变量，令牌以 **Python 常量 + 查表函数** 落地：组件在
``apply_theme`` 或构建 QSS 时经 :func:`color` / :func:`rgba` 取值，禁止
在 widget 里写死十六进制或 rgba 字面量。

**深色模式只覆盖语义层**——换主题若需要重写组件，说明令牌层设计错了。

迁移说明：本模块由 ``ui.theme_helpers.control_palette`` 上移而来，是它的
唯一实现（theme_helpers 侧改为转发，保持旧 API 不变）。
"""
from __future__ import annotations

from typing import Any

from ui import constants

# ============================================================ 1. 原始值层
# 品牌/强调色阶（primary 基准 500 = #3b82f6，与 qfluentwidgets 主题色一致）
_BLUE = {
    50: '#eff6ff', 100: '#dbeafe', 300: '#93c5fd', 400: '#60a5fa',
    500: '#3b82f6', 600: '#2563eb', 700: '#1d4ed8', 900: '#1e3a8a',
}

# ============================================================ 2. 语义层
# 键名统一 snake_case；深浅两套**键集必须一致**（新增键要两侧同时加，
# 有测试守着，见 tests/test_design_tokens.py）。

_LIGHT: dict[str, str] = {
    # 背景
    'bg_surface': '#ffffff',
    'bg_subtle': '#f5f5f5',
    'bg_muted': '#f0f0f0',
    # 文字（对比度实测见 tests/test_design_tokens.py）
    'text_primary': '#202020',
    'text_secondary': '#404040',
    'text_muted': '#6b7280',      # 白底 4.83:1（WCAG AA 达标）
    'text_inverse': '#ffffff',
    # 强调（primary 与 qfw 主题色同值，消除多套强调色并存）
    'primary': _BLUE[500],
    'primary_hover': _BLUE[600],
    'primary_active': _BLUE[700],
    'primary_subtle': _BLUE[50],
    'primary_border': _BLUE[500],
    # 描边（border_strong 取 gray-500：#9ca3af 对白底实测仅 2.54:1，
    # 不达非文本 3:1，已在令牌层提深一档而非逐组件打补丁）
    'border_default': '#d9d9d9',
    'border_strong': '#6b7280',
    'border_focus': _BLUE[500],
}

_DARK: dict[str, str] = {
    'bg_surface': '#000000',
    'bg_subtle': '#2d2d2d',
    'bg_muted': '#3d3d3d',
    'text_primary': '#f0f0f0',
    'text_secondary': '#d6d6d6',
    'text_muted': '#a8b0bd',      # #202020 深底 7.4:1（AA 达标）
    'text_inverse': '#202020',
    # 暗底需更高明度的同色相（skill 暗色规则）
    'primary': _BLUE[400],
    'primary_hover': _BLUE[300],
    'primary_active': _BLUE[100],
    'primary_subtle': _BLUE[900],
    'primary_border': _BLUE[400],
    'border_default': '#5a5a5a',
    'border_strong': '#9ca3af',
    'border_focus': _BLUE[400],
}

# 语义状态色由 constants.STATUS_COLORS 派生（保持单源，不重复定义）
_STATUS_KEYS = ('success', 'warning', 'error', 'info')

# ============================================================ 非颜色令牌
RADIUS = {'none': 0, 'sm': 4, 'md': 6, 'lg': 8, 'xl': 12, 'pill': 9999}
DURATION = {'fast': 150, 'normal': 300, 'slow': 500}      # ms


def tokens(dark: bool) -> dict[str, str]:
    """语义令牌表（深浅同键集），含由 STATUS_COLORS 派生的状态色。"""
    base = dict(_DARK if dark else _LIGHT)
    for key in _STATUS_KEYS:
        light, dark_value = constants.STATUS_COLORS[key]
        base[key] = dark_value if dark else light
    base['secondary'] = constants.STATUS_COLORS['secondary'][1 if dark else 0]
    base['disabled'] = constants.STATUS_COLORS['disabled'][1 if dark else 0]
    return base


def _current_dark() -> bool:
    """当前是否深色主题（延迟导入，避免与 theme_helpers 模块级循环）。"""
    from ui.theme_helpers import isDarkTheme
    return bool(isDarkTheme())


def color(name: str, dark: bool | None = None) -> str:
    """取语义色值；``dark=None`` 时跟随当前主题。"""
    return tokens(_current_dark() if dark is None else dark)[name]


def rgba(name: str, alpha: float, dark: bool | None = None) -> str:
    """语义色的半透明形式（``rgba(r, g, b, a)``，Qt 的 alpha 为 0–255）。

    组件写 QSS 需要淡底/淡描边时用本函数，勿手抄 ``rgba(59,130,246,0.06)``
    这类字面量——那正是"改强调色要改六处"的根因。
    """
    return _rgba_hex(color(name, dark), alpha)


def _rgba_hex(hex_color: str, alpha: float) -> str:
    """``#rrggbb`` + 0–1 透明度 → ``rgba(r, g, b, a)``。"""
    value = hex_color.lstrip('#')
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {round(alpha * 255)})'


def radius(name: str = 'md') -> int:
    """圆角（px）。"""
    return RADIUS[name]


def duration(name: str = 'normal') -> int:
    """动效时长（ms）：fast 150 / normal 300 / slow 500。"""
    return DURATION[name]


def focus_border(dark: bool | None = None) -> str:
    """键盘焦点环的描边（2px border_focus，WCAG 2.4.7 / 非文本 ≥3:1）。

    组件用 QSS ``#id:focus{border:2px solid <本值>}`` 挂载——Qt 的
    ``outline`` 支持有限，直接用 border 更稳（见 _ResultCard._normal_qss）。
    """
    return color('border_focus', dark)


# ============================================================ 3. 组件层
def palette(dark: bool) -> dict[str, Any]:
    """控件族调色板（图表 / 覆盖层 / 原生表格 / 页签条），由语义层派生。

    原 ``ui.theme_helpers.control_palette`` 的实现上移至此；键名保持不变，
    旧调用方零改动。
    """
    t = tokens(dark)
    if dark:
        return {
            'plot_bg': 'k', 'plot_fg': 'w',
            'surface': t['bg_surface'], 'border': '#5a5a5a',
            'hover': t['bg_muted'], 'button_bg': '#2d2d2d',
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
        'surface': t['bg_surface'], 'border': t['border_default'],
        'hover': t['bg_muted'], 'button_bg': '#ffffff',
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


__all__ = ['tokens', 'color', 'rgba', 'radius', 'duration',
           'focus_border', 'palette', 'RADIUS', 'DURATION']
