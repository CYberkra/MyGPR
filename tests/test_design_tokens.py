# -*- coding: utf-8 -*-
"""设计令牌层契约（design-ui-designer 令牌方法论在 Qt 侧的落地防线）。

守住三条：
1. 深浅两套令牌**键集必须一致**——漏一侧会让换主题后 KeyError 或回退默认值；
2. 语义色对比度实测达标（正文 ≥ 4.5:1，大字/非文本 ≥ 3:1）——skill 要求
   "没有数字就不能声称可访问"；
3. ``theme_helpers.control_palette`` 转发到令牌层（单源，不再第二份实现）。
"""
from __future__ import annotations

import pytest

from ui.design_tokens import (DURATION, RADIUS, color, duration, palette,
                              radius, rgba, tokens)

pytestmark = pytest.mark.filterwarnings('ignore::DeprecationWarning')


def _srgb(channel: float) -> float:
    c = channel / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(hex_color: str) -> float:
    value = hex_color.lstrip('#')
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _srgb(r) + 0.7152 * _srgb(g) + 0.0722 * _srgb(b)


def _contrast(fg: str, bg: str) -> float:
    l1, l2 = _luminance(fg), _luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


class TestTokenShape:
    def test_light_and_dark_share_keys(self):
        assert set(tokens(False)) == set(tokens(True))

    def test_primary_is_single_source(self):
        """强调色只有一处定义：与 qfluentwidgets 主题色 #3b82f6 同值。"""
        assert tokens(False)['primary'].lower() == '#3b82f6'
        # 深色取更高明度同色相（skill 暗色规则），不再是同一个值
        assert tokens(True)['primary'] != tokens(False)['primary']

    def test_status_colors_come_from_constants(self):
        """状态色由 constants.STATUS_COLORS 派生，不在令牌层重复定义。"""
        from ui import constants
        for key in ('success', 'warning', 'error', 'info'):
            assert tokens(False)[key] == constants.STATUS_COLORS[key][0]
            assert tokens(True)[key] == constants.STATUS_COLORS[key][1]

    def test_scales_are_on_system(self):
        assert RADIUS['md'] == 6 and RADIUS['pill'] > 100
        assert DURATION['fast'] == 150 and DURATION['slow'] == 500
        assert radius('md') == 6 and duration('normal') == 300


class TestContrastMeasured:
    """实测对比度（skill：没有数字就不能声称可访问）。"""

    def test_text_on_surface_light(self):
        t = tokens(False)
        assert _contrast(t['text_primary'], t['bg_surface']) >= 4.5
        assert _contrast(t['text_muted'], t['bg_surface']) >= 4.5
        assert _contrast(t['text_secondary'], t['bg_surface']) >= 4.5

    def test_text_on_surface_dark(self):
        t = tokens(True)
        assert _contrast(t['text_primary'], t['bg_surface']) >= 4.5
        assert _contrast(t['text_muted'], t['bg_surface']) >= 4.5

    def test_non_text_contrast(self):
        """描边 / 焦点环属非文本元素：≥ 3:1。"""
        for dark in (False, True):
            t = tokens(dark)
            assert _contrast(t['border_strong'], t['bg_surface']) >= 3.0
            assert _contrast(t['border_focus'], t['bg_surface']) >= 3.0


class TestHelpers:
    def test_rgba_output_shape(self):
        out = rgba('primary', 0.06)
        assert out.startswith('rgba(') and out.endswith(')')
        parts = out[5:-1].split(',')
        assert len(parts) == 4 and 0 <= int(parts[3]) <= 255

    def test_radius_and_duration_defaults(self):
        assert radius() == 6
        assert duration() == 300


class TestControlPaletteForwarding:
    """control_palette 的唯一实现已上移到 design_tokens.palette。"""

    def test_palette_matches_forwarded_api(self):
        from ui.theme_helpers import control_palette
        for dark in (False, True):
            assert control_palette(dark) == palette(dark)

    def test_palette_keys_stable(self):
        expected = {'plot_bg', 'surface', 'border', 'hover', 'button_bg',
                    'button_text', 'panel_bg', 'panel_border', 'text',
                    'table_base', 'selection', 'nav_line', 'nav_track'}
        for dark in (False, True):
            assert expected <= set(palette(dark))


class TestNoHardcodedAccentInWidgets:
    """回归锁：widget 层不得再出现强调色/中性灰的硬编码字面量。"""

    _FORBIDDEN = ('#8A8A85', '0,120,212', '#0078D4', '90,156,216', '#E0A83A')

    def test_widgets_free_of_forbidden_literals(self):
        from pathlib import Path
        repo = Path(__file__).resolve().parents[1]
        offenders = []
        for path in (repo / 'ui').rglob('*.py'):
            text = path.read_text(encoding='utf-8', errors='ignore')
            for literal in self._FORBIDDEN:
                if literal in text:
                    offenders.append(f'{path.name}: {literal}')
        assert not offenders, f'仍存在硬编码色值: {offenders}'
