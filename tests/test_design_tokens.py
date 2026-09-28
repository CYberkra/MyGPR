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

pytest.importorskip("PyQt6")     # 后端 CI（无 Qt）自动跳过，见 tests/conftest.py
pytest.importorskip("pyqtgraph")

from ui.design_tokens import (  # noqa: E402  - 须在 importorskip 之后
    DURATION, RADIUS, color, duration, palette, radius, rgba, tokens)

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


class TestStatusContrastAA:
    """浅色主题状态文字对比度整改防线（原值实测仅 2.15–3.76:1）。"""

    def test_status_light_colors_meet_aa_on_white(self):
        from ui import constants
        for key in ('success', 'warning', 'error', 'info'):
            light = constants.STATUS_COLORS[key][0]
            assert _contrast(light, '#ffffff') >= 4.4, key

    def test_badge_light_text_meets_aa_on_tint(self):
        from ui import constants
        for key in ('success', 'warning', 'error', 'info'):
            fg, bg = constants.BADGE_COLOR_SETS[key][0]
            assert _contrast(fg, bg) >= 4.4, key

    def test_click_targets_not_tiny(self):
        """图标钮 ≥ 24px（卡 hover 钮 / chip 添加钮）；chip 内小钮 ≥ 18px。"""
        from pathlib import Path
        repo = Path(__file__).resolve().parents[1]
        grid = (repo / 'ui/widgets/bscan_result_grid.py').read_text(
            encoding='utf-8')
        assert 'setFixedSize(24, 24)' in grid
        assert 'setFixedSize(20, 20)' not in grid
        strip = (repo / 'ui/widgets/chain_strip.py').read_text(
            encoding='utf-8')
        assert 'setFixedSize(12, 12)' not in strip
        assert 'setFixedSize(14, 14)' not in strip
        assert 'setFixedSize(18, 18)' in strip
        assert 'setFixedSize(24, 24)' in strip


class TestComponentStateMatrix:
    """八状态矩阵落地防线：hover 态存在且色值走令牌（docs/design）。"""

    def test_result_card_normal_qss_contains_hover(self):
        from ui.widgets.bscan_result_grid import _ResultCard
        qss = _ResultCard._normal_qss()
        assert ':hover' in qss
        assert 'transparent' in qss          # 常态无描边

    def test_chip_hover_only_when_enabled(self):
        from ui.widgets.chain_strip import _chip_qss
        assert ':hover' in _chip_qss(True)
        # 禁用态不给可点性预示（无 hover 反馈）
        assert ':hover' not in _chip_qss(False)
        assert 'dashed' in _chip_qss(False)  # 虚线 = 禁用形态

    def test_hover_uses_primary_token_not_literal(self):
        """hover 描边必须是 primary 派生 rgba，不允许写死旧色值。"""
        from ui.design_tokens import tokens
        from ui.widgets.bscan_result_grid import _ResultCard
        from ui.widgets.chain_strip import _chip_qss
        primary = tokens(False)['primary'].lstrip('#')
        rgb = ', '.join(str(int(primary[i:i + 2], 16)) for i in (0, 2, 4))
        assert rgb in _ResultCard._normal_qss()
        assert rgb in _chip_qss(True)


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
