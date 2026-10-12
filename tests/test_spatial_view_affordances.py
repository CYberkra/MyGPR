# -*- coding: utf-8 -*-
"""空间页「附属件按视图显隐 + 三维卡折叠」回归测试。

背景（2026-10-06，1600×950 实测量化）：
    空间页左栏 320×902 的垂直构成里，「三维显示」卡 260px = **28.8%**，而它
    的 6 个控件在 ``_connect_internal`` 里全部只连 ``self._3d_view``
    （set_vertical_exaggeration / set_track_drape / set_imagery_enabled /
    地形来源 / 导入 DEM / 清除 DEM）——切到平面地图、高程剖面、深度切片时
    一行都不生效，却仍然占着位置。同理，中栏底部的深度切片控制行只操作
    ``_depth_view`` 的等值线，在平面地图下拖动屏幕不会有任何变化。

本测试钉住三个契约：
    ① 深度切片行仅在「深度切片」段可见，其余三段隐藏；
    ② 三维显示卡默认收起，切到三维视图时自动展开，切走**不**自动收起
       （手动展开查看是主动行为，替他收掉会打断正在做的调参）；
    ③ 折叠不丢控件状态（DoubleSpinBox 的值 / SwitchButton 的开关态）——
       实现走 ``setVisible`` 而非重建控件，这条是防"顺手改成重建"的红线；
    ④ 右栏两卡按内容贴顶，空白由末尾 stretch 归到栏底（不再各占一半）。
"""
import time

import pytest

pytest.importorskip('PyQt6')

_SEG_MAP = 'planMap'
_SEG_PROFILE = 'elevationProfile'
_SEG_3D = 'trajectory3d'
_SEG_DEPTH = 'depthSlice'


@pytest.fixture
def spatial(qapp):
    from ui.pages.spatial_page import SpatialPage

    page = SpatialPage()
    page.resize(1600, 950)
    page.show()
    settle(qapp)
    yield page
    page.deleteLater()
    qapp.processEvents()


def settle(qapp, wait_s: float = 0.35) -> None:
    """离屏下动画不推进——折叠动画必须用 animate=False 或显式等够帧。

    沿用本项目既有教训：离屏里 ``set_collapsed(True)`` 走动画版会读到中间态，
    产生假失败（见 tests/test_spatial_deferred_redraw.py 注释）。
    """
    end = time.perf_counter() + wait_s
    while time.perf_counter() < end:
        qapp.processEvents()


class TestDepthRowVisibility:
    def test_hidden_on_non_depth_segments(self, spatial, qapp):
        """非深度切片段 → 深度行隐藏。"""
        for key in (_SEG_MAP, _SEG_PROFILE, _SEG_3D):
            spatial._switch_view(key)
            settle(qapp, 0.15)
            assert not spatial._depth_row_widget.isVisible(), (
                f'{key} 段不应显示深度切片控制行')

    def test_visible_on_depth_segment(self, spatial, qapp):
        """深度切片段 → 深度行可见。"""
        spatial._switch_view(_SEG_DEPTH)
        settle(qapp, 0.15)
        assert spatial._depth_row_widget.isVisible()

    def test_initial_state_hidden(self, spatial):
        """初始态（默认平面地图段）就不占那一行。"""
        assert not spatial._depth_row_widget.isVisible()

    def test_toggle_is_idempotent(self, spatial, qapp):
        """重复切到同一段不改变显隐（``setVisible`` 幂等）。"""
        spatial._switch_view(_SEG_DEPTH)
        settle(qapp, 0.15)
        spatial._switch_view(_SEG_DEPTH)
        settle(qapp, 0.15)
        assert spatial._depth_row_widget.isVisible()


class TestView3dCardCollapse:
    def test_default_collapsed(self, spatial):
        """默认收起——这正是省下 28.8% 左栏高度的关键。"""
        assert spatial._view3d_card.is_collapsed()

    def test_auto_expands_on_3d_segment(self, spatial, qapp):
        """切到三维视图 → 自动展开（否则用户得先意识到"要展开才能调"）。"""
        assert spatial._view3d_card.is_collapsed()
        spatial._switch_view(_SEG_3D)
        settle(qapp, 0.3)
        assert not spatial._view3d_card.is_collapsed()

    def test_not_auto_collapsed_when_leaving_3d(self, spatial, qapp):
        """切走**不**自动收起：手动展开查看是主动行为。"""
        spatial._switch_view(_SEG_3D)
        settle(qapp, 0.3)
        spatial._switch_view(_SEG_MAP)
        settle(qapp, 0.3)
        assert not spatial._view3d_card.is_collapsed()

    def test_collapse_preserves_control_state(self, spatial, qapp):
        """折叠往返不丢控件状态（红线：防"改成重建控件"）。"""
        spatial._view3d_card.set_collapsed(False, animate=False)
        settle(qapp, 0.3)
        spatial._3d_exag_spin.setValue(2.5)
        spatial._3d_imagery_switch.setChecked(False)
        spatial._3d_drape_switch.setChecked(True)

        spatial._view3d_card.set_collapsed(True, animate=False)
        settle(qapp, 0.3)
        spatial._view3d_card.set_collapsed(False, animate=False)
        settle(qapp, 0.3)

        assert spatial._3d_exag_spin.value() == 2.5
        assert spatial._3d_imagery_switch.isChecked() is False
        assert spatial._3d_drape_switch.isChecked() is True

    def test_collapsed_shrinks_card(self, spatial, qapp):
        """折叠确实省高度（不是只改个isVisible 而布局没变）。"""
        spatial._view3d_card.set_collapsed(False, animate=False)
        settle(qapp, 0.3)
        expanded = spatial._view3d_card.height()
        spatial._view3d_card.set_collapsed(True, animate=False)
        settle(qapp, 0.3)
        collapsed = spatial._view3d_card.height()
        assert collapsed < expanded, (
            f'折叠后应更矮：{collapsed} vs {expanded}')

    def test_toggle_button_flips_tooltip(self, spatial, qapp):
        """chevron 按钮的 tooltip 随状态翻转（唯一可见的展开入口）。"""
        assert spatial._view3d_card._toggle_btn.toolTip() == '展开'
        spatial._view3d_card.set_collapsed(False, animate=False)
        settle(qapp, 0.3)
        assert spatial._view3d_card._toggle_btn.toolTip() == '收起'


class TestRightColumnTopAligned:
    def test_cards_not_stretched_to_half(self, spatial, qapp):
        """两卡按内容贴顶——不再各占一半高度。"""
        from qfluentwidgets import CardWidget

        content = spatial._right_panel.content_widget()
        cards = [w for w in content.findChildren(CardWidget)
                 if w.parentWidget() is not None and w.isVisible()]
        assert len(cards) == 2, f'应有两张卡，实际 {len(cards)}'

        right_h = spatial._right_panel.height()
        for card in cards:
            assert card.height() < right_h * 0.5, (
                f'卡高 {card.height()} 达到右栏 {right_h} 的一半，'
                f'说明仍被 stretch 平分')

    def test_blank_collects_at_bottom(self, spatial, qapp):
        """空白集中在栏底：第二张卡底部到栏底有大块连续留白。"""
        from qfluentwidgets import CardWidget

        content = spatial._right_panel.content_widget()
        cards = sorted(
            (w for w in content.findChildren(CardWidget)
             if w.parentWidget() is not None and w.isVisible()),
            key=lambda w: w.y())
        bottom_card = cards[-1]
        gap = (spatial._right_panel.content_widget().height()
               - (bottom_card.y() + bottom_card.height()))
        assert gap > spatial._right_panel.height() * 0.3, (
            f'栏底留白应占三成以上，实际 {gap}px / '
            f'{spatial._right_panel.height()}px')
