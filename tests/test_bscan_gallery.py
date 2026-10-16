# -*- coding: utf-8 -*-
"""BScanGallery 契约（P1-4 重写）：动态列数 1/2/3/4 档、真实缩略图 +
步序徽章 + 名称、当前卡描边 + 角标、hover 无悬浮按钮、单击设当前并关闭、
空状态居中、错峰淡入、Esc。独立组件保留；2026-10-16 处理页取消结果墙。
"""
from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QKeyEvent  # noqa: E402
from PyQt6.QtWidgets import QToolButton  # noqa: E402

from ui.widgets.bscan_gallery import BScanGallery, _GalleryCard  # noqa: E402


def _sources(n: int, *, bundle=True) -> list:
    out = []
    for i in range(n):
        out.append({'key': f'k{i}', 'title': f'源{i}',
                    'bundle': _bundle() if bundle else None})
    return out


def _bundle():
    import numpy as np
    from types import SimpleNamespace

    return SimpleNamespace(
        matrix=np.full((8, 6), 1.0, dtype=np.float32),
        vmin=0.0, vmax=1.0, title='b', x_label='道数', y_label='采样点',
        trace_axis_m=None, sample_axis=None, sample_axis_label='',
        trace_count=6, sample_count=8, trace_elevation_m=None,
        depth_axis_m=None)


@pytest.fixture
def gallery(qapp):
    w = BScanGallery()
    yield w
    w.close()
    w.deleteLater()


def _card_positions(gallery) -> list:
    """每张卡所在 (row, col)。"""
    out = []
    for card in gallery.cards():
        index = gallery._grid.indexOf(card)
        out.append(gallery._grid.getItemPosition(index)[:2])
    return out


def _press_escape(widget) -> None:
    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape,
                      Qt.KeyboardModifier.NoModifier)
    widget.keyPressEvent(event)


class TestDynamicColumns:
    @pytest.mark.parametrize('n,cols', [(1, 1), (2, 2), (3, 3), (4, 4),
                                        (5, 4), (8, 4), (12, 4)])
    def test_columns_for_matrix(self, gallery, n, cols):
        assert BScanGallery._columns_for(n) == cols

    def test_grid_geometry_follows_columns(self, gallery):
        gallery.set_sources(_sources(5), 'k0')
        assert _card_positions(gallery) == [
            (0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]

    def test_columns_shrink_without_stale_stretch(self, gallery):
        """列数回落（4 档 → 1 档）后空列不残留 stretch（结果网格教训）。"""
        gallery.set_sources(_sources(8), 'k0')
        gallery.set_sources(_sources(1), 'k0')
        assert _card_positions(gallery) == [(0, 0)]
        assert gallery._grid.columnStretch(0) == 1   # 唯一使用列
        for col in range(1, 4):
            assert gallery._grid.columnStretch(col) == 0   # 空列零残留


class TestCardContent:
    def test_card_has_view_badge_and_name(self, gallery):
        gallery.set_sources(_sources(3), 'k0')
        cards = gallery.cards()
        assert all(isinstance(c, _GalleryCard) for c in cards)
        assert all(c._view is not None for c in cards)          # 真实缩略图
        assert [c._badge.text() for c in cards] == ['输入', '1', '2']
        assert [c._name.text() for c in cards] == ['源0', '源1', '源2']

    def test_card_has_no_hover_buttons(self, gallery, qapp):
        """hover 仅淡描边：画廊卡不新增任何悬浮按钮（PRD ③ 边界；视图
        自带的全屏钮属 BScanView 既有 chrome，与结果网格同款，不算）。"""
        gallery.set_sources(_sources(2), 'k0')
        gallery.show()
        qapp.processEvents()
        for card in gallery.cards():
            buttons = [w for w in card.findChildren(QToolButton)
                       if w is not card._view._fullscreen_btn]
            assert buttons == [], f'卡内不应有悬浮按钮：{buttons}'

    def test_bundle_none_shows_placeholder(self, gallery):
        gallery.set_sources(_sources(2, bundle=False), 'k0')
        card = gallery.cards()[1]
        assert card._placeholder.isVisibleTo(card._canvas)
        assert card._placeholder.text() == '预览尚未生成'
        gallery.set_sources(_sources(2, bundle=True), 'k0')
        assert not gallery.cards()[1]._placeholder.isVisibleTo(
            gallery.cards()[1]._canvas)


class TestCurrentHighlight:
    def test_current_card_has_border_and_corner(self, gallery):
        gallery.set_sources(_sources(3), 'k1')
        cards = gallery.cards()
        assert cards[1]._current is True
        assert cards[1]._corner.isVisibleTo(cards[1])
        assert 'border:2px solid' in cards[1].styleSheet()
        assert cards[0]._current is False
        assert not cards[0]._corner.isVisibleTo(cards[0])

    def test_current_respects_key_not_position(self, gallery):
        gallery.set_sources(_sources(3), 'k2')
        assert gallery.cards()[2]._current is True
        # 增量更新：current_key 变化 → 角标跟 key 走
        gallery.set_sources(_sources(3), 'k0')
        assert gallery.cards()[0]._current is True
        assert not gallery.cards()[2]._current


class TestPickAndClose:
    def test_card_click_emits_pick_and_closes(self, gallery, qapp):
        got = []
        gallery.sig_pick.connect(got.append)
        gallery.set_sources(_sources(3), 'k0')
        gallery.show()
        qapp.processEvents()
        gallery.cards()[2].sig_clicked.emit('k2')   # 卡内鼠标路径的等价触发
        assert got == ['k2']
        assert not gallery.isVisible()

    def test_escape_closes(self, gallery, qapp):
        gallery.set_sources(_sources(2), 'k0')
        gallery.show()
        qapp.processEvents()
        assert gallery.isVisible()
        _press_escape(gallery)
        assert not gallery.isVisible()


class TestEmptyState:
    def test_zero_sources_shows_centered_hint(self, gallery):
        gallery.set_sources([], '')
        assert gallery.cards() == []
        assert gallery._empty.isVisibleTo(gallery)

    def test_sources_hide_hint(self, gallery):
        gallery.set_sources(_sources(1), 'k0')
        assert not gallery._empty.isVisibleTo(gallery)


class TestStaggeredFade:
    def test_cards_start_transparent_with_effect(self, gallery):
        gallery.set_sources(_sources(3), 'k0')
        for card in gallery.cards():
            effect = card.graphicsEffect()
            assert effect is not None
            assert effect.opacity() == 0.0     # 淡入起点（错峰启动前）


class TestPageWithoutGallery:
    """The 2026-10-16 workbench retires both automatic and explicit result walls.

    Independent BScanGallery behavior above remains supported and unchanged.
    """

    def test_many_results_do_not_construct_a_gallery(self, qapp, monkeypatch):
        from types import SimpleNamespace
        from ui.pages.processing_page import ProcessingPage

        constructed = []
        monkeypatch.setattr(BScanGallery, '__init__',
                            lambda *args, **kwargs: constructed.append(True))
        page = ProcessingPage()
        try:
            page.set_original_bundle(_bundle())
            artifacts = [SimpleNamespace(
                artifact_id=f'F{i}', line_id='L01', name=f'result{i}',
                method_id='agc', created_at='2026-10-16T10:02:00',
                manifest={'params': {'artifact_kind': 'processing',
                                     'run_group_id': 'G1'}}) for i in range(6)]
            page.set_artifacts(artifacts)
            for art in artifacts:
                page.select_artifact(art.artifact_id)
            page.show_latest_result()
            assert constructed == []
            assert len(page._workspace.views()) == 1
            assert not hasattr(page, '_open_gallery')
            assert not hasattr(page, '_gallery_btn')
            assert page.findChildren(BScanGallery) == []
        finally:
            page.close()
            page.deleteLater()

    def test_catalog_and_preview_updates_preserve_selected_tabs(self, qapp):
        from types import SimpleNamespace
        from ui.pages.processing_page import ProcessingPage

        page = ProcessingPage()
        try:
            page.set_original_bundle(_bundle())
            art = SimpleNamespace(
                artifact_id='F1', line_id='L01', name='run_agc',
                method_id='agc', created_at='2026-10-16T10:02:00',
                manifest={'params': {'artifact_kind': 'processing',
                                     'run_group_id': 'G1'}})
            page.set_artifacts([art])
            page.select_artifact('F1')
            page.select_original()
            before = page._workspace.tab_keys()
            art.name = 'renamed saved result'
            page.set_artifacts([art])
            page.set_artifact_bundle('F1', _bundle())
            assert page._workspace.tab_keys() == before
            assert page._workspace.active_key() == 'original'
            assert page._workspace.tab_bar().tabText(1) == art.name
            assert page.findChildren(BScanGallery) == []
        finally:
            page.close()
            page.deleteLater()
