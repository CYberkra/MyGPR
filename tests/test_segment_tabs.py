# -*- coding: utf-8 -*-
"""SlimSegment（共享瘦版子标签）与 make_segment_card 测试。

覆盖：addItem/setCurrentItem/点击切换、字号跟随全局正文、paintEvent
（有/无选中项）经 grab() 触发不抛错、make_segment_card 单行 header 组装。
"""
import pytest

pytest.importorskip('PyQt6')

from ui import constants  # noqa: E402
from ui.page_scaffold import make_segment_card  # noqa: E402
from ui.widgets.segment_tabs import SlimSegment  # noqa: E402


@pytest.fixture
def segment(qapp):
    seg = SlimSegment()
    yield seg
    seg.deleteLater()


# ============================================================ 页签行为
def test_add_item_and_switch(segment):
    segment.addItem('a', '日志')
    segment.addItem('b', '任务')
    segment.setCurrentItem('a')
    assert segment.currentRouteKey() == 'a'
    segment.setCurrentItem('b')
    assert segment.currentRouteKey() == 'b'
    assert segment.currentItem().text() == '任务'


def test_item_click_switches(qapp, segment):
    segment.addItem('a', '日志')
    segment.addItem('b', '任务')
    segment.setCurrentItem('a')
    segment.widget('b').click()
    assert segment.currentRouteKey() == 'b'


def test_on_click_callback_fires(segment):
    seen = []
    segment.addItem('a', '日志', onClick=lambda: seen.append('a'))
    segment.widget('a').click()
    assert seen == ['a']


# ============================================================ 外观
def test_font_follows_body_size(segment):
    segment.addItem('a', '日志')
    expected = round(constants.FONT_SIZE_BODY * 4 / 3)
    assert segment.widget('a').font().pixelSize() == expected


def test_paint_without_current_item(segment):
    segment.addItem('a', '日志')   # 未 setCurrentItem → 提前返回分支
    assert not segment.grab().isNull()


def test_paint_with_current_item(segment):
    segment.addItem('a', '日志')
    segment.addItem('b', '任务')
    segment.setCurrentItem('b')
    segment.resize(200, 30)
    assert not segment.grab().isNull()


# ============================================================ make_segment_card
def test_make_segment_card_single_row_header(qapp):
    seg = SlimSegment()
    seg.addItem('a', '原始数据')
    card, layout = make_segment_card('数据预览', seg)
    try:
        header = layout.itemAt(0).layout()
        assert header is not None
        widgets = [header.itemAt(i).widget() for i in range(header.count())]
        assert seg in widgets                     # 页签在 header 行内
        assert seg.parent() is card               # 加入布局后重挂父
        assert any(hasattr(w, 'text') and w.text() == '数据预览'
                   for w in widgets)              # 标题在同行
    finally:
        card.deleteLater()
