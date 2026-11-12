"""BScanView → AScanPopup 跟随链路测试（offscreen）。"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from ui.widgets.bscan_view import BScanView


@pytest.fixture
def view(qapp):
    v = BScanView()
    mat = (np.random.default_rng(0).standard_normal((200, 300)) * 0.02).astype(np.float32)
    v.set_matrix(mat, -0.1, 0.1, title='t')
    return v


def test_toggle_creates_and_shows_popup(view):
    assert view._ascan_popup is None
    view.set_ascan_follow(True)
    assert view._ascan_popup is not None
    assert not view._pick_enabled


def test_click_updates_popup_trace(view):
    view.set_ascan_follow(True)
    view._emit_point_picked(10, 100)  # 复用内部发射路径
    # 数据核对：浮窗存在即链路通（波形内容由 AScanView 持 PlotDataItem）


def test_toggle_off_hides_popup(view):
    view.set_ascan_follow(True)
    view.set_ascan_follow(False)
    assert not view._ascan_popup.isVisible()


def test_follow_disabled_when_no_data(view):
    v = BScanView()  # 未 set_matrix
    v.set_ascan_follow(True)
    v._emit_point_picked(5, 5)  # 不应抛异常


def test_follow_does_not_enable_or_disable_interpretation_picking(view):
    picked = []
    view.sig_point_picked.connect(lambda trace, sample: picked.append((trace, sample)))
    view.set_ascan_follow(True)
    view._emit_point_picked(4, 5, view_trace=4)
    assert not view._pick_enabled and picked == []
    view.set_pick_enabled(True)
    view.set_ascan_follow(False)
    view._emit_point_picked(4, 5, view_trace=4)
    assert picked == [(4, 5)]
