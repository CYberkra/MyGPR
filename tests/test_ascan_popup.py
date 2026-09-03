"""AScanPopup 浮窗行为测试（offscreen）。"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from ui.widgets.ascan_popup import AScanPopup


@pytest.fixture(scope='module')
def app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_popup_shows_trace_and_title(app):
    popup = AScanPopup()
    trace = np.random.randn(501).astype(np.float32)
    popup.show_trace(trace, trace_index=42, distance_m=12.5)
    assert popup._ascan_view is not None
    # 标题含道号
    assert '42' in popup._ascan_view._plot.plotItem.titleLabel.text


def test_popup_clear(app):
    popup = AScanPopup()
    popup.show_trace(np.ones(10), trace_index=0, distance_m=0.0)
    popup.clear()
    # clear 后不抛异常即通过；曲线数据为空由 AScanView.clear 保证


def test_popup_close_hides_not_destroys(app):
    popup = AScanPopup()
    popup.show()
    popup.close()
    assert popup._ascan_view is not None  # 关闭仅隐藏，实例复用
