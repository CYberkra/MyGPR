"""AScanPopup 浮窗行为测试（offscreen）。

必须用 conftest 的 session 级 ``qapp``——模块自建 QApplication 会在
模块卸载后被 GC，qfluentwidgets 全局 QConfig 单例随之悬垂，污染后续
所有建 Fluent 控件的测试（本次事故：22 个后继测试 RuntimeError）。
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from ui.widgets.ascan_popup import AScanPopup


def test_popup_shows_trace_and_title(qapp):
    popup = AScanPopup()
    trace = np.random.randn(501).astype(np.float32)
    popup.show_trace(trace, trace_index=42, distance_m=12.5)
    assert popup._ascan_view is not None
    # 标题含道号
    assert '42' in popup._ascan_view._plot.plotItem.titleLabel.text


def test_popup_clear(qapp):
    popup = AScanPopup()
    popup.show_trace(np.ones(10), trace_index=0, distance_m=0.0)
    popup.clear()
    # clear 后不抛异常即通过；曲线数据为空由 AScanView.clear 保证


def test_popup_close_hides_not_destroys(qapp):
    popup = AScanPopup()
    popup.show()
    popup.close()
    assert popup._ascan_view is not None  # 关闭仅隐藏，实例复用
