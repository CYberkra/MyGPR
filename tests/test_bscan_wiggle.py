"""BScanView 三态显示模式测试（offscreen）。"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过

from ui.widgets.bscan_view import BScanDisplayMode, BScanView


@pytest.fixture
def view(qapp):
    v = BScanView()
    v.set_matrix((np.random.randn(100, 150) * 0.02).astype(np.float32),
                 -0.1, 0.1, title='t')
    return v


def test_default_mode_is_grayscale(view):
    assert view.display_mode is BScanDisplayMode.GRAYSCALE


def test_switch_to_wiggle_creates_path_item(view):
    view.set_display_mode(BScanDisplayMode.WIGGLE)
    assert view._wiggle_item is not None
    assert view._wiggle_item.path() is not None
    assert len(view._wiggle_item.path().toSubpathPolygons()) >= 1


def test_switch_back_to_grayscale_hides_wiggle(view):
    view.set_display_mode(BScanDisplayMode.WIGGLE)
    view.set_display_mode(BScanDisplayMode.GRAYSCALE)
    assert not view._wiggle_item.isVisible()


def test_waveform_mode_keeps_image_visible(view):
    view.set_display_mode(BScanDisplayMode.WAVEFORM)
    assert view._image_item.isVisible()


def test_invalid_mode_raises(view):
    with pytest.raises(ValueError):
        view.set_display_mode('bogus')
