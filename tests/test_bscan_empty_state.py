"""An empty scientific preview must not show fabricated axis ranges."""
import numpy as np
import pytest

pytest.importorskip('PyQt6')

from ui.widgets.bscan_view import BScanView


def test_axes_follow_data_lifecycle_without_changing_matrix(qapp):
    view = BScanView()
    view.set_axis_modes('trace', 'sample')
    view.set_colorbar_visible(True)
    for axis in ('bottom', 'left'):
        assert not view._plot.getAxis(axis).isVisible()
    assert not view._colorbar.isVisible()
    matrix = np.arange(24, dtype=np.float32).reshape(6, 4)
    original = matrix.copy()
    view.set_matrix(matrix, 0.0, 23.0)
    for axis in ('bottom', 'left'):
        assert view._plot.getAxis(axis).isVisible()
    assert view._colorbar.isVisible()
    np.testing.assert_array_equal(matrix, original)
    view.clear()
    for axis in ('bottom', 'left'):
        assert not view._plot.getAxis(axis).isVisible()
    view.close()
