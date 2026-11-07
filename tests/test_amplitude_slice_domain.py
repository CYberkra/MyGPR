import numpy as np
import pytest

from mygpr.domain.spatial.amplitude_slice import grid_line_values, window_attribute


def test_magnitude_is_not_transformed_again_and_window_is_half_open():
    data = np.array([[1., 2.], [2., 4.], [3., 6.], [4., 8.]])
    actual = window_attribute(data, [0., 1., 2., 3.], 1., 3., component='mag', attribute='rms')
    np.testing.assert_allclose(actual, np.sqrt([6.5, 26.]))


def test_real_envelope_and_magnitude_reject_unknown_component():
    data = np.array([[2.], [2.], [2.], [2.]])
    np.testing.assert_allclose(window_attribute(data, [0., 1., 2., 3.], 0., 4., component='real'), [2.])
    with pytest.raises(ValueError, match='分量'):
        window_attribute(data, [0., 1., 2., 3.], 0., 4., component='fdata')
    with pytest.raises(ValueError, match='非负'):
        window_attribute(-data, [0., 1., 2., 3.], 0., 4., component='mag')


def test_incomplete_window_and_invalid_samples_are_not_filled():
    data = np.array([[1., 1.], [2., np.nan], [3., 3.], [4., 4.]])
    values = window_attribute(data, [0., 1., 2., 3.], 0., 4., component='mag')
    assert np.isfinite(values[0]) and np.isnan(values[1])
    with pytest.raises(ValueError, match='范围'):
        window_attribute(data, [0., 1., 2., 3.], -1., 2., component='mag')
    with pytest.raises(ValueError, match='时间轴'):
        window_attribute(data, [0., 1., 1., 3.], 0., 2., component='mag')


def test_grid_averages_lines_equally_and_preserves_unobserved_cells():
    grid = grid_line_values(
        np.array([0.1, 0.2, 0.3, 0.4, 2.1]),
        np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        np.array([2., 2., 2., 10., 7.]),
        np.array(['L1', 'L1', 'L1', 'L2', 'L2']), cell_size_m=1.,
    )
    np.testing.assert_allclose(grid.values[0, [0, 2]], [6., 7.])
    assert np.isnan(grid.values[0, 1])
    assert grid.trace_counts[0, 0] == 4
    assert grid.line_counts[0, 0] == 2


def test_grid_rejects_excessive_dimensions():
    with pytest.raises(ValueError, match='网格'):
        grid_line_values([0., 10000.], [0., 10000.], [1., 2.], ['a', 'b'], cell_size_m=1.)
