import numpy as np
import pytest

from mygpr.domain.spatial.curtain import curtain_geometry


def test_curtain_uses_each_track_position_and_preserves_data_values():
    mesh = curtain_geometry([0., 10.], [1., 2.], [100., 101.],
                            [0., 2.], np.array([[1., 2.], [3., 4.]]),
                            vertical_scale=1.)
    np.testing.assert_allclose(mesh.vertices, [[0, 1, 100], [10, 2, 101],
                                               [0, 1, 98], [10, 2, 99]])
    np.testing.assert_array_equal(mesh.faces, [[0, 2, 1], [1, 2, 3]])
    np.testing.assert_array_equal(mesh.values, [1., 2., 3., 4.])


def test_curtain_does_not_bridge_missing_positions_or_samples():
    mesh = curtain_geometry([0., np.nan, 2.], [0., 0., 0.], [0., 0., 0.],
                            [0., 1.], np.ones((2, 3)), vertical_scale=2.)
    assert mesh.faces.shape == (0, 3)
    mesh = curtain_geometry([0., 1.], [0., 0.], [0., 0.], [0., 1.],
                            np.ones((2, 2)), vertical_scale=1., connected=[False])
    assert mesh.faces.shape == (0, 3)
    mesh = curtain_geometry([0., 1.], [0., 0.], [0., 0.],
                            [0., 1.], np.array([[1., 2.], [3., np.nan]]), vertical_scale=1.)
    assert mesh.faces.shape == (0, 3)


def test_curtain_requires_explicit_positive_scale_and_compatible_shapes():
    with pytest.raises(ValueError, match='比例'):
        curtain_geometry([0., 1.], [0., 0.], [0., 0.], [0., 1.], np.ones((2, 2)), vertical_scale=0.)
    with pytest.raises(ValueError, match='形状'):
        curtain_geometry([0., 1.], [0., 0.], [0., 0.], [0., 1.], np.ones((2, 3)), vertical_scale=1.)
