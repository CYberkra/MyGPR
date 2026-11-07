import numpy as np
import pytest

pytest.importorskip('PyQt6')

from ui.widgets.curtain_picking import pick_mesh


def test_picks_frontmost_triangle_and_ignores_empty_screen():
    vertices = np.array([[-.5, -.5, .4], [.5, -.5, .4], [0., .5, .4],
                         [-.5, -.5, -.4], [.5, -.5, -.4], [0., .5, -.4]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    depth, vertex = pick_mesh(vertices, faces, np.eye(4), 100, 100, (50, 50))
    assert depth == pytest.approx(-.4) and vertex in (3, 4, 5)
    assert pick_mesh(vertices, faces, np.eye(4), 100, 100, (1, 1)) is None


def test_empty_and_degenerate_meshes_cannot_be_picked():
    assert pick_mesh(np.zeros((3, 3)), np.array([[0, 1, 2]]), np.eye(4), 100, 100, (50, 50)) is None
    assert pick_mesh(np.zeros((3, 3)), np.empty((0, 3)), np.eye(4), 100, 100, (50, 50)) is None
