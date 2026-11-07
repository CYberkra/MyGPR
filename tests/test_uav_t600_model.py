from types import SimpleNamespace

import numpy as np
import pytest

from ui.widgets.uav_t600_model import place_t600, t600_mesh


def test_t600_mesh_has_finite_triangles_and_colors():
    vertices, faces, colors = t600_mesh()
    assert vertices.shape[1] == 3 and faces.shape[1] == 3
    assert len(vertices) == len(colors) and colors.shape[1] == 4
    assert np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < len(vertices)
    assert len(vertices) < 5000


def test_placement_follows_track_direction_and_preserves_z_origin():
    model = np.array([[1., 0., 0.], [0., 0., 0.]])
    track = np.array([[100., 200., 103.], [100., 205., 103.]])
    placed = place_t600(model, track, np.array([100., 200., 100.]), 10., 2.)
    np.testing.assert_allclose(placed[1], [0., 0., 6.], atol=1e-8)
    np.testing.assert_allclose(placed[0], [0., 2., 6.], atol=1e-8)
    assert place_t600(model, np.repeat(track[:1], 2, axis=0), np.zeros(3), 10.) is None


def test_scene_toggle_removes_and_restores_aircraft(qapp, monkeypatch):
    pytest.importorskip('PyQt6')
    import ui.widgets.trajectory_3d_view as module
    class Scene:
        def __init__(self):
            self.items = []
        def addItem(self, item):
            self.items.append(item)
        def removeItem(self, item):
            self.items.remove(item)
    monkeypatch.setattr(module, '_gl_state', lambda: (SimpleNamespace(GLMeshItem=lambda **kw: kw), None))
    view = module.Trajectory3DView()
    scene = Scene()
    view._gl_view = scene
    view._origin = np.zeros(3)
    view._track_data = [('L01', '#000', np.array([[0., 0., 1.], [1., 0., 1.]]))]
    view._render_uav()
    assert len(scene.items) == 1 and not view._uav_caption.isHidden()
    view.set_uav_visible(False)
    assert not scene.items and view._uav_caption.isHidden()
    view.set_uav_visible(True)
    assert len(scene.items) == 1
    view._clear_uav()
    view._gl_view = None
    view.shutdown()
    view.close()
