from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PyQt6')

from mygpr.application.spatial.amplitude import AmplitudeSource
from mygpr.application.spatial.curtain_preview import CurtainPreview


def test_3d_scene_adds_real_trace_curtain_and_window_marker(qapp, monkeypatch):
    import ui.widgets.trajectory_3d_view as module

    objects = []
    class Scene:
        def addItem(self, item):
            objects.append(item)
        def removeItem(self, item):
            objects.remove(item)
    gl = SimpleNamespace(GLMeshItem=lambda **kw: ('mesh', kw),
                         GLLinePlotItem=lambda **kw: ('line', kw))
    monkeypatch.setattr(module, '_gl_state', lambda: (gl, None))
    view = module.Trajectory3DView()
    view._gl_view = Scene()
    view._origin = np.zeros(3)
    view._extent = 10.
    view._terrain_src = (32648, (0, 0, 1, 1))
    view._track_data = [('L1', '#000000', np.zeros((2, 3)))]
    messages = []
    view.curtain_notice.connect(messages.append)
    preview = CurtainPreview(AmplitudeSource('L1', component='real'), 'EPSG:32648',
                             np.array([[1., 2.], [3., 4.]]), np.array([0., 10.]),
                             np.array([0., 1.]), np.array([0., 0.]), np.array([5., 5.]),
                             np.array([0, 1]), np.array([True]))
    view.set_curtains([preview], 4., 6.)
    assert [item[0] for item in objects] == ['mesh', 'line']
    assert objects[0][1]['faces'].shape == (2, 3)
    assert '不代表地下深度' in messages[-1]
    view.clear_curtains()
    assert objects == []
    view._gl_view = None
    view.shutdown()
    view.close()


def test_bscan_time_band_is_horizontal(qapp):
    from ui.widgets.curtain_bscan_panel import CurtainBScanPanel
    panel = CurtainBScanPanel()
    assert panel._window.orientation == 'horizontal'
    panel.close()


def test_raw_track_uses_shared_vertical_origin_and_exaggeration(qapp, monkeypatch):
    import ui.widgets.trajectory_3d_view as module
    objects = []
    class Scene:
        def addItem(self, item):
            objects.append(item)
        def removeItem(self, item):
            objects.remove(item)
    monkeypatch.setattr(module, '_gl_state', lambda: (SimpleNamespace(
        GLLinePlotItem=lambda **kw: kw), None))
    view = module.Trajectory3DView()
    view._gl_view = Scene()
    view._origin = np.array([0., 0., 100.])
    view._track_data = [('L1', '#000000', np.array([[0., 0., 103.], [1., 0., 105.]]))]
    view._exag = 2.
    view._render_lines()
    np.testing.assert_allclose(objects[0]['pos'][:, 2], [6., 10.])
    view._gl_view = None
    view.shutdown()
    view.close()


def test_vertical_datum_difference_is_reported_not_applied():
    from ui.widgets.terrain_alignment import datum_notice
    payload = {'gx': np.array([0., 1.]), 'gy': np.array([0., 1.]),
               'z': np.full((2, 2), 100.)}
    points = np.array([[.2, .2, 120.], [.3, .3, 120.], [.4, .4, 120.], [.5, .5, 120.]])
    notice = datum_notice(payload, points, 20.)
    assert '+20.0 m' in notice and '未自动平移' in notice
    np.testing.assert_array_equal(payload['z'], np.full((2, 2), 100.))
