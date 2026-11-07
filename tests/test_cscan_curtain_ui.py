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
