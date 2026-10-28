"""No implicit sample/trace thinning anywhere in the desktop B-scan path."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip('PyQt6')

from ui import desktop_backend_facade as facade
from ui.controllers import project_controller as module
from ui.widgets.bscan_view import BScanView


def test_desktop_bundle_keeps_every_trace_and_sample():
    matrix = np.arange(1001 * 1801, dtype='float32').reshape(1001, 1801)
    bundle = facade.build_preview_bundle('L01', matrix)
    np.testing.assert_array_equal(bundle.matrix, matrix)
    assert bundle.matrix.shape == (1001, 1801)


@pytest.mark.parametrize('entry', ['raw', 'artifact', 'comparison'])
def test_controller_reads_full_resolution_for_each_bscan_entry(qapp, monkeypatch, entry):
    matrix = np.arange(501 * 4751, dtype='float32').reshape(501, 4751)
    controller = module.ProjectController()
    controller._current = NS(project_id='P')
    queue, delivered, calls = [], [], []

    def info(*args):
        return NS(shape=matrix.shape, time_window_ns=900., length_m=0., metadata={})

    def read(*args, **kw):
        calls.append(kw)
        r0, r1 = kw.get('sample_start', 0), kw.get('sample_end', matrix.shape[0])
        c0, c1 = kw.get('trace_start', 0), kw.get('trace_end', matrix.shape[1])
        rs = max(1, int(np.ceil((r1-r0)/kw['max_samples'])))
        cs = max(1, int(np.ceil((c1-c0)/kw['max_traces'])))
        return matrix[r0:r1:rs, c0:c1:cs], np.arange(r0, r1, rs), np.arange(c0, c1, cs)

    controller.set_backend(NS(backend=NS(projects=NS(
        get_dataset_info=info, get_artifact_dataset_info=info,
        read_window=read, read_artifact_window=read,
        read_trace_metadata=lambda *args: {},
        read_artifact_trace_metadata=lambda *args: {}))))
    monkeypatch.setattr(controller, '_line_elevation', lambda *args: None)
    monkeypatch.setattr(module, 'run_command', lambda command, **kw: queue.append(command))
    controller.dataset_preview_ready.connect(delivered.append)
    controller.artifact_preview_ready.connect(lambda aid, bundle: delivered.append(bundle))
    controller.comparison_preview_ready.connect(lambda pid, lid, token, bundle, error: delivered.append(bundle))
    if entry == 'raw':
        controller.preview_line('L05')
    elif entry == 'artifact':
        controller.preview_artifact('L05', 'A')
    else:
        controller.preview_comparison_line('P', 'L05', 'compare')
    queue[0].execute()

    assert len(delivered) == 1 and delivered[0] is not None
    np.testing.assert_array_equal(delivered[0].matrix, matrix)
    np.testing.assert_array_equal(delivered[0].trace_indices, np.arange(4751))
    assert max((k['sample_end']-k['sample_start'])*(k['trace_end']-k['trace_start'])*4
               for k in calls) <= 8*1024*1024


def test_imageitem_never_auto_downsamples(qapp):
    view = BScanView()
    try:
        view.set_matrix(np.ones((3, 33000), dtype='float32'), 0, 1)
        assert view._image_item.autoDownsample is False
    finally:
        view.close()


def test_waveform_keeps_more_than_500_samples(qapp, monkeypatch):
    import pyqtgraph as pg

    captured = []
    original = pg.arrayToQPath

    def capture(x, y, connect):
        captured.append((len(x), np.array(y)))
        return original(x, y, connect)

    monkeypatch.setattr(pg, 'arrayToQPath', capture)
    view = BScanView()
    try:
        data = np.sin(np.arange(1501, dtype='float32')[:, None])
        view.set_matrix(data, -1, 1)
        view._render_wiggle(data, filled=False, symmetric=True)
        assert captured[-1][0] == 1502  # every sample plus the closing baseline
        np.testing.assert_array_equal(captured[-1][1][:-1], np.arange(1501))
    finally:
        view.close()


def test_full_resolution_memory_refusal_happens_before_any_read(monkeypatch):
    from mygpr.infrastructure.system import resource_policy

    monkeypatch.setattr(resource_policy, 'available_memory_bytes', lambda: 100)
    calls = []
    with pytest.raises(resource_policy.InsufficientProcessingMemory, match='不会自动降采样'):
        facade.read_full_resolution_bscan((501, 4751), lambda **kw: calls.append(kw))
    assert calls == []


def test_elevation_view_keeps_samples_beyond_8192_rows():
    from ui.widgets.bscan_axes import build_elevation_view

    data = np.zeros((9000, 2), dtype='float32')
    data[-1] = 123
    warped, axis = build_elevation_view(data, np.full(2, 10000.), np.arange(9000))
    assert warped.shape == data.shape
    np.testing.assert_array_equal(warped[-1], [123, 123])
    assert axis[-1] == 1001


def test_elevation_resource_refusal_keeps_complete_original_image(qapp, monkeypatch):
    from core.gui_rendering import make_preview_bundle
    from ui.widgets import bscan_axes, bscan_view_data

    checked, warnings = [], []

    def refuse(shape):
        checked.append(shape)
        raise RuntimeError('资源不足，不会自动降采样')

    monkeypatch.setattr(bscan_axes, 'validate_bscan_display_shape', refuse)
    monkeypatch.setattr(bscan_view_data.InfoBar, 'warning', lambda **kw: warnings.append(kw))
    data = np.arange(20, dtype='float32').reshape(10, 2)
    view = BScanView()
    try:
        view.set_bundle(make_preview_bundle(data, depth_axis_m=np.arange(10),
                                           trace_elevation_m=np.array([100., 105.])))
        view.set_y_axis_mode('elevation')
        assert checked == [(15, 2)]
        assert len(warnings) == 1
        assert '不会自动降采样' in warnings[0]['content']
        assert view._showing_elevation is False
        np.testing.assert_array_equal(view._image_item.image, data)
        assert view._image_item.autoDownsample is False
    finally:
        view.close()


def test_clearing_view_releases_file_cache_even_with_gain_and_envelope(qapp):
    import gc
    from pathlib import Path

    matrix, _, _ = facade.read_full_resolution_bscan((5, 5), lambda **kw: (
        np.ones((5, 5), dtype='float32'), np.arange(5), np.arange(5)))
    path = Path(matrix.filename)
    bundle = facade.build_preview_bundle('L01', matrix)
    view = BScanView()
    try:
        view.set_bundle(bundle)
        view.set_gain('tvg')
        view.set_display_mode('envelope')
        del matrix, bundle
        gc.collect()
        assert path.is_file()
        view.clear()
        gc.collect()
        assert not path.exists()
    finally:
        view.close()
