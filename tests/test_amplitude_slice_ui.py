from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PyQt6')

from mygpr.application.spatial.amplitude import AmplitudeResult, AmplitudeSource, LineObservations
from mygpr.domain.spatial.amplitude_slice import grid_line_values
from mygpr.domain.spatial.models import SpatialTrack, SpatialTrackPoint
from mygpr.application.spatial.amplitude_saved import LoadedAmplitudeSlice
from ui.widgets.amplitude_slice_panel import AmplitudeSlicePanel


def test_processed_marker_uses_exact_large_trace_id_after_reordering(qapp):
    panel = AmplitudeSlicePanel()
    identifier = 2**53 + 1
    panel._bundle = SimpleNamespace(trace_indices=np.array([0, 1]),
        source_trace_ids=np.array([identifier + 1, identifier], dtype=np.uint64),
        sample_axis=None, sample_axis_label='')
    panel._selected_raw_id = ('L1', identifier)
    panel._highlight_trace('L1', 'artifact')
    assert panel._marker_value == 1.5
    panel.close()


def test_panel_requests_selected_sources_and_disables_during_job(qapp):
    panel = AmplitudeSlicePanel()
    requests = []
    panel.preview_requested.connect(lambda *args: requests.append(args))
    panel.set_tracks([SpatialTrack('L1', '测线 1', (SpatialTrackPoint(0, 1., 2.),), 'EPSG:32648')], {})
    panel.set_artifacts('L1', [SimpleNamespace(artifact_id='A', name='处理结果')])
    panel._source.setCurrentIndex(1)
    panel._request()
    assert len(requests) == 1
    assert requests[0][1] == {'L1': 'A'}
    assert requests[0][3:5] == (0., 10.)
    assert not panel._run.isEnabled() and panel._cancel.isEnabled()
    panel.set_failure('时间轴未知')
    assert panel._run.isEnabled() and '时间轴' in panel._status.text()
    panel.close()


def test_reopening_stale_saved_slice_displays_frozen_grid_without_3d(qapp):
    panel = AmplitudeSlicePanel()
    grid = grid_line_values([.1], [.1], [2.], ['L1'], cell_size_m=1.)
    source = AmplitudeSource('L1', component='mag')
    result = AmplitudeResult(grid, (source,), 'EPSG:32648', 10., 20., 'mean')
    track = SpatialTrack('L1', 'Line', (SpatialTrackPoint(0, .1, .1),), 'EPSG:32648')
    panel.show_saved(LoadedAmplitudeSlice(result, 'amplitude_slice_v001', True), [track], {}, ())
    assert panel._map._matrix[0, 0] == 2.
    assert not panel._show_3d.isEnabled() and not panel._save.isEnabled()
    assert '来源已变化' in panel._status.text()
    panel.close()


def test_curtain_selection_requests_correct_artifact_and_source_trace(qapp):
    panel = AmplitudeSlicePanel()
    track = SpatialTrack('L1', 'Line', (SpatialTrackPoint(0, .1, .1),), 'EPSG:32648')
    panel.set_tracks([track], {})
    panel._source_ids['L1'] = 'artifact-1'
    calls = []
    panel.bscan_requested.connect(lambda *args: calls.append(args))
    panel.select_curtain('L1', 'artifact-1', 37, 45.)
    assert calls[-1][0:2] == ('L1', 'artifact-1')
    assert panel._selected_source_trace == ('L1', 'artifact-1', 37)
    assert panel._picked_time_ns == 45.
    count = len(calls)
    panel.select_curtain('L1', 'different-artifact', 20, 10.)
    assert len(calls) == count
    panel.close()


def test_panel_displays_coverage_and_preserves_fixed_color_levels(qapp):
    panel = AmplitudeSlicePanel()
    grid = grid_line_values([0.1, 2.1], [0.1, 0.1], [1., 3.], ['L1', 'L1'], cell_size_m=1.)
    source = AmplitudeSource('L1', component='mag')
    positions = (LineObservations(source, 'EPSG:32648', np.array([0, 2]),
                 np.array([.1, 2.1]), np.array([.1, .1]), np.array([0., 0.])),)
    panel.set_result(AmplitudeResult(grid, (source,), 'EPSG:32648', 0., 10., 'rms'), positions)
    assert panel._map._matrix.shape == (1, 3)
    assert np.isnan(panel._map._matrix[0, 1])
    assert panel._levels == (1., 3.)
    assert '0–10 ns' in panel._status.text()
    assert panel._save.isEnabled()
    panel._end.setValue(11.)
    assert not panel._save.isEnabled()
    assert panel._result is None
    panel.close()
