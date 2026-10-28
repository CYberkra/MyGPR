"""Bounded previews keep the full matrix's indices, labels and pick coordinates."""
import numpy as np
import pytest

pytest.importorskip('PyQt6')

from core.gui_rendering import make_preview_bundle
from ui.controllers.project_controller import ProjectController
from ui.widgets.bscan_view import BScanView


def window_bundle():
    traces = np.arange(0, 4751, 3)
    samples = np.arange(0, 2001, 3)
    data = (samples[:, None] * 10000 + traces[None, :]).astype('float32')
    return ProjectController._bundle_from_window(
        data, 'L05', time_window_ns=900, sample_indices=samples,
        trace_indices=traces, total_samples=2001, total_traces=4751)


def test_bounded_controller_bundle_keeps_full_counts_and_exact_indices(qapp):
    bundle = window_bundle()
    assert bundle.matrix.shape == (667, 1584)
    assert bundle.trace_count == 4751
    assert bundle.sample_count == 2001
    np.testing.assert_array_equal(bundle.trace_indices, np.arange(0, 4751, 3))
    np.testing.assert_array_equal(bundle.sample_indices, np.arange(0, 2001, 3))


def test_in_memory_preview_records_actual_stride_instead_of_endpoint_ratio():
    raw = np.arange(9 * 4751, dtype='float32').reshape(9, 4751)
    bundle = make_preview_bundle(raw)
    np.testing.assert_array_equal(bundle.trace_indices, np.arange(0, 4751, 3))
    np.testing.assert_array_equal(bundle.matrix, raw[:, ::3])


@pytest.fixture
def view(qapp):
    widget = BScanView()
    widget.set_bundle(window_bundle())
    yield widget
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def test_trace_ticks_show_full_input_numbers_and_total_count(view):
    labels = view._bottom_axis.tickStrings([.5, 667.5, 1583.5], 1, 1)
    assert [float(label) for label in labels] == [1, 2002, 4750]
    assert '4751' in view._plot.getAxis('bottom').labelText
    assert '1584' in view._plot.getAxis('bottom').labelText


def test_pointer_and_overlay_use_exact_source_indices(view):
    assert view._view_to_data(667, 666) == (2001, 1998)
    np.testing.assert_allclose(view._data_to_view(2001, 1998), (667, 666))
    view.set_overlay_points([(2001, 1998)])
    x, y = view._scatter.getData()
    np.testing.assert_allclose([x[0], y[0]], [667, 666])


def test_crosshair_uses_full_trace_number_not_preview_column(view):
    view._pending_readout = (667, 10, float(view._matrix[10, 667]))
    view._flush_crosshair_readout()
    assert view._readout.text().splitlines()[0] == '道 2002'


def test_nonuniform_window_indices_are_not_replaced_by_a_ratio(qapp):
    bundle = ProjectController._bundle_from_window(
        np.ones((3, 4), dtype='float32'), 'L01', total_samples=10,
        total_traces=20, sample_indices=np.array([0, 3, 9]),
        trace_indices=np.array([0, 3, 9, 19]))
    widget = BScanView()
    try:
        widget.set_bundle(bundle)
        assert widget._view_to_data(2, 1) == (9, 3)
        np.testing.assert_allclose(widget._data_to_view(9, 3), (2, 1))
    finally:
        widget.close()


def test_direct_matrix_replacement_clears_previous_mapping(view):
    view.set_matrix(np.ones((5, 4), dtype='float32'), 0, 1)
    assert view._view_to_data(2, 3) == (2, 3)
    assert view._bottom_axis.tickStrings([2], 1, 1) == ['2']


def test_bundle_preserves_saved_physical_axes_and_unknown_distance():
    metadata = {'time_axis_ns': [100, 130, 190], 'depth_axis_m': [1, 2, 4],
                'distance_axis_m': [0, 2, 7, 11], 'distance_axis_status': 'provided_length'}
    bundle = ProjectController._bundle_from_window(
        np.ones((2, 2)), 'L01', metadata=metadata, total_samples=3,
        total_traces=4, sample_indices=np.array([0, 2]), trace_indices=np.array([1, 3]))
    np.testing.assert_allclose(bundle.sample_axis, [100, 190])
    np.testing.assert_allclose(bundle.depth_axis_m, [1, 4])
    np.testing.assert_allclose(bundle.trace_axis_m, [2, 11])
    metadata['distance_axis_status'] = 'unknown'
    bundle = ProjectController._bundle_from_window(
        np.ones((2, 2)), 'L01', metadata=metadata, total_samples=3,
        total_traces=4, sample_indices=np.array([0, 2]), trace_indices=np.array([1, 3]))
    assert bundle.trace_axis_m is None


def test_bundle_does_not_invent_missing_time_or_distance_axes():
    bundle = ProjectController._bundle_from_window(
        np.ones((2, 2)), 'L01', metadata={'distance_axis_status': 'unknown'},
        total_samples=2, total_traces=2, sample_indices=np.arange(2),
        trace_indices=np.arange(2))
    assert bundle.sample_axis is None
    assert bundle.trace_axis_m is None


def test_bundle_uses_depth_axis_when_time_unavailable():
    bundle = ProjectController._bundle_from_window(
        np.ones((2, 2)), 'L01', metadata={'depth_axis_m': [2, 4]},
        total_samples=2, total_traces=2, sample_indices=np.arange(2), trace_indices=np.arange(2))
    np.testing.assert_allclose(bundle.sample_axis, [2, 4])
    assert bundle.sample_axis_label == '深度 (m)'
