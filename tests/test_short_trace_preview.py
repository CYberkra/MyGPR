"""Repair masks stay aligned with downsampled B-scan display columns."""
import numpy as np
import pytest

pytest.importorskip('PyQt6')


def test_repair_candidates_and_history_are_visible_at_exact_preview_columns(qapp):
    from ui.controllers.project_controller import ProjectController
    from ui.desktop_backend_facade import trace_quality_preview_mask
    from ui.widgets.bscan_view import BScanView
    metadata = {'short_trace_candidate_mask': np.array([0, 1, 0, 0, 0, 0]),
                'short_trace_repaired_mask': np.array([0, 0, 0, 1, 0, 0])}
    mask = trace_quality_preview_mask(metadata, 6)
    bundle = ProjectController._bundle_from_window(
        np.ones((4, 3), np.float32), 'L01', total_samples=4, total_traces=6,
        sample_indices=np.arange(4), trace_indices=np.array([1, 3, 5]), trace_qc_bad_mask=mask)
    np.testing.assert_array_equal(bundle.trace_qc_bad_mask, [1, 1, 0])
    view = BScanView()
    try:
        view.set_bundle(bundle)
        assert len(view._qc_marks) == 2
        assert [tuple(mark.getRegion()) for mark in view._qc_marks] == [(0., 1.), (1., 2.)]
        assert view._view_to_data(1, 0) == (3, 0)
    finally:
        view.close()
