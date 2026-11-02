"""Deterministic missing/invalid samples are distinct from statistical outliers."""

import numpy as np
import pytest

from tests.test_short_trace_auto_windows import run


def scan():
    wave = (1 + 0.1 * np.sin(np.arange(60) / 4)).astype(np.float32)
    return np.repeat(wave[:, None], 65, axis=1)


def test_zero_end_traces_are_quality_issues_not_partial_or_cross_window_changes():
    raw = scan()
    raw[20:, :] = 0  # quiet late windows throughout the line are not whole-zero traces
    raw[:, -2:] = 0
    out = run(raw, mode="interpolate")
    assert np.flatnonzero(out.trace_metadata["short_trace_zero_mask"]).tolist() == [63, 64]
    assert not out.trace_metadata["short_trace_local_change_mask"][-2:].any()
    assert not out.trace_metadata["short_trace_candidate_mask"][-2:].any()
    assert out.trace_metadata["short_trace_unresolved_mask"][-2:].all()
    assert not out.trace_metadata["short_trace_repaired_mask"].any()
    np.testing.assert_array_equal(out.data, raw)


def test_internal_zero_short_run_is_explicitly_repaired_and_reason_retained():
    clean = scan()
    raw = clean.copy()
    raw[:, 20:22] = 0
    marked = run(raw)
    np.testing.assert_array_equal(marked.data, raw)
    repaired = run(raw, mode="interpolate")
    np.testing.assert_array_equal(repaired.data, clean)
    assert repaired.trace_metadata["short_trace_zero_mask"][20:22].all()
    assert repaired.trace_metadata["short_trace_repaired_mask"][20:22].all()
    assert not repaired.trace_metadata["short_trace_candidate_mask"].any()
    assert not repaired.trace_metadata["short_trace_unresolved_mask"].any()


def test_only_exact_full_trace_zero_is_quality_flagged_without_small_amplitude_floor():
    raw = scan()
    raw[:20, 10] = 0
    raw[:, 20] = np.float32(1e-30)
    out = run(raw)
    assert not out.trace_metadata["short_trace_zero_mask"].any()


def test_internal_zero_run_longer_than_limit_is_not_interpolated():
    raw = scan()
    raw[:, 20:25] = 0
    out = run(raw, mode="interpolate")
    assert out.trace_metadata["short_trace_zero_mask"][20:25].all()
    assert not out.trace_metadata["short_trace_repaired_mask"].any()
    np.testing.assert_array_equal(out.data, raw)


def test_all_zero_record_and_short_record_are_marked_without_extrapolation():
    for samples in (4, 60):
        raw = np.zeros((samples, 65), np.float32)
        out = run(raw, mode="interpolate")
        assert out.trace_metadata["short_trace_zero_mask"].all()
        assert not out.trace_metadata["short_trace_candidate_mask"].any()
        assert not out.trace_metadata["short_trace_local_change_mask"].any()
        assert not out.trace_metadata["short_trace_repaired_mask"].any()
        np.testing.assert_array_equal(out.data, raw)


def test_nonfinite_failure_reports_all_affected_trace_numbers_and_preserves_input():
    raw = np.tile(scan(), (3, 1))
    raw[0, 3] = np.nan
    raw[-1, 61] = np.inf
    before = raw.copy()
    with pytest.raises(ValueError, match="非有限") as caught:
        run(raw)
    assert caught.value.nonfinite_trace_numbers == [4, 62]
    np.testing.assert_array_equal(raw, before)


def test_nonfinite_error_serialization_retains_all_trace_numbers_after_message_truncation():
    import json
    from mygpr.domain.common.errors import error_info_from_exception

    raw = np.ones((180, 80), np.float32)
    indices = np.r_[np.arange(40), 69]
    raw[0, indices] = np.nan
    with pytest.raises(ValueError) as caught:
        run(raw)
    saved = json.loads(
        json.dumps(error_info_from_exception(caught.value, category="job", context={"job_id": "test"}).to_dict())
    )
    assert saved["context"]["nonfinite_trace_numbers"] == (indices + 1).tolist()
    assert saved["context"]["job_id"] == "test"
    assert saved["error_code"] == "MYGPR_SHORT_TRACE_NONFINITE_INPUT"


def test_zero_quality_preview_overrides_old_local_label_and_has_specific_issue():
    from core.quality_trace_locations import (
        trace_quality_preview_mask,
        trace_local_change_preview_mask,
        locate_trace_issues,
    )

    metadata = {"short_trace_zero_mask": np.array([0, 1, 1]), "short_trace_local_change_mask": np.array([0, 1, 1])}
    np.testing.assert_array_equal(trace_quality_preview_mask(metadata, 3), [0, 1, 1])
    assert not trace_local_change_preview_mask(metadata, 3).any()
    issues = locate_trace_issues(metadata, 3)
    assert ("short_trace_zero", 1, 3) in {(i.code, i.trace_start, i.trace_end) for i in issues}


def test_zero_quality_mask_survives_equal_distance_resampling():
    from mygpr.infrastructure.processing.algorithms.motion.trace_utils import resample_trace_metadata

    result = resample_trace_metadata(
        {"trace_distance_m": np.arange(5.0), "short_trace_zero_mask": np.array([0, 1, 0, 0, 0], np.int8)},
        target_trace_distance_m=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
    )
    np.testing.assert_array_equal(result["short_trace_zero_mask"], [0, 1, 1, 1, 0])


def test_nonzero_statistical_classes_stay_separate_from_zero_quality():
    raw = scan()
    raw[:, 20:22] += np.sin(np.arange(60) / 3).astype(np.float32)[:, None]
    raw[:20, 10] += 1
    raw[:, 40:42] = 0
    out = run(raw)
    assert np.flatnonzero(out.trace_metadata["short_trace_candidate_mask"]).tolist() == [20, 21]
    assert np.flatnonzero(out.trace_metadata["short_trace_local_change_mask"]).tolist() == [10]
    assert np.flatnonzero(out.trace_metadata["short_trace_zero_mask"]).tolist() == [40, 41]


@pytest.mark.parametrize("scheme,samples,spec", [("auto", 4, ""), ("manual", 60, "0:19")])
def test_insufficient_independent_windows_do_not_enable_zero_interpolation(scheme, samples, spec):
    raw = scan()[:samples].copy()
    raw[:, 20] = 0
    out = run(
        raw, {"time_axis_ns": np.arange(float(samples))}, mode="interpolate", window_mode=scheme, time_windows_ns=spec
    )
    assert out.trace_metadata["short_trace_zero_mask"][20]
    assert not out.trace_metadata["short_trace_repaired_mask"].any()
    np.testing.assert_array_equal(out.data, raw)
