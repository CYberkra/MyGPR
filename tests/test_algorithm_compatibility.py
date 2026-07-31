import numpy as np

from core.field_processing_bridge import (
    COMPATIBILITY_CHECK_METHOD_IDS,
    check_method_compatibility,
    get_field_method_categories,
    recommended_params,
    run_priority_compatibility_checks,
    run_registered_method,
)
from core.gpr_data_model import GPRDataSet


PRIORITY_METHODS = [
    "dewow",
    "subtracting_average_2D",
    "median_background_2D",
    "svd_bg",
    "frequency_filter_1d",
    "sec_gain",
    "agcGain",
    "trace_median_filter",
    "wavelet_2d",
    "time_to_depth",
]


def test_v0873_priority_method_list_is_explicit_and_stable():
    assert COMPATIBILITY_CHECK_METHOD_IDS == PRIORITY_METHODS


def test_priority_registered_methods_execute_with_finite_output():
    dataset = GPRDataSet.synthetic("L03", rows=160, cols=96, length_m=40.0)
    for method_id in PRIORITY_METHODS:
        params = recommended_params(method_id, dataset)
        result, manifest = run_registered_method(dataset, method_id, params)
        assert result.trace_count == dataset.trace_count, method_id
        assert np.isfinite(result.matrix).all(), method_id
        assert manifest["status"] == "success"
        assert manifest["method_id"] == method_id
        assert manifest["input_shape"] == [160, 96]
        assert manifest["trace_count_changed"] is False
        assert "runtime_params" in manifest


def test_time_to_depth_records_sample_count_change_but_preserves_traces():
    dataset = GPRDataSet.synthetic("L03", rows=160, cols=96, length_m=40.0)
    result, manifest = run_registered_method(
        dataset,
        "time_to_depth",
        recommended_params("time_to_depth", dataset),
    )
    assert result.trace_count == dataset.trace_count
    assert result.sample_count != dataset.sample_count
    assert manifest["sample_count_changed"] is True
    assert manifest["trace_count_changed"] is False


def test_compatibility_records_report_passed_methods():
    records = run_priority_compatibility_checks(
        GPRDataSet.synthetic("L03", rows=160, cols=96, length_m=40.0)
    )
    by_method = {record.method_id: record for record in records}
    assert set(by_method) == set(PRIORITY_METHODS)
    for method_id, record in by_method.items():
        assert record.params_ok, method_id
        assert record.execution_ok, method_id
        assert record.trace_count_preserved, method_id
        assert record.finite_output, method_id
        assert record.status == "通过", method_id


def test_hidden_spacing_changing_method_still_not_exposed():
    categories = get_field_method_categories()
    exposed = {m.method_id for methods in categories.values() for m in methods}
    assert "equidistant_trace_resample" not in exposed
