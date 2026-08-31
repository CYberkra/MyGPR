import numpy as np

from core.field_processing_bridge import (
    HIDDEN_FIELD_METHOD_IDS,
    get_field_method_categories,
    recommended_params,
    run_registered_method,
)
from core.gpr_data_model import GPRDataSet


def test_field_categories_hide_spacing_changing_method():
    categories = get_field_method_categories()
    assert categories
    exposed = {method.method_id for methods in categories.values() for method in methods}
    assert HIDDEN_FIELD_METHOD_IDS.isdisjoint(exposed)
    assert "subtracting_average_2D" in exposed
    assert "dewow" in exposed


def test_registered_dewow_runs_on_dataset():
    dataset = GPRDataSet.synthetic("L03", rows=96, cols=128, length_m=32.0)
    params = recommended_params("dewow", dataset)
    result, manifest = run_registered_method(dataset, "dewow", params)
    assert result.matrix.shape == dataset.matrix.shape
    assert manifest["method_id"] == "dewow"
    assert manifest["input_shape"] == [96, 128]


def test_registered_background_and_gain_run_on_dataset():
    dataset = GPRDataSet.synthetic("L03", rows=96, cols=128, length_m=32.0)
    bg_params = {"ntraces": 21, "time_start_ns": 0.0, "time_end_ns": 0.0}
    bg, bg_manifest = run_registered_method(dataset, "subtracting_average_2D", bg_params)
    assert bg.matrix.shape == dataset.matrix.shape
    assert bg_manifest["method_id"] == "subtracting_average_2D"
    gain, gain_manifest = run_registered_method(bg, "sec_gain", recommended_params("sec_gain", bg))
    assert gain.matrix.shape == dataset.matrix.shape
    assert np.isfinite(gain.matrix).all()
    assert gain_manifest["method_id"] == "sec_gain"
