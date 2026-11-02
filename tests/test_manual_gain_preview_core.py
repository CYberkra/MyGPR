"""Read-only gain input replay and exact full-resolution preview mathematics."""
from types import SimpleNamespace

import numpy as np
import pytest

from mygpr.application.processing.gain_preview import (
    PREFIX_REQUIRING_SAVED_INPUT, prepare_gain_preview,
)
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep, ProcessingRequest
from mygpr.infrastructure.processing.algorithms.methods import AXIS_TRANSFORM_BY_METHOD
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor
from ui.desktop_backend_facade import evaluate_manual_gain_preview, manual_gain_preview_bundle


@pytest.fixture
def project(tmp_path):
    from mygpr.interfaces.backend import MyGPRBackend
    backend = MyGPRBackend.create_default(max_workers=1)
    summary = backend.projects.create_project(tmp_path / "preview", name="Preview")
    raw = np.random.default_rng(15).normal(size=(31, 12)).astype(np.float32)
    backend.projects.save_line_dataset(summary.project_id, "L01", raw, time_window_ns=300, length_m=11)
    yield backend, summary.project_id, raw
    backend.shutdown()


def test_enabled_preceding_steps_match_real_pipeline_and_do_not_save(project):
    backend, pid, raw = project
    prefix = [
        {"method_id": "trace_cut", "params": {"trace_start": 3, "trace_end": 9}},
        {"method_id": "time_cut", "params": {"mode": "keep_range", "time_start_ns": 30, "time_end_ns": 90}},
        {"method_id": "amplitude_scale", "params": {"mode": "scale", "scale": 500}, "enabled": False},
    ]
    prepared = prepare_gain_preview(backend.projects, backend.processing, pid, "L01", "", prefix)
    source = backend.projects.read_dataset(pid, "L01")
    nodes = [{"tau": 0, "db": -6}, {"tau": 1, "db": 20}]
    pipeline = PipelineDefinition(steps=tuple(
        [PipelineStep(s["method_id"], s["params"]) for s in prefix if s.get("enabled", True)]
        + [PipelineStep("manual_gain", {"nodes": nodes})]))
    expected = backend.processing.execute_pipeline(source.data, pipeline,
        header_info=source.header_info, trace_metadata=source.trace_metadata)
    actual = evaluate_manual_gain_preview(prepared["matrix"], nodes)
    np.testing.assert_array_equal(actual, expected.data)
    np.testing.assert_array_equal(prepared["header"]["time_axis_ns"], expected.header_info["time_axis_ns"])
    assert prepared["prefix_count"] == 2 and actual.shape[1] == 7
    np.testing.assert_array_equal(backend.projects.read_dataset(pid, "L01").data, raw)
    assert not backend.projects.list_artifacts(pid, "L01")


def test_preparing_saved_result_uses_that_result_not_the_raw_source(project):
    backend, pid, raw = project
    nodes = [{"tau": 0, "db": 20}, {"tau": 1, "db": 20}]
    saved = backend.project_processing.execute_pipeline(pid, "L01", PipelineDefinition(
        steps=(PipelineStep("manual_gain", {"nodes": nodes}),)))
    prepared = prepare_gain_preview(backend.projects, backend.processing, pid, "L01", saved.artifact_id, [])
    np.testing.assert_allclose(prepared["matrix"], raw * 10, rtol=1e-7)
    assert len(backend.projects.list_artifacts(pid, "L01")) == 1


@pytest.mark.parametrize("shape", [(1, 7), (501, 1918)])
def test_preview_bundle_keeps_every_sample_trace_and_stored_axis(shape):
    raw = np.zeros(shape, np.float32)
    raw[-1, -1] = -3
    time = np.arange(shape[0], dtype=np.float64) * 1.8 + 20
    distance = np.arange(shape[1], dtype=np.float64) * 0.02
    bundle = manual_gain_preview_bundle(raw, {"time_axis_ns": time, "distance_axis_m": distance})
    assert bundle.matrix.shape == shape
    np.testing.assert_array_equal(bundle.matrix, raw)
    np.testing.assert_array_equal(bundle.sample_axis, time)
    np.testing.assert_array_equal(bundle.trace_indices, np.arange(shape[1]))
    assert bundle.sample_axis_label == "时间 (ns)"


def test_preview_uses_the_same_nonfinite_normalization_as_actual_gain():
    raw = np.array([[np.nan, -2], [4, np.inf], [-np.inf, 1]], np.float32)
    before = raw.copy()
    nodes = [{"tau": 0, "db": -6}, {"tau": 1, "db": 20}]
    actual = evaluate_manual_gain_preview(raw, nodes)
    expected = NativeProcessingExecutor().execute(ProcessingRequest(raw, "manual_gain", {"nodes": nodes}))
    np.testing.assert_array_equal(actual, expected.data)
    np.testing.assert_array_equal(raw, before)
    assert expected.runtime_warnings[0]["code"] == "data_sanitized"


def test_expanding_prefixes_are_rejected_before_any_read_or_allocation():
    def forbidden(*args, **kwargs):
        pytest.fail("preview read or expansion started before preflight")
    projects = SimpleNamespace(get_dataset_info=forbidden)
    processing = SimpleNamespace(get_method=lambda mid: SimpleNamespace(name=mid), execute_method=forbidden)
    for method in PREFIX_REQUIRING_SAVED_INPUT:
        with pytest.raises(ValueError, match="先运行"):
            prepare_gain_preview(projects, processing, "P", "L01", "", [{"method_id": method}])


def test_expandable_axis_contracts_require_an_explicit_preview_budget_decision():
    expanding = {mid for mid, axis in AXIS_TRANSFORM_BY_METHOD.items()
                 if axis not in {"identity", "sample_crop", "trace_crop"}}
    assert expanding == PREFIX_REQUIRING_SAVED_INPUT


def test_oversize_input_is_rejected_without_reading_full_data():
    projects = SimpleNamespace(get_dataset_info=lambda *_: SimpleNamespace(shape=(4000, 4000)),
        iter_dataset_blocks=lambda *a, **kw: pytest.fail("oversize matrix read"))
    with pytest.raises(ValueError, match="不会降采样"):
        prepare_gain_preview(projects, None, "P", "L01", "", [])
