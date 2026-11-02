"""Manual curve gain is a processing step, with acquisition-compatible dB rules."""
from __future__ import annotations

import numpy as np
import json
import hashlib
import pytest

from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.algorithms.methods import NATIVE_ALGORITHMS
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor


def test_manual_curve_gain_changes_results_without_changing_source_or_axes():
    assert "manual_gain" in NATIVE_ALGORITHMS
    raw = np.array([[1, -2], [2, -4], [3, -6]], dtype=np.float32)
    before = raw.copy()
    axis = np.array([100, 200, 300], dtype=np.float32)
    trace_ids = np.array([0, 25], dtype=np.int64)
    nodes = [{"tau": 0.0, "db": 0.0}, {"tau": 1.0, "db": 20.0}]
    result = NativeProcessingExecutor().execute(ProcessingRequest(
        data=raw, method_id="manual_gain", params={"nodes": nodes},
        header_info={"time_axis_ns": axis}, trace_metadata={"trace_id": trace_ids},
    ))
    # The acquisition display uses dB interpolation, then 10 ** (dB / 20).
    expected = before * np.array([1.0, np.sqrt(10.0), 10.0])[:, None]
    np.testing.assert_allclose(result.data, expected, rtol=1e-7)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(result.header_info["time_axis_ns"], axis)
    np.testing.assert_array_equal(result.trace_metadata["trace_id"], trace_ids)
    assert result.metadata["nodes"] == nodes
    assert result.metadata["position_basis"] == "current_input_samples"
    assert result.data.dtype == np.float32


@pytest.mark.parametrize("samples", [1, 7, 501])
def test_default_zero_db_is_an_exact_identity(samples):
    raw = np.random.default_rng(4).normal(size=(samples, 4)).astype(np.float32)
    result = NativeProcessingExecutor().execute(ProcessingRequest(
        data=raw, method_id="manual_gain", params={},
    ))
    np.testing.assert_array_equal(result.data, raw)
    assert not np.shares_memory(result.data, raw)


@pytest.mark.parametrize("nodes", [
    [], [{"tau": 0, "db": 0}],
    [{"tau": 0.1, "db": 0}, {"tau": 1, "db": 0}],
    [{"tau": 0, "db": 0}, {"tau": 0.5, "db": 0}, {"tau": 0.5, "db": 0}, {"tau": 1, "db": 0}],
    [{"tau": 0, "db": 0}, {"tau": 1, "db": 61}],
    [{"tau": 0, "db": float("nan")}, {"tau": 1, "db": 0}],
    [{"tau": 0, "db": 0}, {"tau": float("inf"), "db": 0}],
    [{"tau": False, "db": 0}, {"tau": 1, "db": 0}],
    [{"tau": 0, "db": 0}, {"tau": 1, "db": None}],
    "not JSON", "[]", "x" * 16385,
])
def test_invalid_curves_fail_instead_of_repairing_or_clipping(nodes):
    with pytest.raises(ValueError, match="增益"):
        NativeProcessingExecutor().execute(ProcessingRequest(
            data=np.ones((5, 2), np.float32), method_id="manual_gain", params={"nodes": nodes},
        ))


def test_gain_overflow_is_rejected_instead_of_saving_infinity():
    with pytest.raises(ValueError, match="float32"):
        NativeProcessingExecutor().execute(ProcessingRequest(
            data=np.full((3, 2), 3e38, np.float32), method_id="manual_gain",
            params={"nodes": [{"tau": 0, "db": 60}, {"tau": 1, "db": 60}]},
        ))


def test_piecewise_json_nodes_have_a_fixed_signed_amplitude_golden():
    nodes = [{"tau": 0, "db": 0}, {"tau": 0.5, "db": 20}, {"tau": 1, "db": -20}]
    result = NativeProcessingExecutor().execute(ProcessingRequest(
        data=np.tile([1.0, -1.0], (5, 1)).astype(np.float32), method_id="manual_gain",
        params={"nodes": json.dumps(nodes)},
    ))
    known = np.array([[1, -1], [np.sqrt(10), -np.sqrt(10)], [10, -10], [1, -1], [0.1, -0.1]], np.float32)
    assert hashlib.sha256(result.data.tobytes()).digest() == hashlib.sha256(known.tobytes()).digest()


def test_file_backed_gain_keeps_the_full_sample_curve_in_each_column_block(tmp_path):
    from tests.test_native_processing_backend import ArrayBlockSource
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    from mygpr.infrastructure.processing.block_executor import FileBackedBlockPipelineExecutor

    raw = np.random.default_rng(7).normal(size=(501, 83)).astype(np.float32)
    nodes = '[{"tau":0,"db":-6},{"tau":0.25,"db":10},{"tau":1,"db":20}]'
    expected = NativeProcessingExecutor().execute(ProcessingRequest(
        data=raw, method_id="manual_gain", params={"nodes": nodes},
    )).data
    executor = FileBackedBlockPipelineExecutor(tmp_path / "blocks", block_bytes=32768)
    actual, summary = executor.execute(
        ArrayBlockSource(raw), PipelineDefinition(steps=(PipelineStep("manual_gain", {"nodes": nodes}),)),
        header_info={}, trace_metadata={}, consumer=lambda matrix, summary: (matrix.copy(), summary),
    )
    np.testing.assert_array_equal(actual, expected)
    assert summary.step_records[0].method_id == "manual_gain"
    assert list((tmp_path / "blocks").iterdir()) == []


def test_saved_project_result_keeps_nodes_and_raw_unchanged(tmp_path):
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    from mygpr.interfaces.backend import MyGPRBackend

    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "project", name="Gain")
        raw = np.tile([1.0, -1.0], (5, 1)).astype(np.float32)
        backend.projects.save_line_dataset(project.project_id, "L01", raw, time_window_ns=900)
        nodes = '[{"tau":0,"db":0},{"tau":0.5,"db":20},{"tau":1,"db":-20}]'
        pipeline = PipelineDefinition(steps=(PipelineStep("manual_gain", {"nodes": nodes}),))
        saved = backend.project_processing.execute_pipeline(project.project_id, "L01", pipeline)
        reread = backend.projects.read_artifact_dataset(project.project_id, "L01", saved.artifact_id)
        expected = NativeProcessingExecutor().execute(ProcessingRequest(
            data=raw, method_id="manual_gain", params={"nodes": nodes},
        )).data
        np.testing.assert_array_equal(reread.data, expected)
        source = backend.projects.read_dataset(project.project_id, "L01")
        np.testing.assert_array_equal(source.data, raw)
        artifacts = backend.projects.list_artifacts(project.project_id, "L01")
        assert artifacts[0].method_id == "manual_gain"
        # The result package is the supported public API for saved parameters.
        job = backend.submit_processing_result_package(
            project.project_id, "L01", str(tmp_path / "gain.zip"), artifact_id=saved.artifact_id,
        )
        result = backend.jobs.wait(job, timeout=30)
        assert result.status.value == "completed", result.error_message
        import zipfile
        with zipfile.ZipFile(result.result.archive_path) as package:
            manifest = json.loads(package.read("manifest.json"))
            params = json.loads(package.read(manifest["steps"][0]["parameters"]))
            assert params["params"]["pipeline"][0]["params"]["nodes"] == nodes
            assert params["params"]["lineage"][0]["metadata"]["nodes"] == json.loads(nodes)
    finally:
        backend.shutdown()
