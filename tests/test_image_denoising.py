"""Image denoisers operate on signed radar amplitudes, never display pixels."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from mygpr.domain.processing.models import PipelineDefinition, PipelineStep, ProcessingRequest
from mygpr.infrastructure.processing.algorithms.methods import NATIVE_ALGORITHMS
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor


def run(method, data, **params):
    assert method in NATIVE_ALGORITHMS, f"missing native denoiser: {method}"
    return NativeProcessingExecutor().execute(ProcessingRequest(
        data=data, method_id=method, params=params,
        header_info={"time_axis_ns": np.arange(data.shape[0]) * 2.0},
        trace_metadata={"trace_id": np.arange(data.shape[1]) + 100},
    ))


def test_gaussian_matches_scipy_without_rescaling_or_changing_axes():
    raw = np.random.default_rng(5).normal(size=(31, 29)).astype(np.float32) * 20
    before = raw.copy()
    result = run("gaussian_smoothing", raw, sigma_samples=0.7, sigma_traces=1.3)
    expected = gaussian_filter(raw.astype(float), (0.7, 1.3), mode="reflect", truncate=4)
    np.testing.assert_allclose(result.data, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(result.header_info["time_axis_ns"], np.arange(31) * 2)
    np.testing.assert_array_equal(result.trace_metadata["trace_id"], np.arange(29) + 100)
    assert result.data.dtype == np.float32


def test_gaussian_can_smooth_traces_without_mixing_time_samples():
    raw = np.zeros((9, 11), np.float32)
    raw[4, 5] = -100
    result = run("gaussian_smoothing", raw, sigma_samples=0, sigma_traces=1)
    assert np.count_nonzero(result.data[:4]) == np.count_nonzero(result.data[5:]) == 0
    assert result.data[4, 5] < -30
    assert result.data[4].sum() == pytest.approx(-100, abs=1e-5)


def _brute_nlm(raw, patch_samples, patch_traces, search_samples, search_traces, h):
    # Independent small scalar oracle, explicit patch and search loops.
    pr, pc = patch_samples // 2, patch_traces // 2
    hr, hc = pr + search_samples, pc + search_traces
    padded = np.pad(raw.astype(float), ((hr, hr), (hc, hc)), mode="symmetric")
    out = np.empty_like(raw, dtype=float)
    for r, c in np.ndindex(raw.shape):
        r, c = r + hr, c + hc
        reference = padded[r-pr:r+pr+1, c-pc:c+pc+1]
        total, weight_sum = 0.0, 0.0
        for dr in range(-search_samples, search_samples+1):
            for dc in range(-search_traces, search_traces+1):
                patch = padded[r+dr-pr:r+dr+pr+1, c+dc-pc:c+dc+pc+1]
                weight = np.exp(-np.mean((reference-patch)**2) / h**2)
                total += weight * padded[r+dr, c+dc]
                weight_sum += weight
        out[r-hr, c-hc] = total / weight_sum
    return out


def test_nlm_matches_independent_patch_distance_oracle_at_edges():
    raw = np.random.default_rng(6).normal(size=(7, 8)).astype(np.float32) * 4
    params = dict(patch_samples=3, patch_traces=5, search_samples=2, search_traces=1,
                  h=2.5, h_mode="absolute")
    result = run("non_local_means", raw, **params)
    expected = _brute_nlm(raw, 3, 5, 2, 1, 2.5)
    np.testing.assert_allclose(result.data, expected, rtol=1e-6, atol=1e-6)
    assert result.data.min() < 0 < result.data.max()


@pytest.mark.parametrize("method", ["gaussian_smoothing", "non_local_means"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (13, 19)])
def test_constant_signed_data_and_single_axes_are_preserved(method, shape):
    raw = np.full(shape, -123.25, np.float32)
    result = run(method, raw)
    np.testing.assert_array_equal(result.data, raw)
    assert not np.shares_memory(result.data, raw)


def test_nlm_relative_strength_is_scale_equivariant():
    raw = np.random.default_rng(11).normal(size=(23, 27)).astype(np.float32)
    params = dict(h=0.8, search_samples=2, search_traces=2)
    out = run("non_local_means", raw, **params)
    scaled = run("non_local_means", raw * -100, **params)
    np.testing.assert_allclose(scaled.data, out.data * -100, rtol=2e-6, atol=2e-5)
    assert out.metadata["h_effective"] == pytest.approx(0.8 * np.sqrt(np.mean(raw.astype(float)**2)))


def test_nlm_reduces_known_noise_while_preserving_a_signed_reflector():
    rows, cols = np.indices((64, 72))
    clean = (np.exp(-((rows-27)/2)**2) - np.exp(-((rows-31)/2)**2)) * 2
    noisy = (clean + np.random.default_rng(22).normal(0, 0.2, clean.shape)).astype(np.float32)
    out = run("non_local_means", noisy, h_mode="absolute", h=0.3,
              patch_samples=3, patch_traces=5, search_samples=2, search_traces=4).data
    assert np.mean((out-clean)**2) < np.mean((noisy-clean)**2) * 0.55
    assert out[27].mean() > 1.5 and out[31].mean() < -1.5


def test_extreme_impulse_does_not_change_distances_in_remote_patches():
    from mygpr.infrastructure.processing.algorithms.image_denoising import file_nlm
    raw = np.random.default_rng(1).normal(0, 0.2, (157, 31)).astype(np.float32)
    raw[0, 15] = 1e8
    params = dict(patch_samples=3, patch_traces=3, search_samples=1, search_traces=1,
                  h=0.4, h_mode="absolute")
    result = run("non_local_means", raw, **params).data
    tiny_tiles = np.empty_like(raw)
    file_nlm(raw, tiny_tiles, params, None, 1)
    np.testing.assert_array_equal(result, tiny_tiles)
    # The impulse cannot affect a patch that is dozens of rows away.
    crop = raw[35:54, 8:23]
    expected = _brute_nlm(crop, 3, 3, 1, 1, 0.4)
    np.testing.assert_allclose(result[39:50, 12:19], expected[4:-4, 4:-4], atol=2e-8)


def test_tiny_positive_nlm_strength_is_finite_identity_limit():
    raw = np.random.default_rng(17).normal(size=(9, 11)).astype(np.float32)
    with np.errstate(all="raise"):
        result = run("non_local_means", raw, h=1e-200, h_mode="absolute")
    np.testing.assert_array_equal(result.data, raw)


@pytest.mark.parametrize("method", ["gaussian_smoothing", "non_local_means"])
def test_file_backed_estimate_is_bounded_for_large_matrices(method):
    from mygpr.infrastructure.processing.block_executor import FileBackedBlockPipelineExecutor
    pipeline = PipelineDefinition(steps=(PipelineStep(method, {}),))
    estimate = FileBackedBlockPipelineExecutor().estimate((16384, 131072), "float32", pipeline)
    assert estimate.memory_bytes < 512 * 1024**2
    assert estimate.temporary_disk_bytes >= 16 * 1024**3


def test_adding_a_bounded_filter_does_not_reduce_other_operator_memory_budget():
    from mygpr.infrastructure.processing.block_executor import FileBackedBlockPipelineExecutor
    pipeline = PipelineDefinition(steps=(PipelineStep("gaussian_smoothing", {}), PipelineStep("svd_bg", {})))
    estimate = FileBackedBlockPipelineExecutor().estimate((16384, 131072), "float32", pipeline)
    assert estimate.memory_bytes >= 8 * 1024**3 * NATIVE_ALGORITHMS["svd_bg"].memory_multiplier


@pytest.mark.parametrize("method,params", [
    ("gaussian_smoothing", {"sigma_samples": -1}),
    ("gaussian_smoothing", {"sigma_traces": float("nan")}),
    ("non_local_means", {"h": -1}),
    ("non_local_means", {"h": float("inf")}),
    ("non_local_means", {"patch_samples": 4}),
    ("non_local_means", {"patch_traces": 2.5}),
    ("non_local_means", {"search_samples": 100}),
    ("non_local_means", {"h_mode": "uint8"}),
])
def test_invalid_parameters_are_rejected(method, params):
    assert method in NATIVE_ALGORITHMS
    with pytest.raises(ValueError):
        run(method, np.ones((5, 8), np.float32), **params)


@pytest.mark.parametrize("method", ["gaussian_smoothing", "non_local_means"])
def test_file_backed_tiles_have_no_seams_and_preserve_source(tmp_path, method):
    from tests.test_native_processing_backend import ArrayBlockSource
    from mygpr.infrastructure.processing.block_executor import FileBackedBlockPipelineExecutor
    raw = np.random.default_rng(3).normal(size=(157, 311)).astype(np.float32)
    params = ({"sigma_samples": 1.2, "sigma_traces": 2.0} if method == "gaussian_smoothing"
              else {"h": 0.7, "search_samples": 2, "search_traces": 3})
    expected = run(method, raw, **params).data
    if method == "gaussian_smoothing":
        np.testing.assert_allclose(expected, gaussian_filter(raw.astype(float), (1.2, 2.0)), atol=1e-7)
    executor = FileBackedBlockPipelineExecutor(tmp_path / "scratch", block_bytes=4096)
    actual, summary = executor.execute(
        ArrayBlockSource(raw), PipelineDefinition(steps=(PipelineStep(method, params),)),
        header_info={}, trace_metadata={}, consumer=lambda matrix, summary: (matrix.copy(), summary),
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
    assert summary.step_records[0].metadata["tile_count"] > 1
    assert list((tmp_path / "scratch").iterdir()) == []


def test_nlm_polls_cancellation_inside_search_loop():
    assert "non_local_means" in NATIVE_ALGORITHMS
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 6
    with pytest.raises(RuntimeError, match="cancel"):
        run("non_local_means", np.ones((33, 39), np.float32), cancel_checker=cancelled)


def test_cancelled_file_nlm_removes_partially_written_workspace(tmp_path):
    from tests.test_native_processing_backend import ArrayBlockSource
    from mygpr.application.jobs.cancellation import CancellationTokenSource, JobCancelledError
    from mygpr.application.jobs.context import ExecutionContext
    from mygpr.infrastructure.processing.block_executor import FileBackedBlockPipelineExecutor
    cancellation = CancellationTokenSource()
    def progress(done, total, message):
        if message.startswith("non_local_means:"):
            cancellation.cancel()
    context = ExecutionContext(cancellation_token=cancellation.token, progress_callback=progress)
    executor = FileBackedBlockPipelineExecutor(tmp_path / "cancelled")
    raw = np.random.default_rng(12).normal(size=(157, 311)).astype(np.float32)
    with pytest.raises(JobCancelledError):
        executor.execute(ArrayBlockSource(raw), PipelineDefinition(steps=(PipelineStep("non_local_means", {}),)),
                         header_info={}, trace_metadata={}, context=context,
                         consumer=lambda matrix, summary: pytest.fail("cancelled result was delivered"))
    assert list((tmp_path / "cancelled").iterdir()) == []


@pytest.mark.parametrize("method,params", [
    ("gaussian_smoothing", {"sigma_samples": 0, "sigma_traces": 0}),
    ("non_local_means", {"h": 0}),
])
def test_disabled_strength_is_an_exact_identity(method, params):
    raw = np.random.default_rng(9).normal(size=(23, 29)).astype(np.float32)
    np.testing.assert_array_equal(run(method, raw, **params).data, raw)


@pytest.mark.parametrize("method", ["gaussian_smoothing", "non_local_means"])
def test_nonfinite_data_fails_clearly(method):
    raw = np.ones((5, 8), np.float32)
    raw[1, 2] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        run(method, raw)


def test_saved_pipeline_retains_both_filters_parameters_and_original(tmp_path):
    from mygpr.interfaces.backend import MyGPRBackend
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "project", name="Denoise")
        raw = np.random.default_rng(3).normal(size=(31, 37)).astype(np.float32)
        backend.projects.save_line_dataset(project.project_id, "L01", raw, time_window_ns=900)
        pipeline = PipelineDefinition(steps=(
            PipelineStep("gaussian_smoothing", {"sigma_samples": 0, "sigma_traces": 0.8}),
            PipelineStep("non_local_means", {"h": 0.7, "search_samples": 1, "search_traces": 2}),
        ))
        saved = backend.project_processing.execute_pipeline(project.project_id, "L01", pipeline)
        data = backend.projects.read_artifact_dataset(project.project_id, "L01", saved.artifact_id)
        expected = run("gaussian_smoothing", raw, **pipeline.steps[0].params).data
        expected = run("non_local_means", expected, **pipeline.steps[1].params).data
        np.testing.assert_allclose(data.data, expected, atol=2e-7)
        np.testing.assert_array_equal(backend.projects.read_dataset(project.project_id, "L01").data, raw)
        job = backend.submit_processing_result_package(
            project.project_id, "L01", str(tmp_path / "denoise.zip"), artifact_id=saved.artifact_id)
        result = backend.jobs.wait(job, timeout=30)
        assert result.status.value == "completed", result.error_message
        import json
        import zipfile
        with zipfile.ZipFile(result.result.archive_path) as package:
            manifest = json.loads(package.read("manifest.json"))
            record = json.loads(package.read(manifest["steps"][-1]["parameters"]))
            assert [s["method_id"] for s in record["params"]["pipeline"]] == [
                "gaussian_smoothing", "non_local_means"]
            assert record["params"]["lineage"][-1]["metadata"]["h_mode"] == "relative_rms"
    finally:
        backend.shutdown()


def test_filter_forms_are_discoverable_and_roundtrip_parameters(qapp):
    from ui.desktop_backend_facade import method_catalog
    from ui.widgets.param_form import ParamForm
    catalog = {entry.method_id: entry for entry in method_catalog()}
    for mid in ("gaussian_smoothing", "non_local_means"):
        assert mid in catalog
        assert catalog[mid].category == "denoising"
        form = ParamForm()
        form.set_schema(catalog[mid].parameter_schema)
        params = form.values()
        assert params
        if mid == "non_local_means":
            form._editors["patch_samples"].stepUp()
            assert form.values()["patch_samples"] == 5
            assert "弱反射" in form._editors["h"].toolTip()
        assert run(mid, np.ones((9, 11), np.float32), **params).data.shape == (9, 11)
        form.deleteLater()


def test_parameter_step_and_hint_changes_rebuild_existing_form(qapp):
    from ui.widgets.param_form import ParamForm
    form = ParamForm()
    schema = {"name": "patch", "type": "int", "default": 3, "min": 1, "max": 15}
    form.set_schema([schema])
    form.set_schema([{**schema, "step": 2, "tooltip": "必须为奇数"}])
    assert form._editors["patch"].singleStep() == 2
    assert form._editors["patch"].toolTip() == "必须为奇数"
    form.deleteLater()
