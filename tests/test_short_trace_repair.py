"""Public contracts for explicit, auditable short-trace interpolation."""
import numpy as np
import pytest

from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.native_adapter import NativeProcessingCatalog, NativeProcessingExecutor


def test_method_is_visible_experimental_and_defaults_to_mark_only():
    catalog = NativeProcessingCatalog()
    descriptor = catalog.get('short_trace_repair')
    assert descriptor is not None, 'Independent short-run repair is missing from the method library'
    assert descriptor.visibility == 'public'
    assert not descriptor.auto_tune_enabled
    assert descriptor.parameter_schema['mode']['default'] == 'mark'
    assert catalog.raw_metadata('short_trace_repair')['maturity'] == 'experimental'


def _scan():
    time = np.linspace(0., 900., 101)
    wave = (np.cos(time / 13.) * np.exp(-time / 180.)).astype(np.float32)
    clean = np.repeat(wave[:, None], 65, axis=1)
    noisy = clean.copy()
    noisy[:, 20:22] += (0.5 * np.sin(time / 5. + .4)).astype(np.float32)[:, None]
    return noisy, clean, {'time_axis_ns': time}


def _execute(data, header=None, trace_metadata=None, **params):
    return NativeProcessingExecutor().execute(ProcessingRequest(
        data=data, method_id='short_trace_repair', params=params, header_info=header or {}, trace_metadata=trace_metadata or {}))


def test_default_mark_preserves_data_and_reports_candidates_without_repair():
    raw, _, header = _scan()
    snapshot = raw.copy()
    result = _execute(raw, header, time_windows_ns='0:100;100:200;250:850')
    np.testing.assert_array_equal(result.data, snapshot)
    np.testing.assert_array_equal(raw, snapshot)
    assert np.flatnonzero(result.trace_metadata['short_trace_candidate_mask']).tolist() == [20, 21]
    assert not result.trace_metadata['short_trace_repaired_mask'].any()
    assert result.metadata['repaired_trace_count'] == 0


def test_interpolation_reconstructs_short_run_and_preserves_all_other_samples():
    raw, clean, header = _scan()
    result = _execute(raw, header, mode='interpolate', time_windows_ns='0:100;100:200;250:850')
    np.testing.assert_array_equal(result.data, clean)
    assert np.flatnonzero(result.trace_metadata['short_trace_repaired_mask']).tolist() == [20, 21]
    assert result.metadata['repaired_trace_count'] == 2
    assert not result.trace_metadata['short_trace_unresolved_mask'].any()
    assert not np.shares_memory(result.data, raw)


def test_unbounded_or_edge_runs_are_marked_without_extrapolation():
    raw, _, _ = _scan()
    raw[:, :2] += 10
    raw[:, 30:35] += 10
    result = _execute(raw, mode='interpolate', max_run_traces=2)
    np.testing.assert_array_equal(result.data[:, :2], raw[:, :2])
    np.testing.assert_array_equal(result.data[:, 30:35], raw[:, 30:35])
    assert result.trace_metadata['short_trace_unresolved_mask'][:2].all()
    assert result.trace_metadata['short_trace_unresolved_mask'][30:35].all()


def test_anomaly_in_one_gate_does_not_trigger_whole_trace_replacement():
    raw, clean, header = _scan()
    raw[:] = clean
    raw[35:45, 20] += 20  # localized later response, not broad three-gate corruption
    result = _execute(raw, header, mode='interpolate', time_windows_ns='0:100;100:200;250:850')
    np.testing.assert_array_equal(result.data, raw)
    assert not result.trace_metadata['short_trace_candidate_mask'].any()


@pytest.mark.parametrize('options', [
    {'mode': 'silent'}, {'window_traces': 4}, {'window_traces': 1},
    {'robust_z': float('nan')}, {'robust_z': 0}, {'max_run_traces': 0}, {'max_run_traces': 5},
    {'time_windows_ns': '20:10'}, {'time_windows_ns': '0:1000'},
    {'time_windows_ns': '0:1'}, {'time_windows_ns': 'x:y'},
])
def test_rejects_invalid_parameters_instead_of_silently_changing_them(options):
    raw, _, header = _scan()
    with pytest.raises(ValueError):
        _execute(raw, header, **options)


def test_explicit_time_windows_require_a_trustworthy_axis():
    raw, _, _ = _scan()
    with pytest.raises(ValueError, match='时间轴'):
        _execute(raw, {'time_axis_status': 'unknown'}, time_windows_ns='0:100')


def test_nonfinite_inputs_are_not_silently_filled():
    raw, _, _ = _scan()
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match='NaN|Inf|有限'):
        _execute(raw)


def test_preview_combines_masks_without_reclassifying_repaired_traces_as_bad():
    from core import quality_trace_locations
    metadata = {'trace_qc_bad_mask': np.array([0, 1, 0, 0]),
                'short_trace_candidate_mask': np.array([0, 0, 1, 0]),
                'short_trace_repaired_mask': np.array([0, 0, 0, 1])}
    before = metadata['trace_qc_bad_mask'].copy()
    mask = quality_trace_locations.trace_quality_preview_mask(metadata, 4)
    np.testing.assert_array_equal(mask, [0, 1, 1, 1])
    np.testing.assert_array_equal(metadata['trace_qc_bad_mask'], before)
    issues = quality_trace_locations.locate_trace_issues(metadata, 4)
    assert {(i.code, i.trace_start, i.trace_end) for i in issues} >= {
        ('short_trace_candidate', 2, 3), ('short_trace_repaired', 3, 4)}


def test_project_pipeline_keeps_original_and_saves_repair_masks(tmp_path):
    from mygpr.interfaces.backend import MyGPRBackend
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    raw, clean, _ = _scan()
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / 'short', name='short')
        backend.projects.save_line_dataset(project.project_id, 'L01', raw, time_window_ns=900.)
        marked = backend.project_processing.execute_pipeline(project.project_id, 'L01',
            PipelineDefinition(steps=(PipelineStep('short_trace_repair', {}),)))
        mark_data = backend.projects.read_artifact_dataset(project.project_id, 'L01', marked.artifact_id)
        np.testing.assert_array_equal(mark_data.data, raw)
        repaired = backend.project_processing.execute_pipeline(project.project_id, 'L01',
            PipelineDefinition(steps=(PipelineStep('short_trace_repair', {'mode': 'interpolate'}),)),
            input_artifact_id=marked.artifact_id)
        output = backend.projects.read_artifact_dataset(project.project_id, 'L01', repaired.artifact_id)
        original = backend.projects.read_dataset(project.project_id, 'L01')
        np.testing.assert_array_equal(original.data, raw)
        np.testing.assert_array_equal(output.data, clean)
        assert np.flatnonzero(output.trace_metadata['short_trace_repaired_mask']).tolist() == [20, 21]
        np.testing.assert_array_equal(output.header_info['time_axis_ns'], original.header_info['time_axis_ns'])
    finally:
        backend.shutdown()


def test_previous_repair_provenance_survives_a_later_mark_step():
    raw, _, header = _scan()
    first = _execute(raw, header, mode='interpolate')
    second = NativeProcessingExecutor().execute(ProcessingRequest(
        data=first.data, method_id='short_trace_repair', params={}, header_info=header,
        trace_metadata=first.trace_metadata))
    np.testing.assert_array_equal(second.data, first.data)
    np.testing.assert_array_equal(second.trace_metadata['short_trace_repaired_mask'], first.trace_metadata['short_trace_repaired_mask'])


def test_cancellation_does_not_modify_input():
    from mygpr.application.jobs.cancellation import JobCancelledError
    from mygpr.infrastructure.processing.algorithms.extended.short_trace_repair import method_short_trace_repair
    raw, _, _ = _scan()
    snapshot = raw.copy()
    with pytest.raises(JobCancelledError):
        method_short_trace_repair(raw, cancel_checker=lambda: True)
    np.testing.assert_array_equal(raw, snapshot)


def test_compatibility_facade_matches_native_on_the_same_input():
    from PythonModule.short_trace_repair import method_short_trace_repair
    raw, _, header = _scan()
    params = {'mode': 'interpolate', 'time_windows_ns': '0:100;100:200;250:850'}
    expected = _execute(raw, header, **params)
    result, metadata = method_short_trace_repair(raw, header_info=header, **params)
    np.testing.assert_array_equal(result, expected.data)
    np.testing.assert_array_equal(metadata['trace_metadata_updates']['short_trace_candidate_mask'], expected.trace_metadata['short_trace_candidate_mask'])


@pytest.mark.parametrize('data', [np.ones((4, 2)), np.ones((0, 5)), np.ones(5), np.ones((4, 5), dtype=complex)])
def test_invalid_shapes_and_complex_inputs_are_explicitly_rejected(data):
    with pytest.raises(ValueError):
        _execute(data)


@pytest.mark.parametrize('axis', [np.arange(5), np.zeros(101), np.r_[np.arange(100), np.nan]])
def test_malformed_time_axis_is_rejected(axis):
    raw, _, _ = _scan()
    with pytest.raises(ValueError, match='时间轴'):
        _execute(raw, {'time_axis_ns': axis}, time_windows_ns='0:50')


@pytest.mark.parametrize('amplitude', [1e-18, 1., 1e18])
def test_detection_does_not_depend_on_uncalibrated_amplitude_units(amplitude):
    raw = np.zeros((8, 65), np.float32)
    raw[:, 20] = amplitude
    result = _execute(raw, window_mode='full')
    assert np.flatnonzero(result.trace_metadata['short_trace_candidate_mask']).tolist() == [20]


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_ns_gate_includes_declared_boundary_samples_after_axis_storage(dtype):
    raw = np.zeros((11, 65), np.float32)
    raw[1:5, 20] = 1
    result = _execute(raw, {'time_axis_ns': np.linspace(0, 1, 11, dtype=dtype)}, time_windows_ns='0.1:0.4')
    gate = result.metadata['gate_diagnostics'][0]
    assert (gate['sample_start'], gate['sample_stop']) == (1, 5)
    assert np.flatnonzero(result.trace_metadata['short_trace_local_change_mask']).tolist() == [20]
    assert not result.trace_metadata['short_trace_candidate_mask'].any()


@pytest.mark.parametrize('key', ['trace_qc_bad_mask', 'short_trace_unresolved_mask', 'short_trace_repaired_mask'])
def test_known_marked_neighbors_cannot_anchor_another_interpolation(key):
    raw = np.random.default_rng(3).normal(0, .01, (64, 65)).astype(np.float32)
    raw[:, 20] = 1
    raw[:, 19] = .004
    mask = np.zeros(65, np.int8)
    mask[19] = 1
    result = _execute(raw, mode='interpolate', trace_metadata={key: mask})
    assert result.trace_metadata['short_trace_candidate_mask'][20]
    assert result.trace_metadata['short_trace_unresolved_mask'][20]
    np.testing.assert_array_equal(result.data[:, 20], raw[:, 20])


def test_repair_provenance_survives_downstream_equal_distance_interpolation():
    raw = np.random.default_rng(3).normal(0, .01, (64, 65)).astype(np.float32)
    raw[:, 20] = 1
    first = _execute(raw, mode='interpolate')
    second = NativeProcessingExecutor().execute(ProcessingRequest(
        data=first.data, method_id='motion_compensation_speed', params={'spacing_m': 1.5},
        trace_metadata={**first.trace_metadata, 'trace_distance_m': np.arange(65, dtype=float)}))
    # Output distance19.5 contains input19 and repaired20 with nonzero weights.
    np.testing.assert_allclose(second.data[:, 13], .5*(first.data[:, 19]+first.data[:, 20]))
    assert second.trace_metadata['short_trace_repaired_mask'][13]


def test_mask_resampling_uses_nonzero_support_without_bleeding_at_exact_samples():
    from mygpr.infrastructure.processing.algorithms.motion.trace_utils import resample_trace_metadata
    out = resample_trace_metadata({'trace_distance_m': np.arange(5, dtype=float),
        'short_trace_repaired_mask': np.array([0, 1, 0, 0, 0], np.int8)},
        target_trace_distance_m=np.array([0., .5, 1., 1.5, 2., 3.]))
    np.testing.assert_array_equal(out['short_trace_repaired_mask'], [0, 1, 1, 1, 0, 0])


def test_cli_runs_the_same_registered_mark_and_interpolate_method(tmp_path):
    import cli_batch
    raw, clean, _ = _scan()
    source = tmp_path / 'input.csv'
    np.savetxt(source, raw, delimiter=',')
    result = cli_batch.run_job({'id': 'short', 'input': str(source), 'methods': [
        {'key': 'short_trace_repair', 'params': {}},
        {'key': 'short_trace_repair', 'params': {'mode': 'interpolate'}},
    ]}, repo_root=str(tmp_path), output_dir=str(tmp_path / 'output'))
    assert result['status'] == 'ok'
    assert result['final_shape'] == list(clean.shape)
    output, _, _ = cli_batch.load_gpr_csv(str(tmp_path / result['final_csv']))
    np.testing.assert_allclose(output, clean, rtol=1e-6, atol=1e-7)


def test_float64_arithmetic_roundoff_does_not_drop_a_boundary_sample():
    raw = np.zeros((11, 65), np.float32)
    result = _execute(raw, {'time_axis_ns': np.linspace(0, 1, 11)}, time_windows_ns='0:0.3')
    assert result.metadata['gate_diagnostics'][0]['sample_stop'] == 4


def test_physical_windows_do_not_silently_extend_beyond_the_declared_axis():
    with pytest.raises(ValueError, match='时间轴'):
        _execute(np.zeros((11, 65), np.float32), {'time_axis_ns': np.linspace(0, 1, 11)}, time_windows_ns='0:1.0000001')
