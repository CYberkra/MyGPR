"""Execution, axis, and warning contracts for the native processing catalog."""
import numpy as np
import pytest

from core.methods_registry import ALGORITHM_CATALOG
from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.algorithms.methods import NATIVE_ALGORITHMS
from mygpr.infrastructure.processing.native_adapter import (
    NativeProcessingExecutor,
    prepare_native_params,
)


def _header():
    return {
        'a_scan_length': 32,
        'num_traces': 8,
        'total_time_ns': 32.0,
        'time_window_ns': 32.0,
        'trace_interval_m': 0.1,
        'length_m': 0.8,
        'track_length_m': 0.8,
    }


def _trace_metadata():
    return {
        'trace_index': np.arange(8, dtype=np.int32),
        'height_agl_m': np.linspace(1.0, 1.4, 8),
        'trace_distance_m': np.arange(8) * 0.1,
    }


def _data():
    rows = np.arange(32)[:, None]
    return np.sin(rows / 3.0) * np.linspace(0.2, 0.8, 8)[None, :]


def _defaults(method_id):
    return {
        key: value['default']
        for key, value in (NATIVE_ALGORITHMS[method_id].parameter_schema or {}).items()
        if 'default' in value
    }


def test_kirchhoff_defaults_execute_against_real_physical_axes():
    params = _defaults('kirchhoff_migration')
    params['backend'] = 'cpu'
    result = NativeProcessingExecutor().execute(ProcessingRequest(
        _data(), 'kirchhoff_migration', params, _header(), _trace_metadata()))
    assert np.isfinite(result.data).all()
    mapped = result.metadata['mapped_params']
    assert mapped['length_m'] == pytest.approx(0.8)
    assert mapped['time_window_ns'] == pytest.approx(32.0)
    assert any(item.get('code') == 'kirchhoff_trace_grid_resampled'
               for item in result.runtime_warnings)


@pytest.mark.parametrize(('alias', 'canonical', 'value'), [
    ('dx', 'dx_m', 0.2),
    ('v', 'velocity_m_per_ns', 0.12),
    ('depth', 'depth_m', 3.0),
    ('dz', 'dz_m', 0.05),
])
def test_rtm_legacy_aliases_work_without_shadowing(alias, canonical, value):
    params = prepare_native_params(ProcessingRequest(
        _data(), 'rtm_migration', {alias: value}, _header(), _trace_metadata()))
    assert params[canonical] == value


def test_rtm_rejects_conflicting_aliases():
    with pytest.raises(ValueError, match='dx_m'):
        prepare_native_params(ProcessingRequest(
            _data(), 'rtm_migration', {'dx': 0.2, 'dx_m': 0.1}, _header(), _trace_metadata()))


def test_rtm_ui_exposes_only_canonical_spatial_parameters():
    from ui.desktop_backend_facade import method_catalog

    entry = next(item for item in method_catalog() if item.method_id == 'rtm_migration')
    names = {item['name'] for item in entry.parameter_schema}
    assert {'dx_m', 'velocity_m_per_ns', 'depth_m', 'dz_m'} <= names
    assert {'dx', 'v', 'depth', 'dz'}.isdisjoint(names)


def test_every_algorithm_has_an_explicit_axis_transform():
    expected = {
        'time_cut': 'sample_crop',
        'trace_cut': 'trace_crop',
        'trace_qc': 'trace_crop',
        'time_to_depth': 'time_to_depth',
        'motion_compensation_speed': 'trace_resample',
        'motion_compensation_v2': 'trace_resample',
        'stolt_migration': 'migration',
        'kirchhoff_migration': 'migration',
        'rtm_migration': 'migration',
    }
    assert set(NATIVE_ALGORITHMS) == {spec.algorithm_id for spec in ALGORITHM_CATALOG.all()}
    for spec in ALGORITHM_CATALOG.all():
        transform = expected.get(spec.algorithm_id, 'identity')
        assert spec.axis_transform == transform, spec.algorithm_id
        assert spec.preserves_trace_axis == (transform not in {'trace_crop', 'trace_resample'})
        assert spec.preserves_sample_axis == (transform not in {'sample_crop', 'time_to_depth', 'migration'})
        if transform in {'time_to_depth', 'migration'}:
            assert spec.output_contract == 'bscan.depth_x_traces'


def test_ccbs_reference_fallback_warnings_reach_caller():
    result = NativeProcessingExecutor().execute(ProcessingRequest(
        np.ones((8, 4)), 'ccbs',
        {'use_custom_ref': True, 'reference_trace_index': 99},
        {'total_time_ns': 8.0, 'trace_interval_m': 1.0, 'length_m': 4.0},
    ))
    codes = {item.get('code') for item in result.runtime_warnings}
    assert 'ccbs_reference_trace_out_of_range' in codes
    assert 'ccbs_reference_fallback' in codes


def test_new_catalog_members_have_fitness_tasks():
    from scripts.algorithm_fitness_benchmark import TASKS

    methods = {task[1] for task in TASKS}
    expected = {'trace_cut'}
    if 'short_trace_repair' in NATIVE_ALGORITHMS:
        expected.add('short_trace_repair')
    assert expected <= methods
