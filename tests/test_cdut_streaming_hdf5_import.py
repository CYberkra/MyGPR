"""CDUT remote H5 import: displayed samples, complete snapshots and provenance."""
import h5py
import numpy as np
import pytest

from core.acquisition_hdf5 import acquisition_trace_metadata
from core.field_import_preview import build_import_preflight
from core.gpr_data_model import load_gpr_dataset, load_gpr_dataset_for_import


def write_stream(path, *, component='fdata', endpoint=None):
    waveform = np.arange(5 * 17, dtype=np.float32).reshape(5, 17) - 35
    fields = {
        'lon': [104, 104.001, np.nan, 104.003, 104.004],
        'lat': [30, 30.001, np.nan, 30.003, 30.004],
        'alt': [500, 501, np.nan, 503, 504],
        'height': [2, 3, np.nan, 5, 6],
        'roll': [1, 2, np.nan, 4, 5], 'pitch': [2, 3, np.nan, 5, 6],
        'yaw': [3, 4, np.nan, 6, 7],
        'ax': [.1, .2, np.nan, .4, .5], 'ay': [.2, .3, np.nan, .5, .6],
        'az': [1, 1, np.nan, 1, 1],
        'gx': [10, 20, np.nan, 40, 50], 'gy': [20, 30, np.nan, 50, 60],
        'gz': [30, 40, np.nan, 60, 70],
        'timestamp_ms': [1792000000123, 1792000000148, np.nan, 1792000000198, 1792000000223],
    }
    with h5py.File(path, 'w') as handle:
        handle.attrs.update(samples_per_trace=17, time_window_ns=900., store_component=component,
                            committed_trace_count=5, acquisition_id='4f3d3bee-e683-40db-9767-3dd105d58d27',
                            simulation=True, data_source='SIMULATED_SYNTHETIC')
        if endpoint is not None:
            handle.attrs['time_axis_endpoint'] = endpoint
        handle.create_dataset('traces', data=waveform, chunks=(1, 17), maxshape=(None, 17))
        for name, values in fields.items():
            handle.create_dataset('trace_meta/' + name, data=np.asarray(values, dtype=np.float64))
        handle.create_dataset('trace_meta/trace_id', data=np.arange(2**53 + 1, 2**53 + 6, dtype=np.uint64))
    return waveform, fields


def test_remote_fdata_import_preserves_values_and_does_not_relabel_component(tmp_path):
    path = tmp_path / 'ground.h5'
    waveform, fields = write_stream(path)
    preview = build_import_preflight(path)
    assert preview.can_import, preview.message
    assert '显示采样' in preview.source_kind
    assert any('时间轴' in item for item in preview.suggestions)
    for dataset in (load_gpr_dataset(path), load_gpr_dataset_for_import(path, line_id='L01', staging_dir=tmp_path / 'stage')):
        np.testing.assert_array_equal(dataset.matrix, waveform.T)
        assert dataset.metadata['store_component'] == 'fdata'
        assert dataset.metadata['simulation'] is True
        assert dataset.metadata['acquisition_id'] == '4f3d3bee-e683-40db-9767-3dd105d58d27'
        meta = acquisition_trace_metadata(dataset.metadata)
        np.testing.assert_allclose(meta['roll_deg'], fields['roll'], equal_nan=True)
        np.testing.assert_allclose(meta['accel_x_g'], fields['ax'], equal_nan=True)
        np.testing.assert_allclose(meta['angular_rate_z_dps'], fields['gz'], equal_nan=True)
        np.testing.assert_array_equal(meta['trace_id'], np.arange(2**53 + 1, 2**53 + 6, dtype=np.uint64))
        np.testing.assert_allclose(meta['trace_timestamp_s'], np.asarray(fields['timestamp_ms']) / 1000,
                                   rtol=0, atol=1e-7, equal_nan=True)


@pytest.mark.parametrize('component', ['real', 'magnitude'])
def test_existing_components_also_preserve_new_sensor_fields(tmp_path, component):
    path = tmp_path / 'local.h5'
    _, fields = write_stream(path, component=component)
    dataset = load_gpr_dataset(path)
    meta = acquisition_trace_metadata(dataset.metadata)
    np.testing.assert_allclose(meta['pitch_deg'], fields['pitch'], equal_nan=True)
    np.testing.assert_allclose(dataset.time_axis_ns, np.arange(17) * (900 / 17), rtol=1e-6)


@pytest.mark.parametrize('name', ['roll', 'ax', 'gx', 'timestamp_ms', 'trace_id'])
def test_misaligned_new_sensor_fields_are_rejected_in_preflight(tmp_path, name):
    path = tmp_path / 'bad.h5'
    write_stream(path, component='real')
    with h5py.File(path, 'r+') as handle:
        del handle['trace_meta/' + name]
        handle.create_dataset('trace_meta/' + name, data=[1])
    assert not build_import_preflight(path).can_import
    with pytest.raises(ValueError, match=name):
        load_gpr_dataset(path)


def test_partial_commit_is_not_accepted_as_a_finished_line(tmp_path):
    path = tmp_path / 'partial.h5'
    write_stream(path, component='real')
    with h5py.File(path, 'r+') as handle:
        handle.attrs['committed_trace_count'] = 4
    assert not build_import_preflight(path).can_import
    with pytest.raises(ValueError, match='committed_trace_count'):
        load_gpr_dataset(path)


def test_explicit_endpoint_controls_time_axis_and_cli_sampling_interval(tmp_path):
    from core.gpr_io import auto_load_data
    path = tmp_path / 'endpoint.h5'
    write_stream(path, component='real', endpoint=True)
    payload = auto_load_data(str(path))
    assert payload['header_info']['time_step_s'] == pytest.approx(900e-9 / 16)
    np.testing.assert_allclose(payload['header_info']['time_axis_ns'], np.linspace(0, 900, 17))


def test_source_timestamps_are_available_to_existing_sensor_sync(tmp_path):
    from core.sensor_sync_service import infer_trace_timestamps
    path = tmp_path / 'timestamps.h5'
    _, fields = write_stream(path, component='real')
    timestamps = infer_trace_timestamps(load_gpr_dataset(path))
    assert timestamps is not None
    np.testing.assert_allclose(timestamps, np.asarray(fields['timestamp_ms']) / 1000, rtol=0, atol=1e-7, equal_nan=True)


def test_project_reopen_and_processing_keep_stream_snapshot(tmp_path):
    from core.field_project_store import FieldProjectStore
    from core.field_project_operations import import_line_data
    from mygpr.infrastructure.persistence.field_project_adapter import _load_trace_metadata
    path = tmp_path / 'stream.h5'
    waveform, fields = write_stream(path)
    store = FieldProjectStore.create_empty(tmp_path / 'project', name='stream')
    import_line_data(store, path, line_id='L01')
    reopened = FieldProjectStore.open(store.root)
    dataset = reopened.load_gpr_dataset('L01')
    np.testing.assert_array_equal(dataset.matrix, waveform.T)
    sensors = _load_trace_metadata(reopened, 'L01')
    np.testing.assert_allclose(sensors['yaw_deg'], fields['yaw'], equal_nan=True)
    assert int(sensors['trace_id'][0]) == 2**53 + 1
