"""Completed CDUT 1.1.11 files: storage semantics and original ICZT axis."""
import json

import h5py
import numpy as np
import pytest

from core.acquisition_hdf5 import acquisition_trace_metadata
from core.field_import_preview import build_import_preflight
from core.gpr_data_model import load_gpr_dataset, load_gpr_dataset_for_import
from core.gpr_io import auto_load_data

IDENTITY = 'd54b6f93-9d38-4d61-bd08-10882ef54a53'


def write_completed(path, *, component='real', manifest=False):
    values = np.arange(3 * 501, dtype=np.float32).reshape(3, 501) - 700
    if component == 'mag':
        values = np.abs(values)
    with h5py.File(path, 'w') as handle:
        handle.attrs.update(samples_per_trace=501, time_window_ns=900., store_component=component,
                            input_domain='frequency', signal_domain='time',
                            acquisition_id=IDENTITY, committed_trace_count=3)
        handle.create_dataset('traces', data=values)
        handle.create_dataset('trace_meta/height', data=[2., np.nan, 3.])
        handle.create_dataset('trace_meta/trace_id', data=np.arange(3, dtype=np.uint64))
    settings = {'iczt': {'f_start': 20e6, 'f_stop': 170e6, 'n_freq': 501,
                         't_max': 900e-9, 'n_time': 501, 'window': 'kaiser', 'method': 'bluestein'},
                'store_component': component, 'initiate': True, 'interval': .08}
    if manifest:
        save_manifest(path, {'acquisition_id': IDENTITY, 'sfcw': settings})
    return values, settings


def save_manifest(path, contents):
    (path.parent / (path.name + '.sfcw.json')).write_text(json.dumps(contents), encoding='utf-8')


@pytest.mark.parametrize('component', ['real', 'mag'])
def test_new_storage_tags_preserve_waveforms_domains_and_gaps(tmp_path, component):
    path = tmp_path / 'line.h5'
    expected, _ = write_completed(path, component=component)
    preview = build_import_preflight(path)
    assert preview.can_import, preview.message
    for data in (load_gpr_dataset(path), load_gpr_dataset_for_import(path, line_id='L01', staging_dir=tmp_path / 'stage')):
        np.testing.assert_array_equal(data.matrix, expected.T)
        assert data.metadata['store_component'] == component
        assert data.metadata['input_domain'] == 'frequency'
        assert data.metadata['signal_domain'] == 'time'
        np.testing.assert_allclose(acquisition_trace_metadata(data.metadata)['flight_height_m'], [2., np.nan, 3.], equal_nan=True)
    if component == 'mag':
        assert '幅值' in preview.source_kind
        assert any('相位' in message for message in preview.suggestions)


def test_identity_bound_manifest_provides_original_inclusive_axis_to_gui_and_cli(tmp_path):
    path = tmp_path / 'line.h5'
    _, settings = write_completed(path, manifest=True)
    data = load_gpr_dataset(path)
    assert data.metadata['sfcw'] == settings
    assert data.metadata['time_axis_convention'] == 'sfcw_manifest'
    assert data.metadata['time_axis_endpoint'] is True
    np.testing.assert_allclose(data.time_axis_ns, np.linspace(0., 900., 501), rtol=1e-6)
    payload = auto_load_data(str(path))
    assert payload['header_info']['time_step_s'] == pytest.approx(900e-9 / 500)
    assert not any('时窗/N' in message for message in build_import_preflight(path).suggestions)


@pytest.mark.parametrize('fault', ['identity', 'sample_count', 'window', 'component', 'explicit_endpoint', 'invalid_json', 'missing_identity'])
def test_conflicting_manifest_never_silently_changes_physical_axes(tmp_path, fault):
    path = tmp_path / 'line.h5'
    _, settings = write_completed(path, manifest=True)
    manifest = {'acquisition_id': IDENTITY, 'sfcw': settings}
    if fault == 'identity':
        manifest['acquisition_id'] = 'ad5c4e42-45f6-4973-9b7e-74e724e2a0c5'
    elif fault == 'sample_count':
        settings['iczt']['n_time'] = 500
    elif fault == 'window':
        settings['iczt']['t_max'] = 700e-9
    elif fault == 'component':
        settings['store_component'] = 'mag'
    elif fault == 'explicit_endpoint':
        with h5py.File(path, 'r+') as handle:
            handle.attrs['time_axis_endpoint'] = False
    elif fault == 'missing_identity':
        with h5py.File(path, 'r+') as handle:
            del handle.attrs['acquisition_id']
    save_manifest(path, manifest)
    if fault == 'invalid_json':
        (tmp_path / 'line.h5.sfcw.json').write_text('{', encoding='utf-8')
    assert not build_import_preflight(path).can_import
    with pytest.raises(ValueError, match='SFCW'):
        load_gpr_dataset(path)


@pytest.mark.parametrize('domain', ['frequency', 'unknown'])
def test_non_time_signal_cannot_be_presented_as_a_time_domain_ascan(tmp_path, domain):
    path = tmp_path / 'line.h5'
    write_completed(path)
    with h5py.File(path, 'r+') as handle:
        handle.attrs['signal_domain'] = domain
    assert not build_import_preflight(path).can_import
    with pytest.raises(ValueError, match='signal_domain'):
        load_gpr_dataset(path)


def test_missing_optional_companions_keep_main_waveform_importable_with_notice(tmp_path):
    path = tmp_path / 'ground.h5'
    write_completed(path)
    assert build_import_preflight(path).can_import
    assert any('SFCW' in message and '缺' in message for message in build_import_preflight(path).suggestions)
    data = load_gpr_dataset(path)
    assert data.metadata['sfcw_companions']['manifest'] == 'missing'
    assert data.metadata['sfcw_companions']['raw_s21'] == 'missing'
    assert data.metadata['time_axis_convention'] == 'legacy_window_over_samples'


def test_project_reopen_keeps_bound_settings_after_original_companion_is_unavailable(tmp_path):
    from core.field_project_operations import import_line_data
    from core.field_project_store import FieldProjectStore
    path = tmp_path / 'line.h5'
    expected, settings = write_completed(path, manifest=True)
    store = FieldProjectStore.create_empty(tmp_path / 'project', name='1.1.11')
    import_line_data(store, path, line_id='L01')
    (tmp_path / 'line.h5.sfcw.json').unlink()
    reopened = FieldProjectStore.open(store.root)
    data = reopened.load_gpr_dataset('L01')
    np.testing.assert_array_equal(data.matrix, expected.T)
    np.testing.assert_allclose(data.time_axis_ns, np.linspace(0., 900., 501), rtol=1e-6)
    assert data.metadata.get('metadata', data.metadata)['sfcw'] == settings
    assert (store.root / 'raw/L01/line.h5.sfcw.json').is_file()


@pytest.mark.parametrize('replacement_manifest', [False, True])
def test_replacing_same_named_line_never_reuses_old_companions(tmp_path, replacement_manifest):
    from core.field_project_operations import import_line_data
    from core.field_project_store import FieldProjectStore
    first = tmp_path / 'first/line.h5'
    second = tmp_path / 'second/line.h5'
    first.parent.mkdir()
    second.parent.mkdir()
    write_completed(first, manifest=True)
    (first.parent / 'line.h5.s21.jsonl').write_text('old acquisition raw journal', encoding='utf-8')
    _, settings = write_completed(second, manifest=replacement_manifest)
    next_identity = 'cefba604-d80f-451d-a61a-54a3d0d4f567'
    with h5py.File(second, 'r+') as handle:
        handle.attrs['acquisition_id'] = next_identity
    if replacement_manifest:
        save_manifest(second, {'acquisition_id': next_identity, 'sfcw': settings})
    store = FieldProjectStore.create_empty(tmp_path / 'project', name='replace')
    import_line_data(store, first, line_id='L01')
    import_line_data(store, second, line_id='L01')
    raw = store.root / 'raw/L01'
    assert not (raw / 'line.h5.s21.jsonl').exists()
    assert (raw / 'line.h5.sfcw.json').exists() is replacement_manifest
    data = store.load_gpr_dataset('L01')
    assert data.metadata.get('metadata', data.metadata)['acquisition_id'] == next_identity


@pytest.mark.parametrize('failure', ['cancel', 'disk_error'])
def test_companion_copy_failure_rolls_back_waveform_and_old_companions(tmp_path, failure):
    import hashlib
    from core.chunked_gpr_io import ImportCancelled
    from core.field_project_operations import import_line_data
    from core.field_project_store import FieldProjectStore
    source = tmp_path / 'input/line.h5'
    source.parent.mkdir()
    original, _ = write_completed(source, manifest=True)
    journal = source.parent / 'line.h5.s21.jsonl'
    journal.write_bytes(b'old raw journal')
    store = FieldProjectStore.create_empty(tmp_path / 'project', name='copy rollback')
    import_line_data(store, source, line_id='L01')
    raw = store.root / 'raw/L01'
    old_hashes = {name: hashlib.sha256((raw/name).read_bytes()).hexdigest()
                  for name in ('line.h5', 'line.h5.sfcw.json', 'line.h5.s21.jsonl')}
    journal.write_bytes(b'x' * 100000)
    cancelled = False

    def progress(stage, current, total):
        nonlocal cancelled
        if stage == 'copy_source' and total == 100000 and current:
            if failure == 'disk_error':
                raise OSError('injected companion write failure')
            cancelled = True

    expected = ImportCancelled if failure == 'cancel' else OSError
    with pytest.raises(expected):
        import_line_data(store, source, line_id='L01', cancel_requested=lambda: cancelled, progress_callback=progress)
    assert {name: hashlib.sha256((raw/name).read_bytes()).hexdigest() for name in old_hashes} == old_hashes
    np.testing.assert_array_equal(store.load_gpr_dataset('L01').matrix, original.T)


def test_cli_config_results_and_resume_on_another_windows_drive(tmp_path):
    import os
    from pathlib import Path
    import cli_batch
    if os.name != 'nt' or tmp_path.drive.lower() == Path(cli_batch.__file__).drive.lower():
        pytest.skip('requires config/output and application on different Windows drives')
    source = tmp_path / 'line.h5'
    expected, _ = write_completed(source)
    output = tmp_path / 'output'
    config = tmp_path / 'batch.json'
    config.write_text(json.dumps({'output_dir': str(output), 'jobs': [
        {'id': 'line', 'input': str(source),
         'methods': [{'key': 'amplitude_scale', 'params': {'mode': 'constant', 'scale': 2.}}]}]}), encoding='utf-8')
    assert cli_batch.main(['validate', '--config', str(config)]) == 0
    assert cli_batch.main(['run', '--config', str(config)]) == 0
    summary_path = next(output.glob('summary_*.json'))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['stats']['failed'] == 0
    result = summary['results'][0]
    assert Path(result['final_csv']).is_absolute()
    np.testing.assert_allclose(np.loadtxt(result['final_csv'], delimiter=','), expected.T*2, rtol=1e-6, atol=1e-6)
    result['status'] = 'failed'
    summary['stats'] = {'ok': 0, 'failed': 1, 'total': 1}
    summary_path.write_text(json.dumps(summary), encoding='utf-8')
    assert cli_batch.main(['resume', '--summary', str(summary_path)]) == 0
