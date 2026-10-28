"""Stacked CSV declarations must not silently truncate or realign trace data."""
from pathlib import Path

import numpy as np
import pytest

from core.chunked_gpr_io import ImportCancelled
from core.field_import_preview import build_import_preflight
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import load_gpr_dataset, load_gpr_dataset_for_import


def write_stacked(path: Path, *, samples='4', traces='2', window='90', interval='0', rows=None):
    header = [f'Number of Samples = {samples}', f'Time windows (ns) = {window}',
              f'Number of Traces = {traces}', f'Trace interval (m) = {interval}']
    payload = rows if rows is not None else [f',,,{value},' for value in (1, 2, 3, 4, 10, 11, 12, 13)]
    path.write_text('\n'.join(header + payload) + '\n', encoding='utf-8')


@pytest.mark.parametrize('extra', [',,,99,', '1,2,3'])
def test_extra_rows_cannot_be_silently_dropped(tmp_path, extra):
    source = tmp_path / 'line.csv'
    write_stacked(source)
    with source.open('a', encoding='utf-8') as stream:
        stream.write(extra + '\n')
    with pytest.raises(ValueError):
        load_gpr_dataset(source)


def test_incomplete_trace_count_is_rejected(tmp_path):
    source = tmp_path / 'line.csv'
    write_stacked(source, rows=[',,,1,'] * 7)
    with pytest.raises(ValueError, match='insufficient'):
        load_gpr_dataset(source)


@pytest.mark.parametrize('field,value', [('samples', '4.5'), ('traces', '2.5'),
                                        ('window', '-1'), ('interval', '-1')])
def test_invalid_declared_axes_are_rejected_before_import(tmp_path, field, value):
    source = tmp_path / 'line.csv'
    write_stacked(source, **{field: value})
    assert not build_import_preflight(source).can_import
    with pytest.raises(ValueError):
        load_gpr_dataset(source)


def test_stacked_import_observes_cancellation(tmp_path):
    source = tmp_path / 'line.csv'
    write_stacked(source)
    with pytest.raises(ImportCancelled):
        load_gpr_dataset_for_import(source, line_id='L01', staging_dir=tmp_path / 'stage',
                                    cancel_requested=lambda: True)
    assert not (tmp_path / 'stage' / 'matrix.npy').exists()


def test_stacked_import_checks_cancellation_during_parsing(tmp_path):
    source = tmp_path / 'line.csv'
    write_stacked(source, traces='100', rows=[',,,1,'] * 400)
    checks = []

    def cancel():
        checks.append(True)
        return len(checks) >= 3

    with pytest.raises(ImportCancelled):
        load_gpr_dataset_for_import(source, line_id='L01', staging_dir=tmp_path / 'stage',
                                    cancel_requested=cancel)
    assert len(checks) == 3
    assert not (tmp_path / 'stage' / 'matrix.npy').exists()


def test_exact_rows_preserve_amplitudes_and_sensor_gaps(tmp_path):
    source = tmp_path / 'line.csv'
    write_stacked(source)
    for data in (load_gpr_dataset(source), load_gpr_dataset_for_import(
            source, line_id='L01', staging_dir=tmp_path / 'stage')):
        np.testing.assert_array_equal(data.matrix, [[1, 10], [2, 11], [3, 12], [4, 13]])
        assert np.isnan(data.metadata['trajectory_rows'][0]['longitude'])
        np.testing.assert_allclose(data.time_axis_ns, [0, 30, 60, 90])


def test_malformed_replacement_keeps_previous_project_input(tmp_path):
    first, second = tmp_path / 'first.csv', tmp_path / 'second.csv'
    write_stacked(first)
    write_stacked(second, rows=[',,,99,'] * 9)
    store = FieldProjectStore.create_empty(tmp_path / 'project', name='integrity rollback')
    store.import_line_file('L01', first)
    before = np.asarray(store.load_gpr_dataset('L01').matrix).copy()
    with pytest.raises(ValueError):
        store.import_line_file('L01', second)
    np.testing.assert_array_equal(store.load_gpr_dataset('L01').matrix, before)
    assert not (store.root / 'raw' / 'L01' / 'second.csv').exists()
