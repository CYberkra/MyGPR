"""NPZ reopening and project import keep physical axes and their provenance."""
import json
import io
import zipfile
import numpy as np
import pytest

from core.gpr_data_model import (GPRDataSet, load_gpr_dataset,
                                 load_gpr_dataset_for_import, time_to_depth_axis)
from core.chunked_gpr_io import load_npz_for_import
from mygpr.interfaces.backend import MyGPRBackend


def _assert_dataset(dataset, original):
    assert dataset.dielectric_constant == original.dielectric_constant
    assert dataset.time_window_ns == original.time_window_ns
    for name in ('matrix', 'distance_axis_m', 'time_axis_ns', 'depth_axis_m'):
        np.testing.assert_array_equal(getattr(dataset, name), getattr(original, name))
    restored_metadata = dataset.metadata.get('metadata') or dataset.metadata
    for key, value in original.metadata.items():
        assert restored_metadata[key] == value


@pytest.mark.parametrize('endpoint', [False, True])
@pytest.mark.parametrize('offset', [0., 40.])
@pytest.mark.parametrize('samples', [1, 8])
def test_npz_roundtrip_keeps_sampling_metadata_through_project_reopen(
        tmp_path, endpoint, offset, samples):
    axis = offset + np.arange(samples, dtype=np.float32) * 2.
    metadata = dict(time_axis_endpoint=endpoint, time_cut_offset_ns=offset,
                    time_step_s=2e-9, distance_axis_status='known',
                    scientific_provenance={'calibration': 'original', 'accepted': True})
    original = GPRDataSet(
        'L01', np.arange(samples * 3, dtype=np.float32).reshape(samples, 3),
        np.array([0., 3., 6.], np.float32), axis, time_to_depth_axis(axis, 4.),
        time_window_ns=2. * (samples - int(endpoint)), dielectric_constant=4.,
        metadata=metadata)
    path = original.save_npz(tmp_path / 'source.npz')
    _assert_dataset(load_gpr_dataset(path), original)
    staged = load_gpr_dataset_for_import(path, line_id='L02', staging_dir=tmp_path / 'stage')
    try:
        _assert_dataset(staged, original)
    finally:
        staged.matrix._mmap.close()
    backend = MyGPRBackend.create_default(max_workers=1)
    project = backend.projects.create_project(tmp_path / 'project', name='NPZ round trip')
    try:
        backend.projects.import_line_source(project.project_id, 'L01', path)
        store = backend.projects._session(project.project_id)._store
        _assert_dataset(store.load_gpr_dataset('L01'), original)
        backend.projects.close_project(project.project_id)
        project = backend.projects.open_project(project.root_path)
        _assert_dataset(backend.projects._session(project.project_id)._store.load_gpr_dataset('L01'),
                        original)
    finally:
        backend.shutdown()


def test_npz_invalid_axis_closes_matrix_before_removing_staged_output(tmp_path):
    path = tmp_path / 'invalid-axis.npz'
    np.savez(path, matrix=np.ones((8, 3), np.float32),
             time_axis_ns=np.ones((2, 4), np.float32))
    output = tmp_path / 'matrix.npy'
    with pytest.raises(ValueError, match='NPZ axis is invalid'):
        load_npz_for_import(path, output)
    assert not output.exists()


def test_npz_object_metadata_does_not_require_pickle_or_leave_project_staging(tmp_path):
    path = tmp_path / 'object-metadata.npz'
    np.savez(path, matrix=np.ones((8, 3), np.float32),
             metadata=np.array({'unsupported': 'object metadata'}, dtype=object))
    backend = MyGPRBackend.create_default(max_workers=1)
    project = backend.projects.create_project(tmp_path / 'project', name='No pickle import')
    try:
        backend.projects.import_line_source(project.project_id, 'L01', path)
        np.testing.assert_array_equal(backend.projects.read_dataset(project.project_id, 'L01').data,
                                      np.ones((8, 3), np.float32))
        assert not list((tmp_path / 'project' / 'raw' / 'L01').glob('.import_staging_*'))
    finally:
        backend.shutdown()


@pytest.mark.parametrize('field', ['dielectric_constant', 'time_window_ns'])
def test_npz_invalid_numeric_header_releases_import_matrix(tmp_path, field):
    path = tmp_path / 'invalid-header.npz'
    np.savez(path, matrix=np.ones((8, 3), np.float32),
             metadata=np.array(json.dumps({field: 'invalid number'})))
    stage = tmp_path / 'stage'
    with pytest.raises(ValueError, match='invalid number'):
        load_gpr_dataset_for_import(path, line_id='L01', staging_dir=stage)
    assert not (stage / 'matrix.npy').exists()


@pytest.mark.parametrize('staged', [False, True])
def test_npz_oversized_metadata_is_rejected_before_member_allocation(tmp_path, monkeypatch, staged):
    from core import chunked_gpr_io

    monkeypatch.setattr(chunked_gpr_io, 'MAX_NPZ_METADATA_BYTES', 64, raising=False)
    path = tmp_path / 'large-metadata.npz'
    np.savez_compressed(path, matrix=np.ones((8, 3), np.float32), metadata=np.array('x' * 40000))
    with np.load(path, allow_pickle=False) as archive:
        archive_type = type(archive)
    original = archive_type.__getitem__

    def forbid_metadata_read(self, key):
        assert key != 'metadata', 'metadata member was allocated before its size check'
        return original(self, key)

    monkeypatch.setattr(archive_type, '__getitem__', forbid_metadata_read)
    with pytest.raises(ValueError, match='metadata.*too large'):
        if staged:
            load_gpr_dataset_for_import(path, line_id='L01', staging_dir=tmp_path / 'stage')
        else:
            load_gpr_dataset(path)
    assert not (tmp_path / 'stage' / 'matrix.npy').exists()


@pytest.mark.parametrize('staged', [False, True])
def test_npz_forged_metadata_header_cannot_request_an_unbounded_array(tmp_path, monkeypatch, staged):
    matrix = io.BytesIO()
    np.save(matrix, np.ones((8, 3), np.float32))
    metadata = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        metadata, dict(descr='<U3000000', fortran_order=False, shape=()))
    path = tmp_path / 'forged-metadata.npz'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('matrix.npy', matrix.getvalue())
        archive.writestr('metadata.npy', metadata.getvalue())
    with np.load(path, allow_pickle=False) as archive:
        archive_type = type(archive)
    original = archive_type.__getitem__

    def forbid_metadata_read(self, key):
        assert key != 'metadata', 'forged metadata header reached array allocation'
        return original(self, key)

    monkeypatch.setattr(archive_type, '__getitem__', forbid_metadata_read)
    with pytest.raises(ValueError, match='metadata.*too large'):
        if staged:
            load_gpr_dataset_for_import(path, line_id='L01', staging_dir=tmp_path / 'stage')
        else:
            load_gpr_dataset(path)
    assert not (tmp_path / 'stage' / 'matrix.npy').exists()


@pytest.mark.parametrize('metadata', [None, '{invalid'])
def test_legacy_npz_without_valid_metadata_keeps_default_header_and_saved_axes(tmp_path, metadata):
    path = tmp_path / 'legacy.npz'
    axes = dict(time_axis_ns=np.arange(8, dtype=np.float32) * 2.,
                distance_axis_m=np.array([0., 3., 6.], np.float32))
    contents = dict(matrix=np.ones((8, 3), np.float32), **axes)
    if metadata is not None:
        contents['metadata'] = np.array(metadata)
    np.savez(path, **contents)
    regular = load_gpr_dataset(path, dielectric_constant=5.)
    staged = load_gpr_dataset_for_import(path, line_id='legacy', staging_dir=tmp_path / 'stage',
                                        dielectric_constant=5.)
    try:
        for dataset in (regular, staged):
            assert dataset.dielectric_constant == 5.
            assert dataset.time_window_ns == 14.
            np.testing.assert_array_equal(dataset.time_axis_ns, axes['time_axis_ns'])
            np.testing.assert_array_equal(dataset.depth_axis_m,
                                          time_to_depth_axis(axes['time_axis_ns'], 5.))
    finally:
        staged.matrix._mmap.close()
