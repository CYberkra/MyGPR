from __future__ import annotations

import csv

import numpy as np
import pytest

from core.basal_interface_annotations import BasalInterfaceAnnotation
from core.field_project_store import FieldProjectStore


@pytest.mark.parametrize('calibrated', [False, True])
def test_matrix_import_and_annotation_keep_axis_provenance(tmp_path, calibrated):
    source = tmp_path / 'source.npy'
    np.save(source, np.zeros((64, 20), dtype=np.float32))
    original = source.read_bytes()
    store = FieldProjectStore.create_empty(tmp_path / 'project', name='axis provenance')
    try:
        options = {'time_window_ns': 100.0, 'trace_interval_m': 0.25} if calibrated else {}
        store.import_line_file('L01', source, **options)
        dataset = store.load_gpr_dataset('L01')
        assert dataset.metadata['time_axis_status'] == ('known' if calibrated else 'unknown')
        assert store.get_line('L01').distance_override == ('' if calibrated else 'unknown')
        annotation = BasalInterfaceAnnotation('L01', trace_count=20, sample_count=64,
                                               source_result_id='L01_raw', source_mode='raw')
        annotation.set_keypoint(0, 30)
        store.save_basal_interface_annotation('L01', annotation)
        with np.load(store.interface_labels_npz_path('L01')) as labels:
            assert bool(np.isfinite(labels['time_gt_ns'][0])) is calibrated
            assert bool(np.isfinite(labels['depth_gt'][0])) is calibrated
        with store.interface_spatial_path('L01').open(encoding='utf-8-sig', newline='') as file:
            row = next(csv.DictReader(file))
        assert bool(row['distance_m']) is calibrated
        assert bool(row['time_ns']) is calibrated
        assert source.read_bytes() == original
    finally:
        store.close()


def test_generic_hdf5_matrix_accepts_user_axes(tmp_path):
    h5py = pytest.importorskip('h5py')
    source = tmp_path / 'matrix.h5'
    with h5py.File(source, 'w') as handle:
        handle.create_dataset('data', data=np.zeros((64, 20), dtype=np.float32))
    original = source.read_bytes()
    store = FieldProjectStore.create_empty(tmp_path / 'project', name='hdf5 axes')
    try:
        store.import_line_file('L01', source, time_window_ns=100.0, trace_interval_m=0.25)
        dataset = store.load_gpr_dataset('L01')
        assert dataset.metadata['time_axis_status'] == 'known'
        assert dataset.time_axis_ns[-1] == pytest.approx(100.0)
        assert source.read_bytes() == original
    finally:
        store.close()
