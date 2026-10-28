import json
import numpy as np
import pytest

from core.project_storage_backend import LegacyProjectStorageBackend


def _artifact(tmp_path, output_header):
    folder = tmp_path / 'processed' / 'L01'
    folder.mkdir(parents=True)
    np.save(folder / 'L01_processed_1.npy', np.arange(12, dtype=np.float32).reshape(3, 4))
    (folder / 'L01_processing_manifest_1.json').write_text(json.dumps({'output_header': output_header}), encoding='utf-8')
    (folder / 'L01_params_1.json').write_text(json.dumps({'input_dataset': {'length_m': 20, 'time_window_ns': 30}}), encoding='utf-8')
    return LegacyProjectStorageBackend(tmp_path, None).load_processing_artifact('L01', 'L01_processed_1')


def test_legacy_artifact_preserves_saved_axes(tmp_path):
    dataset = _artifact(tmp_path, {'time_axis_ns': [10, 13, 19], 'depth_axis_m': [1, 2, 4], 'distance_axis_m': [5, 6, 9, 12], 'time_window_ns': 20})
    np.testing.assert_array_equal(dataset.time_axis_ns, [10, 13, 19])
    np.testing.assert_array_equal(dataset.depth_axis_m, [1, 2, 4])
    np.testing.assert_array_equal(dataset.distance_axis_m, [5, 6, 9, 12])


def test_legacy_artifact_rejects_invalid_saved_axis(tmp_path):
    with pytest.raises(ValueError, match='distance_axis_m'):
        _artifact(tmp_path, {'distance_axis_m': [1, 2]})


def test_legacy_artifact_without_saved_axes_keeps_fallback(tmp_path):
    dataset = _artifact(tmp_path, {'time_window_ns': 30})
    np.testing.assert_array_equal(dataset.time_axis_ns, [0, 15, 30])
    np.testing.assert_allclose(dataset.distance_axis_m, np.linspace(0, 20, 4))
