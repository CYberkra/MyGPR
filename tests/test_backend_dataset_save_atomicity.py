"""The public matrix-save API preserves the last complete state on failure."""
from pathlib import Path

import numpy as np
import pytest

from mygpr.interfaces.backend import MyGPRBackend


def _snapshot(store):
    files = {path.relative_to(store.root): path.read_bytes()
             for folder in ('raw', 'data') for path in (store.root / folder).rglob('*')
             if path.is_file()}
    return ((store.root / 'project.json').read_bytes(), files,
            store.storage.catalog.list_lines(), store.storage.catalog.list_artifacts())


@pytest.mark.parametrize('replace_existing', [False, True])
@pytest.mark.parametrize('failure', ['matrix', 'metadata', 'project', 'catalog'])
def test_failed_matrix_save_preserves_manifest_catalog_and_data(
        tmp_path, monkeypatch, replace_existing, failure):
    backend = MyGPRBackend.create_default(max_workers=1)
    project = backend.projects.create_project(tmp_path / 'project', name='Atomic matrix API')
    pid = project.project_id
    try:
        if replace_existing:
            backend.projects.save_line_dataset(pid, 'L01', np.ones((16, 4), np.float32),
                                                name='Original', time_window_ns=900.)
        store = backend.projects._session(pid)._store
        before = _snapshot(store)
        with monkeypatch.context() as fault:
            if failure == 'metadata':
                original = store.write_json

                def fail_metadata(path, payload):
                    if Path(path) == store.gpr_metadata_path('L01'):
                        raise OSError('simulated save failure')
                    return original(path, payload)

                fault.setattr(store, 'write_json', fail_metadata)
            else:
                target, method, fail_call = {
                    'matrix': (store.storage, 'save_raw_dataset', 1),
                    'project': (store, 'save_manifest', 2),
                    'catalog': (store.storage.catalog, 'upsert_line', 2),
                }[failure]
                original = getattr(target, method)
                calls = 0

                def fail_after_write(*args, **kwargs):
                    nonlocal calls
                    result = original(*args, **kwargs)
                    calls += 1
                    if calls == fail_call:
                        raise OSError('simulated save failure')
                    return result

                fault.setattr(target, method, fail_after_write)
            with pytest.raises(OSError, match='simulated save failure'):
                backend.projects.save_line_dataset(pid, 'L01', np.full((16, 4), 2, np.float32),
                                                    name='Replacement', time_window_ns=600.)
        assert _snapshot(store) == before
        assert not list((store.root / '.transactions').glob('*/transaction.json'))
        backend.projects.close_project(pid)
        reopened = backend.projects.open_project(project.root_path)
        assert _snapshot(backend.projects._session(reopened.project_id)._store) == before
        if replace_existing:
            np.testing.assert_array_equal(backend.projects.read_dataset(pid, 'L01').data,
                                          np.ones((16, 4), np.float32))
        else:
            assert backend.projects.list_lines(pid) == ()
    finally:
        backend.shutdown()
