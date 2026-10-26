"""Versioned backup semantics, tested against a real committed v1 fixture."""
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import zipfile

import numpy as np
import pytest

from core.field_project_backup import backup_project_archive, restore_project_archive
from core.field_project_errors import FieldProjectOperationError
from core.field_project_store import FieldProjectStore


FIXTURE_ROOT = Path(__file__).parent / 'fixtures' / 'project_backup_v1'


@pytest.fixture
def legacy_validator():
    return runpy.run_path(str(FIXTURE_ROOT / 'reader_v1.py'))['_validated_backup_members']


def test_new_backup_schema_is_v2(tmp_path):
    store = FieldProjectStore.create_empty(tmp_path / 'project', name='Versioned')
    try:
        result = backup_project_archive(store, tmp_path / 'backups')
        with zipfile.ZipFile(result.archive_path) as handle:
            assert json.loads(handle.read('backup_manifest.json'))['schema'] == 'mygpr.project_backup.v2'
    finally:
        store.close()


def test_committed_v1_reader_rejects_new_deletion_snapshot(tmp_path, legacy_validator):
    store = FieldProjectStore.create_empty(tmp_path / 'project', name='Versioned')
    try:
        stale = store.root / 'old_result.txt'
        stale.write_text('old result')
        backup_project_archive(store, tmp_path / 'backups')
        stale.unlink()
        result = backup_project_archive(store, tmp_path / 'backups', incremental=True)
        with zipfile.ZipFile(result.archive_path) as handle:
            with pytest.raises(FieldProjectOperationError, match='清单格式无效'):
                legacy_validator(handle)
    finally:
        store.close()


def test_actual_legacy_v1_fixture_restores_with_exact_raw_matrix(tmp_path, legacy_validator):
    archive = FIXTURE_ROOT / 'project_v1.zip'
    provenance = json.loads((FIXTURE_ROOT / 'provenance.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == provenance['archive_sha256']
    with zipfile.ZipFile(archive) as handle:
        manifest, _ = legacy_validator(handle)
        assert manifest['schema'] == 'mygpr.project_backup.v1'
        assert 'deleted_files' not in manifest
    restored = restore_project_archive(archive, tmp_path / 'restore')
    store = FieldProjectStore.open(restored.project_path, access_mode='read_only')
    try:
        assert [line.line_id for line in store.list_lines()] == ['L01']
        actual = store.load_gpr_dataset('L01')
        np.testing.assert_array_equal(np.asarray(actual.matrix), np.arange(12, dtype=np.float32).reshape(3, 4))
    finally:
        store.close()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == provenance['archive_sha256']


def test_new_v2_incremental_can_extend_an_actual_legacy_v1_base(tmp_path):
    archive = FIXTURE_ROOT / 'project_v1.zip'
    restored = restore_project_archive(archive, tmp_path / 'restore')
    store = FieldProjectStore.open(restored.project_path, access_mode='write')
    try:
        backups = tmp_path / 'backups'
        backups.mkdir()
        base = backups / f'{store.root.name}_backup_legacy.zip'
        shutil.copyfile(archive, base)
        result = backup_project_archive(store, backups, incremental=True, retention_keep=1)
        with zipfile.ZipFile(result.archive_path) as handle:
            manifest = json.loads(handle.read('backup_manifest.json'))
            assert manifest['schema'] == 'mygpr.project_backup.v2'
            assert manifest['incremental_base_archive'] == base.name
        assert base.is_file()
        recovered = restore_project_archive(result.archive_path, tmp_path / 'new_restore')
        assert recovered.verified
    finally:
        store.close()


@pytest.mark.parametrize('schema', ['mygpr.project_backup.v0', 'mygpr.project_backup.v3', 'unrecognized'])
def test_unknown_backup_versions_are_rejected_before_extraction(tmp_path, schema):
    with zipfile.ZipFile(FIXTURE_ROOT / 'project_v1.zip') as source:
        members = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(members['backup_manifest.json'])
    manifest['schema'] = schema
    members['backup_manifest.json'] = json.dumps(manifest).encode()
    archive = tmp_path / 'unsupported.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        for name, content in members.items():
            handle.writestr(name, content)
    destination = tmp_path / 'restore'
    with pytest.raises(FieldProjectOperationError):
        restore_project_archive(archive, destination)
    assert list(destination.iterdir()) == []
