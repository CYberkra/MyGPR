"""Real filesystem checks for portable incremental project snapshots."""
import json
from pathlib import Path
import zipfile

import pytest

from core.field_project_backup import backup_project_archive, restore_project_archive
from core.field_project_errors import FieldProjectOperationError
from core.field_project_store import FieldProjectStore
from core.job_manager import JobCancelled


@pytest.fixture
def project(tmp_path):
    store = FieldProjectStore.create_empty(tmp_path / 'project', name='Recovery')
    yield store
    store.close()


def _manifest(archive):
    with zipfile.ZipFile(archive) as handle:
        return json.loads(handle.read('backup_manifest.json'))


def _rewrite_manifest(archive, update):
    with zipfile.ZipFile(archive) as source:
        content = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(content['backup_manifest.json'])
    update(manifest)
    content['backup_manifest.json'] = json.dumps(manifest).encode()
    with zipfile.ZipFile(archive, 'w') as target:
        for name, data in content.items():
            target.writestr(name, data)


def test_retention_keeps_ancestors_needed_by_incremental_snapshot(project, tmp_path):
    full = backup_project_archive(project, tmp_path / 'backups')
    delta = backup_project_archive(project, tmp_path / 'backups', incremental=True, retention_keep=1)
    assert delta.verified and delta.recovery_tested
    assert Path(full.archive_path).is_file()
    assert restore_project_archive(delta.archive_path, tmp_path / 'restore').verified


def test_relative_default_backup_root_preserves_incremental_ancestors(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MYGPR_BACKUP_ROOT', 'relative-backups')
    full = backup_project_archive(project)
    delta = backup_project_archive(project, incremental=True, retention_keep=1)
    assert Path(full.archive_path).is_file()
    assert restore_project_archive(delta.archive_path, tmp_path / 'restore').verified


def test_case_only_rename_is_replayed_as_delete_then_add(project, tmp_path):
    original = project.root / 'Note.TXT'
    original.write_text('case rename payload')
    backup_project_archive(project, tmp_path / 'backups')
    original.rename(project.root / 'note.txt')
    delta = backup_project_archive(project, tmp_path / 'backups', incremental=True)
    manifest = _manifest(delta.archive_path)
    assert 'Note.TXT' in manifest['deleted_files']
    assert 'note.txt' in {row['path'] for row in manifest['files']}
    restored = restore_project_archive(delta.archive_path, tmp_path / 'restore')
    names = {path.name for path in Path(restored.project_path).iterdir()}
    assert 'note.txt' in names and 'Note.TXT' not in names
    assert (Path(restored.project_path) / 'note.txt').read_text() == 'case rename payload'


@pytest.mark.parametrize('deleted', [['note.txt'], ['OLD.TXT', 'old.txt']])
def test_case_rename_support_keeps_same_spelling_and_duplicate_delete_checks(project, tmp_path, deleted):
    (project.root / 'note.txt').write_text('retained member')
    full = backup_project_archive(project, tmp_path / 'backups')
    _rewrite_manifest(full.archive_path, lambda payload: payload.update(deleted_files=deleted))
    with pytest.raises(FieldProjectOperationError, match='冲突'):
        restore_project_archive(full.archive_path, tmp_path / 'restore')


def test_case_rename_support_does_not_allow_two_case_colliding_members(project, tmp_path):
    import hashlib
    (project.root / 'note.txt').write_text('retained member')
    full = backup_project_archive(project, tmp_path / 'backups')
    data = b'case collision'

    def add_member_row(payload):
        payload['files'].append({'path': 'Note.TXT', 'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})

    _rewrite_manifest(full.archive_path, add_member_row)
    with zipfile.ZipFile(full.archive_path, 'a') as handle:
        handle.writestr('Note.TXT', data)
    with pytest.raises(FieldProjectOperationError, match='大小写冲突'):
        restore_project_archive(full.archive_path, tmp_path / 'restore')


def test_cancel_after_final_member_stops_before_verification_io(project, tmp_path, monkeypatch):
    from core import field_project_backup as backup
    full = backup_project_archive(project, tmp_path / 'backups')
    before = Path(full.archive_path).read_bytes()
    cancel = False
    reads_after_cancel = []
    real_open = zipfile.ZipFile.open

    def observe_open(handle, name, mode='r', *args, **kwargs):
        if cancel and mode == 'r':
            reads_after_cancel.append(str(name))
        return real_open(handle, name, mode, *args, **kwargs)

    def progress(current, total, message):
        nonlocal cancel
        if current == total:
            cancel = True

    monkeypatch.setattr(backup.zipfile.ZipFile, 'open', observe_open)
    with pytest.raises(JobCancelled):
        backup_project_archive(project, tmp_path / 'backups', cancel_requested=lambda: cancel,
                               progress_callback=progress, retention_keep=1)
    assert reads_after_cancel == []
    assert Path(full.archive_path).read_bytes() == before
    assert list((tmp_path / 'backups').iterdir()) == [Path(full.archive_path)]


@pytest.mark.parametrize('stage', ['compression', 'hash_verification', 'restore_verification'])
def test_mid_block_cancel_stops_remaining_large_member_io(project, tmp_path, monkeypatch, stage):
    from core import field_project_backup as backup
    block_size = 8 * 1024 * 1024
    (project.root / 'payload.bin').write_bytes(b'x' * (block_size + 1024 * 1024))
    full = backup_project_archive(project, tmp_path / 'backups')
    before = Path(full.archive_path).read_bytes()
    cancel = False
    observed_sizes = []
    if stage == 'compression':
        original_write = zipfile._ZipWriteFile.write

        def observe_write(writer, data):
            nonlocal cancel
            count = original_write(writer, data)
            if writer._zinfo.filename == 'payload.bin':
                observed_sizes.append(len(data))
                cancel = True
            return count

        monkeypatch.setattr(zipfile._ZipWriteFile, 'write', observe_write)
    else:
        original_read = zipfile.ZipExtFile.read

        def observe_read(reader, size=-1):
            nonlocal cancel
            data = original_read(reader, size)
            if reader.name == 'payload.bin':
                observed_sizes.append(len(data))
                cancel = True
            return data

        monkeypatch.setattr(zipfile.ZipExtFile, 'read', observe_read)
    with pytest.raises(JobCancelled):
        if stage == 'restore_verification':
            backup._restore_project_archive(full.archive_path, tmp_path / 'restore', cancel_requested=lambda: cancel)
        else:
            backup_project_archive(project, tmp_path / 'backups', cancel_requested=lambda: cancel, retention_keep=1)
    assert observed_sizes == [block_size]
    assert Path(full.archive_path).read_bytes() == before
    assert list((tmp_path / 'backups').iterdir()) == [Path(full.archive_path)]
    if stage == 'restore_verification':
        assert list((tmp_path / 'restore').iterdir()) == []


def test_incremental_snapshot_does_not_resurrect_deleted_files(project, tmp_path):
    asset = project.root / 'processed' / 'old_result.txt'
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text('stale result')
    backup_project_archive(project, tmp_path / 'backups')
    asset.unlink()
    delta = backup_project_archive(project, tmp_path / 'backups', incremental=True)
    restored = restore_project_archive(delta.archive_path, tmp_path / 'restore')
    assert not (Path(restored.project_path) / 'processed' / 'old_result.txt').exists()


def test_multilayer_snapshot_reads_unchanged_files_from_earliest_ancestor(project, tmp_path):
    asset = project.root / 'ancestor_only.txt'
    asset.write_text('preserve every generation')
    full = backup_project_archive(project, tmp_path / 'backups')
    deltas = [backup_project_archive(project, tmp_path / 'backups', incremental=True) for _ in range(3)]
    for delta in deltas:
        assert 'ancestor_only.txt' not in {row['path'] for row in _manifest(delta.archive_path)['files']}
    restored = restore_project_archive(deltas[-1].archive_path, tmp_path / 'restore')
    assert (Path(restored.project_path) / asset.name).read_text() == asset.read_text()
    Path(full.archive_path).unlink()
    with pytest.raises(FieldProjectOperationError, match='基准档案'):
        restore_project_archive(deltas[-1].archive_path, tmp_path / 'missing_ancestor')


@pytest.mark.parametrize('base_name', ['../outside.zip', '/outside.zip', 'C:/outside.zip'])
def test_incremental_base_must_be_a_sibling_filename(project, tmp_path, base_name):
    result = backup_project_archive(project, tmp_path / 'backups')
    _rewrite_manifest(result.archive_path, lambda payload: payload.update(incremental=True, incremental_base_archive=base_name))
    with pytest.raises(FieldProjectOperationError, match='不安全|安全|基准档案'):
        restore_project_archive(result.archive_path, tmp_path / 'restore')
    assert not list((tmp_path / 'restore').glob('*.restore'))


def test_incremental_dependency_cycle_is_rejected(project, tmp_path):
    full = backup_project_archive(project, tmp_path / 'backups')
    delta = backup_project_archive(project, tmp_path / 'backups', incremental=True)
    _rewrite_manifest(full.archive_path, lambda payload: payload.update(incremental=True, incremental_base_archive=Path(delta.archive_path).name))
    with pytest.raises(FieldProjectOperationError, match='循环'):
        restore_project_archive(delta.archive_path, tmp_path / 'restore')


def test_incremental_base_from_another_project_is_rejected(project, tmp_path):
    full = backup_project_archive(project, tmp_path / 'backups')
    delta = backup_project_archive(project, tmp_path / 'backups', incremental=True)
    _rewrite_manifest(full.archive_path, lambda payload: payload.update(project_id='different-project'))
    with pytest.raises(FieldProjectOperationError, match='身份'):
        restore_project_archive(delta.archive_path, tmp_path / 'restore')


def test_legacy_backup_without_deleted_files_still_restores(project, tmp_path):
    full = backup_project_archive(project, tmp_path / 'backups')
    def legacy_manifest(payload):
        payload['schema'] = 'mygpr.project_backup.v1'
        payload.pop('deleted_files', None)
    _rewrite_manifest(full.archive_path, legacy_manifest)
    assert restore_project_archive(full.archive_path, tmp_path / 'restore').verified


def test_legacy_incremental_chain_without_deletion_fields_restores_all_layers(project, tmp_path):
    full = backup_project_archive(project, tmp_path / 'backups')
    deltas = [backup_project_archive(project, tmp_path / 'backups', incremental=True) for _ in range(2)]
    def legacy_manifest(payload):
        payload['schema'] = 'mygpr.project_backup.v1'
        payload.pop('deleted_files', None)
    for result in [full, *deltas]:
        _rewrite_manifest(result.archive_path, legacy_manifest)
    assert restore_project_archive(deltas[-1].archive_path, tmp_path / 'restore').verified


def test_unsafe_deleted_path_is_rejected_before_extraction(project, tmp_path):
    full = backup_project_archive(project, tmp_path / 'backups')
    _rewrite_manifest(full.archive_path, lambda payload: payload.update(deleted_files=['../outside.txt']))
    destination = tmp_path / 'restore'
    with pytest.raises(FieldProjectOperationError, match='不安全'):
        restore_project_archive(full.archive_path, destination)
    assert list(destination.iterdir()) == []


def test_hash_failure_removes_restore_staging_and_publishes_no_project(project, tmp_path):
    full = backup_project_archive(project, tmp_path / 'backups')

    def alter_hash(payload):
        payload['files'][0]['sha256'] = '0' * 64

    _rewrite_manifest(full.archive_path, alter_hash)
    destination = tmp_path / 'restore'
    with pytest.raises(FieldProjectOperationError, match='校验失败'):
        restore_project_archive(full.archive_path, destination)
    assert list(destination.iterdir()) == []


def test_cancelled_backup_publishes_nothing_and_preserves_previous_archive(project, tmp_path):
    full = backup_project_archive(project, tmp_path / 'backups')
    before = Path(full.archive_path).read_bytes()
    with pytest.raises(JobCancelled):
        backup_project_archive(project, tmp_path / 'backups', cancel_requested=lambda: True, retention_keep=1)
    assert Path(full.archive_path).read_bytes() == before
    assert list((tmp_path / 'backups').iterdir()) == [Path(full.archive_path)]


def test_backup_cancelled_during_copy_does_not_prune_or_publish(project, tmp_path):
    full = backup_project_archive(project, tmp_path / 'backups')
    before = Path(full.archive_path).read_bytes()
    cancel = False

    def progress(*args):
        nonlocal cancel
        cancel = True

    with pytest.raises(JobCancelled):
        backup_project_archive(project, tmp_path / 'backups',
                               cancel_requested=lambda: cancel, progress_callback=progress, retention_keep=1)
    assert Path(full.archive_path).read_bytes() == before
    assert list((tmp_path / 'backups').iterdir()) == [Path(full.archive_path)]


def test_failed_restore_verification_does_not_publish_verified_backup(project, tmp_path, monkeypatch):
    from core import field_project_backup as backup
    full = backup_project_archive(project, tmp_path / 'backups')
    before = Path(full.archive_path).read_bytes()

    def fail_verification(*args, **kwargs):
        raise FieldProjectOperationError('injected recovery verification failure')

    monkeypatch.setattr(backup, '_restore_project_archive', fail_verification)
    with pytest.raises(FieldProjectOperationError, match='injected'):
        backup_project_archive(project, tmp_path / 'backups', incremental=True, retention_keep=1)
    assert Path(full.archive_path).read_bytes() == before
    assert list((tmp_path / 'backups').iterdir()) == [Path(full.archive_path)]
