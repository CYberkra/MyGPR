"""Project archive dialog state and confirmation."""
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import QObject, pyqtSignal  # noqa: E402

from ui import file_dialogs  # noqa: E402
from ui.project_archive_dialogs import BackupDialog, RestoreDialog  # noqa: E402
from ui.controllers.delivery_controller import DeliveryController, _ArchiveCheckCommand  # noqa: E402


def test_backup_cleanup_requires_opt_in(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(file_dialogs, 'getExistingDirectory', lambda *_args: str(tmp_path))
    dialog = BackupDialog(str(tmp_path))
    assert not dialog._submit.isEnabled()
    dialog._select_destination()
    assert dialog._submit.isEnabled()
    assert dialog.options() == {
        'destination_dir': str(tmp_path), 'incremental': True, 'retention_keep': None,
    }
    dialog._full.setChecked(True)
    dialog._prune.setChecked(True)
    assert dialog.options()['incremental'] is False
    assert dialog.options()['retention_keep'] == 10


def test_restore_requires_latest_archive_check(qapp, tmp_path, monkeypatch):
    class Checker(QObject):
        archive_checked = pyqtSignal(str, str)

        def check_backup_archive(self, token, path):
            self.token = token

    archives = iter((str(tmp_path / 'first.zip'), str(tmp_path / 'second.zip')))
    monkeypatch.setattr(file_dialogs, 'getOpenFileName', lambda *_args: (next(archives), ''))
    checker = Checker()
    dialog = RestoreDialog(str(tmp_path), checker)
    dialog._select_archive()
    first = checker.token
    dialog._select_archive()
    checker.archive_checked.emit(first, '')
    assert not dialog._submit.isEnabled()
    checker.archive_checked.emit(checker.token, '缺少基准档案 base.zip')
    assert not dialog._submit.isEnabled()
    assert 'base.zip' in dialog._status.text()
    checker.archive_checked.emit(checker.token, '')
    assert dialog._submit.isEnabled()
    assert dialog.options()['archive_path'].endswith('second.zip')


def test_invalid_archive_shows_actionable_error_and_blocks_restore(qapp, tmp_path, monkeypatch):
    archive = tmp_path / 'invalid.zip'
    archive.write_text('not a zip', encoding='utf-8')
    monkeypatch.setattr(file_dialogs, 'getOpenFileName', lambda *_args: (str(archive), ''))
    controller = DeliveryController()
    monkeypatch.setattr(controller, 'check_backup_archive',
                        lambda token, path: _ArchiveCheckCommand(controller, token, path).execute())
    dialog = RestoreDialog(str(tmp_path), controller)

    dialog._select_archive()

    assert not dialog._submit.isEnabled()
    assert '请选择其他备份包' in dialog._status.text()
    assert 'MYGPR_ERROR' not in dialog._status.text()
    assert 'File is not a zip file' not in dialog._status.text()
