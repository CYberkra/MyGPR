import zipfile
from pathlib import Path
import pytest
from core.field_project_operations import FieldProjectOperationError, backup_project_archive, restore_project_archive
from core.field_project_store import FieldProjectStore


def test_backup_restore_roundtrip_and_recovery_flag(tmp_path: Path):
    store = FieldProjectStore.create_empty(tmp_path / "project", name="P")
    result = backup_project_archive(store, tmp_path / "backup")
    assert result.verified and result.recovery_tested
    restored = restore_project_archive(result.archive_path, tmp_path / "restore")
    assert restored.verified


def test_restore_rejects_zip_slip(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("backup_manifest.json", '{"files": []}')
        zf.writestr("../escape.txt", "bad")
    with pytest.raises(FieldProjectOperationError):
        restore_project_archive(archive, tmp_path / "restore")
