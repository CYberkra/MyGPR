import json
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


def _add_line(store, line_id: str) -> None:
    from core.field_project_models import FieldLineRecord

    store.upsert_line(FieldLineRecord(line_id, f"Line {line_id}"))


def test_incremental_backup_chain_restore(tmp_path: Path):
    store = FieldProjectStore.create_empty(tmp_path / "project", name="INCR")
    backup_dir = tmp_path / "backup"

    backup_project_archive(store, backup_dir)
    _add_line(store, "L01")
    delta1 = backup_project_archive(store, backup_dir, incremental=True)
    assert delta1.verified
    _add_line(store, "L02")
    delta2 = backup_project_archive(store, backup_dir, incremental=True)
    assert delta2.verified

    with zipfile.ZipFile(delta2.archive_path, "r") as handle:
        manifest = json.loads(handle.read("backup_manifest.json").decode("utf-8"))
    assert manifest["incremental"] is True
    assert manifest["incremental_base_archive"] == Path(delta1.archive_path).name
    delta_paths = {row["path"] for row in manifest["files"]}
    assert "project.json" in delta_paths
    assert "data/lines/L01.json" not in delta_paths  # 未变文件被省略

    restored = restore_project_archive(delta2.archive_path, tmp_path / "restore")
    assert restored.verified and restored.file_count >= 2

    reopened = FieldProjectStore.open(Path(restored.project_path), access_mode="read_only")
    try:
        line_ids = {line.line_id for line in reopened.list_lines()}
        assert line_ids == {"L01", "L02"}
    finally:
        reopened.close()


def test_incremental_without_base_degrades_to_full(tmp_path: Path):
    store = FieldProjectStore.create_empty(tmp_path / "project", name="NODEG")
    result = backup_project_archive(store, tmp_path / "backup", incremental=True)
    assert result.verified
    with zipfile.ZipFile(result.archive_path, "r") as handle:
        manifest = json.loads(handle.read("backup_manifest.json").decode("utf-8"))
    assert manifest["incremental"] is False  # 无基准 → 全量


def test_retention_prunes_oldest_backups(tmp_path: Path):
    store = FieldProjectStore.create_empty(tmp_path / "KEEP", name="KEEP")
    backup_dir = tmp_path / "backup"
    for _ in range(3):
        backup_project_archive(store, backup_dir, retention_keep=2)
    archives = sorted(backup_dir.glob("KEEP_backup_*.zip"))
    assert len(archives) == 2


def test_incremental_restore_missing_base_fails(tmp_path: Path):
    store = FieldProjectStore.create_empty(tmp_path / "project", name="MISS")
    backup_dir = tmp_path / "backup"
    backup_project_archive(store, backup_dir)
    _add_line(store, "L01")
    delta = backup_project_archive(store, backup_dir, incremental=True)
    with zipfile.ZipFile(delta.archive_path, "r") as handle:
        manifest = json.loads(handle.read("backup_manifest.json").decode("utf-8"))
    (backup_dir / manifest["incremental_base_archive"]).unlink()
    with pytest.raises(FieldProjectOperationError, match="基准档案"):
        restore_project_archive(delta.archive_path, tmp_path / "restore")
