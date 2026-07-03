from __future__ import annotations

from pathlib import Path

from core.field_project_models import FieldLineRecord
from core.field_project_operations import backup_project_archive, export_line_manifest_csv
from core.field_project_store import FieldProjectStore


def test_export_line_manifest_and_project_backup(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="beta-test")
    store.upsert_line(FieldLineRecord("L01", "beta-line", 12.5, "通过", "已投影", "已导入", "now"))

    manifest_csv = export_line_manifest_csv(store)
    assert manifest_csv.exists()
    text = manifest_csv.read_text(encoding="utf-8-sig")
    assert "line_id" in text
    assert "L01" in text
    assert "beta-line" in text

    result = backup_project_archive(store, tmp_path / "backups")
    archive = Path(result.archive_path)
    assert archive.exists()
    assert archive.suffix == ".zip"
    assert result.file_count >= 1
    assert result.size_mb >= 0


def test_beta_ui_removes_known_placeholder_backup_and_connects_export() -> None:
    source = Path("ui/field_panels/project_page.py").read_text(encoding="utf-8")
    assert "项目备份入口已保留" not in source
    assert "backup_project_archive" in source
    assert "export_line_manifest_csv" in source
    assert "_action_export_line_manifest" in source
    start = source.index("def _line_list_card")
    end = source.index("def _add_preview_line", start)
    body = source[start:end]
    assert "导出清单" in body
    assert "_action_export_line_manifest" in body


def test_version_and_beta_boundary_documents_exist() -> None:
    assert Path("VERSION").read_text(encoding="utf-8").strip() == "0.9.20"
    assert Path("docs/user/manual_v0.9.0_beta.md").exists()
    assert Path("docs/developer/beta_boundary_v0.9.0.md").exists()
    assert Path("docs/audit/button_callback_audit_v0.9.0.md").exists()
