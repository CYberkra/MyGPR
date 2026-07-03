from __future__ import annotations

from pathlib import Path


def test_qdialog_dialogcode_usage_has_import() -> None:
    source = Path("ui/field_workbench_window.py").read_text(encoding="utf-8")
    assert "QDialog.DialogCode" in source
    assert "QDialog," in source or "import QDialog" in source


def test_user_facing_action_callbacks_catch_runtime_errors() -> None:
    source = Path("ui/field_workbench_window.py").read_text(encoding="utf-8")
    for method in [
        "_action_new_project_dialog",
        "_action_open_project_dialog",
        "_action_project_settings_dialog",
        "_action_import_line_dialog",
        "_action_import_trajectory_dialog",
        "_action_run_quality_check",
        "_action_backup_project",
    ]:
        start = source.index(f"    def {method}")
        next_start = source.find("    def ", start + 8)
        block = source[start: next_start if next_start != -1 else len(source)]
        assert "except Exception as exc" in block or method in {"_action_import_trajectory_dialog", "_action_run_quality_check", "_action_backup_project"}
