from __future__ import annotations

from pathlib import Path


def test_field_workbench_delegates_project_page_callbacks() -> None:
    source = Path("ui/field_workbench_window.py").read_text(encoding="utf-8")
    project_page = Path("ui/field_panels/project_page.py").read_text(encoding="utf-8")
    assert "ProjectPageMixin" in source
    assert "class ProjectPageMixin" in project_page
    assert "def _action_import_line_dialog" not in source
    assert "def _action_import_line_dialog" in project_page
    assert "def _page_project_management" not in source
    assert "def _page_project_management" in project_page
    assert len(source.splitlines()) < 1800


def test_target_actions_are_split_from_interpretation_page() -> None:
    page = Path("ui/field_panels/interpretation_page.py").read_text(encoding="utf-8")
    actions = Path("ui/field_panels/target_actions.py").read_text(encoding="utf-8")
    assert "TargetActionsMixin" in page
    assert "class TargetActionsMixin" in actions
    assert "def _add_preview_target" not in page
    assert "def _add_preview_target" in actions
    assert "def _on_target_canvas_click" not in page
    assert "def _on_target_canvas_click" in actions


def test_version_and_split_audit_document() -> None:
    assert Path("VERSION").read_text(encoding="utf-8").strip() == "0.9.20"
    assert Path("docs/audit/ui_callback_split_v0.9.3.md").exists()
