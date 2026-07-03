from __future__ import annotations

from pathlib import Path


def test_home_table_preview_helpers_are_split_from_main_window() -> None:
    main = Path("ui/field_workbench_window.py").read_text(encoding="utf-8")
    assert "from ui.field_panels.home_page import HomePageMixin" in main
    assert "from ui.field_panels.table_utils import FieldTableMixin" in main
    assert "from ui.field_panels.preview_helpers import FieldPreviewMixin" in main
    assert "class FieldWorkbenchWindow(HomePageMixin, FieldTableMixin, FieldPreviewMixin" in main
    assert "def _build_home_page" not in main
    assert "def _table" not in main
    assert "def _draw_current_line_bscan" not in main


def test_new_helper_modules_contain_expected_methods() -> None:
    home = Path("ui/field_panels/home_page.py").read_text(encoding="utf-8")
    table = Path("ui/field_panels/table_utils.py").read_text(encoding="utf-8")
    preview = Path("ui/field_panels/preview_helpers.py").read_text(encoding="utf-8")
    assert "class HomePageMixin" in home
    assert "def _build_home_page" in home
    assert "class FieldTableMixin" in table
    assert "def _fill_table" in table
    assert "class FieldPreviewMixin" in preview
    assert "def _draw_current_line_strip" in preview


def test_main_window_line_budget_after_ui_helper_split() -> None:
    line_count = len(Path("ui/field_workbench_window.py").read_text(encoding="utf-8").splitlines())
    # v0.9.5 target was 950; v0.9.24 includes collapsible panels, project events,
    # link navigation, deletion ops, and plot viewer dialog.
    assert line_count <= 1250


def test_v095_docs_and_version_are_present() -> None:
    assert Path("VERSION").read_text(encoding="utf-8").strip() == "0.9.24"
    assert Path("docs/audit/ui_helpers_split_v0.9.6.md").exists()
