from __future__ import annotations

from pathlib import Path


def test_processing_page_callbacks_are_split_from_main_window() -> None:
    main = Path("ui/field_workbench_window.py").read_text(encoding="utf-8")
    page = Path("ui/field_panels/processing_page.py").read_text(encoding="utf-8")
    assert "ProcessingPageMixin" in main
    assert "class ProcessingPageMixin" in page
    for name in [
        "def _page_processing",
        "def _run_selected_processing",
        "def _save_processing_result",
        "def _processing_params_card",
        "def _rebuild_processing_params_panel",
    ]:
        assert name not in main
        assert name in page
    assert len(main.splitlines()) < 1300


def test_processing_page_preserves_algorithm_bridge_contract() -> None:
    page = Path("ui/field_panels/processing_page.py").read_text(encoding="utf-8")
    assert "run_registered_method" in page
    assert "save_processed_line" in page
    assert "last_processing_manifest" in page
    assert "_refresh_target_source_options" in page


def test_processing_split_audit_document_exists() -> None:
    assert Path("docs/audit/processing_page_split_v0.9.6.md").exists()
    assert Path("VERSION").read_text(encoding="utf-8").strip() == "0.9.20"
