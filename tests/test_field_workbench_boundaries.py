from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_field_workbench_main_file_has_page_boundaries() -> None:
    main_path = ROOT / "ui" / "field_workbench_window.py"
    text = main_path.read_text(encoding="utf-8")
    line_count = len(text.splitlines())

    assert line_count < 1800
    assert "class FieldWorkbenchWindow(" in text
    for mixin in [
        "HomePageMixin",
        "FieldTableMixin",
        "FieldPreviewMixin",
        "ProjectPageMixin",
        "ProcessingPageMixin",
        "InterpretationPageMixin",
        "SpatialPageMixin",
        "DeliveryPageMixin",
        "QMainWindow",
    ]:
        assert mixin in text
    assert "def _page_interpretation" not in text
    assert "def _page_spatial" not in text
    assert "def _page_delivery" not in text

    for module in [
        "ui/field_panels/interpretation_page.py",
        "ui/field_panels/spatial_page.py",
        "ui/field_panels/delivery_page.py",
        "ui/field_panels/widgets.py",
        "ui/field_panels/plots.py",
    ]:
        assert (ROOT / module).exists(), module


def test_docs_are_partitioned_into_top_level_buckets() -> None:
    docs = ROOT / "docs"
    top_level_files = [p.name for p in docs.iterdir() if p.is_file()]
    assert top_level_files == []
    for bucket in ["user", "developer", "audit", "legacy"]:
        assert (docs / bucket).is_dir()
    assert (docs / "developer" / "document_index_v0.8.82.md").exists()
    assert (docs / "legacy" / "research").is_dir()
