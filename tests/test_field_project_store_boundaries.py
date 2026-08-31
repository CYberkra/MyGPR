from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_field_project_store_is_small_coordinator() -> None:
    main_path = ROOT / "core" / "field_project_store.py"
    text = main_path.read_text(encoding="utf-8")
    assert len(text.splitlines()) < 180
    assert "FieldLineStoreMixin" in text
    assert "FieldTargetStoreMixin" in text
    assert "FieldSpatialStoreMixin" in text
    assert "FieldArtifactStoreMixin" in text
    assert "def save_targets" not in text
    assert "def save_processed_line" not in text
    assert "def export_spatial_targets_xy" not in text


def test_field_project_store_modules_exist() -> None:
    for module in [
        "core/field_project_models.py",
        "core/field_line_store.py",
        "core/field_target_store.py",
        "core/field_spatial_store.py",
        "core/field_artifact_store.py",
        "core/field_project_runtime_store.py",
    ]:
        assert (ROOT / module).exists(), module


def test_field_project_store_public_imports_remain_compatible() -> None:
    from core.field_project_store import FIELD_PROJECT_SCHEMA, FieldLineRecord, FieldProjectStore

    assert FIELD_PROJECT_SCHEMA == "mygpr.field_project.v3"
    assert FieldLineRecord("L99", "测试测线").to_ui_dict()["id"] == "L99"
    assert hasattr(FieldProjectStore, "create_empty")
    assert not hasattr(FieldProjectStore, "create_or_open_demo")
