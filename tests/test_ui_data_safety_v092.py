from __future__ import annotations

import csv
from pathlib import Path

from core.field_project_models import FieldLineRecord, FieldProjectManifest
from core.field_project_store import FieldProjectStore


def test_new_project_default_coordinate_system_is_auto_zone() -> None:
    manifest = FieldProjectManifest()
    assert manifest.coordinate_system == "CGCS2000 / 3-degree GK"
    dialog_source = Path("ui/field_panels/project_dialogs.py").read_text(encoding="utf-8")
    assert "CGCS2000 / 3-degree GK Zone 39" not in dialog_source
    assert "CGCS2000 / 3-degree GK" in dialog_source


def test_target_without_real_trajectory_does_not_write_fake_xy(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="xy-safety")
    store.upsert_line(FieldLineRecord("L01", "line-without-trajectory"))
    store.save_targets(
        "L01",
        [
            {
                "target_id": "T-01",
                "name": "T-01",
                "line_id": "L01",
                "mileage": 12.5,
                "depth": 1.2,
                "type": "疑似管线",
                "confidence": "★★★☆☆",
                "status": "待确认",
            }
        ],
    )
    target_rows = list(csv.DictReader(store.targets_path("L01").open("r", encoding="utf-8-sig")))
    assert target_rows[0]["x"] == ""
    assert target_rows[0]["y"] == ""

    spatial_path = store.root / "spatial" / "L01_targets_xy.csv"
    spatial_rows = list(csv.DictReader(spatial_path.open("r", encoding="utf-8-sig")))
    assert spatial_rows[0]["x"] == ""
    assert spatial_rows[0]["y"] == ""

    loaded = store.load_targets("L01")
    assert loaded[0]["x"] == ""
    assert loaded[0]["y"] == ""


def test_batch_import_dialog_has_close_protection() -> None:
    source = Path("ui/field_panels/batch_import_dialog.py").read_text(encoding="utf-8")
    assert "def closeEvent" in source
    assert "event.ignore()" in source
    assert "request_cancel" in source
    assert "def reject" in source


def test_line_switch_clears_processing_state() -> None:
    source = Path("ui/field_workbench_window.py").read_text(encoding="utf-8")
    assert "def _clear_line_dependent_processing_state" in source
    assert "self.processed_gpr_dataset = None" in source
    select_block = source[source.index("def _select_line_from_table"):source.index("def _setup_ui")]
    assert "self._clear_line_dependent_processing_state()" in select_block


def test_bscan_orientation_fix_always_requires_confirmation() -> None:
    source = Path("ui/field_panels/project_page.py").read_text(encoding="utf-8")
    method = source[source.index("def _action_fix_bscan_orientation"):source.index("def _action_export_line_manifest")]
    assert "确认修正 B-scan 方向" in method
    assert "QMessageBox.question" in method
    assert "转置修正会改写标准化 B-scan 数据" in method


def test_legacy_choose_loose_path_is_not_noop() -> None:
    source = Path("ui/field_workbench_window.py").read_text(encoding="utf-8")
    method = source[source.index("def choose_loose_path"):source.index("def _init_preview_state")]
    assert "pass" not in method
    assert "_action_import_line_dialog" in method
