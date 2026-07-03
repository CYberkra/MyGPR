from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from core.field_project_operations import batch_import_line_data, infer_line_identity_from_filename
from core.field_project_store import FieldProjectStore
from ui.field_workbench_window import FieldWorkbenchWindow


def _app():
    return QApplication.instance() or QApplication([])


def test_infer_field_line_identity_from_common_csv_names() -> None:
    assert infer_line_identity_from_filename("Line9origin(30).csv").line_id == "L09_30"
    assert infer_line_identity_from_filename("Line9origin(30).csv").name == "9号测线（30）"
    assert infer_line_identity_from_filename("Line3origin.csv").line_id == "L03"
    assert infer_line_identity_from_filename("L1origin.csv").line_id == "L01"
    assert infer_line_identity_from_filename("X1origin.csv").line_id == "X1"


def test_batch_import_continues_after_individual_failure(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "batch_project", name="batch-import")
    valid = Path("sample_data/gui_sidecar_all_data_main.csv")
    invalid = tmp_path / "bad.csv"
    invalid.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    summary = batch_import_line_data(store, [valid, invalid])

    assert summary.total == 2
    assert summary.succeeded == 1
    assert summary.failed == 1
    lines = store.list_lines()
    assert len(lines) == 1
    assert store.load_gpr_dataset(lines[0].line_id).matrix.shape == (10, 12)


def test_batch_imported_lines_bind_to_workbench_project_tree(tmp_path: Path) -> None:
    app = _app()
    store = FieldProjectStore.create_empty(tmp_path / "ui_batch_project", name="ui-batch")
    src = Path("sample_data/gui_sidecar_all_data_main.csv")
    first = tmp_path / "Line3origin.csv"
    second = tmp_path / "Line9origin(30).csv"
    first.write_bytes(src.read_bytes())
    second.write_bytes(src.read_bytes())
    summary = batch_import_line_data(store, [first, second])
    assert summary.succeeded == 2

    win = FieldWorkbenchWindow(version_text="MyGPR v0.8.95")
    try:
        win._set_active_project_store(store, status_message="batch test")
        win._post_project_operation_refresh(switch_to="data_management")
        app.processEvents()
        ids = {line["id"] for line in win.line_records}
        assert {"L03", "L09_30"}.issubset(ids)
        assert win.project_tree_widget is not None
        root = win.project_tree_widget.invisibleRootItem()
        texts: list[str] = []
        for i in range(root.childCount()):
            item = root.child(i)
            texts.append(item.text(0))
            for j in range(item.childCount()):
                texts.append(item.child(j).text(0))
        joined = "\n".join(texts)
        assert "3号测线" in joined
        assert "9号测线（30）" in joined
    finally:
        win.close()
