from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from core.field_project_store import FieldProjectStore
from ui.field_workbench_window import FieldWorkbenchWindow


def _app():
    return QApplication.instance() or QApplication([])


def test_empty_project_tree_and_preview_do_not_show_demo_lines(tmp_path: Path) -> None:
    app = _app()
    store = FieldProjectStore.create_empty(tmp_path / "empty_project", name="empty-state")
    win = FieldWorkbenchWindow(version_text="MyGPR v0.8.92")
    try:
        win._set_active_project_store(store, status_message="test empty")
        win._post_project_operation_refresh(switch_to="data_management")
        app.processEvents()
        assert win.line_records == []
        assert win.project_tree_widget is not None
        texts = []
        root = win.project_tree_widget.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            texts.append(item.text(0))
            for j in range(item.childCount()):
                texts.append(item.child(j).text(0))
        joined = "\n".join(texts)
        assert "暂无测线" in joined
        assert "L03   过路口测线" not in joined
        assert win.active_gpr_dataset is None
    finally:
        win.close()


def test_imported_line_rebuilds_project_bound_widgets(tmp_path: Path) -> None:
    app = _app()
    store = FieldProjectStore.create_empty(tmp_path / "import_project", name="import-state")
    line = store.import_line_file("L01", Path("sample_data/gui_sidecar_all_data_main.csv"), name="real-line", copy_into_project=True)
    win = FieldWorkbenchWindow(version_text="MyGPR v0.8.92")
    try:
        win._set_active_project_store(store, status_message="test import")
        win._post_project_operation_refresh(switch_to="data_management")
        app.processEvents()
        assert len(win.line_records) == 1
        assert win.line_records[0]["id"] == line.line_id
        assert win.active_gpr_dataset is not None
        assert win.active_gpr_dataset.matrix.shape == (10, 12)
        assert win.project_tree_widget is not None
        root = win.project_tree_widget.invisibleRootItem()
        all_text = []
        for i in range(root.childCount()):
            item = root.child(i)
            all_text.append(item.text(0))
            for j in range(item.childCount()):
                all_text.append(item.child(j).text(0))
        joined = "\n".join(all_text)
        assert "real-line" in joined
        assert "已导入" in joined
    finally:
        win.close()
