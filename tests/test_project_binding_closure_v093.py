from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QLabel

from core.field_project_store import FieldProjectStore
from ui.field_workbench_window import FieldWorkbenchWindow


def _app():
    return QApplication.instance() or QApplication([])


def _labels(win: FieldWorkbenchWindow) -> list[str]:
    return [label.text() for label in win.findChildren(QLabel)]


def test_empty_project_pages_show_empty_state_not_demo(tmp_path: Path) -> None:
    app = _app()
    store = FieldProjectStore.create_empty(tmp_path / "empty_project", name="empty-linkage")
    win = FieldWorkbenchWindow(version_text="MyGPR v0.8.93")
    try:
        win._set_active_project_store(store, status_message="empty")
        win._post_project_operation_refresh(switch_to="data_management")
        app.processEvents()
        assert win.line_records == []
        assert win.active_gpr_dataset is None
        text = "\n".join(_labels(win))
        assert "暂无测线" in text
        assert "L03   过路口测线" not in text
        win.switch_workspace("processing_lab")
        app.processEvents()
        assert win.active_gpr_dataset is None
        win.switch_workspace("interpretation")
        app.processEvents()
        text = "\n".join(_labels(win))
        assert "目标定位视图（-- 暂无测线）" in text or "暂无目标标注" in text
    finally:
        win.close()


def test_imported_project_drives_processing_spatial_and_delivery_pages(tmp_path: Path) -> None:
    app = _app()
    store = FieldProjectStore.create_empty(tmp_path / "import_project", name="import-linkage")
    store.import_line_file("Line9", Path("sample_data/gui_sidecar_all_data_main.csv"), name="9号测线", copy_into_project=True)
    win = FieldWorkbenchWindow(version_text="MyGPR v0.8.93")
    try:
        win._set_active_project_store(store, status_message="imported")
        win._post_project_operation_refresh(switch_to="data_management")
        app.processEvents()
        assert len(win.line_records) == 1
        assert win.selected_line == "Line9"
        assert win.active_gpr_dataset is not None
        assert win.trajectory_model is not None
        text = "\n".join(_labels(win))
        assert "9号测线" in text
        win.switch_workspace("spatial")
        app.processEvents()
        text = "\n".join(_labels(win))
        assert "Line9" in text
        assert "坐标系" in text
        win.switch_workspace("delivery")
        app.processEvents()
        text = "\n".join(_labels(win))
        assert "import-linkage" in text
    finally:
        win.close()
