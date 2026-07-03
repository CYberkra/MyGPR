from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from core.field_project_operations import batch_import_line_data, diagnose_import_failure
from core.field_project_store import FieldProjectStore
from ui.field_panels.batch_import_dialog import BatchImportProgressDialog


def _app():
    return QApplication.instance() or QApplication([])


def test_batch_import_result_contains_diagnostics_and_artifact_paths(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="diagnostic-test")
    valid = Path("sample_data/gui_sidecar_all_data_main.csv")
    invalid = tmp_path / "broken.csv"
    invalid.write_text("not,a,gpr,file\n", encoding="utf-8")

    summary = batch_import_line_data(store, [valid, invalid])

    assert summary.total == 2
    assert summary.succeeded == 1
    success = summary.results[0]
    failure = summary.results[1]

    assert success.success is True
    assert success.file_size_mb > 0
    assert success.elapsed_s >= 0
    assert success.raw_dir
    assert Path(success.raw_dir).exists()
    assert success.manifest_path
    assert Path(success.manifest_path).exists()
    assert success.shape_text == "10×12"

    assert failure.success is False
    assert failure.diagnosis
    assert "请" in failure.diagnosis or "CSV" in failure.diagnosis


def test_diagnose_import_failure_for_matrix_validation_message(tmp_path: Path) -> None:
    src = tmp_path / "Line9origin.csv"
    src.write_text("1,2,3,4,5\n", encoding="utf-8")
    diagnosis = diagnose_import_failure(src, "CSV numeric content is too small for a B-scan matrix: rows=5, cols=5")
    assert "头信息" in diagnosis
    assert "B-scan" in diagnosis


def test_batch_import_dialog_has_result_table_and_actions(tmp_path: Path) -> None:
    app = _app()
    store = FieldProjectStore.create_empty(tmp_path / "project", name="dialog-test")
    dialog = BatchImportProgressDialog(None, store=store, sources=[Path("sample_data/gui_sidecar_all_data_main.csv")], auto_start=False)
    try:
        assert dialog.result_table.columnCount() >= 8
        headers = [dialog.result_table.horizontalHeaderItem(i).text() for i in range(dialog.result_table.columnCount())]
        assert "诊断/错误" in headers
        assert dialog.open_raw_button.text() == "打开 raw 目录"
        assert dialog.open_manifest_button.text() == "查看 manifest"
        assert dialog.copy_error_button.text() == "复制诊断"
    finally:
        if dialog._worker is not None:
            dialog._worker.request_cancel()
        dialog.close()
        app.processEvents()
