from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from core.field_project_models import FieldLineRecord
from core.field_project_store import FieldProjectStore
from core.field_data_quality import DataOrientation
from core.gpr_data_model import GPRDataSet
from ui.field_panels.project_dialogs import QualityReportDialog


def _app():
    return QApplication.instance() or QApplication([])


def test_transpose_fix_backs_up_dataset_and_reruns_quality(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="transpose-test")
    store.upsert_line(FieldLineRecord("L01", "risk-line"))
    matrix = np.arange(300 * 20, dtype=np.float32).reshape(300, 20)
    dataset = GPRDataSet.from_matrix("L01", matrix, length_m=50.0, time_window_ns=700.0, format_name="test-transpose-risk")
    store.save_gpr_dataset("L01", dataset)

    report = store.run_line_quality_check("L01")
    assert report.orientation == DataOrientation.TRANSPOSE_RISK

    fixed = store.transpose_gpr_dataset("L01")
    fixed_dataset = store.load_gpr_dataset("L01")

    assert fixed_dataset.matrix.shape == (20, 300)
    assert fixed.orientation != DataOrientation.TRANSPOSE_RISK
    manifest = store.root / "raw" / "L01" / "orientation_fix_manifest.json"
    assert manifest.exists()
    backup_dir = store.root / "raw" / "L01" / "orientation_fixes"
    assert any(backup_dir.glob("*before_transpose*.npz"))


def test_quality_report_dialog_formats_report_and_fix_button(tmp_path: Path) -> None:
    app = _app()
    store = FieldProjectStore.create_empty(tmp_path / "project", name="dialog-test")
    store.upsert_line(FieldLineRecord("L01", "risk-line"))
    dataset = GPRDataSet.from_matrix("L01", np.ones((300, 20), dtype=np.float32), length_m=50.0, time_window_ns=700.0)
    store.save_gpr_dataset("L01", dataset)
    report = store.run_line_quality_check("L01")

    dialog = QualityReportDialog(None, line_id="L01", report=report, can_fix_orientation=True)
    try:
        assert dialog.fix_button.isEnabled()
        assert "矩阵尺寸" in dialog.findChild(__import__("PyQt6.QtWidgets").QtWidgets.QTextEdit).toPlainText()
        dialog._request_fix()
        assert dialog.fix_requested is True
    finally:
        dialog.close()
        app.processEvents()


def test_workbench_exposes_quality_detail_and_orientation_fix_actions() -> None:
    source = Path("ui/field_panels/project_page.py").read_text(encoding="utf-8")
    assert "查看质检详情" in source
    assert "修正 B-scan 方向" in source
    assert "transpose_gpr_dataset" in source
