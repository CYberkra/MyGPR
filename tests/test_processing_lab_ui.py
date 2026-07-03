from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from core.ingest_service import IngestService
from core.qc_service import QcService
from ui.processing_lab_page import ProcessingLabPage
from ui.workbench_window import MyGPRWorkbenchWindow

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _formal_project(tmp_path: Path) -> Path:
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(120, dtype=np.float32).reshape(20, 6), delimiter=",")
    temporary = IngestService.open_temporary(source)
    formal = IngestService.formalize(temporary, tmp_path / "formal", name="Formal")
    line_id = formal.list_lines()[0].line_id
    qc = QcService(formal)
    report = qc.run_line_qc(line_id)
    for item in report.items:
        if item.severity == "warning" and not item.acknowledged:
            qc.acknowledge_warning(line_id, item.code, "测试夹具确认该警告不阻断处理")
    formal.close()
    temporary.close()
    return tmp_path / "formal"


def test_workbench_processing_workspace_is_real_processing_lab(tmp_path: Path) -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_project(_formal_project(tmp_path))
        line_id = win.project.list_lines()[0].line_id
        win.selected_line_id = line_id
        win.switch_workspace("processing_lab")

        assert isinstance(win.processing_lab, ProcessingLabPage)
        assert win.workspace_pages["processing_lab"] is win.processing_lab
        win.processing_lab.open_line(win.project, line_id)
        assert win.processing_lab.session is not None
        assert win.processing_lab.chain_table.rowCount() == 0
        assert win.processing_lab.method_combo.count() > 10
        assert win.processing_lab.canvas.figure.axes
    finally:
        win.close()
        app.processEvents()


def test_processing_lab_applies_method_updates_chain_and_can_save(tmp_path: Path) -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_project(_formal_project(tmp_path))
        line_id = win.project.list_lines()[0].line_id
        win.processing_lab.open_line(win.project, line_id)
        win.processing_lab.select_method("amplitude_scale")
        win.processing_lab.params_editor.setPlainText('{"scale": 2.0}')

        win.processing_lab.apply_selected_method()
        assert win.processing_lab.chain_table.rowCount() == 1
        assert "幅值" in win.processing_lab.chain_table.item(0, 1).text()
        win.processing_lab.chain_table.selectRow(0)
        win.processing_lab.toggle_step()
        assert win.processing_lab.chain_table.item(0, 0).text() == "停用"
        win.processing_lab.toggle_step()
        assert win.processing_lab.chain_table.item(0, 0).text() == "启用"
        result = win.processing_lab.save_version("UI Version")
        assert result.name == "UI Version"
        assert win.processing_lab.compare_combo.findData(f"version:{result.result_id}") >= 0
        win.processing_lab.compare_combo.setCurrentIndex(
            win.processing_lab.compare_combo.findData(f"version:{result.result_id}")
        )
        win.processing_lab.refresh_plot()
        assert win.processing_lab.figure.axes[0].get_title().startswith("处理结果")
    finally:
        win.close()
        app.processEvents()


def test_processing_lab_button_operations_use_background_worker(tmp_path: Path) -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_project(_formal_project(tmp_path))
        line_id = win.project.list_lines()[0].line_id
        win.processing_lab.open_line(win.project, line_id)
        win.processing_lab.select_method("amplitude_scale")
        win.processing_lab.params_editor.setPlainText('{"scale": 2.0}')

        win.processing_lab.apply_button.click()
        assert win.processing_lab._task_thread is not None
        deadline = __import__("time").time() + 10
        while win.processing_lab._task_thread is not None and __import__("time").time() < deadline:
            app.processEvents()
        assert win.processing_lab.chain_table.rowCount() == 1
    finally:
        win.close()
        app.processEvents()


def test_processing_lab_runs_and_exports_manual_auto_comparison(
    tmp_path: Path,
) -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_project(_formal_project(tmp_path))
        line_id = win.project.list_lines()[0].line_id
        win.processing_lab.open_line(win.project, line_id)
        win.processing_lab.select_method("dewow")
        win.processing_lab.params_editor.setPlainText('{"window": 1}')

        win.processing_lab.compare_autotune_button.click()
        assert win.processing_lab._task_thread is not None
        deadline = __import__("time").time() + 15
        while win.processing_lab._task_thread is not None and __import__("time").time() < deadline:
            app.processEvents()

        assert win.processing_lab.session.last_manual_auto_comparison is not None
        assert win.processing_lab.export_comparison_button.isEnabled() is True
        win.processing_lab.export_comparison_button.click()
        bundle_root = (
            win.project.root
            / "exports"
            / "auto_tune_comparisons"
            / f"manual_auto_{line_id}"
        )
        assert (bundle_root / "comparison_report.md").exists()
        assert (bundle_root / "evidence_bundle.zip").exists()
    finally:
        win.close()
        app.processEvents()


def test_processing_lab_display_controls_are_display_only(tmp_path: Path) -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_project(_formal_project(tmp_path))
        line_id = win.project.list_lines()[0].line_id
        win.processing_lab.open_line(win.project, line_id)
        original = np.array(win.processing_lab.session.current_data, copy=True)

        win.processing_lab.cmap_combo.setCurrentText("seismic")
        win.processing_lab.symmetric_scale_check.setChecked(True)
        win.processing_lab.percentile_clip_check.setChecked(True)
        win.processing_lab.colorbar_check.setChecked(True)
        win.processing_lab.refresh_plot()

        image = win.processing_lab.figure.axes[0].images[0]
        vmin, vmax = image.get_clim()
        assert image.get_cmap().name == "seismic"
        assert vmin < 0 < vmax
        assert abs(abs(vmin) - abs(vmax)) < 1.0e-6
        assert len(win.processing_lab.figure.axes) == 2
        assert np.array_equal(win.processing_lab.session.current_data, original)

        win.processing_lab.select_method("amplitude_scale")
        win.processing_lab.params_editor.setPlainText('{"scale": 2.0}')
        win.processing_lab.apply_selected_method()
        win.processing_lab.compare_combo.setCurrentIndex(
            win.processing_lab.compare_combo.findData("difference")
        )
        win.processing_lab.refresh_plot()
        diff_image = win.processing_lab.figure.axes[0].images[0]
        diff_vmin, diff_vmax = diff_image.get_clim()
        assert diff_image.get_cmap().name == "coolwarm"
        assert diff_vmin < 0 < diff_vmax
        assert abs(abs(diff_vmin) - abs(diff_vmax)) < 1.0e-6
    finally:
        win.close()
        app.processEvents()


def test_processing_lab_apply_recommendation_is_tied_to_selected_method(
    tmp_path: Path,
) -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        win.open_project(_formal_project(tmp_path))
        line_id = win.project.list_lines()[0].line_id
        win.processing_lab.open_line(win.project, line_id)
        win.processing_lab.select_method("dewow")
        win.processing_lab.session.last_recommendation = {
            "method_key": "dewow",
            "recommended_params": {"window": 1},
        }
        win.processing_lab._sync_controls()
        assert win.processing_lab.apply_recommendation_button.isEnabled() is True

        win.processing_lab.select_method("amplitude_scale")
        assert win.processing_lab.apply_recommendation_button.isEnabled() is False
        assert "重新生成参数推荐" in win.processing_lab.recommendation_text.text()

        win.processing_lab.apply_recommendation()
        assert win.processing_lab.chain_table.rowCount() == 0
        assert "不一致" in win.processing_lab.recommendation_text.text()
    finally:
        win.close()
        app.processEvents()
