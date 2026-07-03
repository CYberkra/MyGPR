from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from ui.simulation_validation_page import SimulationValidationPage, _format_shell_command
from ui.workbench_window import MyGPRWorkbenchWindow, WORKSPACES

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "gprmax_campaign"


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_simulation_validation_page_loads_campaign_and_previews_command() -> None:
    app = _app()
    page = SimulationValidationPage()
    try:
        result = page.load_campaign(FIXTURES / "campaign_valid.yaml")
        assert result.status == "ready"
        assert page.scene_table.rowCount() == 1
        assert page.scene_combo.currentData() == "scene_valid_01"
        assert "GX-RUN-001_valid" in page.status_label.text()

        page.num_runs_spin.setValue(3)
        command = page.build_run_command_preview()
        assert command[:2] == ["python", "scripts/gprmax_campaign_runner.py"]
        assert "--campaign" in command
        assert "--run-scene" in command
        assert "scene_valid_01" in command
        assert command[-2:] == ["--num-runs", "3"]
        assert "gprmax_campaign_runner.py" in page.command_preview.toPlainText()
        assert page.copy_command_button.isEnabled()
    finally:
        page.close()
        app.processEvents()


def test_simulation_validation_page_reports_invalid_scene_and_blocks_copy() -> None:
    app = _app()
    page = SimulationValidationPage()
    try:
        result = page.load_campaign(FIXTURES / "campaign_missing_expected_outputs.yaml")
        assert result.status == "invalid"
        assert page.scene_table.rowCount() == 1
        assert page._selected_scene_status() == "invalid"
        assert "expected_outputs_missing" in page.details_text.toPlainText()
        assert page.build_run_command_preview() == []
        assert "不生成运行命令" in page.command_preview.toPlainText()
        assert not page.copy_command_button.isEnabled()
    finally:
        page.close()
        app.processEvents()


def test_simulation_validation_command_preview_uses_windows_safe_quotes() -> None:
    command = _format_shell_command([
        "python",
        "scripts/gprmax_campaign_runner.py",
        "--campaign",
        r"C:\My Projects\case one\campaign.yaml",
    ])
    assert '"C:\\My Projects\\case one\\campaign.yaml"' in command
    assert "'C:" not in command


def test_workbench_hides_simulation_validation_workspace_by_default() -> None:
    app = _app()
    win = MyGPRWorkbenchWindow()
    try:
        assert "simulation_validation" not in WORKSPACES
        assert "simulation_validation" not in win.workspace_buttons
        assert win.simulation_validation is None
    finally:
        win.close()
        app.processEvents()


def test_simulation_validation_rejects_invalid_gpu_device_text() -> None:
    app = _app()
    page = SimulationValidationPage()
    try:
        page.load_campaign(FIXTURES / "campaign_valid.yaml")
        page.gpu_devices_edit.setText("0 abc -1")
        page.refresh_command_preview()
        assert page.build_run_command_preview() == []
        assert "GPU 设备格式无效" in page.command_preview.toPlainText()
        assert not page.copy_command_button.isEnabled()
    finally:
        page.close()
        app.processEvents()
