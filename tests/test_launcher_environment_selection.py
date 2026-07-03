from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_launcher_reports_v0873_and_keeps_manual_install_policy() -> None:
    launcher = (ROOT / "scripts" / "mygpr_windows_launcher.py").read_text(encoding="utf-8")
    start_bat = (ROOT / "start_mygpr.bat").read_text(encoding="utf-8")
    check_bat = (ROOT / "check_mygpr_environment.bat").read_text(encoding="utf-8")

    assert "read_version" in launcher
    assert "does not install" in launcher
    assert "python -m pip install -r requirements.txt" in launcher
    assert "MyGPR v0.8.89" in start_bat
    assert "MyGPR v0.8.89" in check_bat
    assert "install_mygpr_environment.bat" in start_bat
    assert "requirements.txt" in start_bat
    assert "requirements.txt" in check_bat


def test_launcher_candidate_order_prefers_existing_envs_before_windows_py() -> None:
    launcher = (ROOT / "scripts" / "mygpr_windows_launcher.py").read_text(encoding="utf-8")
    order_tokens = [
        "os.environ.get(\"MYGPR_PYTHON\")",
        "for prefix_name in (\"VIRTUAL_ENV\", \"CONDA_PREFIX\")",
        "for rel in (\".venv\", \"venv\", \"env\", \".env\")",
        "_append_conda_envs(candidates)",
        "_append_path_pythons(candidates)",
        "_append_windows_launcher_pythons(candidates)",
        "_append_common_python_installs(candidates)",
    ]
    positions = [launcher.index(token) for token in order_tokens]
    assert positions == sorted(positions)


def test_field_workbench_wording_guardrails() -> None:
    ui_text = (ROOT / "ui" / "field_workbench_window.py").read_text(encoding="utf-8")
    bridge_text = (ROOT / "core" / "field_processing_bridge.py").read_text(encoding="utf-8")

    assert "单算法处理" not in ui_text
    assert "重采样类" not in ui_text
    assert "equidistant_trace_resample" in bridge_text
    assert "HIDDEN_FIELD_METHOD_IDS" in bridge_text
