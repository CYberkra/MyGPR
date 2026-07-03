from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_requirements_file_exists_and_dev_includes_it() -> None:
    runtime = ROOT / "requirements.txt"
    dev = ROOT / "requirements-dev.txt"
    assert runtime.exists()
    runtime_text = runtime.read_text(encoding="utf-8")
    dev_text = dev.read_text(encoding="utf-8")
    for dependency in ["PyQt6", "PyQt6-Fluent-Widgets", "PyWavelets", "matplotlib"]:
        assert dependency in runtime_text
    assert "-r requirements.txt" in dev_text
    assert "pytest" in dev_text


def test_windows_installer_uses_runtime_requirements() -> None:
    installer = (ROOT / "install_mygpr_environment.bat").read_text(encoding="utf-8")
    launcher = (ROOT / "start_mygpr.bat").read_text(encoding="utf-8")
    checker = (ROOT / "check_mygpr_environment.bat").read_text(encoding="utf-8")
    assert "requirements.txt" in installer
    assert "requirements-dev.txt" not in installer
    assert "requirements.txt" in launcher
    assert "requirements.txt" in checker
