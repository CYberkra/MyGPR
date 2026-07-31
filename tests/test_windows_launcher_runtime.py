from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _launcher_module():
    path = ROOT / "scripts" / "mygpr_windows_launcher.py"
    spec = importlib.util.spec_from_file_location("mygpr_windows_launcher_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_rejects_python_310_and_accepts_supported_range() -> None:
    launcher = _launcher_module()
    assert launcher.is_supported_python_version((3, 10, 9)) is False
    assert launcher.is_supported_python_version("3.10.9") is False
    assert launcher.is_supported_python_version((3, 11, 0)) is True
    assert launcher.is_supported_python_version("3.12.0") is True
    assert launcher.is_supported_python_version((3, 13, 9)) is True
    assert launcher.is_supported_python_version((3, 14, 0)) is False


def test_launcher_environment_probe_includes_version_policy() -> None:
    launcher = (ROOT / "scripts" / "mygpr_windows_launcher.py").read_text(encoding="utf-8")
    assert 'PYTHON_VERSION_LABEL = "Python 3.11-3.13"' in launcher
    assert "version_supported" in launcher
    assert '("PIL", "Pillow")' in launcher
    assert "supported and not missing" in launcher
    assert "No checked Python 3.11-3.13 environment" in launcher


def test_environment_checker_uses_same_python_and_pillow_contract() -> None:
    checker = (ROOT / "scripts" / "check_env.py").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-core.txt").read_text(encoding="utf-8")
    assert "MIN_PYTHON = (3, 11)" in checker
    assert "MAX_PYTHON = (3, 13)" in checker
    assert '("PIL", "Pillow"' in checker
    assert "Pillow==12.3.0" in requirements
