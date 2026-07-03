from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.check_env import build_report
from scripts.run_app import prepare_runtime_environment
from scripts.check_release_hygiene import scan


def test_environment_checker_reports_current_tree_ok() -> None:
    report = build_report(Path.cwd())
    assert report.version == "0.9.24"
    assert report.python_ok is True
    assert report.writable_log_dir is True
    missing = [row.module for row in report.required_modules if not row.ok]
    assert missing == []
    missing_paths = [row.path for row in report.required_paths if not row.ok]
    assert missing_paths == []


def test_check_env_json_cli_is_machine_readable() -> None:
    cp = subprocess.run(
        [sys.executable, "scripts/check_env.py", "--json", "--strict"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    payload = json.loads(cp.stdout)
    assert payload["schema"] == "mygpr.environment_check.v1"
    assert payload["version"] == "0.9.24"
    assert payload["ok"] is True


def test_run_app_check_uses_same_environment_contract() -> None:
    cp = subprocess.run(
        [sys.executable, "scripts/run_app.py", "--check"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "Environment status: OK" in cp.stdout


def test_environment_helpers_do_not_create_release_logs_directory(monkeypatch) -> None:
    root = Path.cwd()
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("MYGPR_LOG_DIR", raising=False)
    if (root / "logs").exists():
        raise AssertionError("test requires a clean release root without top-level logs")

    report = build_report(root)
    env = prepare_runtime_environment(root)

    assert report.writable_log_dir is True
    assert not (root / "logs").exists()
    assert not Path(env["MPLCONFIGDIR"]).is_relative_to(root)


def test_windows_launcher_scripts_are_versioned_and_include_projection_dependency() -> None:
    for name in ["start_mygpr.bat", "check_mygpr_environment.bat"]:
        text = Path(name).read_text(encoding="utf-8")
        assert "v0.9.20" in text
        assert "v0.8.89" not in text
    assert "pyproj" in Path("install_mygpr_environment.bat").read_text(encoding="utf-8")
    assert "pyproj" in Path("scripts/mygpr_windows_launcher.py").read_text(encoding="utf-8")


def test_release_hygiene_scanner_detects_forbidden_runtime_files(tmp_path: Path) -> None:
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "__pycache__").mkdir()
    (tmp_path / "core" / "__pycache__" / "x.pyc").write_bytes(b"bad")
    (tmp_path / "runtime_projects").mkdir()
    (tmp_path / "runtime_projects" / "demo.txt").write_text("bad", encoding="utf-8")
    payload = scan(tmp_path)
    assert payload["ok"] is False
    joined = "\n".join(payload["findings"])
    assert "__pycache__" in joined
    assert "runtime_projects" in joined


def test_release_hygiene_scanner_accepts_clean_minimal_tree(tmp_path: Path) -> None:
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "x.py").write_text("x=1\n", encoding="utf-8")
    payload = scan(tmp_path)
    assert payload["ok"] is True
