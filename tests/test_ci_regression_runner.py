"""CI entry accepts named suites and preserves pytest's exit code."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import pytest


@pytest.fixture
def runner(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_ci_regressions.py"
    spec = importlib.util.spec_from_file_location("ci_regression_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for relative in ("tests/test_ci_regression_runner.py", *(p for group in module.SUITES.values() for p in group)):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    return module


def test_refuses_local_execution_before_launching_pytest(runner, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("local tests launched"))
    with pytest.raises(SystemExit) as caught:
        runner.main(["--suite", "bscan"])
    assert caught.value.code == 2


def test_selected_suite_passes_failure_status_and_has_junit(runner, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    calls = []
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=5)
    monkeypatch.setattr(runner.subprocess, "run", launch)
    assert runner.main(["--suite", "bscan"]) == 5
    command, kwargs = calls[0]
    assert "tests/test_bscan_viewport_levels.py" in command
    assert "tests/test_image_denoising.py" not in command
    assert any(arg.startswith("--junitxml=") for arg in command)
    assert kwargs == {"cwd": runner.ROOT, "check": False}


def test_all_suites_are_deduplicated(runner, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kw:
                        commands.append(command) or SimpleNamespace(returncode=0))
    assert runner.main(["--suite", "all"]) == 0
    paths = [arg for arg in commands[0] if arg.startswith("tests/")]
    assert len(paths) == len(set(paths))
    assert "tests/test_image_denoising.py" in paths
    assert "tests/test_processing_page_v2.py" in paths


def test_unknown_suite_or_missing_test_fails_before_subprocess(runner, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected subprocess"))
    with pytest.raises(SystemExit):
        runner.main(["--suite", "bscan; arbitrary command"])
    (runner.ROOT / "tests/test_bscan_viewport_levels.py").unlink()
    with pytest.raises(SystemExit):
        runner.main(["--suite", "bscan"])
