from __future__ import annotations

import os
from pathlib import Path
import sys

from scripts.process_runner import run_logged_process, safe_log_name


def test_logged_process_captures_output_without_pipe(tmp_path: Path) -> None:
    result = run_logged_process(
        [sys.executable, "-c", "print('runner-ok')"],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=10,
        log_path=tmp_path / "normal.log",
    )
    assert result.returncode == 0
    assert result.timed_out is False
    assert "runner-ok" in result.output_tail
    assert Path(result.log_path).exists()


def test_logged_process_kills_timed_out_process_group(tmp_path: Path) -> None:
    result = run_logged_process(
        [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(30)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=1,
        log_path=tmp_path / "timeout.log",
    )
    assert result.returncode == 124
    assert result.timed_out is True
    assert "TIMEOUT" in result.output_tail


def test_safe_log_name_removes_path_separators() -> None:
    assert safe_log_name("pytest-isolated:tests/test_x.py") == "pytest-isolated-tests-test_x.py"
