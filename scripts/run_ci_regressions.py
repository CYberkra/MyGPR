"""Verify the task, archive and navigation regressions on the workbench source."""
import subprocess
import sys

raise SystemExit(subprocess.call([
    sys.executable, "-m", "pytest", "-q", "--tb=short", "--timeout=120",
    "tests/test_output_panel.py",
    "tests/test_project_archive_dialogs.py",
    "tests/test_main_window_titlebar.py",
    "tests/test_coordinator_chains.py",
    "tests/test_qt_shutdown.py",
]))
