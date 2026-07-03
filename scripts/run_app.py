#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-platform MyGPR application runner.

This is the canonical script entry point for development, terminal starts and
Windows batch wrappers.  It sets runtime-safe environment defaults and can run a
read-only environment check before launching the Qt application.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from scripts.check_env import build_report, default_log_dir, print_human


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def prepare_runtime_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) + ((os.pathsep + existing_pythonpath) if existing_pythonpath else "")
    env.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    env.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    env.setdefault("QT_OPENGL", "software")
    env.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false;qt.qpa.window=false")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("MPLBACKEND", "QtAgg")
    log_dir = Path(env.get("MYGPR_LOG_DIR") or default_log_dir())
    env.setdefault("MYGPR_LOG_DIR", str(log_dir))
    mpl_cache = log_dir.parent / "matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_cache))
    return env


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MyGPR with runtime-safe defaults.")
    parser.add_argument("--check", action="store_true", help="check environment and exit")
    parser.add_argument("--no-env-check", action="store_true", help="skip strict pre-launch environment check")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run app_qt.py")
    args = parser.parse_args(argv)

    root = project_root()
    report = build_report(root)
    if args.check:
        print_human(report)
        return 0 if report.ok else 1
    if not args.no_env_check and not report.ok:
        print_human(report)
        return 1

    app = root / "app_qt.py"
    if not app.exists():
        print(f"ERROR: app_qt.py not found: {app}")
        return 1
    env = prepare_runtime_environment(root)
    cmd = [args.python, "-X", "faulthandler", str(app)]
    return subprocess.call(cmd, cwd=str(root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
