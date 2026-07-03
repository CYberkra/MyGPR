#!/usr/bin/env python3
"""Generate session context for MyGPR development.

Called by Claude Code SessionStart hook.
Reads project state and outputs JSON with additionalContext for the session.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_environment_info() -> dict[str, str]:
    """Detect Python package versions."""
    versions = {}
    for pkg in ("numpy", "scipy", "pandas", "matplotlib", "h5py", "PyWavelets"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg] = "not installed"
    try:
        from PyQt6.QtCore import PYQT_VERSION_STR
        versions["PyQt6"] = PYQT_VERSION_STR
    except ImportError:
        versions["PyQt6"] = "not installed"
    return versions


def find_recent_files(project_dir: Path, count: int = 10) -> list[Path]:
    """Find recently modified Python files in the project."""
    py_files = list(project_dir.rglob("*.py"))
    # Exclude __pycache__, .claude, tests/fixtures
    py_files = [
        p for p in py_files
        if "__pycache__" not in str(p)
        and ".claude" not in str(p)
    ]
    py_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return py_files[:count]


def get_module_summary(project_dir: Path) -> str:
    """Generate a summary of module file counts."""
    modules = {
        "core": "核心算法与业务逻辑",
        "ui": "PyQt6 GUI 界面",
        "tests": "测试套件",
        "PythonModule": "Kirchhoff 偏移模块",
        "scripts": "工具脚本",
        "docs": "文档",
    }
    lines = []
    for mod_dir, desc in modules.items():
        mod_path = project_dir / mod_dir
        if mod_path.is_dir():
            py_count = len(list(mod_path.rglob("*.py")))
            lines.append(f"  {mod_dir}/ ({py_count} .py files) — {desc}")
    return "\n".join(lines)


def main() -> None:
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
    env_info = get_environment_info()
    recent = find_recent_files(project_dir)
    module_summary = get_module_summary(project_dir)

    # Read VERSION
    version = "unknown"
    version_file = project_dir / "VERSION"
    if version_file.exists():
        version = version_file.read_text().strip()

    # Build recent files list
    recent_lines = []
    for p in recent:
        try:
            rel = p.relative_to(project_dir)
        except ValueError:
            rel = p
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        recent_lines.append(f"  - {rel} ({mtime.strftime('%Y-%m-%d %H:%M')})")

    context = f"""MyGPR v{version} beta — GPR data processing workstation.
Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Environment
- numpy {env_info.get('numpy', '?')}, scipy {env_info.get('scipy', '?')}
- matplotlib {env_info.get('matplotlib', '?')}, h5py {env_info.get('h5py', '?')}
- PyQt6 {env_info.get('PyQt6', '?')}, PyWavelets {env_info.get('PyWavelets', '?')}

## Module Structure
{module_summary}

## Recently Modified Files
{chr(10).join(recent_lines)}

## Key Reference
- Entry point: app_qt.py (GUI), cli_batch.py (CLI)
- Test markers: unit (fast core), gui (PyQt6 UI), integration (multi-module), slow, wavelet
- Processing pipeline: dewow → gain → filter → migration → time-depth conversion
- Numerical precision: float64 throughout signal processing
"""

    output = {"additionalContext": context}
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
