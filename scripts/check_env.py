#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MyGPR runtime environment checker.

The checker is cross-platform and intentionally read-only: it never installs
packages and never mutates the user's Python environment.  It is used by the
Windows launcher, by release tests, and by users who need a precise dependency
report before starting MyGPR.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

MIN_PYTHON = (3, 10)
REQUIRED_MODULES: tuple[tuple[str, str, str], ...] = (
    ("PyQt6", "PyQt6", "GUI framework"),
    ("qfluentwidgets", "PyQt6-Fluent-Widgets[full]", "Fluent UI widgets"),
    ("numpy", "numpy", "array processing"),
    ("pandas", "pandas", "table processing"),
    ("scipy", "scipy", "scientific algorithms"),
    ("matplotlib", "matplotlib", "plotting"),
    ("h5py", "h5py", "HDF5 import"),
    ("yaml", "PyYAML", "YAML configuration"),
    ("pywt", "PyWavelets", "wavelet algorithms"),
    ("pyproj", "pyproj", "coordinate projection"),
)
OPTIONAL_MODULES: tuple[tuple[str, str, str], ...] = (
    ("pytest", "pytest", "developer tests"),
)
REQUIRED_PATHS: tuple[str, ...] = (
    "app_qt.py",
    "VERSION",
    "requirements.txt",
    "PythonModule",
    "core",
    "ui",
)


def default_log_dir() -> Path:
    """Return the default writable log directory without touching the release tree."""

    override = os.environ.get("MYGPR_LOG_DIR")
    if override:
        return Path(override)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "MyGPR" / "logs"
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home) / "mygpr" / "logs"
    return Path.home() / ".local" / "state" / "mygpr" / "logs"


@dataclass(frozen=True)
class ModuleCheck:
    module: str
    package: str
    purpose: str
    ok: bool


@dataclass(frozen=True)
class PathCheck:
    path: str
    ok: bool
    kind: str


@dataclass(frozen=True)
class EnvReport:
    schema: str
    project_root: str
    version: str
    executable: str
    python_version: str
    platform: str
    ok: bool
    python_ok: bool
    writable_log_dir: bool
    required_modules: list[ModuleCheck] = field(default_factory=list)
    optional_modules: list[ModuleCheck] = field(default_factory=list)
    required_paths: list[PathCheck] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_version(root: Path) -> str:
    try:
        return (root / "VERSION").read_text(encoding="utf-8-sig").strip() or "unknown"
    except OSError:
        return "unknown"


def _module_checks(rows: Iterable[tuple[str, str, str]]) -> list[ModuleCheck]:
    out: list[ModuleCheck] = []
    for module, package, purpose in rows:
        out.append(ModuleCheck(module=module, package=package, purpose=purpose, ok=importlib.util.find_spec(module) is not None))
    return out


def _path_checks(root: Path) -> list[PathCheck]:
    checks: list[PathCheck] = []
    for rel in REQUIRED_PATHS:
        path = root / rel
        if path.is_dir():
            kind = "dir"
        elif path.is_file():
            kind = "file"
        else:
            kind = "missing"
        checks.append(PathCheck(path=rel, ok=path.exists(), kind=kind))
    return checks


def _writable_log_dir(root: Path) -> bool:
    del root  # The release tree must remain read-only during environment checks.
    log_dir = default_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        probe = log_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def build_report(root: Path | None = None) -> EnvReport:
    root = (root or project_root()).resolve()
    version = read_version(root)
    python_ok = sys.version_info >= MIN_PYTHON
    required_modules = _module_checks(REQUIRED_MODULES)
    optional_modules = _module_checks(OPTIONAL_MODULES)
    required_paths = _path_checks(root)
    writable = _writable_log_dir(root)
    problems: list[str] = []
    if not python_ok:
        problems.append(f"Python 版本过低：需要 {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+，当前 {platform.python_version()}")
    for item in required_modules:
        if not item.ok:
            problems.append(f"缺少模块 {item.module}，请安装包：{item.package}")
    for item in required_paths:
        if not item.ok:
            problems.append(f"缺少项目路径：{item.path}")
    if not writable:
        problems.append("日志目录不可写：请检查项目目录或 LOCALAPPDATA 权限")
    ok = python_ok and writable and all(m.ok for m in required_modules) and all(p.ok for p in required_paths)
    return EnvReport(
        schema="mygpr.environment_check.v1",
        project_root=str(root),
        version=version,
        executable=sys.executable,
        python_version=platform.python_version(),
        platform=platform.platform(),
        ok=ok,
        python_ok=python_ok,
        writable_log_dir=writable,
        required_modules=required_modules,
        optional_modules=optional_modules,
        required_paths=required_paths,
        problems=problems,
    )


def print_human(report: EnvReport) -> None:
    print("==========================================")
    print(f"MyGPR v{report.version} environment check")
    print("==========================================")
    print(f"Project root: {report.project_root}")
    print(f"Python:       {report.executable}")
    print(f"Version:      {report.python_version}")
    print(f"Platform:     {report.platform}")
    print()
    print("Required modules:")
    for item in report.required_modules:
        status = "OK" if item.ok else "MISSING"
        print(f"  {status:7} {item.module:<18} package={item.package}")
    print()
    print("Required project paths:")
    for item in report.required_paths:
        status = "OK" if item.ok else "MISSING"
        print(f"  {status:7} {item.path:<28} {item.kind}")
    print()
    print(f"Writable log directory: {'OK' if report.writable_log_dir else 'FAILED'}")
    if report.problems:
        print()
        print("Problems:")
        for problem in report.problems:
            print(f"  - {problem}")
    print()
    print("Environment status:", "OK" if report.ok else "FAILED")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check MyGPR runtime environment without installing packages.")
    parser.add_argument("--json", action="store_true", help="print JSON report")
    parser.add_argument("--strict", action="store_true", help="return non-zero when required checks fail")
    args = parser.parse_args(argv)
    report = build_report()
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print_human(report)
    return 0 if report.ok or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
