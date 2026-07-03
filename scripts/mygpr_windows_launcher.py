# -*- coding: utf-8 -*-
"""Windows launcher for MyGPR.

This launcher intentionally does *not* install packages; in other words, it does not install packages automatically.  Its job is to find an
already usable Python environment, especially when a user has an existing MyGPR
Conda/venv environment but Windows' py launcher points to a clean system Python.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

REQUIRED_MODULES: Tuple[Tuple[str, str], ...] = (
    ("PyQt6", "PyQt6"),
    ("qfluentwidgets", "qfluentwidgets"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("h5py", "h5py"),
    ("yaml", "PyYAML"),
    ("pywt", "PyWavelets"),
    ("pyproj", "pyproj"),
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_version(root: Path) -> str:
    version_file = root / "VERSION"
    try:
        text = version_file.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return "unknown"
    return text or "unknown"


def log_path(root: Path) -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        log_dir = Path(base) / "MyGPR" / "logs" / "launcher"
    else:
        log_dir = root / "logs" / "launcher"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"start_mygpr_{stamp}.log"


def _append_unique(items: List[Path], item: Optional[os.PathLike[str] | str]) -> None:
    if not item:
        return
    text = os.fspath(item).strip().strip('"')
    if not text:
        return
    p = Path(text)
    # On Windows, Path.exists handles quoted drive paths. In WSL-side validation
    # those paths will not exist, so this check is intentionally only for runtime.
    try:
        if not p.exists() or p.name.lower() != "python.exe":
            return
    except OSError:
        return
    key = str(p.resolve()).lower()
    if key not in {str(x.resolve()).lower() for x in items}:
        items.append(p)


def _run_text(cmd: Sequence[str], timeout: float = 8.0) -> str:
    try:
        cp = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout,
            check=False,
        )
        return cp.stdout or ""
    except Exception:
        return ""


def _append_conda_envs(candidates: List[Path]) -> None:
    conda_json = _run_text(["conda", "env", "list", "--json"], timeout=10.0)
    if conda_json.strip().startswith("{"):
        try:
            data = json.loads(conda_json)
            for env_path in data.get("envs", []):
                _append_unique(candidates, Path(env_path) / "python.exe")
        except Exception:
            pass

    userprofile = os.environ.get("USERPROFILE")
    programdata = os.environ.get("ProgramData") or r"C:\ProgramData"
    common_roots: List[Path] = []
    if userprofile:
        up = Path(userprofile)
        common_roots.extend(
            [
                up / "miniconda3",
                up / "anaconda3",
                up / "mambaforge",
                up / "miniforge3",
                up / "micromamba",
            ]
        )
    common_roots.extend(
        [
            Path(programdata) / "miniconda3",
            Path(programdata) / "anaconda3",
            Path(programdata) / "mambaforge",
            Path(programdata) / "miniforge3",
        ]
    )
    for base in common_roots:
        if not base.exists():
            continue
        _append_unique(candidates, base / "python.exe")
        envs = base / "envs"
        if envs.exists():
            try:
                for child in envs.iterdir():
                    _append_unique(candidates, child / "python.exe")
            except OSError:
                pass


def _append_path_pythons(candidates: List[Path]) -> None:
    for line in _run_text(["where", "python"]).splitlines():
        _append_unique(candidates, line)


def _append_windows_launcher_pythons(candidates: List[Path]) -> None:
    py_list = _run_text(["py", "-0p"], timeout=5.0)
    for line in py_list.splitlines():
        m = re.search(r"([A-Za-z]:\\[^\r\n]*?python\.exe)", line)
        if m:
            _append_unique(candidates, m.group(1))


def _append_common_python_installs(candidates: List[Path]) -> None:
    localapp = os.environ.get("LOCALAPPDATA")
    if not localapp:
        return
    base = Path(localapp) / "Programs" / "Python"
    if not base.exists():
        return
    try:
        for child in base.iterdir():
            if child.is_dir() and child.name.lower().startswith("python"):
                _append_unique(candidates, child / "python.exe")
    except OSError:
        pass


def candidate_pythons(root: Path) -> List[Path]:
    candidates: List[Path] = []

    # 1) Explicit override. This is the safest way for a user to bind MyGPR to
    #    their own existing environment without changing system Python.
    _append_unique(candidates, os.environ.get("MYGPR_PYTHON"))

    # 2) Currently activated virtual environments. These must beat Windows' py
    #    launcher, otherwise a double-click can silently pick Python312.
    for prefix_name in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        prefix = os.environ.get(prefix_name)
        if prefix:
            _append_unique(candidates, Path(prefix) / "Scripts" / "python.exe")
            _append_unique(candidates, Path(prefix) / "python.exe")

    # 3) Local environments next to this package.
    for rel in (".venv", "venv", "env", ".env"):
        _append_unique(candidates, root / rel / "Scripts" / "python.exe")

    # 4) Conda/Mamba environments discovered from the current shell and common
    #    install roots. These should be considered before generic PATH/system
    #    Python entries because field users often keep MyGPR in Conda.
    _append_conda_envs(candidates)

    # 5) PATH python(s). In an Anaconda Prompt this still catches the active env;
    #    on double-click launches it may be a clean system interpreter.
    _append_path_pythons(candidates)

    # 6) Windows Python Launcher entries. Lower priority: these are often clean
    #    system interpreters without MyGPR packages installed.
    _append_windows_launcher_pythons(candidates)

    # 7) Common system Python installs. This is last and only useful for users
    #    who intentionally installed all requirements into system Python.
    _append_common_python_installs(candidates)

    return candidates


def check_modules(py_exe: Path) -> Tuple[bool, List[str], str]:
    mods = [m for m, _pkg in REQUIRED_MODULES]
    code = (
        "import importlib.util, json, sys; "
        f"mods={mods!r}; "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "print(json.dumps({'executable': sys.executable, 'version': sys.version.split()[0], 'missing': missing}, ensure_ascii=False)); "
        "raise SystemExit(0 if not missing else 2)"
    )
    try:
        cp = subprocess.run(
            [str(py_exe), "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return False, [f"无法执行: {exc}"], "unknown"

    version = "unknown"
    missing: List[str] = []
    try:
        data = json.loads((cp.stdout or "").strip().splitlines()[-1])
        version = str(data.get("version") or "unknown")
        missing = [str(x) for x in data.get("missing") or []]
    except Exception:
        missing = ["环境检查失败"]
    return cp.returncode == 0 and not missing, missing, version


def select_python(root: Path, log: Path) -> Tuple[Optional[Path], List[Tuple[Path, bool, List[str], str]]]:
    checked: List[Tuple[Path, bool, List[str], str]] = []
    with log.open("a", encoding="utf-8", errors="ignore") as f:
        f.write("Python candidate scan:\n")
        for py in candidate_pythons(root):
            ok, missing, version = check_modules(py)
            checked.append((py, ok, missing, version))
            if ok:
                f.write(f"  OK      {py}  Python {version}\n")
                return py, checked
            f.write(f"  MISSING {py}  Python {version}  missing={','.join(missing)}\n")
    return None, checked


def module_help_text(missing_modules: Iterable[str]) -> str:
    missing = list(missing_modules)
    pkg_names = []
    for mod in missing:
        for module_name, package_name in REQUIRED_MODULES:
            if mod == module_name:
                pkg_names.append(package_name)
                break
    return ", ".join(pkg_names or missing)


def print_checked_summary(checked: Sequence[Tuple[Path, bool, List[str], str]]) -> None:
    if not checked:
        print("No Python candidates were found.")
        return
    print("Checked Python environments:")
    for py, ok, missing, version in checked:
        status = "OK" if ok else f"missing {module_help_text(missing)}"
        print(f"  - {py}  [Python {version}]  {status}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Launch MyGPR with an existing Python environment.")
    parser.add_argument("--check", action="store_true", help="only check environment and exit")
    parser.add_argument("--no-pause", action="store_true", help="do not wait for Enter on error")
    args = parser.parse_args(argv)

    root = project_root()
    app = root / "app_qt.py"
    log = log_path(root)
    version = read_version(root)

    print("==========================================")
    print(f"MyGPR v{version} one-click launcher")
    print("==========================================")
    print(f"Project root: {root}")
    print(f"Version: {version}")
    print(f"Log file: {log}")
    print()

    with log.open("w", encoding="utf-8", errors="ignore") as f:
        f.write(f"Project root: {root}\n")
        f.write(f"Version: {version}\n")
        f.write(f"Launcher bootstrap Python: {sys.executable}\n")
        f.write(f"Command line: {' '.join(sys.argv)}\n")
        f.write(f"Time: {_dt.datetime.now().isoformat(timespec='seconds')}\n\n")

    if not app.exists():
        print("ERROR: app_qt.py not found in project root.")
        return 1
    if not (root / "PythonModule").exists():
        print("ERROR: PythonModule folder not found in project root.")
        return 1

    print("Searching for an existing MyGPR Python environment...")
    selected, checked = select_python(root, log)
    print_checked_summary(checked)
    print()

    if not selected:
        print("ERROR: No checked Python environment contains all required MyGPR runtime modules.")
        print("This launcher did not install anything and did not change your environment.")
        print()
        print("Recommended fix if you already have a working environment:")
        print(r"  set MYGPR_PYTHON=C:\path\to\your\env\python.exe")
        print("  start_mygpr.bat")
        print()
        print("Or start this launcher from your activated Conda/venv prompt.")
        print()
        print("Manual dependency repair for the selected environment, if needed:")
        print("  python -m pip install -r requirements.txt")
        print(f"See log: {log}")
        if not args.no_pause and os.environ.get("MYGPR_NO_PAUSE") != "1":
            try:
                input("Press Enter to continue . . .")
            except EOFError:
                pass
        return 1

    print(f"Using Python: {selected}")
    if args.check:
        print("Environment check passed.")
        return 0

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) + ((os.pathsep + existing_pythonpath) if existing_pythonpath else "")
    env.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    env.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    env.setdefault("MPLBACKEND", "QtAgg")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("QT_OPENGL", "software")
    env.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false;qt.qpa.window=false")
    # Give Matplotlib a writable per-user cache and avoid repeated font-cache
    # rebuilds on Windows double-click startup.
    local_app_data = env.get("LOCALAPPDATA")
    if local_app_data:
        mpl_cache = Path(local_app_data) / "MyGPR" / "matplotlib_cache"
    else:
        mpl_cache = root / "logs" / "matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_cache))

    print("Starting MyGPR...")
    launch_cmd = [str(selected), "-X", "faulthandler", str(app)]
    with log.open("a", encoding="utf-8", errors="ignore") as f:
        f.write(f"\nSelected Python: {selected}\n")
        f.write(f"Runtime environment: MPLCONFIGDIR={env.get('MPLCONFIGDIR')}; QT_OPENGL={env.get('QT_OPENGL')}\n")
        f.write(f"Launch command: {' '.join(launch_cmd)}\n")
        f.write(f"Starting MyGPR at {_dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.flush()
        rc = subprocess.call(launch_cmd, cwd=str(root), env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\nMyGPR exit code: {rc}\n")

    print()
    if rc != 0:
        print(f"MyGPR exited with error code {rc}.")
        print(f"Log file: {log}")
        if not args.no_pause and os.environ.get("MYGPR_NO_PAUSE") != "1":
            try:
                input("Press Enter to continue . . .")
            except EOFError:
                pass
    else:
        print("MyGPR closed normally.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
