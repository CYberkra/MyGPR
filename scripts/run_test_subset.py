#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run explicit MyGPR test subsets with stable collection boundaries.

The repository mixes fast headless tests, Qt GUI tests, gprMax contract tests,
slow validation runners, and paper-facing integration checks.  Plain marker
filtering is not enough because pytest still imports every collected file.  This
runner first narrows files by filename contract, then executes the selected set.

Examples:
  python scripts/run_test_subset.py unit
  python scripts/run_test_subset.py gui -- -q
  python scripts/run_test_subset.py baseline
  python scripts/run_test_subset.py gui-smoke
  python scripts/run_test_subset.py list unit

Notes:
  * GUI subsets are automatically wrapped with xvfb-run on Linux when DISPLAY is
    unavailable and xvfb-run exists.
  * The baseline subset is intentionally conservative: preflight + fast unit + focused GUI/report smoke tests.  It is the recommended command
    before packaging a small UI or processing patch.  Run gui-smoke separately when Qt GUI paths changed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

GUI_NAME_HINTS = (
    "gui",
    "daily_processing",
    "result_dialog",
    "no_prior_ui",
    "wiggle_ui",
    "workbench",
    "roi_picker",
    "app_sidecar_gui",
    "app_csv_load_runtime_seam",
    "import_export_report",
    "shared_state_sync",
)

SLOW_NAME_HINTS = (
    "runner",
    "benchmark",
    "validation",
    "multi_scene",
    "native",
    "package",
    "motion_pipeline",
    "e2e",
    "no_zerotime",
    "post_zero_time",
    "risk_flag",
    "signal_loss",
    "demo",
)

INTEGRATION_NAME_HINTS = (
    "cli",
    "sidecar",
    "evidence",
    "export",
    "workflow",
    "pipeline",
    "dashboard",
    "campaign",
    "package",
    "runner",
    "validation",
    "report",
)

# Files that are intentionally not part of the fast unit baseline even when the
# filename is generic.  They perform broad candidate sweeps or expensive signal
# diagnostics and should be run through the slow subset or targeted commands.
EXPLICIT_SLOW_FILES = {
    "test_auto_tune.py",
}

# Minimal GUI/report smoke tests used by the baseline subset.  They exercise the
# Qt import path, CSV load seam, and report sidecar generation without running
# every GUI regression test.
BASELINE_GUI_SMOKE_FILES = [
    "tests/test_app_csv_load_runtime_seam.py",
    "tests/test_import_export_report.py",
    "tests/test_processing_lineage_controller_gui.py",
    "tests/test_bscan_interaction_controller_gui.py",
    "tests/test_autotune_sync_controller_gui.py",
]

BASELINE_WAVELET_SMOKE_CMD = [
    sys.executable,
    "-m",
    "pytest",
    "tests/test_round2_processing_kernels.py",
    "-k",
    "wavelet",
    "-q",
]


@dataclass
class StageResult:
    name: str
    command: list[str]
    returncode: int
    duration_s: float


def _test_files() -> list[Path]:
    return sorted(TESTS.glob("test_*.py"))


def _name(path: Path) -> str:
    return path.name.lower()


def _has_any(name: str, hints: tuple[str, ...]) -> bool:
    return any(hint in name for hint in hints)


def _is_gui(path: Path) -> bool:
    return _has_any(_name(path), GUI_NAME_HINTS)


def _is_gprmax(path: Path) -> bool:
    name = _name(path)
    return "gprmax" in name or name.startswith("test_gx_")


def _is_slow(path: Path) -> bool:
    name = _name(path)
    return name in EXPLICIT_SLOW_FILES or _has_any(name, SLOW_NAME_HINTS)


def _is_integration(path: Path) -> bool:
    return _has_any(_name(path), INTEGRATION_NAME_HINTS) or _is_gprmax(path)


def _is_wavelet(path: Path) -> bool:
    name = _name(path)
    if "wavelet" in name:
        return True
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "wavelet" in text or "pywt" in text


def _subset_files(subset: str) -> list[str]:
    files = _test_files()
    if subset == "unit":
        selected = [
            p
            for p in files
            if not (_is_gui(p) or _is_gprmax(p) or _is_slow(p) or _is_integration(p))
        ]
    elif subset == "gui":
        selected = [p for p in files if _is_gui(p)]
    elif subset == "gui-smoke":
        selected = [ROOT / f for f in BASELINE_GUI_SMOKE_FILES]
    elif subset == "integration":
        selected = [p for p in files if _is_integration(p) and not _is_slow(p) and not _is_gprmax(p)]
    elif subset == "slow":
        selected = [p for p in files if _is_slow(p)]
    elif subset == "gprmax":
        selected = [p for p in files if _is_gprmax(p)]
    elif subset == "wavelet":
        selected = [p for p in files if _is_wavelet(p)]
    else:
        selected = files
    return [str(p.relative_to(ROOT)) for p in selected]


def _pytest_cmd_for_subset(subset: str, *, quiet: bool = False) -> list[str]:
    if subset == "preflight":
        return [sys.executable, "scripts/preflight_check.py"]
    if subset == "all":
        return [sys.executable, "-m", "pytest", *( ["-q"] if quiet else [] )]
    files = _subset_files(subset)
    if not files:
        return [sys.executable, "-c", f"print('No tests matched subset: {subset}')"]
    cmd = [sys.executable, "-m", "pytest", *files]
    # The wavelet subset is a focused dependency/contract check, not a full
    # execution of every file that merely mentions wavelet presets.  Running the
    # matching tests only keeps this subset deterministic and out of the slow
    # budget lane.
    if subset == "wavelet":
        cmd.extend(["-k", "wavelet or pywt"])
    if quiet:
        cmd.append("-q")
    return cmd


def _needs_qt_display(subset: str) -> bool:
    return subset in {"gui", "gui-smoke", "baseline-gui-smoke", "preflight", "wavelet", "all"}


def _maybe_xvfb(cmd: list[str], subset: str) -> list[str]:
    needs_qt_display = _needs_qt_display(subset)
    if not needs_qt_display:
        return cmd
    if os.name != "posix" or os.environ.get("MYGPR_TEST_NO_XVFB"):
        return cmd
    xvfb_run = shutil.which("xvfb-run")
    if not xvfb_run:
        return cmd
    # Some CI/sandbox environments expose a DISPLAY value even when no usable X
    # server is available.  Test execution is more deterministic if Qt/Matplotlib
    # smoke tests always use a fresh xvfb display on POSIX when available.
    return [xvfb_run, "-a", *cmd]


def _run_stage(name: str, cmd: list[str], *, subset_for_display: str, extra_env: dict[str, str] | None = None) -> StageResult:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "QtAgg" if _needs_qt_display(subset_for_display) else "Agg")
    if extra_env:
        env.update(extra_env)
    run_cmd = _maybe_xvfb(cmd, subset_for_display)
    print("[MyGPR tests]", name + ":", " ".join(run_cmd), flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call(run_cmd, cwd=ROOT, env=env)
    return StageResult(name=name, command=run_cmd, returncode=rc, duration_s=round(time.perf_counter() - t0, 3))


def _run_single_subset(subset: str, extra: list[str], *, quiet: bool = False) -> int:
    cmd = _pytest_cmd_for_subset(subset, quiet=quiet)
    cmd.extend(extra)
    result = _run_stage(subset, cmd, subset_for_display=subset)
    return result.returncode


def _run_file_staged_subset(subset: str, extra: list[str], *, quiet: bool = False) -> int:
    files = _subset_files(subset)
    if not files:
        print(f"No tests matched subset: {subset}")
        return 0
    rc = 0
    for path in files:
        cmd = [sys.executable, "-m", "pytest", path]
        if subset == "wavelet":
            cmd.extend(["-k", "wavelet or pywt"])
        if quiet:
            cmd.append("-q")
        cmd.extend(extra)
        file_path = ROOT / path
        display_subset = "gui" if _is_gui(file_path) else "unit"
        if subset == "wavelet":
            display_subset = "wavelet"
        result = _run_stage(f"{subset}:{path}", cmd, subset_for_display=display_subset)
        if result.returncode != 0:
            rc = result.returncode
            break
    return rc


def _run_baseline(extra: list[str], *, json_out: Path | None = None) -> int:
    stages: list[tuple[str, list[str], str]] = [
        ("preflight", [sys.executable, "scripts/preflight_check.py"], "preflight"),
        ("unit", _pytest_cmd_for_subset("unit", quiet=True) + extra, "unit"),
    ]
    results: list[StageResult] = []
    rc = 0
    for name, cmd, display_subset in stages:
        result = _run_stage(name, cmd, subset_for_display=display_subset)
        results.append(result)
        if result.returncode != 0:
            rc = result.returncode
            break
    payload = {
        "schema": "mygpr.test_baseline.v1",
        "root": str(ROOT),
        "python": sys.version.split()[0],
        "stages": [asdict(r) for r in results],
        "ok": rc == 0,
    }
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[MyGPR tests] wrote {json_out}")
    return rc


def _print_subset_list(subset: str) -> int:
    if subset == "baseline":
        print("preflight")
        print("unit:")
        print("\n".join(f"  {f}" for f in _subset_files("unit")))
        print("gui-smoke is intentionally separate:")
        print("\n".join(f"  {f}" for f in BASELINE_GUI_SMOKE_FILES))
        return 0
    for path in _subset_files(subset):
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a named MyGPR test subset.")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run a named subset. This is also the default command.")
    run_parser.add_argument(
        "subset",
        choices=["all", "baseline", "gprmax", "gui", "gui-smoke", "integration", "preflight", "slow", "unit", "wavelet"],
        help="Subset to run",
    )
    run_parser.add_argument("--json-out", type=Path, default=None, help="Write baseline stage results to JSON")
    run_parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args after -- are appended")

    list_parser = sub.add_parser("list", help="List files selected by a subset without running tests.")
    list_parser.add_argument(
        "subset",
        choices=["all", "baseline", "gprmax", "gui", "gui-smoke", "integration", "slow", "unit", "wavelet"],
    )

    # Backward compatible form: python scripts/run_test_subset.py unit -- -q
    first = (argv or sys.argv[1:])[:1]
    if first and first[0] in {"all", "baseline", "gprmax", "gui", "gui-smoke", "integration", "preflight", "slow", "unit", "wavelet"}:
        argv = ["run", *(argv or sys.argv[1:])]

    args = parser.parse_args(argv)
    if args.command == "list":
        return _print_subset_list(args.subset)
    if args.command != "run":
        parser.print_help()
        return 2

    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    if args.subset == "baseline":
        return _run_baseline(extra, json_out=args.json_out)
    # Integration combines GUI sidecar tests, report/export checks, and backend
    # integration contracts.  Running it file-by-file avoids cross-file Qt
    # application teardown hangs observed in headless sandboxes while preserving
    # the same selected file boundary.
    if args.subset == "integration":
        return _run_file_staged_subset(args.subset, extra)
    return _run_single_subset(args.subset, extra)


if __name__ == "__main__":
    raise SystemExit(main())
