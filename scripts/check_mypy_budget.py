#!/usr/bin/env python3
"""Enforce a mypy-findings ratchet: counts may decrease but may not increase.

The baseline in ``config/mypy_baseline.json`` records the number of mypy
errors present in the tree when the ratchet was introduced.  CI runs this
script so every PR that adds type errors fails; fixes that reduce the count
lower the baseline via ``--write-baseline``.

A 5% tolerance absorbs platform-level jitter (stub availability differs
slightly between Linux CI and Windows workstations).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config/mypy_baseline.json"
TOLERANCE = 0.05


def count_mypy_errors() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "mypy", "mygpr", "core", "--no-error-summary"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # --no-error-summary 仍然输出 "Found N errors ..." 汇总行；两种形态都兼容。
    stderr_tail = completed.stderr.strip().splitlines()
    for line in reversed(stderr_tail):
        if line.startswith("Found "):
            return int(line.split()[1])
    # 汇总行缺失时逐行统计（每行一个 "file:line: error"）。
    lines = [ln for ln in completed.stdout.splitlines() if ": error:" in ln]
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    current = count_mypy_errors()
    if args.write_baseline or not BASELINE.exists():
        payload = {
            "schema": "mygpr.mypy_baseline.v1",
            "policy": "ratchet; error count may decrease but may not increase",
            "tolerance": TOLERANCE,
            "baseline_errors": current,
        }
        BASELINE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"mypy baseline written: {current} errors")
        return 0

    baseline = int(json.loads(BASELINE.read_text(encoding="utf-8"))["baseline_errors"])
    limit = int(baseline * (1 + TOLERANCE))
    print(f"mypy errors: current={current} baseline={baseline} limit={limit}")
    if current > limit:
        print(
            f"mypy ratchet FAILED: {current} errors exceed baseline {baseline} "
            "(+5% jitter tolerance). Fix the new type errors instead of raising the baseline."
        )
        return 1
    print("mypy ratchet: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
