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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import RatchetGate, add_write_baseline_flag, write_baseline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config/mypy_baseline.json"
TOLERANCE = 0.05
SCHEMA = "mygpr.mypy_baseline.v1"


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
    add_write_baseline_flag(parser)
    args = parser.parse_args()

    current = count_mypy_errors()
    if args.write_baseline or not BASELINE.exists():
        write_baseline(BASELINE, {
            "schema": SCHEMA,
            "policy": "ratchet; error count may decrease but may not increase",
            "tolerance": TOLERANCE,
            "baseline_errors": current,
        })
        print(f"mypy baseline written: {current} errors")
        return 0

    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        print(f"mypy ratchet FAILED: unexpected baseline schema {payload.get('schema')!r}")
        return 1
    gate = RatchetGate(name="mypy errors", current=current,
                       baseline=int(payload["baseline_errors"]),
                       tolerance=float(payload.get("tolerance", TOLERANCE)))
    print(gate.report())
    violation = gate.violation()
    if violation:
        print(f"mypy ratchet FAILED: {violation}. "
              "Fix the new type errors instead of raising the baseline.")
        return 1
    print("mypy ratchet: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
