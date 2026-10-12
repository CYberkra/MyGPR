#!/usr/bin/env python3
"""Backend quality gate — the single checklist shared by CI and `make gate`.

This module is the **only** place the backend checklist is written down. The
`backend` job in `.github/workflows/backend-ci.yml` invokes this script as one
step, so the workflow and `make gate` cannot drift apart — the previous setup
listed the same 14 commands in both places and silently lost
`check_backend_api_contract.py` along the way.

`--require-diff` is off by default here because a local run usually has no
upstream ref to diff against; CI passes it explicitly.
"""
from __future__ import annotations
import argparse
import subprocess
import sys

COMMANDS = [
    [sys.executable, "scripts/check_python_compile.py"],
    [sys.executable, "scripts/check_architecture.py"],
    [sys.executable, "scripts/check_schema_catalog.py"],
    [sys.executable, "scripts/check_backend_api_contract.py"],
    [sys.executable, "-m", "ruff", "check", "."],
    [sys.executable, "scripts/check_mypy_budget.py"],
    [sys.executable, "scripts/check_debt_budget.py"],
    [sys.executable, "scripts/check_complexity_budget.py"],
    [sys.executable, "scripts/check_release_hygiene.py"],
    [sys.executable, "scripts/check_project_format_compatibility.py"],
    [sys.executable, "backend_smoke.py"],
    [sys.executable, "backend_project_smoke.py"],
    [sys.executable, "-m", "pytest", "-q",
     "--cov=mygpr", "--cov=core", "--cov-report=json:coverage.json"],
    [sys.executable, "scripts/check_coverage_policy.py", "coverage.json"],
]


def _diff_coverage_command(require_diff: bool) -> list[str]:
    command = [sys.executable, "scripts/check_diff_coverage.py", "coverage.json"]
    if require_diff:
        command.append("--require-diff")
    return command


def _redundancy_command() -> list[str]:
    return [sys.executable, "scripts/audit_test_redundancy.py"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-diff", action="store_true",
                        help="make incremental coverage mandatory (CI passes this)")
    parser.add_argument("--skip-redundancy", action="store_true",
                        help="skip the (advisory) test redundancy audit")
    args = parser.parse_args(argv)

    commands = [*COMMANDS, _diff_coverage_command(args.require_diff)]
    if not args.skip_redundancy:
        commands.append(_redundancy_command())

    for command in commands:
        print("+", " ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
