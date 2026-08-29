#!/usr/bin/env python3
"""Backend quality gate — the single checklist shared by CI and `make gate`.

Keep this list in sync with the `backend` job in
`.github/workflows/backend-ci.yml`; CI executes the same steps individually.
"""
from __future__ import annotations
import subprocess
import sys

COMMANDS = [
    [sys.executable, "scripts/check_python_compile.py"],
    [sys.executable, "scripts/check_architecture.py"],
    [sys.executable, "scripts/check_schema_catalog.py"],
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

def main() -> int:
    for command in COMMANDS:
        print("+", " ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
