#!/usr/bin/env python3
from __future__ import annotations
import subprocess, sys

COMMANDS = [
    [sys.executable, "scripts/check_python_compile.py"],
    [sys.executable, "scripts/check_architecture.py"],
    [sys.executable, "scripts/check_schema_catalog.py"],
    [sys.executable, "backend_smoke.py"],
    [sys.executable, "backend_project_smoke.py"],
    [sys.executable, "-m", "pytest", "-q"],
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
