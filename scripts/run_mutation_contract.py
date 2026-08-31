#!/usr/bin/env python3
"""Small deterministic mutation gate for central parameter validation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MUTANTS = {
    "accept_unknown_parameters": (
        "if reject_unknown and schema:",
        "if False and reject_unknown and schema:",
    ),
    "disable_minimum_check": (
        'if "min" in spec and float(normalized) < float(spec["min"]):',
        'if False and "min" in spec and float(normalized) < float(spec["min"]):',
    ),
    "disable_cross_field_order": (
        "if low > high:",
        "if False and low > high:",
    ),
}


ORACLE = r"""
from mygpr.application.processing.validation import validate_parameters
from mygpr.domain.common.errors import ParameterValidationError
from mygpr.domain.processing.models import ProcessingMethodDescriptor

d = ProcessingMethodDescriptor(method_id='contract', name='contract', parameter_schema={
    'window': {'type': 'int', 'min': 1, 'max': 101},
    'gain_min': {'type': 'float', 'min': 0.0},
    'gain_max': {'type': 'float', 'min': 0.0},
})

def rejected(params):
    try:
        validate_parameters(d, params)
    except ParameterValidationError:
        return True
    return False

assert rejected({'windwo': 23}), 'unknown parameter survived'
assert rejected({'window': 0}), 'minimum-bound mutant survived'
assert rejected({'gain_min': 7.0, 'gain_max': 6.0}), 'cross-field mutant survived'
"""


def _run_oracle(import_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(import_root)
    return subprocess.run(
        [sys.executable, "-I", "-c", "import sys; sys.path.insert(0, " + repr(str(import_root)) + ");\n" + ORACLE],
        cwd=import_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/test-results/mutation-contract.json")
    args = parser.parse_args(argv)
    source_rel = Path("mygpr/application/processing/validation.py")
    source = (ROOT / source_rel).read_text(encoding="utf-8")
    baseline = _run_oracle(ROOT)
    if baseline.returncode != 0:
        print("[mutation] baseline oracle failed")
        print((baseline.stdout + baseline.stderr)[-1600:])
        return 2
    results = []
    for name, (needle, replacement) in MUTANTS.items():
        if needle not in source:
            results.append({"mutant": name, "status": "invalid", "reason": "source token missing"})
            continue
        with tempfile.TemporaryDirectory(prefix="mygpr-mutant-") as temp:
            temp_root = Path(temp)
            shutil.copytree(ROOT / "mygpr", temp_root / "mygpr")
            mutant_path = temp_root / source_rel
            mutant_path.write_text(source.replace(needle, replacement, 1), encoding="utf-8")
            completed = _run_oracle(temp_root)
            killed = completed.returncode != 0
            results.append({"mutant": name, "status": "killed" if killed else "survived", "returncode": completed.returncode, "output_tail": (completed.stdout + completed.stderr)[-1200:]})
    survivors = [row for row in results if row["status"] != "killed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "mygpr.mutation_contract.v1", "status": "failed" if survivors else "passed", "results": results}, indent=2) + "\n", encoding="utf-8")
    if survivors:
        print("[mutation] FAILED survivors=" + ", ".join(str(row["mutant"]) for row in survivors)); return 1
    print(f"[mutation] PASS killed={len(results)}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
