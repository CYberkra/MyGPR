#!/usr/bin/env python3
"""Check coverage.py JSON against MyGPR critical-module coverage policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pct(covered: int, total: int) -> float:
    return 100.0 if total <= 0 else 100.0 * covered / total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_json", type=Path)
    parser.add_argument("--critical-only", action="store_true")
    args = parser.parse_args(argv)
    policy = json.loads((ROOT / "config/coverage_policy.json").read_text(encoding="utf-8"))
    payload = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    files = payload.get("files", {})
    errors: list[str] = []
    rows: list[dict[str, object]] = []
    for rel, thresholds in policy.get("critical_modules", {}).items():
        record = files.get(rel) or files.get(str((ROOT / rel).resolve()))
        if record is None:
            errors.append(f"critical module absent from coverage data: {rel}")
            continue
        summary = record.get("summary", {})
        line = float(summary.get("percent_covered", 0.0))
        branch = _pct(int(summary.get("covered_branches", 0)), int(summary.get("num_branches", 0)))
        rows.append({"module": rel, "line": line, "branch": branch})
        if line + 1e-9 < float(thresholds["line_min"]):
            errors.append(f"{rel} line coverage {line:.2f}% < {thresholds['line_min']}%")
        if branch + 1e-9 < float(thresholds["branch_min"]):
            errors.append(f"{rel} branch coverage {branch:.2f}% < {thresholds['branch_min']}%")
    if not args.critical_only:
        total = payload.get("totals", {})
        line = float(total.get("percent_covered", 0.0))
        branch = _pct(int(total.get("covered_branches", 0)), int(total.get("num_branches", 0)))
        if line < float(policy["global"]["line_min"]):
            errors.append(f"global line coverage {line:.2f}% < {policy['global']['line_min']}%")
        if branch < float(policy["global"]["branch_min"]):
            errors.append(f"global branch coverage {branch:.2f}% < {policy['global']['branch_min']}%")
    report = args.coverage_json.with_name("coverage-policy-report.json")
    report.write_text(json.dumps({"status": "failed" if errors else "passed", "critical_modules": rows, "errors": errors}, indent=2) + "\n", encoding="utf-8")
    if errors:
        print("[coverage] FAILED")
        for error in errors:
            print(f" - {error}")
        return 1
    print("[coverage] PASS " + ", ".join(f"{row['module']}={row['line']:.1f}/{row['branch']:.1f}" for row in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
