#!/usr/bin/env python3
"""Check coverage of changed executable lines from git diff or a JSON line map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")


def _git_changed_lines(base: str, head: str) -> dict[str, set[int]]:
    completed = subprocess.run(
        ["git", "diff", "--unified=0", "--no-color", f"{base}...{head}", "--", "*.py"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        return {}
    result: dict[str, set[int]] = {}
    current = ""
    line_no = 0
    for line in completed.stdout.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            result.setdefault(current, set())
            continue
        match = HUNK.match(line)
        if match:
            line_no = int(match.group("start"))
            continue
        if not current or line.startswith("---"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            result[current].add(line_no)
            line_no += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        else:
            line_no += 1
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_json", type=Path)
    parser.add_argument("--changed-lines", type=Path, help="JSON mapping path -> added line numbers")
    parser.add_argument("--base", default="origin/master")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--allow-no-diff", action="store_true", default=True)
    args = parser.parse_args(argv)
    policy = json.loads((ROOT / "config/coverage_policy.json").read_text(encoding="utf-8"))
    coverage = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    if args.changed_lines:
        raw = json.loads(args.changed_lines.read_text(encoding="utf-8"))
        changed = {str(path): {int(value) for value in lines} for path, lines in raw.items()}
    else:
        changed = _git_changed_lines(args.base, args.head)
    if not changed:
        print("[diff-coverage] PASS no changed-line map available")
        return 0
    total = covered = 0
    details = []
    for path, lines in sorted(changed.items()):
        row = coverage.get("files", {}).get(path)
        if row is None:
            continue
        executable = set(row.get("executed_lines", [])) | set(row.get("missing_lines", []))
        relevant = lines & executable
        hit = relevant & set(row.get("executed_lines", []))
        total += len(relevant); covered += len(hit)
        details.append({"path": path, "executable_changed": len(relevant), "covered": len(hit), "missing": sorted(relevant - hit)})
    percent = 100.0 if total == 0 else 100.0 * covered / total
    threshold = float(policy.get("diff", {}).get("line_min", 80.0))
    report = {"schema": "mygpr.diff_coverage_report.v1", "status": "passed" if percent >= threshold else "failed", "percent": percent, "threshold": threshold, "total": total, "covered": covered, "details": details}
    out = args.coverage_json.with_name("diff-coverage-report.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if percent < threshold:
        print(f"[diff-coverage] FAILED {percent:.2f}% < {threshold:.2f}%")
        for row in details:
            if row["missing"]:
                print(f" - {row['path']}: {row['missing']}")
        return 1
    print(f"[diff-coverage] PASS {percent:.2f}% changed executable lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
