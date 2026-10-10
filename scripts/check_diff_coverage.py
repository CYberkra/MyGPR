#!/usr/bin/env python3
"""Check coverage of changed executable lines from git diff or a JSON line map.

The default base is ``origin/main`` (this repository's trunk). Note that
``git diff A...B`` needs the base ref to be present locally, so CI checkouts
must use ``fetch-depth: 0``; otherwise the diff silently resolves to nothing and
this gate degrades into a no-op. Use ``--require-diff`` in pull-request CI to
turn that silent pass into a loud failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")
DEFAULT_BASE = "origin/main"


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


def _ref_exists(ref: str) -> bool:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return completed.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_json", type=Path)
    parser.add_argument("--changed-lines", type=Path, help="JSON mapping path -> added line numbers")
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help=f"diff base ref (default: {DEFAULT_BASE})")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--require-diff", action="store_true",
                        help="fail when no changed-line map can be derived "
                             "(use in PR CI so a shallow checkout cannot silently pass)")
    args = parser.parse_args(argv)
    policy = json.loads((ROOT / "config/coverage_policy.json").read_text(encoding="utf-8"))
    coverage = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    if args.changed_lines:
        raw = json.loads(args.changed_lines.read_text(encoding="utf-8"))
        changed = {str(path): {int(value) for value in lines} for path, lines in raw.items()}
        source = str(args.changed_lines)
    else:
        if not _ref_exists(args.base):
            message = (f"[diff-coverage] base ref {args.base!r} is not available locally "
                       "(shallow checkout? fetch-depth: 0)")
            if args.require_diff:
                print(f"{message} — FAILED because --require-diff was set")
                return 1
            print(f"{message} — skipping (diff coverage not enforced)")
            return 0
        changed = _git_changed_lines(args.base, args.head)
        source = f"git diff {args.base}...{args.head}"
    if not changed:
        if args.require_diff:
            print(f"[diff-coverage] FAILED no changed executable lines derived from {source}")
            return 1
        print(f"[diff-coverage] PASS no changed-line map available (source: {source})")
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
    if total == 0 and args.require_diff:
        print(f"[diff-coverage] FAILED changed files carried no executable lines "
              f"(source: {source}); cannot judge incremental coverage")
        return 1
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
