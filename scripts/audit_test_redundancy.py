#!/usr/bin/env python3
"""Audit test-suite redundancy and report static source-contract concentration.

Three dimensions, cheapest-and-hardest first:

1. **exact duplicate bodies** — two test functions with identical AST bodies
   (decorators/name stripped). This is the only dimension that can *fail* the
   gate: it is unambiguous, so a hit is always real duplication.
2. **same test name across files** — usually a naming collision rather than
   duplicated logic, but it is the cheapest signal that two files cover the
   same behaviour and should probably be merged or parameterized.
3. **same fixture name across files** — a `view`/`page` fixture redefined in
   five files is five subtly different objects behind one name, which is a
   readability trap even when the bodies differ.

Dimensions 2 and 3 are advisory: they need human judgement to tell real
overlap from legitimate per-module setup.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _normalized(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    clone = ast.FunctionDef(
        name="test",
        args=node.args,
        body=node.body,
        decorator_list=[],
        returns=node.returns,
        type_comment=node.type_comment,
        type_params=getattr(node, "type_params", []),
    )
    return ast.dump(clone, include_attributes=False)


def _reads_source(tree: ast.AST) -> bool:
    text = ast.dump(tree, include_attributes=False)
    return "read_text" in text and any(token in text for token in (".py", "source", "Path"))


def _collect(tree: ast.AST, rel: str) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Return (test functions, fixtures) as (name, lineno) pairs."""
    tests: list[tuple[str, int]] = []
    fixtures: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("test_"):
            tests.append((node.name, node.lineno))
        is_fixture = any(
            (isinstance(dec, ast.Attribute) and dec.attr == "fixture")
            or (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                and dec.func.attr == "fixture")
            for dec in node.decorator_list
        )
        if is_fixture:
            fixtures.append((node.name, node.lineno))
    return tests, fixtures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/test-results/redundancy.json")
    args = parser.parse_args(argv)

    body_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    name_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    fixture_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    static_modules: list[str] = []
    test_count = 0

    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        rel = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        if _reads_source(tree):
            static_modules.append(rel)
        tests, fixtures = _collect(tree, rel)
        for name, line in tests:
            test_count += 1
            name_groups[name].append({"path": rel, "line": line})
        for name, line in fixtures:
            fixture_groups[name].append({"path": rel, "line": line})
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                key = hashlib.sha256(_normalized(node).encode("utf-8")).hexdigest()
                body_groups[key].append({"path": rel, "name": node.name, "line": node.lineno})

    exact_duplicates = [rows for rows in body_groups.values() if len(rows) > 1]
    # Same name in the *same* file is fine (overloads/parametrize at worst).
    name_collisions = [
        {"name": name, "occurrences": rows}
        for name, rows in sorted(name_groups.items())
        if len({row["path"] for row in rows}) > 1
    ]
    fixture_duplication = [
        {"name": name, "occurrences": rows}
        for name, rows in sorted(fixture_groups.items())
        if len({row["path"] for row in rows}) > 1
    ]

    payload = {
        # v1 keeps the registered contract (config/schema_catalog.json marks it
        # immutable); the added advisory sections ride along as optional keys so
        # existing consumers keep working.
        "schema": "mygpr.test_redundancy_report.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "test_function_count": test_count,
        "exact_duplicate_groups": exact_duplicates,
        "cross_file_name_collisions": name_collisions,
        "cross_file_fixture_duplication": fixture_duplication,
        "static_contract_modules": static_modules,
        "status": "failed" if exact_duplicates else "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if fixture_duplication:
        print(f"[redundancy] NOTE fixture names redefined across files ({len(fixture_duplication)}):")
        for entry in fixture_duplication:
            files = ", ".join(sorted({row["path"] for row in entry["occurrences"]}))
            print(f" - {entry['name']}: {len(entry['occurrences'])}x  ({files})")
    if name_collisions:
        print(f"[redundancy] NOTE test names reused across files ({len(name_collisions)}):")
        for entry in name_collisions:
            files = ", ".join(sorted({row["path"] for row in entry["occurrences"]}))
            print(f" - {entry['name']}: {files}")
    if exact_duplicates:
        print(f"[redundancy] FAILED exact duplicate groups={len(exact_duplicates)}")
        for rows in exact_duplicates:
            print(" - " + " | ".join(f"{row['path']}::{row['name']}" for row in rows))
        return 1
    print(f"[redundancy] PASS tests={test_count} static_contract_modules={len(static_modules)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
