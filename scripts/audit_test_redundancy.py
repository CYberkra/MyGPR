#!/usr/bin/env python3
"""Reject exact duplicate test bodies and report static source-contract concentration."""
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/test-results/redundancy.json")
    parser.add_argument("--fail-on-duplicates", action="store_true", default=True)
    args = parser.parse_args(argv)
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
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
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                test_count += 1
                key = hashlib.sha256(_normalized(node).encode("utf-8")).hexdigest()
                groups[key].append({"path": rel, "name": node.name, "line": node.lineno})
    duplicates = [rows for rows in groups.values() if len(rows) > 1]
    payload = {
        "schema": "mygpr.test_redundancy_report.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_function_count": test_count,
        "exact_duplicate_groups": duplicates,
        "static_contract_modules": static_modules,
        "status": "failed" if duplicates else "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if duplicates:
        print(f"[redundancy] FAILED exact duplicate groups={len(duplicates)}")
        for rows in duplicates:
            print(" - " + " | ".join(f"{row['path']}::{row['name']}" for row in rows))
        return 1
    print(f"[redundancy] PASS tests={test_count} static_contract_modules={len(static_modules)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
