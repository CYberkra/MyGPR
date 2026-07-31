#!/usr/bin/env python3
"""Ratchet high-complexity production functions without blocking normal helpers."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable, TypedDict

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "complexity_baseline.json"
_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
class FunctionMetric(TypedDict):
    path: str
    name: str
    line: int
    decision_points: int


_DECISION_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Match,
    ast.BoolOp,
    ast.IfExp,
    ast.comprehension,
)


def decision_points(node: ast.AST) -> int:
    """Return a stable structural decision score for one function."""
    return sum(isinstance(child, _DECISION_NODES) for child in ast.walk(node))


def iter_python_files(root: Path, roots: Iterable[str]) -> Iterable[Path]:
    for relative_root in roots:
        source_root = root / relative_root
        if not source_root.is_dir():
            continue
        yield from sorted(path for path in source_root.rglob("*.py") if "__pycache__" not in path.parts)


def collect_complexity(root: Path, roots: Iterable[str]) -> tuple[dict[str, int], list[FunctionMetric]]:
    functions: list[FunctionMetric] = []
    for path in iter_python_files(root, roots):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, _FUNCTION_NODES):
                continue
            functions.append({
                "path": path.relative_to(root).as_posix(),
                "name": node.name,
                "line": node.lineno,
                "decision_points": decision_points(node),
            })
    scores = [item["decision_points"] for item in functions]
    metrics = {
        "function_count": len(functions),
        "max_decision_points": max(scores, default=0),
        "functions_over_30": sum(score > 30 for score in scores),
        "functions_over_40": sum(score > 40 for score in scores),
        "functions_over_60": sum(score > 60 for score in scores),
    }
    functions.sort(key=lambda item: (item["decision_points"], item["path"], item["name"]), reverse=True)
    return metrics, functions


def validate_metrics(payload: dict, current: dict[str, int]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != "mygpr.complexity_baseline.v1":
        errors.append("invalid complexity baseline schema")
    baseline = dict(payload.get("metrics") or {})
    for key in ("max_decision_points", "functions_over_30", "functions_over_40", "functions_over_60"):
        allowed = int(baseline.get(key, -1))
        actual = int(current.get(key, 0))
        if allowed < 0:
            errors.append(f"missing complexity metric: {key}")
        elif actual > allowed:
            errors.append(f"complexity regressed: {key}: {actual} > {allowed}")
    return errors


def main() -> int:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    roots = tuple(str(item) for item in payload.get("source_roots") or ("core", "ui", "mygpr", "PythonModule"))
    try:
        metrics, functions = collect_complexity(ROOT, roots)
    except (OSError, SyntaxError, UnicodeError) as exc:
        print(f"complexity budget check failed: {exc}")
        return 1
    errors = validate_metrics(payload, metrics)
    print(json.dumps({"metrics": metrics, "top_functions": functions[:10]}, ensure_ascii=False, indent=2))
    if errors:
        print("\n".join(errors))
        return 1
    print("complexity budget: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
