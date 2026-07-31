#!/usr/bin/env python3
"""Requirement/risk traceability primitives for the industrial test system."""
from __future__ import annotations

from dataclasses import dataclass
import ast
import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class TraceLink:
    test: str
    requirements: tuple[str, ...]
    risks: tuple[str, ...]


def _strings(node: ast.AST) -> tuple[str, ...]:
    values: list[str] = []
    for arg in getattr(node, "args", ()):
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            values.append(arg.value)
    return tuple(values)


def inline_links(path: Path, root: Path = ROOT) -> TraceLink:
    """Read module/function pytest requirement and risk marker arguments."""
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    requirements: set[str] = set()
    risks: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return TraceLink(rel, (), ())
    for node in ast.walk(tree):
        decorators = getattr(node, "decorator_list", ())
        if isinstance(node, ast.Assign):
            values = [node.value]
        else:
            values = list(decorators)
        for value in values:
            candidates = value.elts if isinstance(value, (ast.List, ast.Tuple)) else (value,)
            for candidate in candidates:
                if not isinstance(candidate, ast.Call):
                    continue
                func = candidate.func
                marker = ""
                if isinstance(func, ast.Attribute):
                    marker = func.attr
                if marker == "requirement":
                    requirements.update(_strings(candidate))
                elif marker == "risk":
                    risks.update(_strings(candidate))
    return TraceLink(rel, tuple(sorted(requirements)), tuple(sorted(risks)))


def load_catalog(root: Path = ROOT) -> dict[str, object]:
    return json.loads((root / "config" / "requirements_catalog.json").read_text(encoding="utf-8"))


def load_mappings(root: Path = ROOT) -> tuple[TraceLink, ...]:
    path = root / "config" / "test_traceability.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("mappings", []):
        rows.append(
            TraceLink(
                str(row["test"]),
                tuple(sorted(str(x) for x in row.get("requirements", []))),
                tuple(sorted(str(x) for x in row.get("risks", []))),
            )
        )
    return tuple(rows)


def merge_links(root: Path = ROOT) -> tuple[TraceLink, ...]:
    merged: dict[str, tuple[set[str], set[str]]] = {}
    for link in load_mappings(root):
        reqs, risks = merged.setdefault(link.test, (set(), set()))
        reqs.update(link.requirements)
        risks.update(link.risks)
    for path in (root / "tests").rglob("test_*.py"):
        link = inline_links(path, root)
        if not link.requirements and not link.risks:
            continue
        reqs, risks = merged.setdefault(link.test, (set(), set()))
        reqs.update(link.requirements)
        risks.update(link.risks)
    return tuple(
        TraceLink(test, tuple(sorted(reqs)), tuple(sorted(risks)))
        for test, (reqs, risks) in sorted(merged.items())
    )


def existing_tests(root: Path = ROOT) -> set[str]:
    return {
        path.resolve().relative_to(root.resolve()).as_posix()
        for path in (root / "tests").rglob("test_*.py")
    }


def ids(rows: Iterable[dict[str, object]]) -> set[str]:
    return {str(row["id"]) for row in rows}
