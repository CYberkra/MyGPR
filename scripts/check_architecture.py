#!/usr/bin/env python3
"""Enforce layer direction, migration ownership and compatibility boundaries."""
from __future__ import annotations

import ast
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config/architecture_policy.toml"


@dataclass(frozen=True)
class ImportRef:
    name: str
    line: int


def py_files(entry: str) -> list[Path]:
    path = ROOT / entry
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(item for item in path.rglob("*.py") if "__pycache__" not in item.parts)


def module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative_import(path: Path, node: ast.ImportFrom) -> str:
    if node.level <= 0:
        return node.module or ""
    current = module_name(path).split(".")
    if path.name != "__init__.py" and current:
        current.pop()
    ascend = max(0, node.level - 1)
    if ascend:
        current = current[:-ascend] if ascend <= len(current) else []
    suffix = (node.module or "").split(".") if node.module else []
    return ".".join([*current, *suffix])


def imported_names(tree: ast.AST, path: Path, *, top_level_only: bool = False) -> Iterable[ImportRef]:
    nodes = tree.body if top_level_only and isinstance(tree, ast.Module) else ast.walk(tree)
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield ImportRef(alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            yield ImportRef(_resolve_relative_import(path, node), node.lineno)


def _matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _path_matches(path: Path, prefix: str) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    normalized = prefix.rstrip("/")
    return relative == normalized or relative.startswith(normalized + "/")


def _migration_exception(policy: dict, path: Path) -> dict | None:
    matches = [
        item
        for item in policy.get("migration_exceptions", [])
        if _path_matches(path, str(item.get("path_prefix", "")))
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: len(str(item.get("path_prefix", ""))))


def _validate_exception_metadata(policy: dict) -> list[str]:
    errors: list[str] = []
    for index, item in enumerate(policy.get("migration_exceptions", []), start=1):
        for field in ("path_prefix", "owner", "remove_after", "reason"):
            if not str(item.get(field, "")).strip():
                errors.append(f"migration exception #{index}: missing {field}")
        prefix = str(item.get("path_prefix", ""))
        if prefix and not (ROOT / prefix).exists():
            errors.append(f"migration exception #{index}: path does not exist: {prefix}")
    return errors


def _local_import_allowed(name: str, allowed: tuple[str, ...]) -> bool:
    return any(_matches_prefix(name, prefix) for prefix in allowed)


def _check_layers(policy: dict) -> tuple[list[str], dict[str, set[str]]]:
    errors: list[str] = []
    graph: dict[str, set[str]] = defaultdict(set)
    project_prefixes = tuple(policy["project_imports"]["prefixes"])
    layers = policy.get("layers", {})

    for layer_name, spec in layers.items():
        default_allowed = tuple(spec.get("allowed_local_import_prefixes", []))
        for root in spec.get("paths", []):
            for path in py_files(root):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                exception = _migration_exception(policy, path) or {}
                allowed = default_allowed + tuple(exception.get("allowed_local_import_prefixes", []))
                current_module = module_name(path)
                for imported in imported_names(tree, path):
                    if not imported.name or not any(
                        _matches_prefix(imported.name, prefix) for prefix in project_prefixes
                    ):
                        continue
                    graph[current_module].add(imported.name)
                    if not _local_import_allowed(imported.name, allowed):
                        errors.append(
                            f"{path.relative_to(ROOT)}:{imported.line}: "
                            f"layer {layer_name} may not import {imported.name}"
                        )
    return errors, graph


def _check_layer_cycles(graph: dict[str, set[str]]) -> list[str]:
    """Detect cycles only inside the new ``mygpr`` architecture packages."""
    nodes = {name for name in graph if name.startswith("mygpr.")}
    adjacency: dict[str, set[str]] = {name: set() for name in nodes}
    for source, targets in graph.items():
        if source not in nodes:
            continue
        for target in targets:
            candidates = [node for node in nodes if node == target or node.startswith(target + ".")]
            if candidates:
                adjacency[source].add(min(candidates, key=len))

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    errors: list[str] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbour in adjacency.get(node, set()):
            if neighbour not in indices:
                strongconnect(neighbour)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbour])
            elif neighbour in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbour])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while stack:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1:
                errors.append("new-layer import cycle: " + " -> ".join(sorted(component)))

    for node in sorted(nodes):
        if node not in indices:
            strongconnect(node)
    return errors


def _check_legacy_core(policy: dict) -> list[str]:
    errors: list[str] = []
    spec = policy["legacy_core"]
    forbidden = tuple(spec["forbidden_import_prefixes"])
    for root in spec["paths"]:
        for path in py_files(root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for imported in imported_names(tree, path):
                if any(_matches_prefix(imported.name, prefix) for prefix in forbidden):
                    errors.append(
                        f"{path.relative_to(ROOT)}:{imported.line}: forbidden legacy-core import {imported.name}"
                    )
    return errors


def _check_entrypoint(policy: dict) -> list[str]:
    entry = policy.get("entrypoint")
    if not entry:
        return []
    path = ROOT / entry["path"]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = tuple(entry["forbidden_top_level_import_prefixes"])
    return [
        f"{path.name}:{imported.line}: heavy top-level import {imported.name}"
        for imported in imported_names(tree, path, top_level_only=True)
        if any(_matches_prefix(imported.name, prefix) for prefix in forbidden)
    ]


def _check_sys_path(policy: dict) -> list[str]:
    errors: list[str] = []
    for root in policy["path_policy"]["forbid_sys_path_mutation_under"]:
        for path in py_files(root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr not in {"insert", "append"}:
                    continue
                target = node.func.value
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "sys"
                    and target.attr == "path"
                ):
                    errors.append(f"{path.relative_to(ROOT)}:{node.lineno}: sys.path mutation")
    return errors


def _check_compatibility_registry(policy: dict) -> list[str]:
    spec = policy.get("compatibility")
    if not spec:
        return []
    registry = (ROOT / spec["registry"]).read_text(encoding="utf-8")
    errors: list[str] = []
    for required in spec["required_for_paths"]:
        module = required[:-3].replace("/", ".") if required.endswith(".py") else required.replace("/", ".")
        if module not in registry:
            errors.append(f"compatibility contract missing: {required}")
    return errors


def _is_silent_handler(node: ast.ExceptHandler) -> bool:
    body = [
        item
        for item in node.body
        if not (
            isinstance(item, ast.Expr)
            and isinstance(item.value, ast.Constant)
            and isinstance(item.value.value, str)
        )
    ]
    return not body or all(isinstance(item, (ast.Pass, ast.Continue)) for item in body)


def _check_new_code_quality(policy: dict) -> list[str]:
    spec = policy["new_code_quality"]
    errors: list[str] = []
    for root in spec["paths"]:
        for path in py_files(root):
            exception = _migration_exception(policy, path) or {}
            max_module = int(exception.get("max_module_lines", spec["max_module_lines"]))
            max_class = int(exception.get("max_class_lines", spec["max_class_lines"]))
            max_function = int(exception.get("max_function_lines", spec["max_function_lines"]))
            text = path.read_text(encoding="utf-8")
            line_count = len(text.splitlines())
            if line_count > max_module:
                errors.append(f"{path.relative_to(ROOT)}: module lines {line_count} > {max_module}")
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    size = int(getattr(node, "end_lineno", node.lineno)) - node.lineno + 1
                    if size > max_class:
                        errors.append(f"{path.relative_to(ROOT)}:{node.lineno}: class {node.name} lines {size} > {max_class}")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    size = int(getattr(node, "end_lineno", node.lineno)) - node.lineno + 1
                    if size > max_function:
                        errors.append(f"{path.relative_to(ROOT)}:{node.lineno}: function {node.name} lines {size} > {max_function}")
                elif (
                    spec.get("forbid_silent_exception_handlers")
                    and isinstance(node, ast.ExceptHandler)
                    and _is_silent_handler(node)
                ):
                    errors.append(f"{path.relative_to(ROOT)}:{node.lineno}: silent exception handler")
    return errors


def _check_frozen_modules(policy: dict) -> list[str]:
    errors: list[str] = []
    for item in policy.get("frozen_modules", []):
        path = ROOT / item["path"]
        if not path.exists():
            errors.append(f"frozen module missing: {item['path']}")
            continue
        lines = len(path.read_text(encoding="utf-8").splitlines())
        limit = int(item["max_lines"])
        if lines > limit:
            errors.append(f"{item['path']}: frozen module grew to {lines} lines > {limit}")
        if not str(item.get("owner", "")).strip() or not str(item.get("reason", "")).strip():
            errors.append(f"{item['path']}: frozen module requires owner and reason")
    return errors


def main() -> int:
    policy = tomllib.loads(POLICY.read_text(encoding="utf-8"))
    errors: list[str] = []
    errors.extend(_validate_exception_metadata(policy))
    layer_errors, graph = _check_layers(policy)
    errors.extend(layer_errors)
    errors.extend(_check_layer_cycles(graph))
    errors.extend(_check_legacy_core(policy))
    errors.extend(_check_entrypoint(policy))
    errors.extend(_check_sys_path(policy))
    errors.extend(_check_compatibility_registry(policy))
    errors.extend(_check_new_code_quality(policy))
    errors.extend(_check_frozen_modules(policy))
    if errors:
        print("\n".join(errors))
        return 1
    print("architecture policy: PASS (layers, cycles, migration ownership, growth limits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
