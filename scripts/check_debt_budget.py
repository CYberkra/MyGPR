#!/usr/bin/env python3
"""Enforce a current technical-debt ratchet and report the long-term target gap."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate import RatchetGate, add_write_baseline_flag, read_baseline, write_baseline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config/debt_baseline.json"
REDUCTION_TARGET = ROOT / "config/debt_reduction_target.json"
SCOPES = ("core", "ui", "mygpr", "PythonModule", "compatibility", "scripts")
ENFORCED = {
    "broad_exception_handlers",
    "silent_exception_handlers",
    "sys_path_mutations",
    "modules_over_1000_lines",
    "classes_over_1000_lines",
    "functions_over_100_lines",
}


def source_files() -> list[Path]:
    paths: list[Path] = []
    for scope in SCOPES:
        paths.extend(path for path in (ROOT / scope).rglob("*.py") if "__pycache__" not in path.parts)
    paths.append(ROOT / "cli_batch.py")
    return sorted({path for path in paths if path.exists()})


def _empty_metrics() -> dict[str, int]:
    return {
        "python_files": 0,
        "source_lines": 0,
        "broad_exception_handlers": 0,
        "silent_exception_handlers": 0,
        "sys_path_mutations": 0,
        "modules_over_1000_lines": 0,
        "classes_over_1000_lines": 0,
        "functions_over_100_lines": 0,
    }


def _is_broad_handler(node: ast.ExceptHandler) -> bool:
    return node.type is None or (
        isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}
    )


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


def _is_sys_path_mutation(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in {"insert", "append"}:
        return False
    value = node.func.value
    return (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "sys"
        and value.attr == "path"
    )


def metrics() -> dict[str, int]:
    result = _empty_metrics()
    for path in source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        result["python_files"] += 1
        result["source_lines"] += len(lines)
        if len(lines) > 1000:
            result["modules_over_1000_lines"] += 1
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and _is_broad_handler(node):
                result["broad_exception_handlers"] += 1
                if _is_silent_handler(node):
                    result["silent_exception_handlers"] += 1
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if getattr(node, "end_lineno", node.lineno) - node.lineno + 1 > 100:
                    result["functions_over_100_lines"] += 1
            if isinstance(node, ast.ClassDef):
                if getattr(node, "end_lineno", node.lineno) - node.lineno + 1 > 1000:
                    result["classes_over_1000_lines"] += 1
            if isinstance(node, ast.Call) and _is_sys_path_mutation(node):
                result["sys_path_mutations"] += 1
    return result


def target_gaps(current: dict[str, int], target: dict[str, int]) -> dict[str, int]:
    return {
        key: max(0, int(current.get(key, 0)) - int(target[key]))
        for key in sorted(ENFORCED & target.keys())
    }


def build_gates(current: dict[str, int], baseline: dict[str, int],
                tolerances: dict[str, float] | None = None) -> list[RatchetGate]:
    """只对ENFORCED 里的键组棘轮；baseline 缺的键记missing_ok（不阻断）。

    与原 ``evaluate_budget`` 语义一致——``value is not None and limit is not
    None`` 才比较——即 baseline 未记录的指标既不算通过也不算失败，只是
    不在棘轮覆盖内。

    ``tolerances`` 来自 baseline 文件的 ``tolerances`` 段：收紧 baseline 到
    current 附近后，零余量会让「顺手多写一个 101 行函数」直接红 CI。给几个
    百分点的余量吸收正常波动，同时保留「不可涨」的方向性——限额仍远低于
    旧 baseline（如 functions_over_100_lines 124 → 46）。
    """
    tolerances = tolerances or {}
    return [
        RatchetGate(name=key, current=int(current.get(key, 0)),
                    baseline=baseline.get(key),
                    tolerance=float(tolerances.get(key, 0.0)),
                    missing_ok=True)
        for key in sorted(ENFORCED)
    ]


def _read_tolerances() -> dict[str, float]:
    """读 baseline 文件的 ``tolerances`` 段（缺项即 0，即零余量）。"""
    if not BASELINE.exists():
        return {}
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    return {str(key): float(value)
            for key, value in dict(payload.get("tolerances") or {}).items()}


def _write_current_baseline(current: dict[str, int]) -> None:
    """把当前实测值写为新 baseline，**保留**既有 tolerances 段。

    收紧时不能顺手把容差清零——``--write-baseline`` 顾的是数值，容差是
    独立的策略选择，清掉会让日常开发被零余量卡住。
    """
    tolerances = _read_tolerances()
    write_baseline(BASELINE, {
        "schema": "mygpr.debt_baseline.v1",
        "policy": "release-ratchet; values may decrease but may not increase",
        "metrics": current,
        "tolerances": tolerances,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    add_write_baseline_flag(parser)
    parser.add_argument("--strict-target", action="store_true")
    args = parser.parse_args()

    current = metrics()
    if args.write_baseline:
        _write_current_baseline(current)

    baseline = read_baseline(BASELINE)
    tolerances = _read_tolerances()
    target = read_baseline(REDUCTION_TARGET, field_name="target_metrics") \
        if REDUCTION_TARGET.exists() else {}
    gates = build_gates(current, baseline, tolerances)
    errors = [text for gate in gates if (text := gate.violation())]
    gaps = target_gaps(current, target)
    result: dict[str, Any] = {
        "current": current,
        "release_baseline": baseline,
        "tolerances": tolerances,
        "reduction_target": target,
        "target_gaps": gaps,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for gate in gates:
        print(gate.report())
    if errors:
        print("\n".join(errors))
        return 1
    if args.strict_target and any(gaps.values()):
        print("historical debt-reduction target not yet reached")
        return 2
    print("debt budget: PASS (release ratchet enforced; reduction target reported)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
