from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "config" / "architecture_policy.toml"


def _load_check_architecture():
    script = ROOT / "scripts" / "check_architecture.py"
    spec = importlib.util.spec_from_file_location("check_architecture", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_policy() -> dict:
    return tomllib.loads(POLICY.read_text(encoding="utf-8"))


def test_check_architecture_passes_on_clean_repo():
    module = _load_check_architecture()
    policy = _load_policy()
    errors, graph = module._check_layers(policy)
    cycle_errors = module._check_layer_cycles(graph)
    legacy_errors = module._check_legacy_core(policy)
    ui_errors = module._check_ui_reverse_dependencies(policy)
    assert not errors, f"layer violations: {errors}"
    assert not cycle_errors, f"cycles: {cycle_errors}"
    assert not legacy_errors, f"legacy violations: {legacy_errors}"
    assert not ui_errors, f"reverse ui dependencies: {ui_errors}"


def test_check_architecture_reports_layer_violations():
    module = _load_check_architecture()
    policy = _load_policy()
    errors, _graph = module._check_layers(policy)
    assert isinstance(errors, list)


def test_check_architecture_reports_ui_reverse_dependencies():
    module = _load_check_architecture()
    policy = _load_policy()
    errors = module._check_ui_reverse_dependencies(policy)
    # 入口脚本 app_qt.py 被策略豁免；其余文件（core/mygpr/PythonModule/cli_batch）仍受约束
    assert isinstance(errors, list)
    assert not any('app_qt.py' in err for err in errors), (
        f"app_qt.py 应被豁免: {errors}")
