from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_check_compile():
    script = ROOT / "scripts" / "check_python_compile.py"
    spec = importlib.util.spec_from_file_location("check_python_compile", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_check_compile_passes_on_clean_repo():
    module = _load_check_compile()
    result = module.check_compile(ROOT.resolve(), ["core", "ui", "scripts", "tests"])
    assert result["ok"] is True
    assert result["file_count"] > 0
    assert result["errors"] == []
