from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_version_checker():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "check_version_consistency.py"
    spec = importlib.util.spec_from_file_location("check_version_consistency", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_version_file_changelog_and_packaging_spec_are_consistent():
    root = Path(__file__).resolve().parents[1]
    expected = (root / "VERSION").read_text(encoding="utf-8-sig").strip()
    checker = _load_version_checker()
    assert checker.check_version_consistency() == expected
