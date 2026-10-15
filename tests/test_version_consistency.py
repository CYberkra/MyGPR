from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pytest


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


@pytest.mark.parametrize("relative", [
    "README.md", "AGENTS.md", "CLAUDE.md", "GUI_README.md", "CURRENT_STATE.md",
    "TODO.md", "CHANGELOG.md", "config/backend_source_manifest.json",
])
def test_version_checker_rejects_stale_release_identity(tmp_path, monkeypatch, relative):
    root = Path(__file__).resolve().parents[1]
    checker = _load_version_checker()
    paths = ["VERSION", "pyproject.toml", "README.md", "AGENTS.md", "CLAUDE.md",
             "GUI_README.md", "CURRENT_STATE.md", "TODO.md", "CHANGELOG.md",
             "config/backend_source_manifest.json"]
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    expected = (tmp_path / "VERSION").read_text(encoding="utf-8-sig").strip()
    target = tmp_path / relative
    target.write_text(target.read_text(encoding="utf-8").replace(expected, "0.0.0"), encoding="utf-8")
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    with pytest.raises(AssertionError, match=relative.replace(".", r"\.")):
        checker.check_version_consistency()
