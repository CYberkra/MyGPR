#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyInstaller packaging contract tests."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_HIDDEN_IMPORT_PREFIXES = ("core.", "ui.", "PythonModule.")
LOCAL_HIDDEN_IMPORTS = {"read_file_data"}


def _read_analysis_keyword(keyword_name: str):
    spec_path = ROOT / "gpr_gui.spec"
    tree = ast.parse(spec_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Analysis":
            for keyword in node.keywords:
                if keyword.arg == keyword_name:
                    return ast.literal_eval(keyword.value)

    raise AssertionError(f"gpr_gui.spec does not define Analysis({keyword_name}=...)")


def _read_spec_datas() -> list[tuple[str, str]]:
    datas = _read_analysis_keyword("datas")
    return [(str(src), str(dst)) for src, dst in datas]


def _read_spec_hiddenimports() -> list[str]:
    return [str(name) for name in _read_analysis_keyword("hiddenimports")]


def _is_local_hidden_import(module_name: str) -> bool:
    return (
        module_name in LOCAL_HIDDEN_IMPORTS
        or module_name.startswith(LOCAL_HIDDEN_IMPORT_PREFIXES)
    )


def _local_module_exists(module_name: str) -> bool:
    module_path = ROOT / Path(*module_name.split("."))
    return module_path.with_suffix(".py").exists() or (
        module_path / "__init__.py"
    ).exists()


def test_pyinstaller_datas_exist():
    missing = [
        src
        for src, _dst in _read_spec_datas()
        if not (ROOT / src).exists()
    ]

    assert missing == []


def test_local_hiddenimports_exist():
    missing = [
        module_name
        for module_name in _read_spec_hiddenimports()
        if _is_local_hidden_import(module_name)
        and not _local_module_exists(module_name)
    ]

    assert missing == []
