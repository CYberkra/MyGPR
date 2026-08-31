"""Regression contract for project/session shutdown ordering.

This test is deliberately free of Qt imports so it can run in lightweight
packaging environments as well as the full GUI test environment.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.app_paths import get_tile_cache_dir


_PAGE_COORDINATOR = Path(__file__).resolve().parents[1] / "ui" / "page_coordinator.py"
_APP_ENTRY = Path(__file__).resolve().parents[1] / "app_qt.py"


def _project_close_call_lines() -> dict[str, int]:
    tree = ast.parse(_PAGE_COORDINATOR.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "PageCoordinator":
            continue
        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if method.name != "_on_close_project_requested":
                continue
            calls: dict[str, int] = {}
            for child in ast.walk(method):
                if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
                    continue
                if child.func.attr in {"close_session", "close_current"}:
                    calls[child.func.attr] = child.lineno
            return calls
    raise AssertionError("PageCoordinator._on_close_project_requested not found")


class ProjectCloseLifecycleContractTests(unittest.TestCase):
    def test_interpretation_session_closes_before_project(self) -> None:
        calls = _project_close_call_lines()
        self.assertIn("close_session", calls)
        self.assertIn("close_current", calls)
        self.assertLess(calls["close_session"], calls["close_current"])


class AppPathContractTests(unittest.TestCase):
    def test_tile_cache_uses_the_configured_user_data_root(self) -> None:
        with TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"LOCALAPPDATA": temporary}, clear=False):
                cache_root = Path(get_tile_cache_dir())
            self.assertEqual(cache_root, Path(temporary) / "MyGPR" / "tile_cache")
            self.assertTrue(cache_root.is_dir())

    def test_smoke_output_does_not_use_a_posix_tmp_literal(self) -> None:
        source = _APP_ENTRY.read_text(encoding="utf-8")
        self.assertNotIn("SMOKE_SHOTS_DIR = '/tmp/mygpr_shots'", source)

    def test_smoke_closes_the_window_before_exiting(self) -> None:
        tree = ast.parse(_APP_ENTRY.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_smoke"
        )
        call_lines = {
            child.func.attr: child.lineno
            for child in ast.walk(function)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
            and child.func.attr in {"close", "exit"}
        }
        self.assertIn("close", call_lines)
        self.assertIn("exit", call_lines)
        self.assertLess(call_lines["close"], call_lines["exit"])


if __name__ == "__main__":
    unittest.main()
