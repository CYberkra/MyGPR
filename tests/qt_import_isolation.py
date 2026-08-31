"""Helpers for proving backend operations do not introduce Qt imports.

Pytest imports every selected test module during collection. GUI modules can
therefore place PyQt6 in ``sys.modules`` before a backend-only test starts.
Comparing snapshots is the stable contract: backend work must not add any Qt
module beyond what the surrounding test process had already loaded.
"""
from __future__ import annotations

import sys


def qt_module_snapshot() -> frozenset[str]:
    return frozenset(
        name for name in sys.modules
        if name == "PyQt6" or name.startswith("PyQt6.")
    )


def assert_qt_imports_unchanged(before: frozenset[str]) -> None:
    after = qt_module_snapshot()
    added = sorted(after - before)
    assert not added, f"backend operation imported Qt modules: {added}"
