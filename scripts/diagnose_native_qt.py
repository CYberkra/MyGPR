"""Temporary CI-only A/B diagnosis of the resource-tree predecessor crash."""
from __future__ import annotations

import gc
import os
from pathlib import Path
import sys
import types

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from PyQt6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication, QStyleFactory  # noqa: E402
from qfluentwidgets.components.widgets import tree_view  # noqa: E402


variant = sys.argv[1]
if variant == 'owned-style':
    def owned_style(widget):
        # Same Windows style as Fluent; change only its native ownership.
        style = QStyleFactory.create('windowsvista')
        style.setParent(widget)
        widget.setStyle(style)

    tree_view.updateDynamicStyle = owned_style
elif variant == 'shared-style':
    tree_view.updateDynamicStyle = lambda widget: widget.setStyle(QApplication.style())
elif variant == 'simple-doubles':
    import tests.test_ui_migration_regressions as migration

    original = migration.MyGPRMainWindow

    class PureWindow:
        _require_line = original._require_line

        def __new__(cls):
            return types.SimpleNamespace()

    migration.MyGPRMainWindow = PureWindow


class CleanupPlugin:
    def pytest_runtest_teardown(self, item):
        if (variant == 'flush-delete'
                and item.name == 'test_project_tree_does_not_repeat_global_project_identity'):
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            QApplication.instance().processEvents()
            gc.collect()


raise SystemExit(pytest.main([
    'tests/test_ui_content_simplification.py::test_project_tree_does_not_repeat_global_project_identity',
    'tests/test_ui_migration_regressions.py',
    '-q', '-s', '-x', '--tb=short', '--timeout=45', '--timeout-method=thread',
    f'--junitxml=output/qt-native-diagnosis/{variant}.xml',
], plugins=[CleanupPlugin()]))
