"""Private Fluent styles survive queued teardown without retaining old bases."""
import gc
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip('PyQt6')
pytest.importorskip('qfluentwidgets')

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QObject
from PyQt6.QtWidgets import QStyleFactory, QWidget
from qfluentwidgets import (PlainTextEdit, TableView, TableWidget, TextBrowser,
                           TextEdit, TreeView, TreeWidget)
from qfluentwidgets.common import style_sheet
from qfluentwidgets.components.widgets import command_bar, line_edit, table_view, tree_view

from ui.fluent_style_lifetime import patch_dynamic_style_lifetime
from ui import theme_helpers  # noqa: F401 - install the production patches

ROOT = Path(__file__).resolve().parents[1]
ALIASES = (command_bar, line_edit, table_view, tree_view)


def _dispose(widget, qapp):
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    gc.collect()
    assert sip.isdeleted(widget)


def test_tree_predecessor_and_warmup_exit_naturally(tmp_path):
    """Replay the original 3-second Windows AV in one shared Qt session."""
    program = textwrap.dedent('''
        import atexit
        import pytest
        atexit.register(lambda: print('TREE_WARMUP_ATEXIT', flush=True))
        result = pytest.main([
            'tests/test_ui_content_simplification.py::test_project_tree_does_not_repeat_global_project_identity',
            'tests/test_ui_migration_regressions.py', '-q', '-x', '--tb=short',
            '--timeout=30', '--timeout-method=thread'])
        print('TREE_WARMUP_RETURN=' + str(result), flush=True)
        raise SystemExit(result)
    ''')
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen',
                       MYGPR_RUNTIME_ROOT=str(tmp_path/'runtime'),
                       MYGPR_SETTINGS_FILE=str(tmp_path/'settings.json'),
                       MYGPR_LOG_DIR=str(tmp_path/'logs'), PYTEST_ADDOPTS='')
    result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', program],
                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=45)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert 'TREE_WARMUP_RETURN=0' in output and 'TREE_WARMUP_ATEXIT' in output
    assert 'fatal exception' not in output.lower() and 'Fatal Python error' not in output


@pytest.mark.parametrize('widget_type', [TreeWidget, TreeView, TableWidget, TableView,
                                       TextEdit, PlainTextEdit, TextBrowser])
def test_real_fluent_widgets_own_factory_bases_until_native_deletion(qapp, monkeypatch, widget_type):
    created = []
    original_create = QStyleFactory.create

    def capture(name):
        style = original_create(name)
        created.append(style)
        return style

    monkeypatch.setattr(QStyleFactory, 'create', staticmethod(capture))
    shared, owner = qapp.style(), qapp.style().parent()
    widget = widget_type()
    try:
        for style in created:
            if style is not None:
                assert style.parent() is widget
                assert style.objectName().lower() == 'windowsvista'
        assert 'setStyle' not in widget.__dict__
        assert shared.parent() is owner
    finally:
        _dispose(widget, qapp)
    assert all(style is None or sip.isdeleted(style) for style in created)
    assert not sip.isdeleted(shared) and shared.parent() is owner


def test_repeated_border_refresh_releases_replaced_private_styles(qapp, monkeypatch):
    created = []
    original_create = QStyleFactory.create

    def capture(name):
        style = original_create(name)
        created.append(style)
        return style

    monkeypatch.setattr(QStyleFactory, 'create', staticmethod(capture))
    widget = TreeWidget()
    try:
        for visible in (True, False, True):
            widget.setBorderVisible(visible)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        bases = [style for style in created if style is not None]
        if bases:
            assert len(bases) == 4
            assert all(sip.isdeleted(style) for style in bases[:-1])
            assert not sip.isdeleted(bases[-1]) and bases[-1].parent() is widget
    finally:
        _dispose(widget, qapp)


@pytest.mark.parametrize('kind', ['command', 'completion'])
def test_repeated_popup_refresh_keeps_only_the_current_dynamic_base(qapp, monkeypatch, kind):
    from qfluentwidgets import LineEdit

    created = []
    original_create = QStyleFactory.create

    def capture(name):
        style = original_create(name)
        created.append(style)
        return style

    monkeypatch.setattr(QStyleFactory, 'create', staticmethod(capture))
    editor = LineEdit()
    menu = command_bar.CommandViewMenu() if kind == 'command' else line_edit.CompleterMenu(editor)
    monkeypatch.setattr(menu, 'exec', lambda *_args, **_kwargs: None)
    if kind == 'completion':
        menu.setItems(['L01', 'L02'])
    try:
        for down in (True, False, True):
            if kind == 'command':
                menu.setDropDown(down)
            else:
                menu.popup()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        alive = [style for style in created if style is not None and not sip.isdeleted(style)]
        assert all(style.parent() in (menu, menu.view) for style in alive)
        assert sum(style.parent() is menu.view for style in alive) <= 1
    finally:
        _dispose(menu, qapp)
        _dispose(editor, qapp)
    assert all(style is None or sip.isdeleted(style) for style in created)


@pytest.mark.parametrize('source', ['private', 'shared', 'existing-owner', 'none'])
def test_patch_preserves_shared_external_and_none_styles(qapp, monkeypatch, source):
    owner = QObject()
    style = None if source == 'none' else qapp.style() if source == 'shared' else QStyleFactory.create('fusion')
    if source == 'existing-owner':
        style.setParent(owner)

    def upstream(widget):
        widget.setStyle(style)

    monkeypatch.setattr(style_sheet, 'updateDynamicStyle', upstream)
    for module in ALIASES:
        monkeypatch.setattr(module, 'updateDynamicStyle', upstream)
    patch_dynamic_style_lifetime()
    patched = style_sheet.updateDynamicStyle
    assert all(module.updateDynamicStyle is patched for module in ALIASES)
    patch_dynamic_style_lifetime()
    assert style_sheet.updateDynamicStyle is patched
    shared_owner = qapp.style().parent()
    widget = QWidget()
    try:
        patched(widget)
        assert 'setStyle' not in widget.__dict__
        if source == 'private':
            assert style.parent() is widget
        elif source == 'existing-owner':
            assert style.parent() is owner
        assert qapp.style().parent() is shared_owner
    finally:
        _dispose(widget, qapp)
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_patch_restores_existing_instance_setter_when_upstream_raises(qapp, monkeypatch):
    def fail(widget):
        raise RuntimeError('upstream failed')

    monkeypatch.setattr(style_sheet, 'updateDynamicStyle', fail)
    patch_dynamic_style_lifetime()
    widget = QWidget()
    setter = widget.setStyle
    widget.setStyle = setter
    try:
        with pytest.raises(RuntimeError, match='upstream failed'):
            style_sheet.updateDynamicStyle(widget)
        assert widget.__dict__['setStyle'] is setter
    finally:
        _dispose(widget, qapp)
