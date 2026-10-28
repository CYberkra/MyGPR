"""Menu base styles live until their native owners finish closing/destruction."""
from pathlib import Path
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip('PyQt6')
pytest.importorskip('qfluentwidgets')

from PyQt6.QtCore import QCoreApplication, QEvent, QObject
from PyQt6.QtWidgets import QMenu, QStyleFactory, QWidget
from qfluentwidgets import Action, RoundMenu
from qfluentwidgets.components.widgets.combo_box import ComboBoxMenu
from qfluentwidgets.components.widgets.menu import EditMenu

from ui.widgets.context_menus import SafeRoundMenu

ROOT = Path(__file__).resolve().parents[1]


def test_original_menu_workflow_exits_naturally_in_a_fresh_process(tmp_path):
    """The original three-case native AV, without a full-suite warmup."""
    program = textwrap.dedent('''
        import atexit
        import sys
        sys.path.insert(0, sys.argv[1])
        atexit.register(lambda: print('QT_SHUTDOWN_ATEXIT', flush=True))
        import pytest
        result = pytest.main([
            'tests/test_processing_workspace_flow.py', '-q',
            '-k', 'source_menu or history_group_menu or replace_run_draft',
            '--basetemp=' + sys.argv[2]])
        print('QT_SHUTDOWN_RETURN=' + str(result), flush=True)
        raise SystemExit(result)
    ''')
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen',
                       MYGPR_RUNTIME_ROOT=str(tmp_path/'runtime'),
                       MYGPR_SETTINGS_FILE=str(tmp_path/'settings.json'),
                       MYGPR_LOG_DIR=str(tmp_path/'logs'))
    result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', program,
                             str(ROOT), str(tmp_path/'pytest')], cwd=ROOT,
                            env=environment, capture_output=True, text=True, timeout=45)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert 'QT_SHUTDOWN_RETURN=0' in output and 'QT_SHUTDOWN_ATEXIT' in output
    assert 'fatal exception' not in output.lower() and 'Fatal Python error' not in output


@pytest.mark.parametrize('menu_type', [RoundMenu, ComboBoxMenu, EditMenu, SafeRoundMenu])
def test_library_and_local_menus_own_the_created_base_style(qapp, monkeypatch, menu_type):
    original_create = QStyleFactory.create
    created = []

    def capture(name):
        style = original_create(name)
        created.append(style)
        return style

    monkeypatch.setattr(QStyleFactory, 'create', staticmethod(capture))
    host = QWidget()
    try:
        menu = menu_type(parent=host)
        menu.addAction(Action('inspect'))
        assert len(created) == 1
        # menu.style() can be a QSS proxy. Verify the actual factory-created
        # base, and never transfer the shared QApplication style.
        assert created[0].parent() is menu
        assert created[0] is not qapp.style()
        assert 'setStyle' not in menu.__dict__
    finally:
        host.close()
        host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()


def test_later_explicit_shared_style_keeps_its_existing_owner(qapp):
    host = QWidget()
    menu = RoundMenu(parent=host)
    shared = qapp.style()
    owner = shared.parent()
    try:
        menu.setStyle(shared)
        assert shared.parent() is owner
        assert 'setStyle' not in menu.__dict__
    finally:
        host.close()
        host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()


def test_menu_constructor_accepts_the_factory_none_fallback(qapp, monkeypatch):
    monkeypatch.setattr(QStyleFactory, 'create', staticmethod(lambda _name: None))
    menu = RoundMenu()
    try:
        assert 'setStyle' not in menu.__dict__
    finally:
        menu.close()
        menu.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()


def test_constructor_exception_restores_an_existing_instance_setter(qapp, monkeypatch):
    menu = RoundMenu.__new__(RoundMenu)
    from functools import partial
    setter = partial(QMenu.setStyle, menu)
    menu.setStyle = setter

    def fail(_name):
        raise RuntimeError('factory unavailable')

    monkeypatch.setattr(QStyleFactory, 'create', staticmethod(fail))
    try:
        with pytest.raises(RuntimeError, match='factory unavailable'):
            RoundMenu.__init__(menu)
        assert menu.__dict__['setStyle'] is setter
    finally:
        menu.close()
        menu.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()


def test_constructor_keeps_a_style_with_an_existing_native_owner(qapp, monkeypatch):
    owner = QObject()
    style = QStyleFactory.create('fusion')
    style.setParent(owner)
    monkeypatch.setattr(QStyleFactory, 'create', staticmethod(lambda _name: style))
    menu = RoundMenu()
    try:
        assert style.parent() is owner
    finally:
        menu.close()
        menu.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_menu_style_patch_is_idempotent(qapp):
    from ui.theme_helpers import patch_round_menu_style_lifetime
    constructor = RoundMenu.__init__
    patch_round_menu_style_lifetime()
    patch_round_menu_style_lifetime()
    assert RoundMenu.__init__ is constructor


@pytest.mark.parametrize('ready,cancel_first', [(False, False), (True, False), (True, True)])
def test_real_entry_releases_accepted_window_before_return(tmp_path, ready, cancel_first):
    """Init shutdown joins its worker; an ignored close leaves the window alive."""
    program = textwrap.dedent('''
        import atexit
        import sys
        import weakref
        sys.path.insert(0, sys.argv[1])
        from PyQt6 import sip
        from PyQt6.QtCore import QObject, QSettings, QTimer, pyqtSlot
        from PyQt6.QtWidgets import QApplication
        import app_qt
        import ui.pages.project_page as project_page
        project_page.QSettings = lambda *_args: QSettings(sys.argv[4], QSettings.Format.IniFormat)
        atexit.register(lambda: print('ENTRY_SHUTDOWN_ATEXIT', flush=True))
        record = {'returned': False, 'cancelled': False, 'destroyed': False}
        window_ref = None

        class Observer(QObject):
            @pyqtSlot()
            def mark(self):
                record['destroyed'] = True
                record['destroyed_before_return'] = not record['returned']
                record['app_alive_at_destroy'] = QApplication.instance() is not None

        observer = Observer()

        class Window(app_qt.MyGPRMainWindow):
            def __init__(self, *args, **kwargs):
                global window_ref
                super().__init__(*args, **kwargs)
                window_ref = weakref.ref(self)
                self.destroyed.connect(observer.mark)
                QTimer.singleShot(0, self.request_close)

            def request_close(self):
                if sys.argv[3] == 'True':
                    if not self._backend_ready:
                        QTimer.singleShot(0, self.request_close)
                        return
                    self.ensure_pages_ready()
                if sys.argv[2] == 'True':
                    project = self.page_coordinator.project
                    original = project.confirm_interpretation_close
                    project.confirm_interpretation_close = lambda: False
                    assert not self.close()
                    assert self.isVisible() and not sip.isdeleted(self)
                    record['cancelled'] = True
                    project.confirm_interpretation_close = original
                assert self.close()

            def closeEvent(self, event):
                controller = self.backend_controller
                thread = controller._thread
                super().closeEvent(event)
                if event.isAccepted():
                    record['init_thread_stopped'] = thread is None or not thread.isRunning()

        app_qt.MyGPRMainWindow = Window
        status = app_qt.main()
        record['returned'] = True
        print('ENTRY_SHUTDOWN_STATE=' + repr(record), flush=True)
        assert status == 0 and record['init_thread_stopped']
        assert record['cancelled'] == (sys.argv[2] == 'True')
        assert record['destroyed'] and record['destroyed_before_return'] and record['app_alive_at_destroy']
        assert window_ref() is None or sip.isdeleted(window_ref())
        print('ENTRY_SHUTDOWN_RETURN=0', flush=True)
    ''')
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen',
                       MYGPR_RUNTIME_ROOT=str(tmp_path/'runtime'),
                       MYGPR_SETTINGS_FILE=str(tmp_path/'settings.json'),
                       MYGPR_LOG_DIR=str(tmp_path/'logs'))
    result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', program,
                             str(ROOT), str(cancel_first), str(ready), str(tmp_path/'legacy.ini')], cwd=ROOT, env=environment,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ENTRY_SHUTDOWN_RETURN=0' in result.stdout and 'ENTRY_SHUTDOWN_ATEXIT' in result.stdout
