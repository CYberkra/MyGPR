"""Windows must keep the main native window when the first lazy GL view arrives."""
from __future__ import annotations

import os
from importlib.util import find_spec
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native surface regression")
def test_first_gl_view_does_not_hide_or_recreate_main_window(tmp_path):
    if find_spec("qfluentwidgets") is None or find_spec("OpenGL") is None:
        pytest.skip("Native GUI/OpenGL dependencies are unavailable")
    environment = dict(os.environ, QT_QPA_PLATFORM="windows:darkmode=0")
    for key in ("APPDATA", "LOCALAPPDATA", "TEMP", "TMP"):
        path = tmp_path / key
        path.mkdir()
        environment[key] = str(path)
    script = textwrap.dedent(r'''
        import sys
        from PyQt6.QtCore import QEvent, QObject
        from PyQt6.QtGui import QOpenGLContext
        from PyQt6.QtWidgets import QApplication
        from ui import constants
        from ui.settings_manager import SettingsManager
        from ui.theme_helpers import apply_theme
        import ui.main_window as mw
        original_excepthook = sys.excepthook
        def report_gl_context(exc_type, exc, traceback):
            try:
                context = QOpenGLContext.currentContext()
                if context is None:
                    print('OpenGL diagnostic: no current Qt context', file=sys.stderr, flush=True)
                else:
                    fmt = context.format()
                    print(f'OpenGL diagnostic: valid={context.isValid()} '
                          f'version={fmt.majorVersion()}.{fmt.minorVersion()} '
                          f'profile={fmt.profile().name} renderable={fmt.renderableType().name}',
                          file=sys.stderr, flush=True)
            except Exception as diagnostic_error:
                print(f'OpenGL diagnostic unavailable: {diagnostic_error!r}', file=sys.stderr, flush=True)
            finally:
                original_excepthook(exc_type, exc, traceback)
        sys.excepthook = report_gl_context
        app = QApplication([])
        app.setStyle("Fusion")
        apply_theme(constants.THEME_LIGHT)
        for name in ("BackendController", "ProjectController", "ProcessingController", "InterpretationController", "DeliveryController"):
            setattr(mw, name, None)
        window = mw.MyGPRMainWindow(settings=SettingsManager(sys.argv[1]))
        window.ensure_pages_ready()
        window.splashScreen.close()
        window.show()
        app.processEvents()
        events = []
        class Watch(QObject):
            def eventFilter(self, obj, event):
                if event.type() in (QEvent.Type.Hide, QEvent.Type.Show, QEvent.Type.WinIdChange):
                    events.append(event.type().name)
                return False
        watch = Watch()
        window.installEventFilter(watch)
        before = int(window.winId())
        from ui.widgets.trajectory_3d_view import Trajectory3DView
        view = window.pages["spatialInterface"]._3d_view
        assert view._ensure_gl_view()
        window.stackedWidget.setCurrentWidget(window.pages["spatialInterface"])
        window.pages["spatialInterface"]._view_stack.setCurrentWidget(view)
        app.processEvents()
        assert view._gl_view.isValid(), "Native OpenGL context was not created"
        assert int(window.winId()) == before, (before, int(window.winId()), events)
        assert not events, events
        window.close()
        app.processEvents()
    ''')
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "settings.json")],
                            env=environment, capture_output=True, text=True, timeout=35)
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
