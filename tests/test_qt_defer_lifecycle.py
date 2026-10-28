"""Deferred callbacks keep Python state until the native timer finishes or dies."""
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

import pytest


@pytest.mark.parametrize('mode', [
    'host-gc', 'spatial-gc', 'normal', 'destroy', 'stop', 'restart',
    'callback-destroy', 'callback-restart', 'callback-error', 'invalid-start',
])
def test_deferred_callback_survives_gc_and_releases_pending_state(tmp_path, mode):
    pytest.importorskip('PyQt6')
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen',
        MYGPR_RUNTIME_ROOT=str(tmp_path / 'runtime'), MYGPR_LOG_DIR=str(tmp_path / 'logs'),
        MYGPR_SETTINGS_FILE=str(tmp_path / 'settings.json'),
        LOCALAPPDATA=str(tmp_path / 'local'), APPDATA=str(tmp_path / 'roaming'))
    program = dedent('''
        import gc
        import socket
        import sys
        import weakref
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QApplication, QWidget
        from ui import constants, qt_defer
        from ui.theme_helpers import apply_theme

        def deny_network(*args, **kwargs):
            raise AssertionError('Network forbidden in timer lifecycle verification')
        socket.socket.connect = deny_network
        errors, called = [], []
        sys.excepthook = lambda kind, error, tb: errors.append(str(error))
        app = QApplication([])
        app.setStyle('Fusion')
        apply_theme(constants.THEME_LIGHT)
        mode = sys.argv[1]

        class Host(QWidget):
            def __init__(self):
                super().__init__()
                self.timer = qt_defer.defer_call(self, lambda key='route': self.flush(key), 0)

            def flush(self, key):
                called.append(key)

        def spatial_host():
            from ui.pages.spatial_page import SpatialPage
            page = SpatialPage()
            page._defer(lambda key='absent-route': page._flush_if_current(key))
            return page

        if mode in ('host-gc', 'spatial-gc'):
            for _ in range(4):
                host = spatial_host() if mode == 'spatial-gc' else Host()
                host.close()
                host.deleteLater()
                del host
                gc.collect()
                app.processEvents()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                app.processEvents()
        elif mode == 'destroy':
            host = Host()
            reference = weakref.ref(host)
            host.deleteLater()
            del host
            gc.collect()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            gc.collect()
            assert called == [], called
            assert reference() is None
        elif mode == 'stop':
            host = Host()
            reference = weakref.ref(host)
            host.timer.stop()
            del host
            gc.collect()
            app.processEvents()
            assert called == [], called
            assert reference() is None, 'cancelled timer retained its host'
        elif mode.startswith('callback-'):
            from PyQt6 import sip
            host = QWidget()
            def callback():
                called.append(mode)
                if mode == 'callback-destroy':
                    sip.delete(host)
                elif mode == 'callback-restart' and len(called) == 1:
                    timer.start(0)
                elif mode == 'callback-error':
                    raise ValueError('intentional callback error')
            timer = qt_defer.defer_call(host, callback)
            for _ in range(4):
                app.processEvents()
            assert len(called) == (2 if mode == 'callback-restart' else 1), called
            if mode == 'callback-error':
                assert errors == ['intentional callback error'], errors
                errors.clear()
            if not sip.isdeleted(host):
                host.deleteLater()
            del host, timer
            gc.collect()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
        elif mode == 'invalid-start':
            host = Host()
            host.timer.stop()
            try:
                host.timer.start('invalid delay')
            except TypeError:
                pass
            else:
                raise AssertionError('QTimer start must still reject invalid arguments')
            del host
            gc.collect()
            app.processEvents()
        else:
            host = Host()
            if mode == 'restart':
                host.timer.stop()
                host.timer.start(0)
            app.processEvents()
            app.processEvents()
            assert called == ['route'], called
            host.close()
            host.deleteLater()
            del host
            gc.collect()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
        gc.collect()
        print('Deferred callback errors:', errors, file=sys.stderr)
        assert errors == [], errors
        assert not getattr(qt_defer, '_PENDING_CALLS', {}), 'pending callback leaked'
    ''')
    result = subprocess.run([sys.executable, '-B', '-c', program, mode],
        env=environment, cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
