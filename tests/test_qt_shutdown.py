"""The shared Qt test runtime must return through normal Python shutdown."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


def test_gui_smoke_returns_normally_and_runs_atexit(tmp_path):
    pytest.importorskip('PyQt6')
    pytest.importorskip('qfluentwidgets')
    root = Path(__file__).resolve().parents[1]
    script = '''
import atexit
import sys
import app_qt
atexit.register(lambda: print("gui-smoke-atexit", flush=True))
sys.argv = ["app_qt.py", "--smoke"]
status = app_qt.main()
print("gui-smoke-returned", status, flush=True)
raise SystemExit(status)
'''
    env = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen',
           'MYGPR_SETTINGS_FILE': str(tmp_path / 'settings.json'),
           'MYGPR_LOG_DIR': str(tmp_path / 'logs'),
           'MYGPR_SMOKE_SHOTS_DIR': str(tmp_path / 'shots')}
    result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', script],
                            cwd=root, env=env, capture_output=True, text=True, timeout=45)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert '[smoke] OK, 8 screenshots' in result.stdout, output
    assert 'gui-smoke-returned 0' in result.stdout, output
    assert 'gui-smoke-atexit' in result.stdout, output
    assert 'fatal exception' not in output.lower(), output
    assert 'fatal python error' not in output.lower(), output
    assert len(list((tmp_path / 'shots').glob('*.png'))) == 8


def test_qt_pytest_runtime_returns_and_runs_atexit(tmp_path):
    pytest.importorskip("PyQt6")
    root = Path(__file__).resolve().parents[1]
    child_test = tmp_path / "test_qt_lifecycle_child.py"
    child_test.write_text(
        '''
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QWidget
from ui.widgets.chain_strip import ChainStrip

def test_native_widget_lifecycle(qapp):
    parent = QWidget()
    strip = ChainStrip(parent)
    strip.set_steps([{"label": "Lifecycle", "enabled": True}])
    destroyed = []
    strip.destroyed.connect(lambda: destroyed.append(True))
    parent.show()
    qapp.processEvents()
    parent.close()
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    assert destroyed == [True]
''', encoding="utf-8",
    )
    script = '''
import atexit
import importlib.util
import pathlib
import sys
import pytest

root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("mygpr_qt_test_runtime", root / "tests/conftest.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
atexit.register(lambda: print("qt-lifecycle-atexit", flush=True))
status = pytest.main([sys.argv[2], "-q", "--confcutdir", sys.argv[3]], plugins=[runtime])
print("qt-lifecycle-pytest-returned", flush=True)
raise SystemExit(status)
'''
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    for key in ("MYGPR_RUNTIME_ROOT", "MYGPR_LOG_DIR", "MYGPR_SETTINGS_FILE"):
        env.pop(key, None)
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script,
         str(root), str(child_test), str(tmp_path)],
        cwd=root, env=env, capture_output=True, text=True, timeout=45,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "1 passed" in output, output
    assert "qt-lifecycle-pytest-returned" in result.stdout, output
    assert "qt-lifecycle-atexit" in result.stdout, output
    assert "fatal exception" not in output.lower(), output
    assert "fatal python error" not in output.lower(), output
