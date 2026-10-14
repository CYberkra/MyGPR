"""Native menu destruction must survive GC before Qt deferred deletion."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

pytest.importorskip("PyQt6")


@pytest.mark.parametrize("mode", ["overflow", "helper-parent", "helper-close"])
def test_menu_survives_gc_until_native_destruction(mode):
    script = '''
import gc
import sys
import weakref
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6 import sip
from ui.widgets.chain_strip import ChainStrip
from ui.widgets.context_menus import make_menu, add_checkable_submenu, _LIVE_MENUS

app = QApplication([])
mode = sys.argv[1]
for iteration in range(8):
    received = []
    if mode == "overflow":
        parent = ChainStrip()
        parent.set_steps([{'label': str(i), 'enabled': True} for i in range(3)])
        parent.sig_step_selected.connect(received.append)
        menu = parent._build_overflow_menu()
        menu._step_submenus[2].actions()[0].trigger()
        assert received == [2]
        parent.sig_step_editor_requested.connect(received.append)
        menu._step_submenus[1].actions()[1].trigger()
        assert received == [2, 1]
    else:
        parent = QWidget()
        menu = make_menu(parent)
        submenu = add_checkable_submenu(menu, 'Choice', ['a', 'b'], 'a', received.append)
        submenu.actions()[1].trigger()
        assert received == ['b']
        del submenu
    destroyed = []
    menu.destroyed.connect(lambda: destroyed.append(True))
    ref = weakref.ref(menu)
    if mode == "helper-close":
        menu.show()
        app.processEvents()
        menu.close()
    else:
        parent.deleteLater()
    del menu
    gc.collect()
    assert ref() is not None, 'menu wrapper collected before native destruction'
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    assert destroyed == [True], 'menu did not release with its owner or close event'
    assert not _LIVE_MENUS, 'native destruction did not release retained menus'
    if mode == "helper-close":
        assert not parent.findChildren(type(ref())), 'closed menu still owned by parent'
    del parent
    gc.collect()
    assert ref() is None or sip.isdeleted(ref()), 'native menu leaked after destruction'
print('menu lifetime passed')
'''
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script, mode],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "menu lifetime passed" in result.stdout
