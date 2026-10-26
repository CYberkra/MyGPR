"""Native menu lifecycle must remain safe across repeated context requests."""
from types import SimpleNamespace as NS

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent

from ui.widgets.file_tree_panel import FileTreePanel


@pytest.fixture
def live_panel(qapp):
    panel = FileTreePanel()
    panel.set_project_info(NS(name='Menu lifetime', project_id='menu-test', root_path=''))
    panel.set_lines([NS(line_id='L01', name='L01', updated_at='', length_m=0,
                        processing_status='', sample_count=72, trace_count=48)])
    panel.resize(400, 600)
    panel.show()
    qapp.processEvents()
    yield panel
    panel.close()
    panel.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def open_line_menu(panel):
    leaf = panel._line_id_by_item['L01']
    panel._on_context_menu(panel._tree.visualItemRect(leaf).center())
    return panel._context_menu


def test_closed_menu_does_not_poison_next_right_click(qapp, live_panel):
    panel = live_panel
    for _ in range(3):
        menu = open_line_menu(panel)
        qapp.processEvents()
        assert menu is not None and menu.isVisible()
        assert any(action.text() == '打包处理结果…' for action in menu.actions())
        menu.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(menu)
        assert panel._context_menu is None


def test_direct_qt_deletion_is_safe_even_without_a_close_signal(qapp, live_panel):
    panel = live_panel
    menu = open_line_menu(panel)
    menu.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert sip.isdeleted(menu)
    assert panel._context_menu is None
    qapp.processEvents()  # Consume popup focus changes before the next user request.
    # Equivalent to another pointer/keyboard context-menu request.
    next_menu = open_line_menu(panel)
    qapp.processEvents()
    assert next_menu is not None and next_menu.isVisible()


def test_late_old_menu_destruction_cannot_clear_a_new_menu(qapp, live_panel):
    panel = live_panel
    first = open_line_menu(panel)
    first.close()
    second = open_line_menu(panel)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert sip.isdeleted(first)
    assert panel._context_menu is second
    assert not sip.isdeleted(second)
