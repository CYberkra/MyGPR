"""Object menus retain identity and never borrow a stale blank-area selection."""
from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PyQt6.QtWidgets import QApplication


@pytest.fixture
def pages(qapp):
    created = []

    def make(page_type):
        page = page_type()
        created.append(page)
        return page

    yield make
    for page in created:
        page.close()
        page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def action(menu, name):
    return next(entry for entry in menu.actions() if entry.data() == name)


def test_interpretation_point_menu_does_not_pick_and_revalidates_snapshot(pages):
    from ui.pages.interpretation_page import InterpretationPage
    page = pages(InterpretationPage)
    page.set_session_active(True)
    page.set_points([(2, 8), (5, 13)])
    table = page._points_table
    table.selectRow(0)
    point = table.visualItemRect(table.item(1, 0)).center()
    changed = []
    page.points_changed.connect(changed.append)
    menu = page._point_menu(point)
    assert page._points == [(2, 8), (5, 13)]
    assert table.currentRow() == 0
    page.set_points([(7, 17)])
    action(menu, 'remove-point').trigger()
    assert page._points == [(7, 17)]
    assert changed == []


def test_interpretation_remove_uses_hit_point_and_blank_menu_has_no_old_target(pages):
    from ui.pages.interpretation_page import InterpretationPage
    page = pages(InterpretationPage)
    page.set_session_active(True)
    page.set_points([(2, 8), (5, 13)])
    table = page._points_table
    table.selectRow(0)
    point = table.visualItemRect(table.item(1, 0)).center()
    action(page._point_menu(point), 'remove-point').trigger()
    assert page._points == [(2, 8)]
    blank = page._point_menu(QPoint(-10, -10))
    assert all(entry.data() != 'remove-point' for entry in blank.actions())
    page.set_busy(True)
    locked = page._point_menu(table.visualItemRect(table.item(0, 0)).center())
    assert not action(locked, 'remove-point').isEnabled()
    assert '忙' in action(locked, 'remove-point').text()


def test_spatial_line_menu_visibility_is_display_only_and_id_is_frozen(pages):
    from ui.pages.spatial_page import SpatialPage
    page = pages(SpatialPage)
    page.set_auto_prefetch_enabled(False)
    page.set_tracks([SimpleNamespace(line_id='L1', name='One', points=()),
                     SimpleNamespace(line_id='L2', name='Two', points=())])
    listing = page._line_list
    listing.setCurrentRow(0)
    menu = page._line_menu(listing.visualItemRect(listing.item(1)).center())
    selected = []
    page.current_line_requested.connect(selected.append)
    assert listing.currentRow() == 0
    action(menu, 'toggle-visible').trigger()
    assert listing.item(1).checkState() == Qt.CheckState.Unchecked
    assert selected == []
    action(menu, 'set-current').trigger()
    assert selected == ['L2']
    page.set_tracks([])
    action(menu, 'set-current').trigger()
    assert selected == ['L2']
    assert all(a.data() != 'set-current' for a in page._line_menu(QPoint(-1, -1)).actions())


def test_delivery_result_menu_has_stable_id_and_truthful_unsupported_actions(pages):
    from ui.pages.delivery_page import DeliveryPage
    page = pages(DeliveryPage)
    page.set_spatial_results([dict(result_id='surface-v1', name='Surface', line_ids=['L1'])])
    table = page._spatial_table
    menu = page._spatial_result_menu(table.visualItemRect(table.item(0, 0)).center())
    action(menu, 'copy-id').trigger()
    assert QApplication.clipboard().text() == 'surface-v1'
    assert not action(menu, 'open-result').isEnabled()  # 未提供项目根目录时不可打开
    assert action(menu, 'export-result').isEnabled()
    assert action(menu, 'delete-result').isEnabled()
    assert action(menu, 'set-current-result').isEnabled()
    page.set_spatial_results([dict(result_id='surface-v2', name='Other')])
    QApplication.clipboard().setText('unchanged')
    action(menu, 'copy-id').trigger()
    assert QApplication.clipboard().text() == 'unchanged'
    blank = page._spatial_result_menu(QPoint(-1, -1))
    assert all(a.data() != 'delete-result' for a in blank.actions())


def test_settings_text_keeps_native_menu_policy(pages):
    from ui.pages.settings_page import SettingsPage

    settings = pages(SettingsPage)
    assert settings._root_edit.contextMenuPolicy() == Qt.ContextMenuPolicy.DefaultContextMenu
    native = settings._root_edit.createStandardContextMenu()
    assert native.actions()
    native.deleteLater()
