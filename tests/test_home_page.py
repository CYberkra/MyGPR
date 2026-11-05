"""Home page prerequisites and narrow-window readability."""
from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')

from ui.pages.home_page import HomePage
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest


@pytest.fixture
def probes(monkeypatch):
    commands = []
    monkeypatch.setattr('ui.controllers.recent_projects_controller.run_command',
                        lambda command, **kwargs: commands.append(command))
    return commands


@pytest.fixture
def page(qapp, probes):
    page = HomePage()
    yield page
    page.close()
    page.deleteLater()


def test_import_requires_writable_project(page):
    assert not page.import_btn.isEnabled()
    page.set_current_project(SimpleNamespace(read_only=False))
    assert page.import_btn.isEnabled()
    page.set_current_project(SimpleNamespace(read_only=True))
    assert not page.import_btn.isEnabled()
    page.set_current_project(None)
    assert not page.import_btn.isEnabled()


def test_narrow_startup_keeps_primary_actions_readable(page, qapp):
    page.resize(600, 850)
    page.show()
    qapp.processEvents()
    assert page.open_btn.isVisible()
    assert page.new_btn.isVisible()
    assert page.rect().contains(page.open_btn.mapTo(page, page.open_btn.rect().center()))
    page.resize(1100, 850)
    qapp.processEvents()
    assert page._details.maximumWidth() == 250
    assert page._content.width() <= page.viewport().width()
    assert page.mini_jobs() is None


@pytest.mark.parametrize('activation', ['button', 'double_click', 'enter'])
def test_selection_does_not_open_and_activation_sends_one_request(page, qapp, tmp_path, activation):
    path = str(tmp_path)
    page.set_recent_projects([path])
    page.resize(1200, 800)
    page.show()
    qapp.processEvents()
    opened = []
    page.open_recent_project_requested.connect(opened.append)
    row = page._rows[path][1]
    QTest.mouseClick(row, Qt.MouseButton.LeftButton)
    assert page._selected_path == path
    assert page._detail_path.text() == path
    assert opened == []
    if activation == 'button':
        page.detail_open_btn.click()
    elif activation == 'double_click':
        QTest.mouseDClick(row, Qt.MouseButton.LeftButton)
    else:
        QTest.keyClick(page._recent_list, Qt.Key.Key_Return)
    assert opened == [path]


def test_search_and_visible_pin_preserve_identity_and_settings(page, tmp_path):
    from ui.settings_manager import SettingsManager
    first, second = str(tmp_path / 'Alpha'), str(tmp_path / 'Beta')
    settings = SettingsManager(str(tmp_path / 'settings.json'))
    settings.set('recent_projects', [first, second])
    settings.set('project_display_names', {second: '校园复测'})
    page.set_settings_manager(settings)
    opened = []
    page.open_recent_project_requested.connect(opened.append)
    page._select_path(second)
    page._rows[second][1].pin_button.click()
    assert page._selected_path == second
    assert page._recent_list.item(0).data(Qt.ItemDataRole.UserRole) == second
    reloaded = SettingsManager(str(tmp_path / 'settings.json'))
    assert reloaded.get('pinned_recent_projects') == [second]
    assert reloaded.get('recent_projects') == [first, second]
    page.refresh_recents()
    page._toggle_pin(first)
    assert settings.get('recent_projects') == [first, second]
    page._toggle_pin(first)
    page.search.setText('校园')
    assert page._recent_list.count() == 1
    assert page._selected_path == second
    page.search.setText('ALPHA')
    assert page._selected_path == first
    assert page._recent_list.count() == 1
    page.search.setText('no-match')
    assert page._recent_list.count() == 0
    assert page._details.isHidden()
    assert page._recent_hint.text() == '没有匹配的项目'
    page._empty_action.click()
    assert page.search.text() == ''
    page.pinned_btn.click()
    assert page._selected_path == second
    assert page._recent_list.count() == 1
    assert opened == []


def test_busy_and_backend_gates_all_open_entry_points(page, tmp_path):
    path = str(tmp_path)
    page.set_recent_projects([path])
    opened, dialogs = [], []
    page.open_recent_project_requested.connect(opened.append)
    page.open_project_requested.connect(lambda: dialogs.append('open'))
    page.new_project_requested.connect(lambda: dialogs.append('new'))
    page.set_backend_ready(False)
    page.detail_open_btn.click()
    page.open_btn.click()
    page.new_btn.click()
    page._request_open(path)
    assert opened == dialogs == []
    page.set_backend_ready(True)
    page.open_recent_project_requested.connect(lambda _path: page.set_busy(True))
    page.detail_open_btn.click()
    assert page.detail_open_btn.text() == '正在打开…'
    page._request_open(path)
    page.open_btn.click()
    page.new_btn.click()
    assert opened == [path]
    assert dialogs == []
    page.set_busy(False)
    page.set_startup_error('项目格式不受支持')
    assert page.detail_open_btn.isEnabled()
    assert page._startup_error.text() == '项目格式不受支持'


def test_directory_results_are_deferred_and_removed_paths_cannot_return(page, probes, tmp_path):
    missing = str(tmp_path / 'missing')
    existing = str(tmp_path)
    page.set_recent_projects([missing, existing])
    assert '检查' in page._detail_state.text()
    probes.pop(0).execute()
    assert '目录不存在' in page._detail_state.text()
    assert page.detail_open_btn.text() == '重试打开'
    page._remove_btn.click()
    assert page._selected_path == existing
    assert page._detail_state.isHidden()
    page._on_availability(missing, 'late error')
    assert missing not in page._availability
    assert page._recent_list.count() == 1


def test_restore_skips_missing_and_opens_first_accessible_once(page, probes, tmp_path):
    missing, existing = str(tmp_path / 'missing'), str(tmp_path)
    opened = []
    page.open_recent_project_requested.connect(opened.append)
    page.set_recent_projects([missing, existing])
    page.restore_last_project([missing, existing])
    assert opened == []
    # The first request was superseded; only its completion starts the latest.
    probes.pop(0).execute()
    probes.pop(0).execute()
    assert opened == [existing]


@pytest.mark.parametrize('action', ['search', 'select', 'dialog', 'close'])
def test_manual_interaction_or_shutdown_prevents_late_auto_restore(page, probes, tmp_path, action):
    path = str(tmp_path)
    opened = []
    page.open_recent_project_requested.connect(opened.append)
    page.set_recent_projects([path])
    page.restore_last_project([path])
    if action == 'search':
        page.search.setText('no-match')
    elif action == 'select':
        page._select_path(path)
    elif action == 'dialog':
        page.cancel_pending_restore()
    else:
        page.shutdown()
    while probes:
        probes.pop(0).execute()
    assert opened == []


@pytest.mark.parametrize('dark', [False, True])
@pytest.mark.parametrize('width', [360, 768, 1280])
def test_theme_and_responsive_layout_keep_actions_and_identity_readable(page, qapp, tmp_path, dark, width):
    from ui.design_tokens import color
    path = str(tmp_path / ('长目录名称' * 35))
    page.set_recent_projects([path])
    page.apply_theme(dark)
    page.resize(width, 1000)
    page.show()
    qapp.processEvents()
    assert page.horizontalScrollBar().maximum() == 0
    assert page.open_btn.width() >= page.open_btn.minimumSizeHint().width()
    assert page._detail_path.text() == path
    assert color('bg_surface', dark) in page._content.styleSheet()
    assert color('text_primary', dark) in page._rows[path][1].styleSheet()
    assert page._rows[path][1].pin_button.accessibleName()
    # Filtering recreates rows: the explicitly selected palette must survive.
    page.search.setText('长目录')
    assert color('text_primary', dark) in page._rows[path][1].styleSheet()


def test_keyboard_menu_and_pointer_menu_share_actions(page, qapp, tmp_path):
    path = str(tmp_path)
    page.set_recent_projects([path])
    page.show()
    qapp.processEvents()
    page._menu_binding.show_keyboard()
    menu = page._menu_binding.menu
    assert [a.text() for a in menu.actions() if not a.isSeparator()] == [
        '打开项目', '固定项目', '复制项目路径', '从最近列表移除', '永久删除项目…']
    page._menu_binding.close()


@pytest.mark.parametrize('menu_kind', ['project', 'context'])
def test_project_menu_cancels_restore_before_pin_order_can_replace_history_order(
        page, probes, tmp_path, qapp, menu_kind):
    from ui.settings_manager import SettingsManager
    first, pinned = str(tmp_path / 'first'), str(tmp_path / 'pinned')
    (tmp_path / 'first').mkdir()
    (tmp_path / 'pinned').mkdir()
    settings = SettingsManager(str(tmp_path / 'settings.json'))
    settings.set('recent_projects', [first, pinned])
    settings.set('pinned_recent_projects', [pinned])
    page.set_settings_manager(settings)
    opened = []
    page.open_recent_project_requested.connect(opened.append)
    page.restore_last_project([first, pinned])
    if menu_kind == 'project':
        menu = page.project_menu(page.open_btn, can_close=False, close_requested=lambda: None)
    else:
        page.show()
        qapp.processEvents()
        page._menu_binding.show_keyboard()
        menu = page._menu_binding.menu
    while probes:
        probes.pop(0).execute()
    assert opened == []
    if menu_kind == 'context':
        page._menu_binding.close()
    else:
        menu.close()


@pytest.mark.parametrize('dark', [False, True])
def test_row_label_palette_preserves_name_path_hierarchy(page, qapp, tmp_path, dark):
    from PyQt6.QtGui import QColor, QPalette
    from qfluentwidgets import Theme, qconfig, setTheme
    from ui.design_tokens import color
    previous = qconfig.theme
    try:
        setTheme(Theme.DARK if dark else Theme.LIGHT)
        path = str(tmp_path)
        page.set_recent_projects([path])
        page.apply_theme(dark)
        page.show()
        qapp.processEvents()
        row = page._rows[path][1]
        for label, token in ((row.name_label, 'text_primary'), (row.path_label, 'text_muted')):
            label.ensurePolished()
            assert label.palette().color(QPalette.ColorRole.WindowText) == QColor(color(token, dark))
    finally:
        setTheme(previous)
