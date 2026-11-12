"""Algorithm search and continuous adding inside the processing tools."""
import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from tests.test_processing_workspace_flow import bundle
from ui.pages.processing_page import ProcessingPage


def _methods() -> list:
    return [
        {'method_id': 'dewow', 'name': 'dewow', 'display_name': '去直流滤波',
         'category': 'filter', 'category_label': '滤波', 'tags': [],
         'parameter_schema': []},
        {'method_id': 'agc', 'name': 'agc', 'display_name': '自动增益控制',
         'category': 'gain', 'category_label': '增益', 'tags': [],
         'parameter_schema': []},
    ]



@pytest.fixture
def page(qapp):
    value = ProcessingPage()
    value.set_line_label('L01')
    value.set_original_bundle(bundle())
    methods = _methods()
    methods[0]['parameter_schema'] = [
        {'name': 'window', 'label': '窗口', 'type': 'int', 'default': 5, 'min': 1, 'max': 100},
    ]
    value.set_methods(methods)
    value.resize(1100, 720)
    value.show()
    value.activateWindow()
    qapp.processEvents()
    yield value
    value.close()
    value.deleteLater()


@pytest.mark.parametrize('search', ['agc', '无匹配'])
def test_filter_clears_selection_and_rejects_hidden_method(page, search):
    page._open_method_picker()
    browser = page._method_browser
    browser.select_method('dewow')
    assert page._commit_step_btn.isEnabled()
    browser._search.setText(search)
    assert browser.current_method_id() is None
    assert not page._commit_step_btn.isEnabled()
    page._commit_step_form()
    browser.sig_add_requested.emit('dewow')
    assert page.current_pipeline()['steps'] == []
    browser._search.clear()
    assert browser.current_method_id() is None


def test_search_matches_names_and_restores_category_expansion(page):
    page._open_method_picker()
    browser = page._method_browser
    tree = browser._tree
    filter_group, gain_group = (tree.topLevelItem(i) for i in range(2))
    assert filter_group.isExpanded() and not gain_group.isExpanded()
    browser._search.setText('AGC')
    assert filter_group.isHidden() and not gain_group.isHidden() and gain_group.isExpanded()
    browser._search.setText('去直')
    assert not filter_group.isHidden() and gain_group.isHidden()
    browser._search.clear()
    assert filter_group.isExpanded() and not gain_group.isExpanded()


def test_inline_library_keeps_search_after_configured_add_and_default_enter(page, qapp):
    page._tools_btn.setChecked(False)
    page._chain_strip.add_button().click()
    browser = page._method_browser
    assert browser.window() is page and browser.isVisibleTo(page)
    assert page._tools_panel.currentWidget() is page._chain_scroll
    browser._search.setText('dewow')
    browser.select_method('dewow')
    page._param_form._editors['window'].setValue(23)
    page._commit_step_btn.click()
    assert browser.isVisibleTo(page)
    assert browser._search.text() == 'dewow'
    assert page._selected_step == 0
    assert page.current_pipeline()['steps'][0]['params'] == {'window': 23}
    browser._tree.setFocus()
    qapp.processEvents()
    QTest.keyClick(browser._tree, Qt.Key.Key_Return)
    assert [step['params'] for step in page.current_pipeline()['steps']] == [{'window': 23}, {'window': 5}]
    assert page._selected_step == 1 and browser.isVisibleTo(page)
    QTest.keyClick(browser._tree, Qt.Key.Key_Escape)
    assert browser.isHidden()
    assert page._editor_uid == page._pipeline_list.step_uid(1)


def test_add_requires_active_writable_input(qapp):
    page = ProcessingPage()
    try:
        assert not page._chain_strip.add_button().isEnabled()
        page._open_method_picker()
        assert page._method_browser.isHidden()
        page.set_line_label('L01')
        page.set_original_bundle(bundle())
        page.set_project_read_only(True)
        assert not page._chain_strip.add_button().isEnabled()
        page._open_method_picker()
        assert page._method_browser.isHidden()
    finally:
        page.close()
        page.deleteLater()


def test_refresh_live_library_preserves_filter_but_clears_removed_selection(page):
    page._open_method_picker()
    browser = page._method_browser
    browser._search.setText('dewow')
    browser.select_method('dewow')
    page.set_methods(_methods()[1:])
    assert browser._search.text() == 'dewow'
    assert browser.current_method_id() is None
    assert not page._commit_step_btn.isEnabled()
