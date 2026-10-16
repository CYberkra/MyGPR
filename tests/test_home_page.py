"""Home page prerequisites and narrow-window readability."""
from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')

from ui.pages.home_page import HomePage


@pytest.fixture
def page(qapp):
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
    assert page._content.width() <= 760
    assert page.mini_jobs() is None
