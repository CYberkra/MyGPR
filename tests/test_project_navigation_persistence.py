"""Layout-driven header changes must not synchronously persist preferences."""
from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QHeaderView

from ui.pages.project_page import ProjectPage
from ui.settings_manager import SettingsManager


@pytest.fixture
def ledger(qapp, monkeypatch, tmp_path):
    # No legacy application/user registry participates in these isolated cases.
    class EmptyLegacy:
        def __init__(self, *_args):
            pass

        def value(self, _key):
            return None

    monkeypatch.setattr('ui.pages.project_page.QSettings', EmptyLegacy)
    page = ProjectPage()
    manager = SettingsManager(str(tmp_path / 'preferences.json'))
    page.set_settings_manager(manager)
    page.set_project_info(SimpleNamespace(project_id='P1', name='Navigation'))
    page.set_lines([SimpleNamespace(line_id='L01', name='Line 1')])
    page.resize(1000, 650)
    page.show()
    qapp.processEvents()
    writes = []
    save = manager.save

    def record_save():
        writes.append(manager.get_all())
        return save()

    monkeypatch.setattr(manager, 'save', record_save)
    yield page, manager, writes
    page.close()
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize('table_name,stretch', [('lines', 1), ('artifacts', 0)])
def test_layout_resize_and_page_show_do_not_write_preferences(ledger, qapp, table_name, stretch):
    page, _manager, writes = ledger
    if table_name == 'artifacts':
        page._detail_tabs.setCurrentIndex(2)
        qapp.processEvents()
        writes.clear()
    header = getattr(page, f'_{table_name}_table').horizontalHeader()
    assert header.sectionResizeMode(stretch) == QHeaderView.ResizeMode.Stretch
    changed = []
    header.sectionResized.connect(lambda index, _old, _new: changed.append(index))
    for width in (790, 1100, 910):
        page.hide()
        page.resize(width, 650)
        page.show()
        qapp.processEvents()
    assert stretch in changed, 'scenario must exercise automatic header resizing'
    assert writes == [], 'navigation/layout must not fsync preference files'


@pytest.mark.parametrize('table_name,column', [('lines', 0), ('artifacts', 1)])
def test_user_header_drag_persists_and_restores_width(ledger, qapp, table_name, column):
    page, manager, writes = ledger
    if table_name == 'artifacts':
        page._detail_tabs.setCurrentIndex(2)
        qapp.processEvents()
        writes.clear()
    header = getattr(page, f'_{table_name}_table').horizontalHeader()
    previous = header.sectionSize(column)
    edge = header.sectionViewportPosition(column) + previous - 1
    QTest.mousePress(header.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(edge, 10))
    QTest.mouseMove(header.viewport(), QPoint(edge + 31, 10), delay=10)
    QTest.mouseRelease(header.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(edge + 31, 10))
    qapp.processEvents()
    assert header.sectionSize(column) != previous
    assert writes, 'a user adjustment must still reach durable settings'
    expected = header.sectionSize(column)
    loaded = SettingsManager(manager.settings_file)
    reopened = ProjectPage()
    try:
        reopened.set_settings_manager(loaded)
        restored = getattr(reopened, f'_{table_name}_table').horizontalHeader()
        assert restored.sectionSize(column) == expected
    finally:
        reopened.close()
        reopened.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_unchanged_column_preferences_do_not_write_again(ledger):
    page, _manager, writes = ledger
    page._save_column_widths(page._lines_table, 'lines')
    count = len(writes)
    page._save_column_widths(page._lines_table, 'lines')
    assert len(writes) == count


def test_failed_save_can_retry_the_same_widths(ledger, monkeypatch):
    page, manager, _writes = ledger
    outcomes = iter((False, True))
    attempts = []

    def save():
        attempts.append(True)
        return next(outcomes)

    monkeypatch.setattr(manager, 'save', save)
    page._save_column_widths(page._lines_table, 'lines')
    page._save_column_widths(page._lines_table, 'lines')
    assert len(attempts) == 2


def test_restoring_column_preferences_does_not_save_intermediate_widths(ledger, qapp):
    page, manager, writes = ledger
    widths = [100, 220, 75, 90, 99, 110, 125]
    manager.set(page._column_widths_key('lines'), widths)
    page._restore_column_widths(page._lines_table, 'lines')
    qapp.processEvents()
    assert writes == []
    assert page._lines_table.horizontalHeader().sectionSize(0) == 100


def test_failed_legacy_migration_then_other_save_keeps_latest_user_width(ledger, qapp, monkeypatch, tmp_path):
    from PyQt6.QtCore import QSettings

    page, manager, _writes = ledger
    legacy_path = str(tmp_path / 'legacy.ini')
    legacy = QSettings(legacy_path, QSettings.Format.IniFormat)
    for name, column, width in (('lines', 0, 111), ('artifacts', 1, 135)):
        header = getattr(page, f'_{name}_table').horizontalHeader()
        widths = [header.sectionSize(i) for i in range(header.count())]
        widths[column] = width
        legacy.setValue(page._column_widths_key(name), widths)
    legacy.sync()
    monkeypatch.setattr('ui.pages.project_page.QSettings',
                        lambda *_args: QSettings(legacy_path, QSettings.Format.IniFormat))
    save = manager.save
    attempts = []

    def fail_first_save():
        attempts.append(True)
        return False if len(attempts) == 1 else save()

    monkeypatch.setattr(manager, 'save', fail_first_save)
    page._restore_column_widths(page._lines_table, 'lines')
    assert legacy.contains(page._column_widths_key('lines'))
    # The second table succeeds, durably writing both tables' settings.
    page._restore_column_widths(page._artifacts_table, 'artifacts')
    header = page._lines_table.horizontalHeader()
    edge = header.sectionViewportPosition(0) + header.sectionSize(0) - 1
    QTest.mousePress(header.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(edge, 10))
    QTest.mouseMove(header.viewport(), QPoint(edge + 31, 10), delay=10)
    QTest.mouseRelease(header.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(edge + 31, 10))
    qapp.processEvents()
    latest = header.sectionSize(0)
    assert latest != 111
    assert SettingsManager(manager.settings_file).get(page._column_widths_key('lines'))[0] == latest
    legacy.sync()
    reopened = ProjectPage()
    try:
        reopened.set_settings_manager(SettingsManager(manager.settings_file))
        assert reopened._lines_table.horizontalHeader().sectionSize(0) == latest
        assert not legacy.contains(page._column_widths_key('lines'))
        assert not legacy.contains(page._column_widths_key('artifacts'))
    finally:
        reopened.close()
        reopened.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
