"""Persist user column preferences without saving automatic ledger layout."""
from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QHeaderView, QTableWidget


class ProjectColumnPreferences:
    """Own failed-save retry and legacy migration state for one project page.

    The page supplies its current settings manager and legacy factory at each
    call; this object holds no widget, manager or native settings handle alive.
    """

    def __init__(self) -> None:
        self._save_failed = False
        self._migration_keys: set[str] = set()

    @staticmethod
    def key(table_name: str) -> str:
        return f'ui/project_page/{table_name}_column_widths'

    def save(self, table: QTableWidget, table_name: str, sm, legacy_factory, *,
             logical_index: int | None = None) -> None:
        header = table.horizontalHeader()
        # Stretch sections change during page show/layout and file-tree width
        # restoration. They are layout output, not a user column preference.
        if (logical_index is not None
                and header.sectionResizeMode(logical_index) != QHeaderView.ResizeMode.Interactive):
            return
        widths = [int(header.sectionSize(i)) for i in range(table.columnCount())]
        if sm is None:
            return
        key = self.key(table_name)
        if (sm.get(key) == widths and not self._save_failed
                and not self._migration_keys):
            return
        sm.set(key, widths)
        self._save_failed = not sm.save()
        if not self._save_failed:
            self._clear_migrated(legacy_factory)

    def _clear_migrated(self, legacy_factory) -> None:
        # One successful settings save commits every pending table, not just
        # the table whose signal triggered it. Never replay a stale legacy key.
        if self._migration_keys:
            legacy = legacy_factory('MyGPR', 'MyGPR')
            for key in self._migration_keys:
                legacy.remove(key)
            self._migration_keys.clear()

    def restore(self, table: QTableWidget, table_name: str, sm, legacy_factory) -> None:
        """Restore widths, migrating valid legacy settings after one save."""
        key = self.key(table_name)
        widths = None
        migrate = False
        legacy = legacy_factory('MyGPR', 'MyGPR')
        stored = legacy.value(key)
        if isinstance(stored, list):
            widths = [int(w) for w in stored
                      if isinstance(w, (int, float)) and int(w) > 0]
            migrate = sm is not None
        elif sm is not None:
            stored = sm.get(key)
            if isinstance(stored, list):
                widths = [int(w) for w in stored
                          if isinstance(w, (int, float)) and int(w) > 0]
        if not widths:
            return
        header = table.horizontalHeader()
        # Replaying preferences must not save each intermediate header state.
        with QSignalBlocker(header):
            for i, w in enumerate(widths):
                if 0 <= i < table.columnCount():
                    header.resizeSection(i, w)
        if migrate:
            self._migration_keys.add(key)
            self.save(table, table_name, sm, legacy_factory)
