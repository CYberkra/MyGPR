"""Virtualized cluster membership; no thousands-of-actions popup menus."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDialog, QLineEdit, QVBoxLayout, QAbstractItemView
from qfluentwidgets import isDarkTheme
from ui.widgets.marker_panel import MarkerTableModel, MarkerTableView, MarkerFilter
from ui.widgets.marker_style import marker_style


class MarkerCluster(QDialog):
    selected = pyqtSignal(str)

    def __init__(self, rows, parent=None):
        super().__init__(parent)
        self.setObjectName('markerEditor')
        self.setWindowTitle(f'{len(rows)} 条标记')
        available = self.screen().availableGeometry()
        self.resize(min(520, available.width() - 32), min(360, available.height() - 48))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        search = QLineEdit(self)
        search.setPlaceholderText('搜索名称或备注')
        search.setClearButtonEnabled(True)
        layout.addWidget(search)
        self.model = MarkerTableModel(self)
        self.model.set_rows(rows)
        self.proxy = MarkerFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table = MarkerTableView(self)
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().hide()
        layout.addWidget(self.table)
        search.textChanged.connect(self._filter)
        self.table.clicked.connect(self._choose)
        self.table.activated.connect(self._choose)
        self.apply_theme(isDarkTheme())

    def apply_theme(self, dark):
        self.setStyleSheet(marker_style(dark))

    def _filter(self, text):
        self.proxy.query = text.strip().casefold()
        self.proxy.invalidateFilter()

    def _choose(self, index):
        row = index.data(Qt.ItemDataRole.UserRole)
        if row:
            self.selected.emit(row['uid'])
            self.accept()
