"""Virtualized cluster membership; no thousands-of-actions popup menus."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDialog, QLineEdit, QTableView, QVBoxLayout, QAbstractItemView
from ui.widgets.marker_panel import MarkerTableModel, MarkerFilter


class MarkerCluster(QDialog):
    selected = pyqtSignal(str)

    def __init__(self, rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'{len(rows)} 条标记')
        self.resize(520, 360)
        layout = QVBoxLayout(self)
        search = QLineEdit(self)
        search.setPlaceholderText('搜索名称或备注')
        layout.addWidget(search)
        self.model = MarkerTableModel(self)
        self.model.set_rows(rows)
        self.proxy = MarkerFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView(self)
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().hide()
        layout.addWidget(self.table)
        search.textChanged.connect(self._filter)
        self.table.clicked.connect(self._choose)
        self.table.activated.connect(self._choose)

    def _filter(self, text):
        self.proxy.query = text.strip().casefold()
        self.proxy.invalidateFilter()

    def _choose(self, index):
        row = index.data(Qt.ItemDataRole.UserRole)
        if row:
            self.selected.emit(row['uid'])
            self.accept()
