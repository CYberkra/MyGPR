"""Project-wide marker browser used by project, spatial and delivery pages."""
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QPushButton, QVBoxLayout

from ui.widgets.marker_panel import MarkerPanel


class MarkerBrowserButton(QPushButton):
    operation = pyqtSignal(str, object)
    navigate = pyqtSignal(str, object)
    selection_changed = pyqtSignal(str, str, str)

    def __init__(self, parent=None):
        super().__init__('标记与区间', parent)
        self.project_id = ''
        self.dialog = None
        self.panel = None
        self._selecting = False
        self._selected_id = ''
        self.clicked.connect(self.open_browser)

    def open_browser(self):
        if self.dialog is None:
            self.dialog = QDialog(self.window())
            self.dialog.setWindowTitle('项目标记')
            available = self.screen().availableGeometry()
            self.dialog.resize(min(650, available.width() - 32), min(620, available.height() - 48))
            layout = QVBoxLayout(self.dialog)
            self.panel = MarkerPanel(self.dialog, embedded=False)
            layout.addWidget(self.panel)
            self.panel.operation.connect(self.operation)
            self.panel.navigate.connect(self._navigate)
            self.panel.selected.connect(self._selected)
            self.panel.locate.connect(self._locate)
        self.dialog.show()
        self.dialog.raise_()
        if self.project_id:
            self.panel.request('project', {'project_id': self.project_id})

    def _selected(self, uid):
        if self._selecting:
            return
        row = next((row for row in self.panel.model.rows if row['uid'] == uid), None)
        if row:
            self.selection_changed.emit(self.project_id, row['line_id'], uid)

    def _locate(self, uid):
        row = next((row for row in self.panel.model.rows if row['uid'] == uid), None)
        if row:
            self._navigate('processing', row)

    def _navigate(self, page, row):
        if self.dialog:
            self.dialog.hide()
        self.navigate.emit(page, row)

    def deliver(self, token, result, error):
        if self.panel:
            self.panel.deliver(token, result, error)
            if self._selected_id:
                self.select(self._selected_id)

    def select(self, uid, *, project_id='', line_id=''):
        if project_id and project_id != self.project_id:
            return
        self._selected_id = uid
        if self.panel and not self._selecting:
            self._selecting = True
            try:
                self.panel.select_id(uid)
            finally:
                self._selecting = False

    def changed(self, updated):
        if self.panel and self.panel.isVisible() and updated.document.project_id == self.project_id:
            self.panel.request('project', {'project_id': self.project_id})

    def set_project(self, project_id):
        self.project_id = project_id
        self._selected_id = ''
        self.setEnabled(bool(project_id))
        if self.panel:
            self.panel.set_view(None)
            self.panel._requests.clear()
        if self.dialog:
            self.dialog.hide()
