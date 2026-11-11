"""Shared marker list and editor wiring for grouped or standalone B-scans."""
from dataclasses import replace

from PyQt6.QtCore import QEvent, QObject, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMenu, QMessageBox

from ui.desktop_backend_facade import EventInterval, MarkerRecord, marker_rows_for_view
from ui.widgets.marker_panel import MarkerPanel


class MarkerWorkspaceTools(QObject):
    operation = pyqtSignal(str, object)
    selection_changed = pyqtSignal(str, str, str)
    navigate = pyqtSignal(str, object)

    def __init__(self, host, views, active):
        super().__init__(host)
        self.host, self.views, self.active = host, views, active
        self.panel = None
        self.documents = {}
        self._selecting = False
        host.installEventFilter(self)

    def connect_view(self, view):
        view.sig_marker_context_changed.connect(lambda: self.sync_view(view))
        view.sig_event_marker_selected.connect(lambda uid: self._selected(view, uid))
        view.marker_strip.edit_requested.connect(lambda uid: self.edit_id(view, uid))
        view.marker_strip.range_created.connect(lambda a, b, uid: self.range_created(view, a, b, uid))

    def sync_view(self, view):
        current = view._marker_view
        if current:
            cached = self.documents.get((current.document.project_id, current.document.line_id))
            if cached and cached.source_id == current.document.source_id and (
                    cached.revision > current.document.revision or cached.timezone != current.timezone):
                updated = replace(current, document=cached, timezone=cached.timezone)
                view.set_marker_view(replace(updated, rows=marker_rows_for_view(updated)))
        if self.panel and view is self.active():
            self.panel.set_view(view._marker_view)

    def changed(self, updated):
        document = updated.document
        self.documents[(document.project_id, document.line_id)] = document
        for view in self.views():
            self.sync_view(view)

    def refresh_links(self, contexts):
        contexts = {(item.document.project_id, item.document.line_id): item for item in contexts}
        for view in self.views():
            current = view._marker_view
            key = (current.document.project_id, current.document.line_id) if current else None
            updated = contexts.get(key)
            if current and updated and (current.interpretation_links != updated.interpretation_links
                                       or current.timezone != updated.timezone
                                       or current.processing_history != updated.processing_history):
                document = replace(current.document, timezone=updated.timezone)
                view.set_marker_view(replace(current, interpretation_links=updated.interpretation_links,
                                             document=document, timezone=updated.timezone,
                                             processing_history=updated.processing_history))

    def show_panel(self):
        if self.panel is None:
            self.panel = MarkerPanel(self.host)
            self.panel.operation.connect(self.operation)
            self.panel.selected.connect(self._panel_selected)
            self.panel.locate.connect(lambda uid: self._panel_selected(uid, locate=True))
            self.panel.adjust.connect(lambda uid: self.active().marker_strip.begin_adjust(uid))
            self.panel.close_requested.connect(self.hide_panel)
            self.panel.navigate.connect(self.navigate)
        view = self.active()
        self.panel.set_view(view._marker_view if view else None)
        self.panel.show()
        self._geometry()
        self.panel.raise_()
        return self.panel

    def _panel_selected(self, uid, *, locate=False):
        if self._selecting:
            return
        row = next((row for row in self.panel.model.rows if row['uid'] == uid), None)
        if row is None:
            return
        context = self.panel._views.get(row['line_id'])
        if context is None:
            return
        active = self.active()
        if locate and (not active or not active._marker_view or active._marker_view.document.line_id != row['line_id']):
            self.navigate.emit('processing', row)
            return
        pid = context.document.project_id
        self.select(uid, locate=locate, project_id=pid, line_id=row['line_id'])
        self.selection_changed.emit(pid, row['line_id'], uid)

    def hide_panel(self):
        self.panel.hide()
        self.host.layout().setContentsMargins(0, 0, 0, 0)

    def _geometry(self):
        if self.panel and self.panel.isVisible():
            width = min(320, self.host.width())
            self.host.layout().setContentsMargins(0, 0, width if self.host.width() >= 900 else 0, 0)
            self.panel.setGeometry(self.host.width() - width, 0, width, self.host.height())

    def eventFilter(self, watched, event):
        if watched is self.host and event.type() == QEvent.Type.Resize:
            self._geometry()
        return False

    def menu(self, view, position):
        menu = QMenu(self.host)
        toggle = menu.addAction('显示标记条')
        toggle.setCheckable(True)
        toggle.setChecked(view._acquisition_markers_visible)
        toggle.toggled.connect(view.set_acquisition_markers_visible)
        menu.addAction('标记列表', self.show_panel)
        current = view._marker_view
        editable = bool(current and not current.read_only)
        menu.addAction('新增点标记', lambda: self.show_panel().edit(MarkerRecord(anchor=current.anchor(0)))).setEnabled(editable)
        menu.addAction('新建区间', view.marker_strip.begin_range).setEnabled(editable)
        menu.addAction('导出带标记图件', lambda: self.export_figure(view)).setEnabled(bool(current))
        menu.exec(position)

    def export_figure(self, view):
        path, _ = QFileDialog.getSaveFileName(self.host, '导出带标记图件', 'markers.png', 'PNG (*.png)')
        if path and not view.grab().save(path, 'PNG'):
            QMessageBox.warning(self.host, '导出失败', '图件未保存，请检查目标路径')

    def _selected(self, view, uid):
        if self._selecting or not view._marker_view:
            return
        document = view._marker_view.document
        self.selection_changed.emit(document.project_id, document.line_id, uid)

    def select(self, uid, *, locate=False, project_id='', line_id=''):
        if self._selecting:
            return
        self._selecting = True
        try:
            for view in self.views():
                current = view._marker_view
                if current and (not project_id or current.document.project_id == project_id) and (
                        not line_id or current.document.line_id == line_id):
                    view.select_event_marker(uid, locate=locate and view is self.active())
            if self.panel:
                self.panel.select_id(uid)
        finally:
            self._selecting = False
        active = self.active()
        if not project_id and active and active._marker_view:
            doc = active._marker_view.document
            self.selection_changed.emit(doc.project_id, doc.line_id, uid)

    def edit_id(self, view, uid):
        row = next((row for row in view._marker_view.rows if row['uid'] == uid), None)
        if row:
            self.show_panel().edit(row['record'], view._marker_view)

    def range_created(self, view, start, end, uid):
        current = view._marker_view
        if not current:
            return
        previous = next((row['record'] for row in current.rows if row['uid'] == uid), EventInterval())
        # A manual boundary adjustment replaces the point references explicitly.
        record = replace(previous, start=current.anchor(start), end=current.anchor(end),
                         start_marker_id='', end_marker_id='', start_marker_revision=0, end_marker_revision=0,
                         time_start='', time_end='', review_state='unreviewed')
        self.show_panel().edit(record, current)

    def deliver(self, token, result, error):
        if self.panel:
            self.panel.deliver(token, result, error)
