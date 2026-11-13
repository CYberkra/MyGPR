"""Shared marker list and editor wiring for grouped or standalone B-scans."""
from dataclasses import replace

from PyQt6.QtCore import QEvent, QObject, pyqtSignal

from ui.desktop_backend_facade import EventInterval, MarkerRecord, marker_rows_for_view
from ui.widgets.context_menus import add_action, make_menu, show_menu
from ui.widgets.marker_panel import MarkerPanel


class MarkerWorkspaceTools(QObject):
    operation = pyqtSignal(str, object)
    selection_changed = pyqtSignal(str, str, str)
    navigate = pyqtSignal(str, object)

    def __init__(self, host, views, active, *, allow_processing=False):
        super().__init__(host)
        self.allow_processing = allow_processing
        self.host, self.views, self.active = host, views, active
        self.panel = None
        self.documents = {}
        self._selecting = False
        host.installEventFilter(self)

    def connect_view(self, view):
        view.sig_marker_context_changed.connect(lambda: self.sync_view(view))
        view.sig_event_marker_selected.connect(lambda uid: self._selected(view, uid))
        view.marker_strip.edit_requested.connect(lambda uid: self.edit_id(view, uid))
        view.marker_strip.menu_requested.connect(lambda uid, pos: self.record_menu(view, uid, pos))
        view.marker_strip.range_created.connect(lambda a, b, uid: self.range_created(view, a, b, uid))
        view._canvas_menu_sections.append(lambda menu, pos: self.add_canvas_action(menu, view, pos))

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

    def _ensure_panel(self):
        if self.panel is None:
            self.panel = MarkerPanel(self.host, allow_processing=self.allow_processing)
            if self.allow_processing:
                self.panel.processing_source = self._active_marker_view
            self.panel.operation.connect(self.operation)
            self.panel.selected.connect(self._panel_selected)
            self.panel.locate.connect(lambda uid: self._panel_selected(uid, locate=True))
            self.panel.adjust.connect(lambda uid: self.active().marker_strip.begin_adjust(uid))
            self.panel.close_requested.connect(self.hide_panel)
            self.panel.navigate.connect(self.navigate)

    def _active_marker_view(self):
        view = self.active()
        return view._marker_view if view else None

    def show_panel(self, view=None):
        self._ensure_panel()
        view = view or self.active()
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
        menu = make_menu(self.host)
        visibility = add_action(menu, None, '显示图上标记', view.set_acquisition_markers_visible)
        visibility.setCheckable(True)
        visibility.setChecked(view._acquisition_markers_visible)
        add_action(menu, None, '打开标记列表', lambda: self.show_panel(view))
        current = view._marker_view
        editable = bool(current and not current.read_only)
        add_action(menu, None, '新增点标记…', lambda: self._create_point(view, current), enabled=editable)
        add_action(menu, None, '新建区间…', lambda: view.marker_strip.begin_range()
                   if current is view._marker_view else None, enabled=editable)
        show_menu(menu, position)

    def _create_point(self, view, context, trace=None):
        if context is not None and view._marker_view is context and not context.read_only:
            self.show_panel(view).edit(MarkerRecord(anchor=context.anchor(trace) if trace is not None else None), context)

    def add_canvas_action(self, menu, view, position):
        context = view._marker_view
        if not context or view._image_shape is None:
            return
        trace = None
        if position is not None:
            point = view._plot.vb.mapSceneToView(position)
            x, y = point.x(), point.y()
            width, height = view._image_shape
            if not (0 <= x < width and 0 <= y < height):
                return
            trace = int(view._trace_indices[int(x)]) if view._trace_indices is not None else int(x)
            if not 0 <= trace < context.count:
                return
        add_action(menu, None, '在此道新增点标记…' if trace is not None else '新增点标记…',
                   lambda: self._create_point(view, context, trace), enabled=not context.read_only)

    def record_menu(self, view, uid, position):
        context = view._marker_view
        row = next((item for item in context.rows if item['uid'] == uid), None) if context else None
        if row:
            self._ensure_panel()
            menu = self.panel.record_menu(row, context, lambda: view._marker_view is context,
                locate=lambda: view.select_event_marker(uid, locate=True),
                adjust=lambda: view.marker_strip.begin_adjust(uid))
            show_menu(menu, position)

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
                         time_start='', time_end='')
        self.show_panel().edit(record, current)

    def deliver(self, token, result, error):
        if self.panel:
            self.panel.deliver(token, result, error)
