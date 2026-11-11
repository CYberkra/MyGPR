"""One ID selection channel across project, B-scan and spatial presentations."""
from ui.controllers.marker_controller import MarkerController
from uuid import uuid4


class MarkerChain:
    def __init__(self, coordinator):
        self.co = coordinator
        self.tools = []
        self.controller = None
        self._project_token = ''

    def connect_all(self):
        co = self.co
        self.controller = MarkerController(co.backend_controller)
        processing = co.page('processingInterface')
        interpretation = co.page('interpretationInterface')
        candidates = [getattr(getattr(processing, '_workspace', None), 'marker_tools', None),
                      getattr(interpretation, 'marker_tools', None)]
        for name in ('projectInterface', 'spatialInterface', 'deliveryInterface'):
            candidates.append(getattr(co.page(name), 'marker_browser', None))
        self.tools = [item for item in candidates if item is not None]
        for item in self.tools:
            item.operation.connect(self.controller.request)
            item.navigate.connect(self.navigate)
            item.selection_changed.connect(self.select)
            self.controller.ready.connect(item.deliver)
            self.controller.changed.connect(item.changed)
        self.controller.artifact_saved.connect(self._artifact_saved)
        self.controller.changed.connect(lambda _: self.reload_project())
        self.controller.ready.connect(self._project_ready)
        co.project_controller.project_opened.connect(lambda summary: self._project(summary.project_id))
        co.project_controller.project_closed.connect(lambda: self._project(''))
        co.project_controller.lines_updated.connect(lambda _: self.reload_project())

    def _project(self, project_id):
        self._project_token = ''
        for item in self.tools:
            if hasattr(item, 'set_project'):
                item.set_project(project_id)
            else:
                item.documents.clear()
                if item.panel:
                    item.panel.set_view(None)
                    item.panel._requests.clear()
        if project_id:
            self.reload_project()
        spatial = self.co.page('spatialInterface')
        if hasattr(spatial, 'set_marker_rows'):
            spatial.set_marker_rows(())

    def reload_project(self):
        project_id = self.co.current_project_id()
        if project_id:
            self._project_token = uuid4().hex
            self.controller.request('project', {'token': self._project_token, 'project_id': project_id})

    def _project_ready(self, token, result, error):
        if token != self._project_token:
            return
        if error:
            self.co.log_message('标记列表读取失败：' + error)
            return
        if not result or result[0].document.project_id != self.co.current_project_id():
            return
        points = sum(len(view.document.markers) for view in result)
        intervals = sum(len(view.document.intervals) for view in result)
        for item in self.tools:
            if hasattr(item, 'setText'):
                item.setText(f'标记 {points} · 区间 {intervals}')
        self.co.page('spatialInterface').set_marker_rows(tuple(row for view in result for row in view.rows))

    def select(self, project_id, line_id, uid):
        if project_id != self.co.current_project_id():
            return
        for item in self.tools:
            if hasattr(item, 'select'):
                item.select(uid, project_id=project_id, line_id=line_id)
        self.co.page('spatialInterface').select_marker(uid)

    def navigate(self, page, row):
        if page == 'processing':
            self.co.project.on_line_process_requested(row['line_id'])
        elif page == 'spatial':
            self.co.goto_page('spatialInterface')
            spatial = self.co.page('spatialInterface')
            spatial.locate_marker(row)

    def _artifact_saved(self, project_id, line_id):
        if project_id == self.co.current_project_id():
            self.co.project_controller.refresh_artifacts(line_id)
            self.co.project_controller.refresh_all_artifacts()
