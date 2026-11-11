"""Worker-command boundary for marker records, preview and registered jobs."""
from dataclasses import replace

from PyQt6.QtCore import QObject, pyqtSignal

from ui.controllers.backend_controller import EXPECTED_BACKEND_ERRORS, friendly_error_message, run_command
from ui.desktop_backend_facade import marker_rows_for_view, export_marker_views


class MarkerController(QObject):
    ready = pyqtSignal(str, object, str)
    changed = pyqtSignal(object)
    job_submitted = pyqtSignal(str)
    artifact_saved = pyqtSignal(str, str)
    _completed = pyqtSignal(str, object, object, str)

    def __init__(self, backend_controller, parent=None):
        super().__init__(parent)
        self.backend_controller = backend_controller
        self._completed.connect(self._deliver)

    def request(self, action, payload):
        backend = getattr(self.backend_controller, 'backend', None)
        if backend is None:
            self.ready.emit(payload['token'], None, '后端尚未就绪')
            return
        run_command(_MarkerCommand(self, backend, action, dict(payload)), name='mygpr-markers-' + action)

    def _deliver(self, action, payload, result, error):
        if not error and action in ('save', 'delete'):
            self.changed.emit(result)
        if not error and action == 'execute':
            bridge = self.backend_controller.job_bridge
            job_id = result
            line_id = payload['preview']['line_id']
            project_id = payload['preview']['project_id']
            def finished(done, success, message, artifact):
                if done != job_id:
                    return
                bridge.job_completed.disconnect(finished)
                if success:
                    self.artifact_saved.emit(project_id, line_id)
            bridge.job_completed.connect(finished)
            bridge.watch(job_id, title='事件区间处理')
            self.job_submitted.emit(job_id)
        self.ready.emit(payload['token'], result, error)


class _MarkerCommand:
    def __init__(self, controller, backend, action, payload):
        self.controller, self.backend, self.action, self.payload = controller, backend, action, payload

    def execute(self):
        result, error = None, ''
        action, payload, backend = self.action, self.payload, self.backend
        try:
            if action in ('save', 'delete'):
                view = payload['view']
                with backend.projects.acquire_lease(view.document.project_id):
                    saved = backend.markers.edit(view, payload.get('record'), delete_id=payload.get('uid', ''))
                    updated = replace(view, document=saved)
                    result = replace(updated, rows=marker_rows_for_view(updated))
            elif action == 'project':
                result = backend.markers.project_views(payload['project_id'])
            elif action == 'load':
                result = backend.markers.view(payload['project_id'], payload['line_id'], payload.get('artifact_id', ''))
            elif action == 'preview':
                result = backend.markers.preview(payload['view'], payload['ids'], payload['mode'])
            elif action == 'execute':
                result = backend.submit_marker_selection(payload['preview'])
            elif action == 'export':
                result = export_marker_views(backend, payload['views'], payload['path'], payload['format'])
            else:
                raise ValueError('不支持的标记操作')
        except (*EXPECTED_BACKEND_ERRORS, MemoryError) as exc:
            error = friendly_error_message(exc)
        self.controller._completed.emit(action, payload, result, error)
