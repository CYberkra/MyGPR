"""GUI-thread results for short name writes and cancellable directory copies."""

import sqlite3
from uuid import uuid4

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from ui.controllers.backend_controller import run_command, friendly_error_message
from ui.desktop_backend_facade import CancellationTokenSource, ExecutionContext


class ProjectIdentityController(QObject):
    started = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str, bool, str, object, str, str)
    error = pyqtSignal(str)
    _ready = pyqtSignal(object)
    _progress_ready = pyqtSignal(str, int, int, str)

    def __init__(self, project):
        super().__init__(project)
        self.project = project
        self.pending = None
        self._ready.connect(self._deliver)
        self._progress_ready.connect(self._show_progress)

    def rename_project(self, project_id, name):
        return self._start("rename_project", project_id, (name,))

    def rename_line(self, project_id, line_id, name):
        return self._start("rename_line", project_id, (line_id, name))

    def relocate_project(self, project_id, destination):
        return self._start("relocate_project", project_id, (destination,))

    def _start(self, operation, project_id, args):
        pc = self.project
        if pc._busy or self.pending or str(project_id) != pc.current_project_id:
            self.error.emit("项目已切换或正在使用中，请稍后重试。")
            return False
        backend = pc._backend()
        if backend is None:
            return False
        if operation == "relocate_project":
            pc._project_context_generation += 1
            pc._preview_generation += 1
            pc._depth_preview_generation += 1
            pc.invalidate_artifact_previews()
        request = dict(
            operation=operation,
            project_id=str(project_id),
            args=args,
            root=str(pc.current_project.root_path),
            generation=pc._project_context_generation,
            token=uuid4().hex,
            cancellation=CancellationTokenSource(),
        )
        self.pending = request
        pc._set_busy(True)
        self.started.emit(operation)
        run_command(_IdentityCommand(self, backend.projects, request), name="mygpr-project-identity")
        return True

    def cancel(self):
        if self.pending:
            self.pending["cancellation"].cancel()

    @pyqtSlot(str, int, int, str)
    def _show_progress(self, token, done, total, message):
        if self.pending and self.pending["token"] == token:
            self.progress.emit(min(100, round(100 * done / max(1, total))), message)

    @pyqtSlot(object)
    def _deliver(self, payload):
        request, result, error = payload
        if self.pending is not request:
            return
        pc = self.project
        current = (
            pc.current_project_id == request["project_id"] and pc._project_context_generation == request["generation"]
        )
        success = current and not error
        if success and request["operation"] == "rename_line":
            pc.refresh_lines()
        if success and request["operation"] != "rename_line":
            pc._current = result
        self.pending = None
        try:
            self.finished.emit(
                request["operation"],
                success,
                error if current else "项目上下文已经变化，结果未应用到当前页面。",
                result,
                request["root"],
                str(request["args"][0]) if request["operation"] == "rename_line" else "",
            )
        finally:
            pc._set_busy(False)


class _IdentityCommand:
    def __init__(self, controller, projects, request):
        self.controller, self.projects, self.request = controller, projects, request

    def execute(self):
        c, r = self.controller, self.request
        result, error = None, ""
        try:
            operation = getattr(self.projects, r["operation"])
            if r["operation"] == "relocate_project":
                context = ExecutionContext(
                    cancellation_token=r["cancellation"].token,
                    progress_callback=lambda done, total, message: c._progress_ready.emit(
                        r["token"], done, total, message
                    ),
                )
                result = operation(r["project_id"], *r["args"], context=context)
            else:
                result = operation(r["project_id"], *r["args"])
        except (OSError, ValueError, RuntimeError, LookupError, sqlite3.Error) as exc:
            error = "操作已取消，原项目保留。" if r["cancellation"].token.is_cancelled else friendly_error_message(exc)
        finally:
            if result is None and not error:
                error = "操作未完成，请检查项目状态和日志后重试。"
            c._ready.emit((r, result, error))
