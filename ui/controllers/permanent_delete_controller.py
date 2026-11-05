"""Permanent deletion: worker preflight, GUI confirmation, worker commit."""
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from ui.controllers.backend_controller import run_command


class PermanentDeleteController(QObject):
    confirmation_requested = pyqtSignal(object)
    started = pyqtSignal(object)
    finished = pyqtSignal(object, bool, str)
    _ready = pyqtSignal(object)

    def __init__(self, project):
        super().__init__(project)
        self.project = project
        self.pending = None
        self._ready.connect(self._deliver)

    def request(self, kind, targets=(), *, line_id='', root=''):
        pc = self.project
        backend = pc._backend()
        if pc._busy or self.pending or backend is None:
            return False
        self.pending = {'generation': pc._project_context_generation, 'plan': None,
                        'project_id': pc.current_project_id, 'stage': 'preview'}
        pc._set_busy(True)
        pc.log_message.emit('正在计算删除范围…')
        run_command(_DeleteCommand(self, backend.maintenance.deletion, 'preview',
                    (kind, pc.current_project_id, tuple(targets)), {'line_id': line_id, 'root': root}),
                    name='mygpr-delete-preflight')
        return True

    def confirm(self, plan, accepted):
        if not self.pending or self.pending['plan'] is not plan:
            return
        pc = self.project
        if not accepted:
            self.pending = None
            pc._set_busy(False)
            return
        if self.pending['generation'] != pc._project_context_generation:
            self._complete(plan, False, '项目上下文已变化，请重新选择')
            return
        self.pending['stage'] = 'execute'
        pc._preview_generation += 1
        pc._depth_preview_generation += 1
        pc.invalidate_artifact_previews()
        self.started.emit(plan)
        run_command(_DeleteCommand(self, pc._backend().maintenance.deletion, 'execute', (plan,), {}),
                    name='mygpr-permanent-delete')

    def _complete(self, plan, success, message):
        self.pending = None
        self.project._set_busy(False)
        self.project.log_message.emit(message)
        self.finished.emit(plan, success, message)

    @pyqtSlot(object)
    def _deliver(self, payload):
        request, stage, result, error, closed = payload
        if self.pending is not request:
            return
        pc = self.project
        if stage == 'preview':
            if error:
                self._complete(None, False, error)
            elif request['generation'] != pc._project_context_generation:
                self._complete(None, False, '项目上下文已变化，请重新选择')
            else:
                request['plan'] = result
                self.confirmation_requested.emit(result)
            return
        plan = request['plan']
        pc._preview_generation += 1
        pc._depth_preview_generation += 1
        pc.invalidate_artifact_previews()
        if closed and plan.project_id == pc.current_project_id:
            pc._project_context_generation += 1
            pc._current = None
            pc.project_closed.emit()
        elif plan.kind == 'artifacts':
            # Invalidate all confirmed targets, even on partial failure.
            for aid in plan.targets:
                pc.preview_invalidated.emit(aid)
            pc.refresh_artifacts(plan.line_id)
            pc.refresh_all_artifacts()
        elif plan.kind == 'lines':
            pc.refresh_lines()
            pc.refresh_all_artifacts()
        message = error or f'已永久删除 {len(plan.targets)} 项'
        self._complete(plan, not bool(error), message)


class _DeleteCommand:
    def __init__(self, controller, service, stage, args, kwargs):
        self.controller, self.service, self.stage = controller, service, stage
        self.args, self.kwargs = args, kwargs
        self.request = controller.pending

    def execute(self):
        result, error, closed = None, '', False
        try:
            result = getattr(self.service, self.stage)(*self.args, **self.kwargs)
        except Exception as exc:  # Report residual progress, not a false rollback.
            error = str(exc)
        finally:
            if self.stage == 'execute' and self.args[0].kind == 'project':
                closed = self.args[0].project_id not in {
                    p.project_id for p in self.service.projects.list_open_projects()}
            self.controller._ready.emit((self.request, self.stage, result, error, closed))
