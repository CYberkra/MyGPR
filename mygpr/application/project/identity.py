"""Exclusive naming and relocation use cases on existing project sessions."""

from pathlib import Path
from contextlib import contextmanager
from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.project.errors import ProjectBusyError, ProjectApplicationError
from mygpr.application.project.ports import ProjectSessionPort
from mygpr.domain.project.models import ProjectLine, ProjectSummary


class ProjectIdentityServiceMixin:
    def _check_maintenance(self, project_id: str) -> None:
        if str(project_id) in self._maintenance:
            raise ProjectBusyError("项目正在迁移或删除，请等待操作完成。")

    def _identity_session(self, project_id: str) -> ProjectSessionPort:
        self._check_maintenance(project_id)
        if self._lease_counts.get(str(project_id), 0):
            raise ProjectBusyError("项目有正在运行或排队的任务，请结束任务后再操作。")
        return self._session(project_id)

    def rename_project(self, project_id: str, name: str) -> ProjectSummary:
        with self._lock:
            return self._identity_session(project_id).rename_project(name)

    def rename_line(self, project_id: str, line_id: str, name: str) -> ProjectLine:
        with self._lock:
            return self._identity_session(project_id).rename_line(line_id, name)

    def relocate_project(
        self, project_id: str, destination: str | Path, *, context: ExecutionContext | None = None
    ) -> ProjectSummary:
        context = context or ExecutionContext.null()
        normalized = str(project_id)
        with self._lock:
            session = self._identity_session(normalized)
            self._maintenance.add(normalized)
        try:
            summary = session.relocate(Path(destination), context=context)
            with self._lock:
                self._roots[normalized] = summary.root_path
            return summary
        finally:
            with self._lock:
                self._maintenance.discard(normalized)

    @contextmanager
    def exclusive_delete(self, project_id: str, *, task_lease: bool = False):
        """Block new leases and wait for session readers before deletion."""
        normalized = str(project_id)
        with self._lock:
            self._check_maintenance(normalized)
            if self._lease_counts.get(normalized, 0) > int(task_lease):
                raise ProjectBusyError('项目有正在运行或排队的任务，请结束任务后再删除。')
            session = self._session(normalized)
            if session.summary.read_only:
                raise ProjectApplicationError("只读项目无法永久删除")
            self._maintenance.add(normalized)
        try:
            yield session
        finally:
            with self._lock:
                self._maintenance.discard(normalized)

    def delete_project_permanently(self, root: str | Path, *, expected_project_id: str = "", validate=None) -> str:
        """Delete an open or closed project, without switching the UI project."""
        resolved = str(Path(root).expanduser().resolve())
        with self._lock:
            existing = next((pid for pid, value in self._roots.items() if value == resolved), "")
        summary = self.get_summary(existing) if existing else self.open_project(root)
        pid = summary.project_id
        try:
            if expected_project_id and pid != expected_project_id:
                raise ProjectApplicationError("项目身份已变化，请重新确认删除范围")
            with self.exclusive_delete(pid) as session:
                if validate is not None:
                    validate(session)
                try:
                    return session.delete_permanently()
                finally:
                    # Core deletion releases its handles before rmtree, including
                    # on partial failure. Never expose that closed session again.
                    session.close()
                    with self._lock:
                        self._sessions.pop(pid, None)
                        self._roots.pop(pid, None)
                        self._lease_counts.pop(pid, None)
        finally:
            if not existing and pid in self._sessions:
                self.close_project(pid)
