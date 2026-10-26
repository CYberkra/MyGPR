"""Exclusive naming and relocation use cases on existing project sessions."""

from pathlib import Path
from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.project.errors import ProjectBusyError
from mygpr.application.project.ports import ProjectSessionPort
from mygpr.domain.project.models import ProjectLine, ProjectSummary


class ProjectIdentityServiceMixin:
    def _check_maintenance(self, project_id: str) -> None:
        if str(project_id) in self._maintenance:
            raise ProjectBusyError("项目正在迁移，请等待迁移完成或取消。")

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
