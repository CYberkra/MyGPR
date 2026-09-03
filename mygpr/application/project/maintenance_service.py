#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project metadata, QC, source-evidence and destructive maintenance use cases."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.project.service import ProjectService
from mygpr.domain.project.models import (
    BatchImportSummary,
    LineDeleteResult,
    LineQualityReport,
    ProjectMetadata,
    SourceFileStatus,
)


class ProjectMaintenanceService:
    """Application boundary for project maintenance previously owned by legacy Qt."""

    def __init__(self, projects: ProjectService) -> None:
        self._projects = projects

    def get_metadata(self, project_id: str) -> ProjectMetadata:
        return self._projects._session(project_id).get_metadata()

    def update_metadata(
        self,
        project_id: str,
        **changes: str | None,
    ) -> ProjectMetadata:
        return self._projects._session(project_id).update_metadata(**changes)

    def list_quality_reports(
        self,
        project_id: str,
    ) -> tuple[LineQualityReport, ...]:
        return tuple(self._projects._session(project_id).list_quality_reports())

    def run_line_quality_check(
        self,
        project_id: str,
        line_id: str,
        *,
        context: ExecutionContext | None = None,
    ) -> LineQualityReport:
        if context is not None:
            context.raise_if_cancelled()
        return self._projects._session(project_id).run_line_quality_check(line_id)

    def run_project_quality_check(
        self,
        project_id: str,
        *,
        context: ExecutionContext | None = None,
    ) -> tuple[LineQualityReport, ...]:
        session = self._projects._session(project_id)
        return tuple(
            session.run_project_quality_check(
                context=context or ExecutionContext.null()
            )
        )

    def check_source_files(
        self,
        project_id: str,
        *,
        context: ExecutionContext | None = None,
    ) -> tuple[SourceFileStatus, ...]:
        session = self._projects._session(project_id)
        return tuple(
            session.check_source_files(context=context or ExecutionContext.null())
        )

    def relink_line_source(
        self,
        project_id: str,
        line_id: str,
        new_source: str | Path,
        *,
        allow_mismatch: bool = False,
        context: ExecutionContext | None = None,
    ) -> SourceFileStatus:
        source_path = Path(new_source).expanduser().resolve()
        return self._projects._session(project_id).relink_line_source(
            line_id,
            source_path,
            allow_mismatch=allow_mismatch,
            context=context or ExecutionContext.null(),
        )

    def export_source_manifest(
        self,
        project_id: str,
        destination: str | Path | None = None,
    ) -> str:
        path = None if destination is None else Path(destination).expanduser().resolve()
        return str(self._projects._session(project_id).export_source_manifest(path))

    def transpose_line_dataset(
        self,
        project_id: str,
        line_id: str,
        *,
        context: ExecutionContext | None = None,
    ) -> LineQualityReport:
        return self._projects._session(project_id).transpose_line_dataset(
            line_id,
            context=context or ExecutionContext.null(),
        )

    def delete_line(
        self,
        project_id: str,
        line_id: str,
        *,
        reason: str = "用户删除测线",
    ) -> LineDeleteResult:
        return self._projects._session(project_id).delete_line(line_id, reason=reason)

    def batch_import_lines(
        self,
        project_id: str,
        sources: Sequence[str | Path],
        *,
        context: ExecutionContext | None = None,
    ) -> BatchImportSummary:
        resolved = tuple(Path(item).expanduser().resolve() for item in sources)
        return self._projects._session(project_id).batch_import_lines(
            resolved,
            context=context or ExecutionContext.null(),
        )
