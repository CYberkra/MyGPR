#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI-independent report generation use cases."""
from __future__ import annotations

from typing import Any

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.project.service import ProjectService
from mygpr.domain.reporting.models import ReportPackage


class ReportingService:
    def __init__(self, projects: ProjectService) -> None:
        self._projects = projects

    def generate_package(
        self,
        project_id: str,
        *,
        package_name: str | None = None,
        report_profile: dict[str, Any] | None = None,
        context: ExecutionContext | None = None,
    ) -> ReportPackage:
        return self._projects.generate_report(
            project_id,
            package_name=package_name,
            report_profile=report_profile,
            context=context or ExecutionContext.null(),
        )


__all__ = ["ReportingService"]
