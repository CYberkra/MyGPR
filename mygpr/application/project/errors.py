"""Stable project lifecycle errors, shared by project use cases."""

from mygpr.domain.common.errors import MyGPRError


class ProjectApplicationError(MyGPRError):
    error_code = "MYGPR_PROJECT_ERROR"
    category = "project"


class ProjectBusyError(ProjectApplicationError):
    error_code = "MYGPR_PROJECT_BUSY"
