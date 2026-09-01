#!/usr/bin/env python3
"""Freeze the public Backend API v1 surface and DTO field order."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "backend_api_v1.json"

PUBLIC_TYPES = (
    "mygpr.interfaces.backend.MyGPRBackend",
    "mygpr.application.jobs.models.JobEvent",
    "mygpr.application.jobs.models.JobSnapshot",
    "mygpr.application.jobs.models.JobResultSummary",
    "mygpr.domain.processing.models.ProcessingMethodDescriptor",
    "mygpr.domain.processing.models.ProcessingRequest",
    "mygpr.domain.processing.models.ProcessingResult",
    "mygpr.domain.processing.models.ResourceEstimate",
    "mygpr.domain.processing.models.PipelineStep",
    "mygpr.domain.processing.models.PipelineDefinition",
    "mygpr.domain.project.models.ProjectSummary",
    "mygpr.domain.project.models.ProjectLine",
    "mygpr.domain.project.models.ProjectArtifact",
    "mygpr.domain.reporting.models.ReportPackage",
)

PUBLIC_METHODS = (
    "create_default",
    "submit_processing",
    "submit_pipeline",
    "submit_autotune",
    "submit_line_import",
    "submit_batch_line_import",
    "submit_project_quality_check",
    "submit_line_quality_check",
    "submit_line_source_relink",
    "submit_source_file_check",
    "submit_line_transpose",
    "submit_sensor_sync",
    "submit_project_pipeline",
    "submit_project_report",
    "submit_project_backup",
    "submit_spatial_result",
    "build_georeference_3d",
    "export_artifact_segy",
    "submit_project_restore",
    "shutdown",
)

JOB_TRANSITIONS = {
    "queued": ["running", "cancelled", "failed"],
    "running": ["completed", "cancelled", "failed"],
    "completed": [],
    "cancelled": [],
    "failed": [],
}


def _resolve(path: str) -> Any:
    module_name, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module_name), name)


def _signature(value: Any) -> str:
    return str(inspect.signature(value)).replace("'", "")


def build_contract() -> dict[str, Any]:
    from mygpr.interfaces.backend import BACKEND_API_VERSION, MyGPRBackend

    types: dict[str, Any] = {}
    for path in PUBLIC_TYPES:
        value = _resolve(path)
        entry: dict[str, Any] = {"kind": "dataclass" if is_dataclass(value) else "class"}
        if is_dataclass(value):
            entry["fields"] = [item.name for item in fields(value)]
        types[path] = entry
    methods = {
        name: _signature(getattr(MyGPRBackend, name))
        for name in PUBLIC_METHODS
    }
    from mygpr.application.jobs.runner import JobRunnerClosedError
    from mygpr.interfaces.backend import BackendShutdownError
    from mygpr.domain.common.errors import (
        AutoTuneScoringError,
        EvidenceExportError,
        GprMaxConversionError,
        InputDataError,
        ParameterValidationError,
        ProcessingMethodError,
        ProjectBusyContractError,
    )
    error_codes = {
        item.__name__: item.error_code
        for item in (
            InputDataError,
            ProcessingMethodError,
            ParameterValidationError,
            ProjectBusyContractError,
            EvidenceExportError,
            GprMaxConversionError,
            AutoTuneScoringError,
            JobRunnerClosedError,
            BackendShutdownError,
        )
    }
    return {
        "schema": "mygpr.backend_api_contract.v1",
        "api_version": BACKEND_API_VERSION,
        "compatibility": {
            "add_optional_fields": True,
            "rename_or_remove_fields": False,
            "change_existing_field_type": False,
            "change_error_code_semantics": False,
        },
        "facade_methods": methods,
        "types": types,
        "job_state_transitions": JOB_TRANSITIONS,
        "stable_error_schema": "mygpr.error_info.v1",
        "stable_error_codes": error_codes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    current = build_contract()
    if args.write:
        CONTRACT.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(CONTRACT)
        return 0
    expected = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if current != expected:
        print("Backend API v1 contract drift detected.")
        print(json.dumps({"expected": expected, "current": current}, ensure_ascii=False, indent=2))
        return 1
    print(f"Backend API contract OK: {current['api_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
