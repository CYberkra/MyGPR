#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public result contracts for formal project report packages."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

REPORT_PACKAGE_SCHEMA = "mygpr.report_package.v3"

@dataclass(frozen=True)
class ReportPackageResult:
    """Result returned after generating one report package."""

    package_dir: str
    manifest_path: str
    html_path: str
    summary_json_path: str
    line_manifest_csv_path: str
    quality_csv_path: str
    targets_csv_path: str
    processing_csv_path: str
    spatial_csv_path: str
    file_count: int
    generated_at: str
    pdf_path: str = ""
    interfaces_csv_path: str = ""
    xlsx_path: str = ""
    sensor_sync_csv_path: str = ""
    gis_layers_csv_path: str = ""
    audit_csv_path: str = ""
    checksums_path: str = ""
    figures_dir: str = ""
    seal_path: str = ""
    delivery_zip_path: str = ""
    delivery_zip_sha256_path: str = ""
    spatial_result_id: str = ""
    snapshot_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

__all__ = ["REPORT_PACKAGE_SCHEMA", "ReportPackageResult"]
