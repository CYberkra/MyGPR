#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable report result contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


MANUAL_ACQUISITION_FIELDS = {
    'antenna': '天线型号／配置',
    'platform': '平台／采集方式',
    'survey_date': '采集日期',
    'frequency': '频段或中心频率',
    'power': '功率（设定或实测）',
    'source': '补录依据',
}
ANALYSIS_FIELDS = {
    'observation': '观察',
    'interpretation': '判断与依据',
    'verification': '验证结果与依据',
}


@dataclass(frozen=True, slots=True)
class ReportPackage:
    package_dir: str
    manifest_path: str
    generated_at: str
    file_count: int
    pdf_path: str = ""
    html_path: str = ""
    xlsx_path: str = ""
    delivery_zip_path: str = ""
    delivery_zip_sha256_path: str = ""
    seal_path: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.package_dir).strip():
            raise ValueError("package_dir must not be empty")
        object.__setattr__(self, "file_count", max(0, int(self.file_count)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))
