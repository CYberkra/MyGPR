#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable report result contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


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
