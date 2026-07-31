#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source-evidence, line-manifest, and trajectory attachment operations."""
from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

from core.app_errors import error_info_from_exception
from core.field_project_errors import FieldProjectOperationError
from core.field_project_models import local_now, validate_line_id
from core.field_project_store import FieldLineRecord, FieldProjectStore
from core.source_file_registry import (
    check_all_source_files,
    export_source_file_manifest_csv,
    record_source_file,
    relink_line_source_file,
    source_summary,
)
from core.storage_primitives import atomic_output_path
from core.tabular_security import safe_tabular_row

logger = logging.getLogger(__name__)


def check_project_source_files(
    store: FieldProjectStore,
    *,
    cancel_requested=None,
    progress_callback=None,
):
    """Check recorded source evidence with cooperative cancellation."""
    records = check_all_source_files(
        store, cancel_requested=cancel_requested, progress_callback=progress_callback
    )
    summary = source_summary(records)
    store.append_log(
        "源文件检查: "
        f"total={summary['total']}, available={summary['available']}, "
        f"missing={summary['missing']}, modified={summary['modified']}"
    )
    return records


def relink_project_line_source(
    store: FieldProjectStore,
    line_id: str,
    new_source: str | Path,
    *,
    allow_mismatch: bool = False,
    cancel_requested=None,
    progress_callback=None,
):
    record = relink_line_source_file(
        store,
        line_id,
        new_source,
        allow_mismatch=allow_mismatch,
        cancel_requested=cancel_requested,
        progress_callback=progress_callback,
    )
    store.append_log(f"重新定位源文件 {line_id}: {record.source_path}; status={record.status}")
    return record


def export_project_source_manifest_csv(store: FieldProjectStore, destination: str | Path | None = None) -> Path:
    out = export_source_file_manifest_csv(store, destination)
    store.append_log(f"导出源文件清单: {out}")
    return out


def export_line_manifest_csv(store: FieldProjectStore, destination: str | Path | None = None) -> Path:
    """Export the current line manifest atomically for quick field review."""
    out = Path(destination) if destination is not None else store.root / "reports" / "line_manifest.csv"
    headers = [
        "line_id",
        "name",
        "length_m",
        "data_quality",
        "rtk_status",
        "processing_status",
        "raw_path",
        "gpr_dataset_path",
        "trajectory_path",
        "data_format",
        "updated_at",
    ]
    with atomic_output_path(out, suffix=".csv.tmp") as temporary:
        with temporary.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            for line in store.list_lines():
                writer.writerow(safe_tabular_row({key: getattr(line, key, "") for key in headers}))
            fh.flush()
    store.append_log(f"导出测线清单: {out}")
    return out


def import_trajectory_file(
    store: FieldProjectStore,
    source: str | Path,
    *,
    line_id: str,
) -> Path:
    """Copy an RTK/IMU evidence file into the project and bind provenance."""
    line_id = validate_line_id(line_id)
    src = Path(source).expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise FieldProjectOperationError(f"轨迹文件不存在：{src}")
    dest_dir = store.root / "raw" / line_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if src != dest:
        shutil.copy2(src, dest)
    try:
        line = store.get_line(line_id)
    except KeyError:
        line = FieldLineRecord(line_id=line_id, name=line_id)
    line.trajectory_path = dest.relative_to(store.root).as_posix()
    line.updated_at = local_now()
    store.upsert_line(line)
    try:
        record_source_file(
            store,
            line_id,
            src,
            role="trajectory",
            import_mode="copied_to_project",
            project_raw_path=line.trajectory_path,
        )
    except (OSError, ValueError, TypeError) as provenance_exc:
        error_info = error_info_from_exception(provenance_exc)
        logger.warning("Trajectory source file record failed: %s [%s]", error_info.user_message, error_info.error_code)
        store.append_log(f"测线 {line_id} 轨迹来源文件记录失败: {error_info.user_message}")
    store.append_log(f"附加 RTK/IMU 文件 {line_id}: {dest.name}")
    return dest


__all__ = [
    "check_project_source_files",
    "export_line_manifest_csv",
    "export_project_source_manifest_csv",
    "import_trajectory_file",
    "relink_project_line_source",
]
