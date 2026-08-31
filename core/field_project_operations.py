#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User-facing project operation helpers for the MyGPR field workbench.

This module contains the durable, dialog-independent logic behind the project
management buttons: creating a formal project, opening an existing project,
importing line data and maintaining the recent-project list.  GUI classes should
call this layer rather than writing project/file handling directly in button
callbacks.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import logging

from core.field_project_models import FieldProjectManifest, local_now, validate_line_id
from core.field_project_store import FieldLineRecord, FieldProjectStore
from core.gpr_format_registry import get_format_spec, supported_file_dialog_filter
from core.field_import_preview import build_import_preflight, ImportPreflightResult
from core.source_file_registry import (
    load_source_registry,
    record_source_file,
    remove_line_source_records,
)
from mygpr.domain.common.errors import error_info_from_exception
from core.storage_primitives import atomic_write_json, utc_now
from core.field_project_errors import FieldProjectOperationError
from core.field_project_backup import (
    ProjectBackupResult,
    ProjectRestoreResult,
    backup_project_archive,
    restore_project_archive,
)
from core.field_project_source_ops import (
    check_project_source_files,
    export_line_manifest_csv,
    export_project_source_manifest_csv,
    import_trajectory_file,
    relink_project_line_source,
)
from core.project_root_guard import validate_project_root_marker

DIRECT_READABLE_EXTENSIONS = {".csv", ".txt", ".npy", ".npz", ".h5", ".hdf5"}
PROJECT_MANIFEST_NAME = FieldProjectStore.MANIFEST_NAME
RECENT_PROJECTS_SCHEMA = "mygpr.recent_projects.v1"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectMetadataUpdate:
    """Editable project metadata used by creation and settings dialogs."""

    name: str
    location: str = ""
    operator: str = "操作员"
    project_no: str = ""
    device_model: str = ""
    coordinate_system: str = ""
    vertical_datum: str = ""


@dataclass(frozen=True)
class InferredLineIdentity:
    """Line ID/name inferred from a field CSV filename."""

    line_id: str
    name: str
    confidence: str = "medium"


@dataclass(frozen=True)
class BatchImportItemResult:
    """Result of importing one file during a batch import operation."""

    source: str
    line_id: str
    name: str
    success: bool
    message: str
    sample_count: int = 0
    trace_count: int = 0
    length_m: float = 0.0
    file_size_mb: float = 0.0
    elapsed_s: float = 0.0
    raw_dir: str = ""
    manifest_path: str = ""
    diagnosis: str = ""

    @property
    def shape_text(self) -> str:
        return f"{self.sample_count}×{self.trace_count}" if self.sample_count and self.trace_count else "--"


@dataclass(frozen=True)
class BatchImportSummary:
    """User-facing summary for a batch import operation."""

    total: int
    succeeded: int
    failed: int
    results: tuple[BatchImportItemResult, ...]

    def to_log_lines(self) -> list[str]:
        lines = [f"批量导入完成：成功 {self.succeeded} / {self.total}，失败 {self.failed}。"]
        for row in self.results:
            status = "成功" if row.success else "失败"
            detail = f"{Path(row.source).name} → {row.line_id} {row.name}: {status}；{row.message}"
            if row.success and row.sample_count and row.trace_count:
                detail += f"；矩阵 {row.sample_count}×{row.trace_count}；长度 {row.length_m:.2f} m；耗时 {row.elapsed_s:.2f}s"
            if row.diagnosis:
                detail += f"；诊断：{row.diagnosis}"
            lines.append(detail)
        return lines


def _clean_text(value: str | None) -> str:
    return str(value or "").strip()


def _safe_project_slug(name: str) -> str:
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", name.strip())
    return text.strip("._-") or "MyGPR_Project"


def default_recent_projects_path() -> Path:
    """Return the platform-appropriate recent-project registry path."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "MyGPR" / "recent_projects.json"
    return Path.home() / ".local" / "share" / "MyGPR" / "recent_projects.json"




@dataclass(frozen=True)
class LineDeleteResult:
    """Result of permanently deleting one line's project-local artifacts."""

    line_id: str
    line_name: str
    deleted_paths: tuple[str, ...]
    remaining_line_count: int


@dataclass(frozen=True)
class ProjectDeleteResult:
    """Result of permanently deleting a MyGPR project directory."""

    project_name: str
    original_path: str
    deleted_path: str
    removed_recent_count: int


@dataclass(frozen=True)
class ProjectDeletePreflight:
    """User-facing preflight summary before deleting a project folder."""

    project_name: str
    project_path: str
    size_mb: float
    file_count: int
    line_count: int
    processed_count: int
    report_file_count: int
    external_source_count: int
    missing_recent_count: int = 0

    def to_lines(self) -> list[str]:
        return [
            f"项目名称：{self.project_name}",
            f"项目路径：{self.project_path}",
            f"项目大小：{self.size_mb:.2f} MB",
            f"文件数量：{self.file_count}",
            f"测线数量：{self.line_count}",
            f"处理结果：{self.processed_count}",
            f"报告文件：{self.report_file_count}",
            f"外部源文件：{self.external_source_count} 个，不会删除",
        ]


@dataclass
class RecentProjectRecord:
    path: str
    name: str
    version: str = ""
    last_opened_at: str = field(default_factory=local_now)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RecentProjectRecord":
        return cls(
            path=str(payload.get("path", "")),
            name=str(payload.get("name", "")),
            version=str(payload.get("version", "")),
            last_opened_at=str(payload.get("last_opened_at", local_now())),
        )


class RecentProjectsStore:
    """Small JSON-backed recent project registry."""

    def __init__(self, path: str | Path | None = None, *, limit: int = 12) -> None:
        self.path = Path(path) if path is not None else default_recent_projects_path()
        self.limit = int(limit)

    def load(self) -> list[RecentProjectRecord]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            error_info = error_info_from_exception(exc, category="config")
            logger.warning("Failed to load recent projects: %s [%s]", error_info.user_message, error_info.error_code)
            return []
        rows = payload.get("projects", []) if isinstance(payload, dict) else []
        records = [RecentProjectRecord.from_dict(row) for row in rows if isinstance(row, dict)]
        return [record for record in records if record.path]

    def save(self, records: list[RecentProjectRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": RECENT_PROJECTS_SCHEMA,
            "updated_at": local_now(),
            "projects": [asdict(record) for record in records[: self.limit]],
        }
        tmp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, store: FieldProjectStore) -> None:
        project_path = str(store.root.resolve())
        project_name = store.manifest.name
        records = [record for record in self.load() if str(Path(record.path).resolve()) != project_path]
        records.insert(
            0,
            RecentProjectRecord(
                path=project_path,
                name=project_name,
                version=str(store.manifest.schema),
                last_opened_at=local_now(),
            ),
        )
        self.save(records)

    def remove(self, project_path: str | Path) -> int:
        """Remove one project path from the recent-project registry."""
        target = str(Path(project_path).expanduser().resolve())
        records = self.load()
        kept: list[RecentProjectRecord] = []
        removed = 0
        for record in records:
            try:
                same = str(Path(record.path).expanduser().resolve()) == target
            except Exception as exc:
                error_info = error_info_from_exception(exc, category="config")
                logger.warning("Failed to resolve project path: %s [%s]", error_info.user_message, error_info.error_code)
                same = str(record.path) == str(project_path)
            if same:
                removed += 1
            else:
                kept.append(record)
        if removed:
            self.save(kept)
        return removed

    def prune_missing(self) -> int:
        """Remove recent-project entries whose project.json no longer exists."""
        records = self.load()
        kept: list[RecentProjectRecord] = []
        removed = 0
        for record in records:
            try:
                manifest = Path(record.path).expanduser().resolve() / PROJECT_MANIFEST_NAME
                exists = manifest.exists()
            except Exception as exc:
                error_info = error_info_from_exception(exc, category="config")
                logger.warning("Failed to check manifest existence: %s [%s]", error_info.user_message, error_info.error_code)
                exists = False
            if exists:
                kept.append(record)
            else:
                removed += 1
        if removed:
            self.save(kept)
        return removed


def validate_project_root(root: str | Path) -> Path:
    root_path = Path(root).expanduser().resolve()
    manifest = root_path / PROJECT_MANIFEST_NAME
    if not manifest.exists():
        raise FieldProjectOperationError(f"未找到 MyGPR 项目文件：{manifest}")
    try:
        FieldProjectManifest.from_dict(json.loads(manifest.read_text(encoding="utf-8")))
    except Exception as exc:
        raise FieldProjectOperationError(f"项目文件无效或版本不兼容：{exc}") from exc
    return root_path


def create_project(
    parent_or_root: str | Path | None,
    *,
    name: str = "",
    location: str = "",
    operator: str = "操作员",
    project_no: str = "",
    device_model: str = "",
    coordinate_system: str = "",
    vertical_datum: str = "",
    create_child_dir: bool = True,
    recent_store: RecentProjectsStore | None = None,
) -> FieldProjectStore:
    """Create a formal project without demo measurements.

    Every business field is optional.  Empty names and storage locations are
    resolved to a timestamped project name below the user's Documents folder,
    matching the project-management workflow agreed for the five-page UI.
    """
    resolved_name = str(name or "").strip() or datetime.now().strftime("未命名项目_%Y%m%d_%H%M%S")
    if parent_or_root is None or not str(parent_or_root).strip():
        documents = Path.home() / "Documents"
        base = (documents if documents.exists() else Path.home()) / "MyGPR Projects"
    else:
        base = Path(parent_or_root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    root = base / _safe_project_slug(resolved_name) if create_child_dir else base
    if (root / PROJECT_MANIFEST_NAME).exists():
        raise FieldProjectOperationError(f"该目录已存在 MyGPR 项目：{root}")
    store = FieldProjectStore.create_empty(
        root,
        name=resolved_name,
        location=location.strip(),
        operator=operator.strip() or "操作员",
        project_no=project_no.strip(),
        device_model=device_model.strip(),
        coordinate_system=coordinate_system.strip(),
        vertical_datum=vertical_datum.strip(),
    )
    (recent_store or RecentProjectsStore()).add(store)
    return store


def update_project_metadata(
    store: FieldProjectStore,
    *,
    name: str | None = None,
    location: str | None = None,
    operator: str | None = None,
    project_no: str | None = None,
    device_model: str | None = None,
    coordinate_system: str | None = None,
    vertical_datum: str | None = None,
    recent_store: RecentProjectsStore | None = None,
) -> FieldProjectStore:
    """Update editable project metadata and persist ``project.json``.

    This is the durable counterpart of the project settings dialog; GUI code
    should call this helper instead of mutating ``store.manifest`` directly.
    """
    cleaned_name = _clean_text(name) if name is not None else store.manifest.name
    if not cleaned_name:
        raise FieldProjectOperationError("项目名称不能为空。")
    store.manifest.name = cleaned_name
    if location is not None:
        store.manifest.location = _clean_text(location) or "未填写"
    if operator is not None:
        store.manifest.operator = _clean_text(operator) or "操作员"
    if project_no is not None:
        store.manifest.project_no = _clean_text(project_no) or store.manifest.project_no
    if device_model is not None:
        store.manifest.device_model = _clean_text(device_model) or "未填写"
    if coordinate_system is not None:
        store.manifest.coordinate_system = _clean_text(coordinate_system) or "未填写"
    if vertical_datum is not None:
        store.manifest.vertical_datum = _clean_text(vertical_datum) or "未填写"
    store.save_manifest()
    store.append_log(f"更新项目设置：{store.manifest.name}")
    (recent_store or RecentProjectsStore()).add(store)
    return store


def open_project(root: str | Path, *, recent_store: RecentProjectsStore | None = None) -> FieldProjectStore:
    root_path = validate_project_root(root)
    store = FieldProjectStore.open(root_path)
    (recent_store or RecentProjectsStore()).add(store)
    return store


def next_line_id(store: FieldProjectStore) -> str:
    used = {line.line_id for line in store.list_lines()}
    index = 1
    while True:
        candidate = f"L{index:02d}"
        if candidate not in used:
            return candidate
        index += 1


def validate_import_source(source: str | Path) -> tuple[Path, str, str]:
    """Validate a line-data source and return ``(path, support, message)``."""
    src = Path(source).expanduser().resolve()
    if not src.exists():
        raise FieldProjectOperationError(f"数据文件不存在：{src}")
    if not src.is_file():
        raise FieldProjectOperationError(f"当前导入入口只支持文件，不支持目录：{src}")
    suffix = src.suffix.lower()
    spec = get_format_spec(src)
    if suffix in DIRECT_READABLE_EXTENSIONS:
        return src, "native", "当前版本可直接读取并归一化为 GPRDataSet。"
    if spec is not None:
        raise FieldProjectOperationError(
            f"已识别 {spec.display_name}（{src.suffix}），但当前现场工作台尚未直接解码该格式。"
            f"建议先转换为 CSV / NPY / NPZ / H5 后导入。"
        )
    raise FieldProjectOperationError(
        f"暂不支持该数据格式：{src.suffix or '无扩展名'}。"
        f"当前正式导入入口支持 CSV / NPY / NPZ / H5。"
    )



def preview_import_source(
    source: str | Path,
    *,
    line_id: str = "L01",
    dielectric_constant: float = 9.0,
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> ImportPreflightResult:
    """Return lightweight user-facing diagnostics without materialising a full matrix."""
    safe_line_id = validate_line_id(line_id)
    return build_import_preflight(
        source,
        line_id=safe_line_id,
        dielectric_constant=dielectric_constant,
        cancel_requested=cancel_requested,
        progress_callback=progress_callback,
    )

def infer_line_identity_from_filename(source: str | Path, *, fallback_index: int | None = None) -> InferredLineIdentity:
    """Infer a stable MyGPR line id/name from common field CSV filenames.

    Examples:
    - ``Line9origin(30).csv`` -> ``L09_30`` / ``9号测线（30）``
    - ``Line3origin.csv`` -> ``L03`` / ``3号测线``
    - ``L1origin.csv`` -> ``L01`` / ``L1号测线``
    - ``X1origin.csv`` -> ``X1`` / ``X1号测线``
    """
    stem = Path(source).stem.strip()
    normalized = stem.replace(" ", "")
    paren_match = re.search(r"\(([^()]+)\)", normalized)
    variant = re.sub(r"[^0-9A-Za-z]+", "", paren_match.group(1)) if paren_match else ""
    lower = normalized.lower()
    patterns = [
        # YingShan exports may use LineL1origin / LineX1origin in addition to
        # ordinary Line9origin.  Check the prefixed L/X forms before generic
        # numeric LineN matching, otherwise these files fall back to L05/L06.
        (r"^linel\s*0*(\d+)", "l"),
        (r"^linex\s*0*(\d+)", "x"),
        (r"line\s*0*(\d+)", "line"),
        (r"^l\s*0*(\d+)", "l"),
        (r"^x\s*0*(\d+)", "x"),
        (r"(\d+)\s*号", "cn"),
    ]
    for pattern, kind in patterns:
        match = re.search(pattern, lower if kind != "cn" else normalized, flags=re.IGNORECASE)
        if not match:
            continue
        num = int(match.group(1))
        if kind == "x":
            base_id = f"X{num}"
            base_name = f"X{num}号测线"
        elif kind == "l":
            base_id = f"L{num:02d}"
            base_name = f"L{num}号测线"
        else:
            base_id = f"L{num:02d}"
            base_name = f"{num}号测线"
        if variant:
            return InferredLineIdentity(f"{base_id}_{variant}", f"{base_name}（{variant}）", "high")
        return InferredLineIdentity(base_id, base_name, "high")
    idx = fallback_index if fallback_index is not None else 1
    return InferredLineIdentity(f"L{int(idx):02d}", f"导入测线 L{int(idx):02d}", "low")


def _unique_line_id(store: FieldProjectStore, preferred: str, reserved: set[str]) -> str:
    safe_preferred = validate_line_id(preferred)
    used = {validate_line_id(line.line_id) for line in store.list_lines()} | {validate_line_id(item) for item in reserved}
    if safe_preferred not in used:
        return safe_preferred
    index = 2
    while True:
        candidate = validate_line_id(f"{safe_preferred}_{index}")
        if candidate not in used:
            return candidate
        index += 1


def _rollback_failed_line_import(store: FieldProjectStore, line_id: str, previous_lines: list[FieldLineRecord], backup_dir: Path | None) -> None:
    """Restore project manifest and raw line directory after a failed formal import."""
    raw_dir = store.root / "raw" / line_id
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    if backup_dir is not None and backup_dir.exists():
        shutil.move(str(backup_dir), str(raw_dir))
    store.manifest.set_lines(previous_lines)
    store.save_manifest()


def import_line_data(
    store: FieldProjectStore,
    source: str | Path,
    *,
    line_id: str | None = None,
    name: str | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> FieldLineRecord:
    """Import one readable line file into the active project transactionally."""
    src, _support, _message = validate_import_source(source)
    target_line_id = validate_line_id(line_id or next_line_id(store))
    target_name = name or f"导入测线 {target_line_id}"
    previous_lines = store.list_lines()
    raw_dir = store.root / "raw" / target_line_id
    backup_dir: Path | None = None
    if raw_dir.exists():
        backup_dir = store.root / "raw" / f".{target_line_id}.rollback_{uuid.uuid4().hex}"
        shutil.copytree(raw_dir, backup_dir)
    try:
        store.import_line_file(
            target_line_id, src, name=target_name, copy_into_project=True,
            cancel_requested=cancel_requested, progress_callback=progress_callback,
        )
        line = store.get_line(target_line_id)
        try:
            record_source_file(
                store,
                target_line_id,
                src,
                role="gpr",
                import_mode="copied_to_project",
                project_raw_path=line.raw_path,
            )
        except Exception as provenance_exc:
            error_info = error_info_from_exception(provenance_exc)
            logger.warning("Line source file record failed: %s [%s]", error_info.user_message, error_info.error_code)
            store.append_log(f"测线 {target_line_id} 来源文件记录失败: {error_info.user_message}")
        # Formal import requires a normalized dataset.  ``import_line_file`` accepts
        # sidecar/evidence files defensively, so enforce the user-facing contract here.
        if not line.gpr_dataset_path:
            raise FieldProjectOperationError(f"文件已复制但未生成可显示的 GPR 矩阵，请检查数据内容：{src.name}")
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir)
        return line
    except Exception:
        _rollback_failed_line_import(store, target_line_id, previous_lines, backup_dir)
        raise

def diagnose_import_failure(source: str | Path, message: str) -> str:
    """Return a concise user-facing diagnosis for a failed measurement import."""
    src = Path(source)
    suffix = src.suffix.lower()
    text = str(message or "")
    if not src.exists():
        return "文件不存在或路径已失效。"
    if "too small for a B-scan matrix" in text or "numeric content is too small" in text:
        return "CSV 被识别为数值表，但未匹配 MyGPR 头信息或标准 B-scan 矩阵；请检查 Number of Samples / Number of Traces 等头信息。"
    if "insufficient data rows" in text:
        return "CSV 头信息声明的采样点数×道数大于实际数据行数，文件可能截断或导出不完整。"
    if "Unsupported" in text or "不支持" in text:
        return f"当前导入器尚未支持 {suffix or '该'} 格式。"
    if "No 2D matrix" in text:
        return "HDF5/NPZ 内未找到二维矩阵数据集。"
    if "could not convert" in text or "ValueError" in text:
        return "文件中存在无法解析的数值或列格式异常。"
    return "请查看错误信息；可将该文件单独导入以获得更详细的预检结果。"


def batch_import_line_data(
    store: FieldProjectStore,
    sources: list[str | Path],
    *,
    progress_callback: Callable[[int, int, BatchImportItemResult], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> BatchImportSummary:
    """Import multiple line files, continuing after individual failures.

    ``progress_callback`` is invoked after each file-level result.  It is used by
    the Qt background-import worker to update the progress dialog without
    touching project/UI state from the GUI thread.  ``cancel_requested`` is
    checked between files; already imported files are kept, remaining files are
    reported as cancelled so the summary still covers the complete selection.
    """
    results: list[BatchImportItemResult] = []
    reserved: set[str] = set()
    total = len(sources)
    for idx, source in enumerate(sources, start=1):
        src = Path(source)
        inferred = infer_line_identity_from_filename(src, fallback_index=idx)
        line_id = _unique_line_id(store, inferred.line_id, reserved)
        reserved.add(line_id)
        name = inferred.name if line_id == inferred.line_id else f"{inferred.name}（{line_id}）"
        file_size_mb = round(src.stat().st_size / (1024 * 1024), 3) if src.exists() and src.is_file() else 0.0
        started = time.perf_counter()
        if cancel_requested is not None and cancel_requested():
            result = BatchImportItemResult(
                source=str(src),
                line_id=line_id,
                name=name,
                success=False,
                message="用户取消，未导入。",
                file_size_mb=file_size_mb,
                elapsed_s=round(time.perf_counter() - started, 3),
                diagnosis="用户取消后续文件；已成功导入的文件会保留。",
            )
            results.append(result)
            if progress_callback is not None:
                progress_callback(idx, total, result)
            continue
        try:
            # Full content validation already occurs inside the transactional import.
            # Avoid a redundant whole-file preflight scan for every batch item.
            validate_import_source(src)
            line = import_line_data(
                store, src, line_id=line_id, name=name, cancel_requested=cancel_requested
            )
            dataset = store.load_gpr_dataset(line.line_id)
            raw_dir = store.root / "raw" / line.line_id
            manifest_path = raw_dir / "import_manifest.json"
            result = BatchImportItemResult(
                source=str(src),
                line_id=line.line_id,
                name=line.name,
                success=True,
                message="已导入并归一化",
                sample_count=int(dataset.sample_count),
                trace_count=int(dataset.trace_count),
                length_m=float(dataset.length_m),
                file_size_mb=file_size_mb,
                elapsed_s=round(time.perf_counter() - started, 3),
                raw_dir=str(raw_dir),
                manifest_path=str(manifest_path) if manifest_path.exists() else "",
                diagnosis="",
            )
            results.append(result)
        except Exception as exc:
            result = BatchImportItemResult(
                source=str(src),
                line_id=line_id,
                name=name,
                success=False,
                message=str(exc),
                file_size_mb=file_size_mb,
                elapsed_s=round(time.perf_counter() - started, 3),
                raw_dir=str(store.root / "raw" / line_id),
                manifest_path="",
                diagnosis=diagnose_import_failure(src, str(exc)),
            )
            results.append(result)
        if progress_callback is not None:
            progress_callback(idx, total, results[-1])
    succeeded = sum(1 for row in results if row.success)
    summary = BatchImportSummary(total=total, succeeded=succeeded, failed=total - succeeded, results=tuple(results))
    store.append_log("；".join(summary.to_log_lines()))
    return summary




def _safe_timestamp_for_path() -> str:
    return local_now().replace(":", "-").replace(" ", "_")


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(2, 10_000):
        candidate = path.with_name(f"{path.name}_{idx}")
        if not candidate.exists():
            return candidate
    raise FieldProjectOperationError(f"无法创建唯一目录：{path}")


def _ensure_within_project(path: Path, project_root: Path) -> Path:
    """Return ``path`` resolved after confirming it is inside ``project_root``.

    Delete operations must never follow manifest/source metadata to user-owned
    original input files outside the MyGPR project directory.
    """
    root = project_root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FieldProjectOperationError(f"拒绝删除项目目录之外的文件：{resolved}") from exc
    if resolved == root:
        raise FieldProjectOperationError("拒绝通过测线删除操作删除整个项目目录。")
    return resolved


def _delete_project_local_path(src: Path, project_root: Path, deleted: list[str]) -> None:
    """Delete a project-local file/directory if present and record its path."""
    if not src.exists():
        return
    target = _ensure_within_project(src, project_root)
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    deleted.append(target.as_posix())


def delete_project_line(store: FieldProjectStore, line_id: str, *, reason: str = "用户删除测线") -> LineDeleteResult:
    """Move one line's project-local artifacts into the project recycle bin."""
    store.assert_writable()
    safe_line_id = validate_line_id(line_id)
    lines = store.list_lines()
    target_line = next((line for line in lines if line.line_id == safe_line_id), None)
    if target_line is None:
        raise FieldProjectOperationError(f"未找到测线：{safe_line_id}")
    project_root = store.root.resolve()
    stamp = utc_now().replace(":", "").replace("+", "_")
    trash_root = project_root / ".trash" / "lines" / f"{stamp}_{safe_line_id}"
    trash_root.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []

    def move_path(path: Path) -> None:
        if not path.exists():
            return
        source = _ensure_within_project(path, project_root)
        rel = source.relative_to(project_root)
        destination = trash_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved.append(source.as_posix())

    move_path(store.root / "raw" / safe_line_id)
    move_path(store.root / "processed" / safe_line_id)
    if getattr(getattr(store, "storage", None), "is_hybrid", False):
        move_path(store.storage.line_container_path(safe_line_id))
    move_path(store.root / "targets" / f"{safe_line_id}_targets.csv")
    spatial_dir = store.root / "spatial"
    if spatial_dir.exists():
        for path in sorted(spatial_dir.glob(f"{safe_line_id}_*")):
            move_path(path)
        move_path(spatial_dir / "project_spatial_coordinates.csv")

    source_records = [record.to_dict() for record in load_source_registry(store) if record.line_id == safe_line_id]
    remove_line_source_records(store, safe_line_id)
    remaining = [line for line in lines if line.line_id != safe_line_id]
    if getattr(getattr(store, "storage", None), "is_hybrid", False):
        store.storage.catalog.delete_line(safe_line_id)
    store.manifest.set_lines(remaining)
    if isinstance(store.manifest.reports, dict):
        store.manifest.reports.update({
            "status": "需重新生成",
            "stale_reason": f"测线 {safe_line_id} 已删除并移入回收站",
            "updated_at": local_now(),
        })
    atomic_write_json(trash_root / "trash_manifest.json", {
        "schema": "mygpr.line_trash.v1",
        "line": asdict(target_line),
        "source_records": source_records,
        "original_paths": moved,
        "reason": reason,
        "trashed_at": utc_now(),
    })
    store.save_manifest()
    store.append_log(f"测线移入回收站 {safe_line_id}: files={len(moved)}, reason={reason}, trash={trash_root}")
    return LineDeleteResult(
        line_id=safe_line_id,
        line_name=target_line.name,
        deleted_paths=tuple(moved),
        remaining_line_count=len(remaining),
    )


def remove_recent_project(project_path: str | Path, *, recent_store: RecentProjectsStore | None = None) -> int:
    """Remove a project from the recent-project list without touching files."""
    return (recent_store or RecentProjectsStore()).remove(project_path)


def _assert_safe_project_root_for_removal(store: FieldProjectStore) -> Path:
    root = store.root.resolve()
    manifest = root / PROJECT_MANIFEST_NAME
    if root.is_symlink() or not manifest.is_file() or manifest.is_symlink():
        raise FieldProjectOperationError(f"当前目录不是安全的 MyGPR 项目：{root}")
    forbidden = {Path(root.anchor).resolve(), Path.home().resolve()}
    for candidate in (Path.home() / "Desktop", Path.home() / "Documents", Path.home() / "Downloads"):
        forbidden.add(candidate.resolve())
    if root in forbidden or len(root.parts) < 3:
        raise FieldProjectOperationError(f"拒绝删除不安全的项目路径：{root}")
    try:
        validate_project_root_marker(root, store.manifest.project_id)
    except ValueError as exc:
        raise FieldProjectOperationError(str(exc)) from exc
    return root


def delete_project_permanently(
    store: FieldProjectStore,
    *,
    recent_store: RecentProjectsStore | None = None,
) -> ProjectDeleteResult:
    """Permanently delete the active MyGPR project directory.

    This removes only the MyGPR project folder.  Original input files outside
    that folder are not touched.  The function requires a valid project manifest
    in the target directory and refuses obvious unsafe roots.
    """
    root = _assert_safe_project_root_for_removal(store)
    project_name = store.manifest.name
    store.append_log("准备直接删除项目文件夹。")
    store.close()
    shutil.rmtree(root)
    removed = (recent_store or RecentProjectsStore()).remove(root)
    return ProjectDeleteResult(
        project_name=project_name,
        original_path=str(root),
        deleted_path=str(root),
        removed_recent_count=removed,
    )


archive_project_line = delete_project_line


def delete_project_to_trash(
    store: FieldProjectStore,
    *,
    recent_store: RecentProjectsStore | None = None,
    reason: str = "用户删除项目",
) -> ProjectDeleteResult:
    """Move the whole project to a sibling recycle-bin directory."""
    root = _assert_safe_project_root_for_removal(store)
    project_name = store.manifest.name
    stamp = utc_now().replace(":", "").replace("+", "_")
    trash_parent = root.parent / ".mygpr_trash"
    trash_parent.mkdir(parents=True, exist_ok=True)
    destination = _unique_destination(trash_parent / f"{stamp}_{root.name}")
    store.append_log(f"项目移入回收站：reason={reason}")
    store.close()
    shutil.move(str(root), str(destination))
    atomic_write_json(destination / "project_trash_record.json", {
        "schema": "mygpr.project_trash.v1",
        "project_name": project_name,
        "original_path": str(root),
        "trashed_path": str(destination),
        "reason": reason,
        "trashed_at": utc_now(),
    })
    removed = (recent_store or RecentProjectsStore()).remove(root)
    return ProjectDeleteResult(project_name, str(root), str(destination), removed)


def prune_missing_recent_projects(*, recent_store: RecentProjectsStore | None = None) -> int:
    """Remove stale recent project entries whose manifest is gone."""
    return (recent_store or RecentProjectsStore()).prune_missing()


def preflight_project_delete(
    store: FieldProjectStore,
    *,
    recent_store: RecentProjectsStore | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ProjectDeletePreflight:
    """Collect project-delete facts before the destructive operation starts."""
    root = store.root.resolve()
    if not (root / PROJECT_MANIFEST_NAME).exists():
        raise FieldProjectOperationError(f"当前目录不是有效 MyGPR 项目：{root}")
    file_count = 0
    total_bytes = 0
    for dir_path, _dir_names, file_names in os.walk(root):
        if cancel_requested is not None and cancel_requested():
            from core.job_manager import JobCancelled
            raise JobCancelled("项目删除预检已取消")
        for file_name in file_names:
            path = Path(dir_path) / file_name
            file_count += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
        if progress_callback is not None:
            progress_callback(file_count, 0, f"扫描项目文件：{Path(dir_path).name}")
    lines = store.list_lines()
    processed_count = sum(1 for line in lines if bool(line.processed_result))
    reports_dir = root / "reports"
    report_file_count = sum(1 for path in reports_dir.rglob("*") if path.is_file()) if reports_dir.exists() else 0
    source_records = load_source_registry(store)
    external_source_count = 0
    for record in source_records:
        try:
            src = Path(record.source_path).expanduser().resolve()
            src.relative_to(root)
        except Exception:
            external_source_count += 1
    missing_recent = 0
    if recent_store is not None:
        missing_recent = sum(
            1
            for record in recent_store.load()
            if not (Path(record.path).expanduser().resolve() / PROJECT_MANIFEST_NAME).exists()
        )
    return ProjectDeletePreflight(
        project_name=store.manifest.name,
        project_path=str(root),
        size_mb=round(total_bytes / (1024 * 1024), 3),
        file_count=file_count,
        line_count=len(lines),
        processed_count=processed_count,
        report_file_count=report_file_count,
        external_source_count=external_source_count,
        missing_recent_count=missing_recent,
    )


def project_dialog_filter() -> str:
    return supported_file_dialog_filter()


__all__ = [
    "DIRECT_READABLE_EXTENSIONS",
    "infer_line_identity_from_filename",
    "batch_import_line_data",
    "InferredLineIdentity",
    "BatchImportSummary",
    "BatchImportItemResult",
    "ProjectBackupResult",
    "ProjectRestoreResult",
    "LineDeleteResult",
    "ProjectDeleteResult",
    "ProjectDeletePreflight",
    "diagnose_import_failure",
    "FieldProjectOperationError",
    "ProjectMetadataUpdate",
    "RecentProjectRecord",
    "RecentProjectsStore",
    "create_project",
    "default_recent_projects_path",
    "export_line_manifest_csv",
    "backup_project_archive",
    "restore_project_archive",
    "delete_project_line",
    "delete_project_permanently",
    "remove_recent_project",
    "import_line_data",
    "import_trajectory_file",
    "next_line_id",
    "open_project",
    "preview_import_source",
    "update_project_metadata",
    "check_project_source_files",
    "export_project_source_manifest_csv",
    "preflight_project_delete",
    "prune_missing_recent_projects",
    "relink_project_line_source",
    "project_dialog_filter",
    "validate_import_source",
    "validate_project_root",
]
