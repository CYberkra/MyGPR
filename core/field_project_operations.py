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
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
import logging

from core.field_project_models import FieldProjectManifest, local_now, validate_line_id
from core.field_project_store import FieldLineRecord, FieldProjectStore
from core.gpr_format_registry import get_format_spec, supported_file_dialog_filter
from core.field_import_preview import build_import_preflight, ImportPreflightResult
from core.source_file_registry import (
    check_all_source_files,
    export_source_file_manifest_csv,
    load_source_registry,
    record_source_file,
    relink_line_source_file,
    remove_line_source_records,
    source_summary,
)
from core.app_errors import MyGPRError, error_info_from_exception

DIRECT_READABLE_EXTENSIONS = {".csv", ".txt", ".npy", ".npz", ".h5", ".hdf5"}
PROJECT_MANIFEST_NAME = FieldProjectStore.MANIFEST_NAME
RECENT_PROJECTS_SCHEMA = "mygpr.recent_projects.v1"

logger = logging.getLogger(__name__)


class FieldProjectOperationError(MyGPRError):
    """Raised when a user-facing project operation cannot be completed."""


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
class ProjectBackupResult:
    """Result of creating a portable project backup archive."""

    archive_path: str
    file_count: int
    size_mb: float


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
    parent_or_root: str | Path,
    *,
    name: str,
    location: str = "",
    operator: str = "操作员",
    project_no: str = "",
    device_model: str = "",
    coordinate_system: str = "",
    vertical_datum: str = "",
    create_child_dir: bool = True,
    recent_store: RecentProjectsStore | None = None,
) -> FieldProjectStore:
    """Create a formal project without demo measurements."""
    if not name.strip():
        raise FieldProjectOperationError("项目名称不能为空。")
    base = Path(parent_or_root).expanduser().resolve()
    root = base / _safe_project_slug(name) if create_child_dir else base
    if (root / PROJECT_MANIFEST_NAME).exists():
        raise FieldProjectOperationError(f"该目录已存在 MyGPR 项目：{root}")
    store = FieldProjectStore.create_empty(
        root,
        name=name.strip(),
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



def preview_import_source(source: str | Path, *, line_id: str = "L01", dielectric_constant: float = 9.0) -> ImportPreflightResult:
    """Return user-facing import diagnostics without mutating the project."""
    safe_line_id = validate_line_id(line_id)
    return build_import_preflight(source, line_id=safe_line_id, dielectric_constant=dielectric_constant)

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
        store.import_line_file(target_line_id, src, name=target_name, copy_into_project=True)
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
            preview = preview_import_source(src, line_id=line_id)
            if not preview.can_import:
                raise FieldProjectOperationError(preview.message)
            line = import_line_data(store, src, line_id=line_id, name=name)
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
    """Permanently delete one line's project-local artifacts and manifest entry.

    Only files under ``store.root`` are removed.  Original source files selected
    by the user during import are not touched unless they themselves live inside
    the active MyGPR project directory.
    """
    safe_line_id = validate_line_id(line_id)
    lines = store.list_lines()
    target_line = next((line for line in lines if line.line_id == safe_line_id), None)
    if target_line is None:
        raise FieldProjectOperationError(f"未找到测线：{safe_line_id}")

    deleted: list[str] = []
    project_root = store.root.resolve()
    _delete_project_local_path(store.root / "raw" / safe_line_id, project_root, deleted)
    _delete_project_local_path(store.root / "processed" / safe_line_id, project_root, deleted)
    _delete_project_local_path(store.root / "targets" / f"{safe_line_id}_targets.csv", project_root, deleted)

    spatial_dir = store.root / "spatial"
    if spatial_dir.exists():
        for path in sorted(spatial_dir.glob(f"{safe_line_id}_*")):
            _delete_project_local_path(path, project_root, deleted)
        # Aggregate coordinate exports become stale when the line set changes;
        # delete them so users regenerate current spatial coordinates.
        _delete_project_local_path(spatial_dir / "project_spatial_coordinates.csv", project_root, deleted)

    remove_line_source_records(store, safe_line_id)
    remaining = [line for line in lines if line.line_id != safe_line_id]
    store.manifest.set_lines(remaining)
    if isinstance(store.manifest.reports, dict):
        store.manifest.reports["status"] = "需重新生成"
        store.manifest.reports["stale_reason"] = f"测线 {safe_line_id} 已删除"
        store.manifest.reports["updated_at"] = local_now()
    store.save_manifest()
    store.append_log(f"删除测线 {safe_line_id}: files={len(deleted)}, reason={reason}")
    return LineDeleteResult(
        line_id=safe_line_id,
        line_name=target_line.name,
        deleted_paths=tuple(deleted),
        remaining_line_count=len(remaining),
    )


def remove_recent_project(project_path: str | Path, *, recent_store: RecentProjectsStore | None = None) -> int:
    """Remove a project from the recent-project list without touching files."""
    return (recent_store or RecentProjectsStore()).remove(project_path)


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
    root = store.root.resolve()
    if not (root / PROJECT_MANIFEST_NAME).exists():
        raise FieldProjectOperationError(f"当前目录不是有效 MyGPR 项目：{root}")
    if root == root.parent or root == Path.home().resolve():
        raise FieldProjectOperationError(f"拒绝删除不安全的项目路径：{root}")
    project_name = store.manifest.name
    store.append_log("准备直接删除项目文件夹。")
    removed = (recent_store or RecentProjectsStore()).remove(root)
    shutil.rmtree(root)
    return ProjectDeleteResult(
        project_name=project_name,
        original_path=str(root),
        deleted_path=str(root),
        removed_recent_count=removed,
    )


# Backward-compatible aliases for callers that have not yet been updated.  The
# behavior is direct deletion, not archiving/trash moving.
archive_project_line = delete_project_line
delete_project_to_trash = delete_project_permanently


def prune_missing_recent_projects(*, recent_store: RecentProjectsStore | None = None) -> int:
    """Remove stale recent project entries whose manifest is gone."""
    return (recent_store or RecentProjectsStore()).prune_missing()


def preflight_project_delete(
    store: FieldProjectStore,
    *,
    recent_store: RecentProjectsStore | None = None,
) -> ProjectDeletePreflight:
    """Collect project-delete facts before the destructive operation starts."""
    root = store.root.resolve()
    if not (root / PROJECT_MANIFEST_NAME).exists():
        raise FieldProjectOperationError(f"当前目录不是有效 MyGPR 项目：{root}")
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_file():
            file_count += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
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


def check_project_source_files(store: FieldProjectStore):
    """Check all recorded source files and return updated records."""
    records = check_all_source_files(store)
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
):
    record = relink_line_source_file(store, line_id, new_source, allow_mismatch=allow_mismatch)
    store.append_log(f"重新定位源文件 {line_id}: {record.source_path}; status={record.status}")
    return record


def export_project_source_manifest_csv(store: FieldProjectStore, destination: str | Path | None = None) -> Path:
    out = export_source_file_manifest_csv(store, destination)
    store.append_log(f"导出源文件清单: {out}")
    return out


def export_line_manifest_csv(store: FieldProjectStore, destination: str | Path | None = None) -> Path:
    """Export the current line manifest to CSV for quick field review."""
    import csv

    out = Path(destination) if destination is not None else store.root / "reports" / "line_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
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
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for line in store.list_lines():
            writer.writerow({key: getattr(line, key, "") for key in headers})
    store.append_log(f"导出测线清单: {out}")
    return out


def backup_project_archive(store: FieldProjectStore, destination_dir: str | Path | None = None) -> ProjectBackupResult:
    """Create a ZIP backup of the opened project, excluding runtime caches."""
    backup_dir = Path(destination_dir) if destination_dir is not None else store.root / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = local_now().replace(":", "-").replace(" ", "_")
    archive = backup_dir / f"{store.root.name}_backup_{stamp}.zip"
    excluded_parts = {".git", ".venv", "__pycache__", "backups"}
    file_count = 0
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in store.root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(store.root)
            if any(part in excluded_parts for part in rel.parts):
                continue
            zf.write(path, rel.as_posix())
            file_count += 1
    size_mb = round(archive.stat().st_size / (1024 * 1024), 3)
    store.append_log(f"创建项目备份: {archive}, files={file_count}, size={size_mb:.3f}MB")
    return ProjectBackupResult(str(archive), file_count, size_mb)

def import_trajectory_file(
    store: FieldProjectStore,
    source: str | Path,
    *,
    line_id: str,
) -> Path:
    """Copy an RTK/IMU trajectory evidence file into the raw line directory.

    Full semantic parsing remains in ``TrajectoryModel``.  This operation gives
    the user a formal place to attach source evidence without pretending that all
    vendor trajectory formats have been decoded.
    """
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
    except Exception as provenance_exc:
        error_info = error_info_from_exception(provenance_exc)
        logger.warning("Trajectory source file record failed: %s [%s]", error_info.user_message, error_info.error_code)
        store.append_log(f"测线 {line_id} 轨迹来源文件记录失败: {error_info.user_message}")
    store.append_log(f"附加 RTK/IMU 文件 {line_id}: {dest.name}")
    return dest


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
