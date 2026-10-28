"""Identity-preserving metadata writes and adoption of a verified project copy."""

import os
from dataclasses import replace
from core.field_project_models import local_now
from core.field_project_store import FieldProjectStore
from core.project_mutation import project_mutation
from core.project_integrity import ProjectIntegrityAuditor
from core.security_paths import resolve_managed_path
from core.storage_uri import is_h5_uri, parse_h5_uri
from core.processing_artifact_index import index_processing_artifacts
from core.report_package_versions import ReportPackageVersionService
from core.spatial_result_versions import SpatialResultVersionService


def rename_project_store(store, name):
    store.assert_writable()
    if store.manifest.name != name:
        with project_mutation(store, "rename-project"):
            store.manifest.name = name
            store.save_manifest()


def rename_line_record(store, line_id, name):
    store.assert_writable()
    line = store.get_line(line_id)
    if line.name != name:
        with project_mutation(store, "rename-line"):
            line = replace(line, name=name, updated_at=local_now())
            store.upsert_line(line)
    return line


def _portable(root, value, label):
    if not value:
        return
    path = parse_h5_uri(value).file_path if is_h5_uri(value) else str(value)
    try:
        resolve_managed_path(root, path, require_exists=True)
    except (ValueError, OSError) as exc:
        raise ValueError(f"迁移受阻：{label} 含外部、绝对或失效路径引用，请先收纳或修正：{value}") from exc


def validate_relocation_references(store):
    """Check current dependencies; historical source/provenance paths stay intact."""
    root = store.root
    if getattr(store.storage, "is_hybrid", False):
        _portable(root, store.manifest.catalog_path or "catalog.sqlite", "项目数据库")
    fields = (
        "raw_path",
        "gpr_dataset_path",
        "trajectory_path",
        "processed_result",
        "params_path",
        "sensor_sync_manifest_path",
        "trace_metadata_path",
    )
    for line in store.list_lines():
        for field in fields:
            _portable(root, getattr(line, field, ""), f"{line.line_id} / {field}")
        for artifact in index_processing_artifacts(root, line.line_id):
            for field in ("data_path", "params_path", "manifest_path"):
                _portable(root, getattr(artifact, field, ""), f"处理成果 / {field}")
    if getattr(store.storage, "is_hybrid", False):
        for row in store.storage.catalog.list_artifacts():
            _portable(root, row.get("h5_path", ""), "处理成果容器")
    for folder in ("reports/versions", "spatial/results"):
        resolve_managed_path(root, folder)
    reports = ReportPackageVersionService(store)
    for manifest in (root / "reports/versions").glob("*/report_manifest.json"):
        record = reports.load_report(manifest.parent.name)
        for key, value in record.result.items():
            if key == "package_dir" or key.endswith("_path"):
                _portable(root, value, f"报告 / {key}")
    spatial = SpatialResultVersionService(store)
    for manifest in (root / "spatial/results").glob("*/manifest.json"):
        spatial.load_result(manifest.parent.name)


def open_relocated_store(old, stage, target, check_cancelled):
    probe = FieldProjectStore.open(stage, access_mode="read_only")
    try:
        report = ProjectIntegrityAuditor(probe).audit(
            repair_context=False, clean_staging=False, staging_min_age_s=3600.0, deep_hash=False
        )
        if not report.healthy:
            raise ValueError("项目完整性检查未通过，请先修复项目后迁移。")
        if probe.manifest.project_id != old.manifest.project_id:
            raise ValueError("迁移副本项目身份不匹配。")
    finally:
        probe.close()
    check_cancelled()
    if target.exists():
        raise FileExistsError("目标目录已被占用，请选择其他目录。")
    os.rename(stage, target)
    # Keep the verified target if opening it fails; the original remains open.
    new_store = FieldProjectStore.open(target, access_mode="write")
    adopted = False
    try:
        old.close()
        adopted = True
        return new_store
    finally:
        if not adopted:
            new_store.close()
