#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable, versioned spatial-result snapshots for field projects.

The UI may redraw live project data at any time, but formal GIS deliverables must
be reproducible.  This module freezes the exact annotation/processing/trajectory
inputs used for one spatial result and writes plain, inspectable GIS files under
``spatial/results/<result_id>/``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.field_project_models import local_now, validate_line_id
from core.security_paths import ensure_direct_child, resolve_managed_path, safe_relative_path
from core.storage_uri import is_h5_uri, resolve_h5_uri

SPATIAL_RESULT_SCHEMA = "mygpr.spatial_result.v1"
SPATIAL_RESULT_INDEX_SCHEMA = "mygpr.spatial_result_index.v1"


@dataclass(frozen=True)
class SpatialResultRecord:
    result_id: str
    name: str
    revision: int
    created_at: str
    status: str
    coordinate_system: str
    vertical_datum: str
    line_ids: tuple[str, ...]
    options: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    source_fingerprint: str = ""
    stale: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, stale: bool = False) -> "SpatialResultRecord":
        return cls(
            result_id=str(payload.get("result_id") or ""),
            name=str(payload.get("name") or payload.get("result_id") or ""),
            revision=int(payload.get("revision") or 1),
            created_at=str(payload.get("created_at") or ""),
            status=str(payload.get("status") or "generated"),
            coordinate_system=str(payload.get("coordinate_system") or ""),
            vertical_datum=str(payload.get("vertical_datum") or ""),
            line_ids=tuple(str(item) for item in payload.get("line_ids", [])),
            options=dict(payload.get("options") or {}),
            sources=dict(payload.get("sources") or {}),
            files=dict(payload.get("files") or {}),
            summary=dict(payload.get("summary") or {}),
            source_fingerprint=str(payload.get("source_fingerprint") or ""),
            stale=bool(stale),
        )


class SpatialResultVersionService:
    """Create, list, verify and export immutable spatial-result versions."""

    def __init__(self, project_store: Any):
        self.project = project_store
        self.root = Path(project_store.root) / "spatial" / "results"
        self.index_path = self.root / "index.json"

    def list_results(self) -> list[SpatialResultRecord]:
        records: list[SpatialResultRecord] = []
        if not self.root.exists():
            return records
        for manifest_path in self.root.glob("*/manifest.json"):
            try:
                records.append(self.load_result(manifest_path.parent.name))
            except (OSError, UnicodeError, ValueError, TypeError, KeyError):
                continue
        records.sort(key=lambda item: (item.created_at, item.revision, item.result_id), reverse=True)
        return records

    def current_result_id(self) -> str:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            return str(payload.get("current_result_id") or "")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return ""

    def set_current(self, result_id: str) -> None:
        record = self.load_result(result_id)
        self._atomic_json(
            self.index_path,
            {
                "schema": SPATIAL_RESULT_INDEX_SCHEMA,
                "current_result_id": record.result_id,
                "updated_at": local_now(),
            },
        )

    def load_result(self, result_id: str) -> SpatialResultRecord:
        safe = _safe_result_id(result_id)
        path = self.root / safe / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(path)
        result_dir = ensure_direct_child(self.root, path.parent)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != SPATIAL_RESULT_SCHEMA:
            raise ValueError(f"Unsupported spatial result schema: {payload.get('schema')!r}")
        if str(payload.get("result_id") or "") != safe:
            raise ValueError(f"空间成果清单编号与目录不一致：{safe}")
        files = payload.get("files") or {}
        if not isinstance(files, dict):
            raise ValueError("空间成果 files 字段必须是对象。")
        for label, raw_rel in files.items():
            rel = safe_relative_path(str(raw_rel or ""))
            candidate = resolve_managed_path(result_dir, rel, require_file=True)
            if candidate.parent != result_dir:
                raise ValueError(f"空间成果文件必须直接位于版本目录：{label}={raw_rel}")
        return SpatialResultRecord.from_dict(
            payload,
            stale=self._is_payload_stale(payload),
        )

    def preflight(
        self,
        *,
        line_ids: Iterable[str] | None = None,
        generate_surface: bool = True,
    ) -> dict[str, Any]:
        requested = [validate_line_id(item) for item in (line_ids or [line.line_id for line in self.project.list_lines()])]
        usable: list[str] = []
        warnings: list[str] = []
        errors: list[str] = []
        located = 0
        confirmed = 0
        for line_id in requested:
            try:
                annotation = self.project.load_basal_interface_annotation(line_id)
            except Exception:
                annotation = None
            if annotation is None:
                warnings.append(f"{line_id}：缺少基覆界面标注")
                continue
            try:
                trajectory = self.project.load_trajectory(line_id)
            except Exception:
                trajectory = None
            if trajectory is None or not trajectory.points:
                warnings.append(f"{line_id}：缺少 RTK/GNSS 轨迹，不能生成空间曲线")
                continue
            located += 1
            if str(getattr(annotation, "status", "draft")) == "confirmed":
                confirmed += 1
            else:
                warnings.append(f"{line_id}：标注仍为草稿")
            usable.append(line_id)
        if not usable:
            errors.append("没有同时具备界面标注与空间轨迹的测线。")
        coordinate_system = str(getattr(self.project.manifest, "coordinate_system", "") or "")
        if not coordinate_system or "未" in coordinate_system:
            warnings.append("项目坐标系未完整配置，成果将保留风险标记。")
        surface_allowed = len(usable) >= 3 and _has_spatial_spread(self.project, usable)
        if generate_surface and not surface_allowed:
            warnings.append("测线数量或空间分布不足，连续界面曲面不会生成。")
        return {
            "schema": "mygpr.spatial_preflight.v1",
            "requested_line_ids": requested,
            "usable_line_ids": usable,
            "located_line_count": located,
            "confirmed_annotation_count": confirmed,
            "surface_requested": bool(generate_surface),
            "surface_allowed": bool(surface_allowed),
            "warnings": warnings,
            "errors": errors,
            "passed": not errors,
        }

    def create_result(
        self,
        *,
        name: str = "basal_surface",
        line_ids: Iterable[str] | None = None,
        velocity_m_per_ns: float = 0.08,
        generate_depth_curve: bool = True,
        generate_elevation_curve: bool = True,
        generate_surface: bool = True,
        max_extrapolation_distance_m: float = 4.0,
        max_extrapolation_angle_deg: float = 15.0,
        cancel_requested=None,
        progress_callback=None,
    ) -> SpatialResultRecord:
        self.project.assert_writable()
        preflight = self.preflight(line_ids=line_ids, generate_surface=generate_surface)
        if not preflight["passed"]:
            raise ValueError("；".join(preflight["errors"]))
        usable = list(preflight["usable_line_ids"])
        base_name = _slug(name or "basal_surface")
        revision = self._next_revision(base_name)
        result_id = f"{base_name}_v{revision:03d}"
        staging = self.root / f".{result_id}.{uuid.uuid4().hex}.staging"
        destination = self.root / result_id
        if destination.exists():
            raise FileExistsError(destination)
        staging.mkdir(parents=True, exist_ok=False)
        created_at = local_now()
        profiles_path = staging / "profiles.csv"
        interfaces_path = staging / "interfaces.geojson"
        trajectories_path = staging / "trajectories.geojson"
        surface_path = staging / "surface_control_points.geojson"
        source_manifest_path = staging / "source_manifest.json"
        profile_fields = [
            "result_id", "line_id", "trace_index", "distance_m", "sample_index",
            "time_ns", "depth_m", "x", "y", "surface_z", "interface_z",
            "visibility", "is_no_interface", "is_ignored", "annotation_version",
            "annotation_status", "processing_result", "trajectory_path",
        ]
        interface_features: list[dict[str, Any]] = []
        trajectory_features: list[dict[str, Any]] = []
        surface_features: list[dict[str, Any]] = []
        source_lines: dict[str, Any] = {}
        point_count = 0
        committed = False
        try:
            with profiles_path.open("w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=profile_fields)
                writer.writeheader()
                for line_index, line_id in enumerate(usable, start=1):
                    _check_cancel(cancel_requested)
                    if progress_callback is not None:
                        progress_callback(line_index - 1, max(len(usable), 1), f"冻结 {line_id} 空间曲线")
                    annotation = self.project.load_basal_interface_annotation(line_id)
                    line = self.project.get_line(line_id)
                    spatial_csv = self.project.export_spatial_interface_curve(line_id, annotation)
                    rows: list[dict[str, Any]] = []
                    with spatial_csv.open("r", encoding="utf-8-sig", newline="") as source_fh:
                        for row_index, row in enumerate(csv.DictReader(source_fh), start=1):
                            if row_index % 1024 == 0:
                                _check_cancel(cancel_requested)
                            depth = _float_or_none(row.get("depth_m"))
                            time_ns = _float_or_none(row.get("time_ns"))
                            if depth is None and time_ns is not None:
                                depth = time_ns * float(velocity_m_per_ns) / 2.0
                            surface_z = _float_or_none(row.get("surface_z"))
                            interface_z = _float_or_none(row.get("interface_z"))
                            if interface_z is None and surface_z is not None and depth is not None:
                                interface_z = surface_z - depth
                            enriched = {
                                "result_id": result_id,
                                "line_id": line_id,
                                "trace_index": row.get("trace_index", ""),
                                "distance_m": row.get("distance_m", ""),
                                "sample_index": row.get("sample_index", ""),
                                "time_ns": row.get("time_ns", ""),
                                "depth_m": "" if depth is None else f"{depth:.6f}",
                                "x": row.get("x", ""),
                                "y": row.get("y", ""),
                                "surface_z": row.get("surface_z", ""),
                                "interface_z": "" if interface_z is None else f"{interface_z:.6f}",
                                "visibility": row.get("visibility", "unknown"),
                                "is_no_interface": row.get("is_no_interface", "0"),
                                "is_ignored": row.get("is_ignored", "0"),
                                "annotation_version": str(getattr(annotation, "version", "")),
                                "annotation_status": str(getattr(annotation, "status", "draft")),
                                "processing_result": line.processed_result,
                                "trajectory_path": line.trajectory_path,
                            }
                            writer.writerow(enriched)
                            rows.append(enriched)
                            if enriched["x"] not in (None, "") and enriched["y"] not in (None, ""):
                                point_count += 1
                                if enriched["is_no_interface"] not in ("1", 1, True) and interface_z is not None:
                                    surface_features.append(
                                        _point_feature(
                                            float(enriched["x"]), float(enriched["y"]), interface_z,
                                            {
                                                "result_id": result_id,
                                                "line_id": line_id,
                                                "trace_index": _int_or_none(enriched["trace_index"]),
                                                "visibility": enriched["visibility"],
                                                "depth_m": depth,
                                            },
                                        )
                                    )
                    interface_features.extend(_rows_to_interface_features(result_id, line_id, rows))
                    try:
                        trajectory = self.project.load_trajectory(line_id)
                    except Exception:
                        trajectory = None
                    if trajectory is not None and trajectory.points:
                        coords = [[float(p.x), float(p.y), float(p.z)] for p in trajectory.points]
                        trajectory_features.append(
                            {
                                "type": "Feature",
                                "geometry": {"type": "LineString", "coordinates": coords},
                                "properties": {"result_id": result_id, "line_id": line_id, "role": "trajectory"},
                            }
                        )
                    source_lines[line_id] = self._line_source_payload(line_id)
            source_fingerprint = _fingerprint_source_payload(source_lines)
            source_state_token = _state_token_from_source_payload(source_lines)
            _write_geojson(interfaces_path, interface_features, self.project.manifest.coordinate_system)
            _write_geojson(trajectories_path, trajectory_features, self.project.manifest.coordinate_system)
            surface_written = bool(generate_surface and preflight["surface_allowed"] and surface_features)
            if surface_written:
                _write_geojson(surface_path, surface_features, self.project.manifest.coordinate_system)
            sources = {
                "line_sources": source_lines,
                "source_fingerprint": source_fingerprint,
                "source_state_token": source_state_token,
                "created_at": created_at,
            }
            self._atomic_json(source_manifest_path, sources)
            options = {
                "velocity_m_per_ns": float(velocity_m_per_ns),
                "generate_depth_curve": bool(generate_depth_curve),
                "generate_elevation_curve": bool(generate_elevation_curve),
                "generate_surface": bool(generate_surface),
                "surface_generated": surface_written,
                "max_extrapolation_distance_m": float(max_extrapolation_distance_m),
                "max_extrapolation_angle_deg": float(max_extrapolation_angle_deg),
            }
            files = {
                "profiles_csv": "profiles.csv",
                "interfaces_geojson": "interfaces.geojson",
                "trajectories_geojson": "trajectories.geojson",
                "source_manifest": "source_manifest.json",
            }
            if surface_written:
                files["surface_control_points_geojson"] = "surface_control_points.geojson"
            payload = {
                "schema": SPATIAL_RESULT_SCHEMA,
                "result_id": result_id,
                "name": name or base_name,
                "revision": revision,
                "created_at": created_at,
                "status": "generated_with_warnings" if preflight["warnings"] else "generated",
                "coordinate_system": str(getattr(self.project.manifest, "coordinate_system", "") or ""),
                "vertical_datum": str(getattr(self.project.manifest, "vertical_datum", "") or ""),
                "line_ids": usable,
                "options": options,
                "sources": sources,
                "files": files,
                "summary": {
                    "line_count": len(usable),
                    "interface_feature_count": len(interface_features),
                    "trajectory_feature_count": len(trajectory_features),
                    "spatial_point_count": point_count,
                    "surface_control_point_count": len(surface_features) if surface_written else 0,
                    "preflight_warnings": list(preflight["warnings"]),
                },
                "source_fingerprint": source_fingerprint,
                "source_state_token": source_state_token,
            }
            self._atomic_json(staging / "manifest.json", payload)
            _check_cancel(cancel_requested)
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(destination)
            committed = True
            self.set_current(result_id)
            if hasattr(self.project, "append_log"):
                self.project.append_log(f"生成空间成果版本 {result_id}：{len(usable)} 条测线，{point_count} 个空间点")
            if progress_callback is not None:
                progress_callback(max(len(usable), 1), max(len(usable), 1), "空间成果版本已生成")
            return self.load_result(result_id)
        finally:
            if not committed:
                shutil.rmtree(staging, ignore_errors=True)

    def export_result(self, result_id: str, destination: str | Path, *, format_name: str = "zip") -> Path:
        record = self.load_result(result_id)
        result_dir = self.root / record.result_id
        output = Path(destination)
        fmt = str(format_name or output.suffix.lstrip(".") or "zip").lower()
        if fmt in {"zip", "package"}:
            if output.suffix.lower() != ".zip":
                output = output.with_suffix(".zip")
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
            try:
                with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for path in sorted(result_dir.rglob("*")):
                        if path.is_file():
                            archive.write(path, arcname=f"{record.result_id}/{path.relative_to(result_dir).as_posix()}")
                tmp.replace(output)
            finally:
                tmp.unlink(missing_ok=True)
            return output
        mapping = {
            "geojson": record.files.get("interfaces_geojson", "interfaces.geojson"),
            "csv": record.files.get("profiles_csv", "profiles.csv"),
            "kml": "interfaces.geojson",
        }
        if fmt in {"geojson", "csv"}:
            source = resolve_managed_path(result_dir, mapping[fmt], require_file=True)
            if output.suffix.lower() != f".{fmt}":
                output = output.with_suffix(f".{fmt}")
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
            return output
        if fmt == "kml":
            if output.suffix.lower() != ".kml":
                output = output.with_suffix(".kml")
            _geojson_to_kml(resolve_managed_path(result_dir, record.files.get("interfaces_geojson", "interfaces.geojson"), require_file=True), output, record)
            return output
        if fmt in {"gpkg", "shp"}:
            return _export_with_fiona(resolve_managed_path(result_dir, record.files.get("interfaces_geojson", "interfaces.geojson"), require_file=True), output, fmt, record.coordinate_system)
        raise ValueError(f"Unsupported spatial export format: {format_name}")

    def move_to_recycle_bin(self, result_id: str) -> Path:
        self.project.assert_writable()
        record = self.load_result(result_id)
        source = self.root / record.result_id
        recycle = Path(self.project.root) / "spatial" / "recycle_bin"
        recycle.mkdir(parents=True, exist_ok=True)
        destination = recycle / f"{record.result_id}_{uuid.uuid4().hex[:8]}"
        source.replace(destination)
        if self.current_result_id() == record.result_id:
            remaining = self.list_results()
            self._atomic_json(
                self.index_path,
                {
                    "schema": SPATIAL_RESULT_INDEX_SCHEMA,
                    "current_result_id": remaining[0].result_id if remaining else "",
                    "updated_at": local_now(),
                },
            )
        return destination

    def _next_revision(self, base_name: str) -> int:
        pattern = re.compile(rf"^{re.escape(base_name)}_v(\d+)$")
        revisions = []
        for record in self.list_results():
            match = pattern.fullmatch(record.result_id)
            if match:
                revisions.append(int(match.group(1)))
        return max(revisions, default=0) + 1

    def _line_source_payload(self, line_id: str) -> dict[str, Any]:
        annotation = self.project.load_basal_interface_annotation(line_id)
        line = self.project.get_line(line_id)
        paths = {
            "annotation": self.project.interface_annotation_path(line_id),
            "gpr_dataset": _project_path(self.project.root, line.gpr_dataset_path),
            "trajectory": _project_path(self.project.root, line.trajectory_path),
            "processing_result": _project_path(self.project.root, line.processed_result),
            "processing_params": _project_path(self.project.root, line.params_path),
        }
        return {
            "annotation_version": str(getattr(annotation, "version", "")),
            "annotation_status": str(getattr(annotation, "status", "draft")),
            "source_result_id": str(getattr(annotation, "source_result_id", "")),
            "processing_result": line.processed_result,
            "trajectory_path": line.trajectory_path,
            "files": {
                key: {
                    "path": _relative_or_absolute(self.project.root, path),
                    "sha256": _hash_path(path),
                    "size": _path_size(path),
                    "mtime_ns": _path_mtime_ns(path),
                }
                for key, path in paths.items() if path is not None and path.exists()
            },
        }

    def _current_source_fingerprint(self, line_ids: Iterable[str]) -> str:
        payload: dict[str, Any] = {}
        for raw_line_id in line_ids:
            try:
                line_id = validate_line_id(raw_line_id)
                payload[line_id] = self._line_source_payload(line_id)
            except Exception as exc:
                payload[str(raw_line_id)] = {"error": str(exc)}
        return _fingerprint_source_payload(payload)

    def _current_source_state_token(self, line_ids: Iterable[str]) -> str:
        state: dict[str, Any] = {}
        for raw_line_id in line_ids:
            try:
                line_id = validate_line_id(raw_line_id)
                line = self.project.get_line(line_id)
                paths = {
                    "annotation": self.project.interface_annotation_path(line_id),
                    "gpr_dataset": _project_path(self.project.root, line.gpr_dataset_path),
                    "trajectory": _project_path(self.project.root, line.trajectory_path),
                    "processing_result": _project_path(self.project.root, line.processed_result),
                    "processing_params": _project_path(self.project.root, line.params_path),
                }
                state[line_id] = {
                    key: {
                        "path": _relative_or_absolute(self.project.root, path),
                        "size": _path_size(path),
                        "mtime_ns": _path_mtime_ns(path),
                    }
                    for key, path in paths.items() if path is not None and path.exists()
                }
            except Exception as exc:
                state[str(raw_line_id)] = {"error": str(exc)}
        return hashlib.sha256(
            json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _is_payload_stale(self, payload: dict[str, Any]) -> bool:
        line_ids = payload.get("line_ids", [])
        stored_state = str(payload.get("source_state_token") or (payload.get("sources") or {}).get("source_state_token") or "")
        if stored_state:
            return self._current_source_state_token(line_ids) != stored_state
        # Compatibility fallback for early V1 manifests.  This path may hash
        # source artifacts once, but newly created results always use the cheap
        # size/mtime token for normal UI refreshes.
        return self._current_source_fingerprint(line_ids) != str(payload.get("source_fingerprint") or "")

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def _safe_result_id(value: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", text):
        raise ValueError(f"非法空间成果编号：{value!r}")
    return text


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip()).strip("_").lower()
    if not text:
        text = "basal_surface"
    if not text[0].isalpha():
        text = f"result_{text}"
    return text[:96]


def _project_path(root: str | Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    if is_h5_uri(value):
        path, _dataset_path = resolve_h5_uri(root, value)
        return path if path.exists() else None
    # Project manifests must never turn an absolute or escaped path into an
    # implicitly trusted source.  All lineage inputs are project-managed.
    return resolve_managed_path(root, str(value), require_exists=True)


def _relative_or_absolute(root: str | Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _hash_path(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            with child.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(chunk)
    else:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _path_size(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    if path.is_dir():
        return int(sum(child.stat().st_size for child in path.rglob("*") if child.is_file()))
    return 0


def _path_mtime_ns(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_mtime_ns)
    if path.is_dir():
        mtimes = [path.stat().st_mtime_ns, *(child.stat().st_mtime_ns for child in path.rglob("*") if child.is_file())]
        return int(max(mtimes, default=0))
    return 0


def _fingerprint_source_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _state_token_from_source_payload(payload: dict[str, Any]) -> str:
    state: dict[str, Any] = {}
    for line_id, line_payload in payload.items():
        files = line_payload.get("files", {}) if isinstance(line_payload, dict) else {}
        state[line_id] = {
            key: {
                "path": item.get("path", ""),
                "size": int(item.get("size") or 0),
                "mtime_ns": int(item.get("mtime_ns") or 0),
            }
            for key, item in files.items() if isinstance(item, dict)
        }
    return hashlib.sha256(
        json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _check_cancel(cancel_requested) -> None:
    if cancel_requested is not None and cancel_requested():
        from core.job_manager import JobCancelled
        raise JobCancelled("空间成果生成已取消")


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_geojson(path: Path, features: list[dict[str, Any]], crs: str) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {"type": "name", "properties": {"name": crs}} if crs else None,
        "features": features,
    }
    if payload["crs"] is None:
        payload.pop("crs")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _point_feature(x: float, y: float, z: float | None, properties: dict[str, Any]) -> dict[str, Any]:
    coordinates = [x, y] if z is None else [x, y, z]
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": coordinates}, "properties": properties}


def _rows_to_interface_features(result_id: str, line_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    current: list[list[float]] = []
    current_visibility = "unknown"
    segment_index = 0

    def flush() -> None:
        nonlocal current, segment_index
        if len(current) >= 2:
            segment_index += 1
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": current},
                    "properties": {
                        "result_id": result_id,
                        "line_id": line_id,
                        "segment_id": f"{line_id}-S{segment_index:04d}",
                        "visibility": current_visibility,
                        "role": "basal_interface",
                    },
                }
            )
        current = []

    for row in rows:
        visibility = str(row.get("visibility") or "unknown")
        no_interface = str(row.get("is_no_interface") or "0") in {"1", "true", "True"}
        ignored = str(row.get("is_ignored") or "0") in {"1", "true", "True"}
        x = _float_or_none(row.get("x")); y = _float_or_none(row.get("y")); z = _float_or_none(row.get("interface_z"))
        if no_interface or ignored or x is None or y is None or z is None:
            flush()
            current_visibility = "unknown"
            continue
        if current and visibility != current_visibility:
            flush()
        current_visibility = visibility
        current.append([x, y, z])
    flush()
    return features


def _has_spatial_spread(project: Any, line_ids: list[str]) -> bool:
    centroids: list[tuple[float, float]] = []
    for line_id in line_ids:
        try:
            trajectory = project.load_trajectory(line_id)
        except (OSError, ValueError, TypeError, KeyError):
            continue
        if not trajectory.points:
            continue
        xs = [float(point.x) for point in trajectory.points]
        ys = [float(point.y) for point in trajectory.points]
        centroids.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    if len(centroids) < 3:
        return False
    x_span = max(item[0] for item in centroids) - min(item[0] for item in centroids)
    y_span = max(item[1] for item in centroids) - min(item[1] for item in centroids)
    return x_span > 1e-6 or y_span > 1e-6


def _geojson_to_kml(source: Path, destination: Path, record: SpatialResultRecord) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    placemarks: list[str] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        props = feature.get("properties") or {}
        coords = geometry.get("coordinates") or []
        coord_text = " ".join(",".join(str(value) for value in point[:3]) for point in coords)
        name = str(props.get("segment_id") or props.get("line_id") or "interface")
        placemarks.append(
            f"<Placemark><name>{_xml_escape(name)}</name><description>{_xml_escape(json.dumps(props, ensure_ascii=False))}</description>"
            f"<LineString><altitudeMode>absolute</altitudeMode><coordinates>{coord_text}</coordinates></LineString></Placemark>"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<kml xmlns=\"http://www.opengis.net/kml/2.2\"><Document>"
        f"<name>{_xml_escape(record.name)}</name>{''.join(placemarks)}</Document></kml>",
        encoding="utf-8",
    )


def _xml_escape(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _export_with_fiona(source: Path, destination: Path, fmt: str, crs: str) -> Path:
    try:
        import fiona
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("导出 Shapefile/GeoPackage 需要 Fiona。") from exc
    payload = json.loads(source.read_text(encoding="utf-8"))
    features = [feature for feature in payload.get("features", []) if (feature.get("geometry") or {}).get("type") == "LineString"]
    driver = "GPKG" if fmt == "gpkg" else "ESRI Shapefile"
    if fmt == "gpkg":
        if destination.suffix.lower() != ".gpkg":
            destination = destination.with_suffix(".gpkg")
    else:
        if destination.suffix.lower() != ".shp":
            destination = destination.with_suffix(".shp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        "geometry": "3D LineString",
        "properties": {
            "result_id": "str:80",
            "line_id": "str:32",
            "segment_id": "str:48",
            "visibility": "str:24",
            "role": "str:32",
        },
    }
    crs_arg = crs if crs.upper().startswith("EPSG:") else None
    with fiona.open(destination, "w", driver=driver, schema=schema, crs=crs_arg, encoding="UTF-8") as sink:
        for feature in features:
            properties = feature.get("properties") or {}
            sink.write(
                {
                    "geometry": feature.get("geometry"),
                    "properties": {key: str(properties.get(key, "")) for key in schema["properties"]},
                }
            )
    return destination


__all__ = [
    "SPATIAL_RESULT_SCHEMA",
    "SPATIAL_RESULT_INDEX_SCHEMA",
    "SpatialResultRecord",
    "SpatialResultVersionService",
]
