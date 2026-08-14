#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Line, raw-data and trajectory persistence mixin for field projects."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from core.field_project_models import FieldLineRecord, count_csv_rows, local_now, validate_line_id
from core.coordinate_projection import ProjectionError, project_lonlat_to_xy
from core.gpr_data_model import GPRDataSet, load_gpr_dataset, load_gpr_dataset_for_import
from core.chunked_gpr_io import (
    ImportCancelled,
    LARGE_DATASET_THRESHOLD_BYTES,
    copy_file_chunked,
    save_dataset_directory,
)
from core.trajectory_model import TrajectoryModel, TrajectoryPoint
from core.field_data_quality import LineDataQualityReport, evaluate_line_data_quality


class FieldLineStoreMixin:
    """Manage measurement-line records and normalized raw artifacts."""

    @staticmethod
    def _safe_line_id(line_id: str) -> str:
        return validate_line_id(line_id)

    def list_lines(self) -> list[FieldLineRecord]:
        return self.manifest.line_records()

    def upsert_line(self, line: FieldLineRecord) -> None:
        lines = self.list_lines()
        for idx, existing in enumerate(lines):
            if existing.line_id == line.line_id:
                lines[idx] = line
                break
        else:
            lines.append(line)
        self.manifest.set_lines(lines)
        self.save_manifest()
        if getattr(self.storage, "is_hybrid", False):
            h5_path = self.storage.line_container_relative_path(line.line_id) if self.storage.line_container_path(line.line_id).exists() else ""
            self.storage.catalog.upsert_line(line, h5_path=h5_path)

    def get_line(self, line_id: str) -> FieldLineRecord:
        line_id = self._safe_line_id(line_id)
        for line in self.list_lines():
            if line.line_id == line_id:
                return line
        raise KeyError(line_id)

    def import_line_file(
        self,
        line_id: str,
        source: str | Path,
        *,
        name: str | None = None,
        copy_into_project: bool = True,
        dielectric_constant: float = 9.0,
        cancel_requested=None,
        progress_callback=None,
    ) -> FieldLineRecord:
        """Import one line with cooperative cancellation and bounded memory.

        Matrix normalization is performed through a project-local staging
        directory.  The caller's transaction removes the entire raw directory
        on failure/cancellation, while this method also cleans its own staging
        artifacts defensively.
        """
        line_id = self._safe_line_id(line_id)
        src = Path(source).resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        dest = self.root / "raw" / line_id / src.name if copy_into_project else src
        staging = self.root / "raw" / line_id / f".import_staging_{uuid.uuid4().hex}"
        if copy_into_project:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src != dest:
                copy_file_chunked(
                    src, dest, cancel_requested=cancel_requested, progress_callback=progress_callback
                )
        try:
            line = self.get_line(line_id)
        except KeyError:
            line = FieldLineRecord(line_id=line_id, name=name or line_id)
        size_mb = dest.stat().st_size / (1024 * 1024)
        line.name = name or line.name
        line.raw_path = dest.relative_to(self.root).as_posix() if dest.is_relative_to(self.root) else str(dest)
        # Avoid scanning a multi-gigabyte CSV a second time only to populate a
        # decorative row count.  The normalized dataset supplies sample_count.
        line.raw_rows = 0
        line.raw_size_mb = round(size_mb, 3)
        line.processing_status = "已导入"
        line.updated_at = local_now()
        self.upsert_line(line)
        self.append_log(f"导入测线 {line_id}: {dest.name}, size={line.raw_size_mb:.3f}MB")
        try:
            if src.suffix.lower() not in {".csv", ".txt", ".npy", ".npz", ".h5", ".hdf5"}:
                return self.get_line(line_id)
            dataset = load_gpr_dataset_for_import(
                dest,
                line_id=line_id,
                staging_dir=staging,
                length_m=line.length_m or None,
                dielectric_constant=float(dielectric_constant),
                cancel_requested=cancel_requested,
                progress_callback=progress_callback,
            )
            trajectory_rows = dataset.metadata.pop("trajectory_rows", None) if isinstance(dataset.metadata, dict) else None
            self.save_gpr_dataset(
                line_id, dataset, cancel_requested=cancel_requested, progress_callback=progress_callback
            )
            projection_payload = {"status": "not_applicable"}
            if trajectory_rows:
                lon = [float(row.get("longitude", 0.0)) for row in trajectory_rows]
                lat = [float(row.get("latitude", 0.0)) for row in trajectory_rows]
                try:
                    x_values, y_values, projection = project_lonlat_to_xy(
                        lon, lat, coordinate_system=getattr(self.manifest, "coordinate_system", ""),
                    )
                    projection_payload = {
                        "status": "ok", "coordinate_system": projection.description,
                        "epsg": projection.epsg, "zone": projection.zone,
                        "source_epsg": projection.source_epsg, "is_auto": projection.is_auto,
                    }
                except ProjectionError as exc:
                    x_values, y_values = lon, lat
                    projection_payload = {
                        "status": "failed",
                        "coordinate_system": getattr(self.manifest, "coordinate_system", ""),
                        "error": str(exc),
                    }
                    self.append_log(f"测线 {line_id} 坐标投影失败，空间成果将标记为未投影：{exc}")
                coord_text = str(projection_payload.get("coordinate_system") or "未投影")
                points = [
                    TrajectoryPoint(
                        distance_m=float(row.get("distance_m", 0.0)),
                        x=float(x_values[idx]), y=float(y_values[idx]),
                        # CSV 第 3 列 elevation 是地表高程、第 5 列 height_m 是
                        # 飞行高度（离地）；轨迹海拔 = 地表 + 飞行高度，
                        # 与 sensor_sync 的 local_z = ground + flight 语义一致
                        z=float(row.get("elevation", 0.0)) + float(row.get("height_m", 0.0)),
                        flight_height_m=float(row.get("height_m", 0.0)),
                        quality="已投影" if projection_payload.get("status") == "ok" else "未投影",
                        longitude=float(row.get("longitude", 0.0)),
                        latitude=float(row.get("latitude", 0.0)),
                        coordinate_system=coord_text,
                    )
                    for idx, row in enumerate(trajectory_rows)
                ]
                self.save_trajectory(line_id, TrajectoryModel(points))
                projected_line = self.get_line(line_id)
                projected_line.rtk_status = "已投影" if projection_payload.get("status") == "ok" else "未投影"
                self.upsert_line(projected_line)
            self.write_json(
                self.root / "raw" / line_id / "import_manifest.json",
                {
                    "schema": "mygpr.import_manifest.v2",
                    "line_id": line_id, "source_path": str(src),
                    "copied_path": dest.relative_to(self.root).as_posix() if dest.is_relative_to(self.root) else str(dest),
                    "format_name": dataset.format_name, "sample_count": dataset.sample_count,
                    "trace_count": dataset.trace_count, "length_m": dataset.length_m,
                    "time_window_ns": dataset.time_window_ns, "has_trajectory": bool(trajectory_rows),
                    "projection": projection_payload,
                    "columns": dataset.metadata.get("columns", []) if isinstance(dataset.metadata, dict) else [],
                    "storage_mode": (
                        "hdf5_line_container" if getattr(self.storage, "is_hybrid", False)
                        else ("memmap_directory" if Path(self.root / self.get_line(line_id).gpr_dataset_path).is_dir() else "npz")
                    ),
                    "imported_at": local_now(),
                },
            )
            try:
                self.run_line_quality_check(line_id)
            except Exception as qc_exc:
                self.append_log(f"测线 {line_id} 数据质检失败: {qc_exc}")
            return self.get_line(line_id)
        except ImportCancelled:
            self.append_log(f"测线 {line_id} 导入已取消，正在回滚。")
            raise
        except Exception as exc:
            self.append_log(f"测线 {line_id} 未归一化为矩阵数据: {exc}")
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)


    def gpr_dataset_path(self, line_id: str) -> Path:
        line_id = self._safe_line_id(line_id)
        if getattr(self.storage, "is_hybrid", False):
            return self.storage.line_container_path(line_id)
        return self.root / "raw" / line_id / f"{line_id}_gpr_dataset.npz"

    def gpr_chunked_dataset_path(self, line_id: str) -> Path:
        line_id = self._safe_line_id(line_id)
        return self.root / "raw" / line_id / f"{line_id}_gpr_dataset"

    def gpr_metadata_path(self, line_id: str) -> Path:
        line_id = self._safe_line_id(line_id)
        return self.root / "raw" / line_id / f"{line_id}_gpr_meta.json"

    def trajectory_path(self, line_id: str) -> Path:
        line_id = self._safe_line_id(line_id)
        return self.root / "raw" / line_id / f"{line_id}_trajectory.csv"

    def quality_report_path(self, line_id: str) -> Path:
        line_id = self._safe_line_id(line_id)
        return self.root / "raw" / line_id / f"{line_id}_quality_report.json"

    def save_quality_report(self, line_id: str, report: LineDataQualityReport) -> Path:
        line_id = self._safe_line_id(line_id)
        path = self.quality_report_path(line_id)
        self.write_json(path, report.to_dict())
        try:
            line = self.get_line(line_id)
        except KeyError:
            line = FieldLineRecord(line_id=line_id, name=line_id)
        line.data_quality = report.status_label
        line.updated_at = report.checked_at
        self.upsert_line(line)
        self.append_log(f"数据质检 {line_id}: {report.status_label}; {report.orientation_message}")
        return path

    def load_quality_report(self, line_id: str) -> LineDataQualityReport | None:
        line_id = self._safe_line_id(line_id)
        path = self.quality_report_path(line_id)
        if not path.exists():
            return None
        try:
            import json
            return LineDataQualityReport.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def run_line_quality_check(self, line_id: str) -> LineDataQualityReport:
        line_id = self._safe_line_id(line_id)
        dataset = self.load_gpr_dataset(line_id)
        try:
            trajectory = self.load_trajectory(line_id)
        except Exception:
            trajectory = None
        report = evaluate_line_data_quality(dataset, trajectory)
        self.save_quality_report(line_id, report)
        return report

    def run_project_quality_check(
        self,
        *,
        cancel_requested=None,
        progress_callback=None,
    ) -> list[LineDataQualityReport]:
        reports: list[LineDataQualityReport] = []
        lines = [line for line in self.list_lines() if line.gpr_dataset_path]
        for index, line in enumerate(lines, start=1):
            if cancel_requested is not None and cancel_requested():
                from core.job_manager import JobCancelled
                raise JobCancelled("项目质检已取消")
            try:
                reports.append(self.run_line_quality_check(line.line_id))
            except Exception as exc:
                self.append_log(f"数据质检 {line.line_id} 失败: {exc}")
            if progress_callback is not None:
                progress_callback(index, max(len(lines), 1), f"质检 {line.line_id}")
        return reports

    def save_gpr_dataset(
        self,
        line_id: str,
        dataset: GPRDataSet,
        *,
        cancel_requested=None,
        progress_callback=None,
    ) -> Path:
        line_id = self._safe_line_id(line_id)
        matrix_bytes = int(dataset.matrix.size * np.dtype(dataset.matrix.dtype).itemsize)
        if getattr(self.storage, "is_hybrid", False):
            path, data_sha256 = self.storage.save_raw_dataset(
                line_id, dataset, cancel_requested=cancel_requested, progress_callback=progress_callback
            )
            storage_mode = "hdf5_line_container"
        else:
            npz_path = self.gpr_dataset_path(line_id)
            chunked_path = self.gpr_chunked_dataset_path(line_id)
            if matrix_bytes >= LARGE_DATASET_THRESHOLD_BYTES:
                source_npy = dataset.metadata.get("staging_matrix_path") if isinstance(dataset.metadata, dict) else None
                path = save_dataset_directory(
                    chunked_path, matrix=dataset.matrix,
                    distance_axis_m=dataset.distance_axis_m, time_axis_ns=dataset.time_axis_ns,
                    depth_axis_m=dataset.depth_axis_m, metadata=dataset.to_metadata(),
                    cancel_requested=cancel_requested, progress_callback=progress_callback,
                    source_npy=source_npy if source_npy and Path(source_npy).exists() else None,
                )
                npz_path.unlink(missing_ok=True)
                storage_mode = "memmap_directory"
            else:
                path = dataset.save_npz(
                    npz_path,
                    cancel_checker=cancel_requested,
                    progress_callback=(
                        (lambda current, total, message: progress_callback(message, current, total))
                        if progress_callback is not None else None
                    ),
                )
                if chunked_path.exists():
                    shutil.rmtree(chunked_path)
                storage_mode = "npz"
        metadata = dataset.to_metadata()
        metadata["storage_mode"] = storage_mode
        metadata["matrix_bytes"] = matrix_bytes
        if getattr(self.storage, "is_hybrid", False):
            metadata["data_sha256"] = data_sha256
            metadata["dataset_path"] = "/raw/bscan"
        self.write_json(self.gpr_metadata_path(line_id), metadata)
        try:
            line = self.get_line(line_id)
        except KeyError:
            line = FieldLineRecord(line_id=line_id, name=line_id)
        line.length_m = round(float(dataset.length_m), 2) if dataset.length_m else line.length_m
        line.raw_rows = int(dataset.sample_count)
        line.trace_count = int(dataset.trace_count)
        line.raw_size_mb = round(self._path_size(path) / (1024 * 1024), 3)
        line.gpr_dataset_path = path.relative_to(self.root).as_posix()
        line.data_format = dataset.format_name
        line.processing_status = "已导入" if line.processing_status in {"未处理", "未定位", ""} else line.processing_status
        line.updated_at = local_now()
        self.upsert_line(line)
        self.append_log(
            f"保存归一化 GPR 数据 {line_id}: {dataset.sample_count}×{dataset.trace_count}, "
            f"length={dataset.length_m:.2f}m, storage={storage_mode}"
        )
        return path

    @staticmethod
    def _path_size(path: Path) -> int:
        if path.is_dir():
            return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        return path.stat().st_size

    def load_gpr_dataset(self, line_id: str) -> GPRDataSet:
        line_id = self._safe_line_id(line_id)
        line = self.get_line(line_id)
        if getattr(self.storage, "is_hybrid", False):
            path = self.storage.line_container_path(line_id)
            if not path.exists():
                raise FileNotFoundError(path)
            return self.storage.load_raw_dataset(line_id)
        path = self.root / line.gpr_dataset_path if line.gpr_dataset_path else self.gpr_dataset_path(line_id)
        if not path.exists():
            alternative = self.gpr_chunked_dataset_path(line_id)
            if alternative.exists():
                path = alternative
            else:
                raise FileNotFoundError(path)
        return load_gpr_dataset(path, line_id=line_id, length_m=line.length_m or None, mmap_mode="r")

    def transpose_gpr_dataset(
        self,
        line_id: str,
        *,
        cancel_requested=None,
        progress_callback=None,
    ) -> LineDataQualityReport:
        """Transpose a normalized B-scan through a background-safe transaction.

        The original normalized dataset is backed up before the replacement is
        committed.  Large datasets use a directory-backed, cancellable backup;
        compact datasets keep the historical NPZ backup contract.
        """
        from core.job_manager import JobCancelled

        def check_cancel() -> None:
            if cancel_requested is not None and cancel_requested():
                raise JobCancelled("B-scan 方向修正已取消")

        def report(current: int, total: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(int(current), int(total), str(message))

        line_id = self._safe_line_id(line_id)
        check_cancel()
        report(0, 100, "读取当前标准化 B-scan")
        dataset = self.load_gpr_dataset(line_id)
        old_shape = [int(dataset.sample_count), int(dataset.trace_count)]
        matrix_bytes = int(dataset.matrix.size * dataset.matrix.dtype.itemsize)
        fix_dir = self.root / "raw" / line_id / "orientation_fixes"
        fix_dir.mkdir(parents=True, exist_ok=True)
        timestamp = local_now().replace(":", "-").replace(" ", "_")

        report(5, 100, "备份修正前数据")
        if matrix_bytes >= LARGE_DATASET_THRESHOLD_BYTES:
            backup_path = fix_dir / f"{line_id}_before_transpose_{timestamp}"
            save_dataset_directory(
                backup_path,
                matrix=dataset.matrix,
                distance_axis_m=dataset.distance_axis_m,
                time_axis_ns=dataset.time_axis_ns,
                depth_axis_m=dataset.depth_axis_m,
                metadata=dataset.to_metadata(),
                cancel_requested=cancel_requested,
                progress_callback=(
                    (lambda _stage, current, total: report(5 + int(25 * current / max(total, 1)), 100, "分块备份原始矩阵"))
                    if progress_callback is not None else None
                ),
            )
        else:
            backup_path = fix_dir / f"{line_id}_before_transpose_{timestamp}.npz"
            dataset.save_npz(
                backup_path,
                cancel_checker=cancel_requested,
                progress_callback=lambda current, total, message: report(5 + 25 * current, 100, message),
            )

        check_cancel()
        metadata = dict(dataset.metadata or {})
        fixes = list(metadata.get("orientation_fixes", [])) if isinstance(metadata.get("orientation_fixes", []), list) else []
        fixes.append(
            {
                "operation": "transpose",
                "applied_at": local_now(),
                "old_shape": old_shape,
                "backup_path": backup_path.relative_to(self.root).as_posix(),
                "reason": "manual_quality_action",
            }
        )
        metadata["orientation_fixes"] = fixes
        metadata["orientation_corrected"] = True
        transposed = GPRDataSet.from_matrix(
            line_id,
            dataset.matrix.T,
            length_m=float(dataset.length_m),
            time_window_ns=float(dataset.time_window_ns),
            dielectric_constant=float(dataset.dielectric_constant),
            source_path=dataset.source_path,
            format_name=f"{dataset.format_name}+transpose",
            metadata=metadata,
        )
        report(35, 100, "分块写入转置结果")
        saved_path = self.save_gpr_dataset(
            line_id,
            transposed,
            cancel_requested=cancel_requested,
            progress_callback=(
                (lambda current, total, _message: report(35 + int(50 * current / max(total, 1)), 100, "写入转置矩阵"))
                if progress_callback is not None else None
            ),
        )
        # The replacement has committed.  Do not honour a late cancellation as
        # an apparent rollback; complete metadata and quality records atomically.
        manifest_path = self.root / "raw" / line_id / "orientation_fix_manifest.json"
        payload = {
            "schema": "mygpr.orientation_fix.v2",
            "line_id": line_id,
            "operation": "transpose",
            "applied_at": local_now(),
            "old_shape": old_shape,
            "new_shape": [int(transposed.sample_count), int(transposed.trace_count)],
            "backup_path": backup_path.relative_to(self.root).as_posix(),
            "dataset_path": saved_path.relative_to(self.root).as_posix(),
            "axis_rebuild_policy": "matrix_transposed_axes_rebuilt_from_previous_length_and_time_window",
            "axis_warning": "Distance/time/depth axes were rebuilt from the previous normalized dataset metadata; verify against source CSV before using as final engineering deliverable.",
        }
        self.write_json(manifest_path, payload)
        report(90, 100, "重新执行数据质检")
        quality = self.run_line_quality_check(line_id)
        self.append_log(
            f"B-scan 方向修正 {line_id}: {old_shape[0]}×{old_shape[1]} -> {transposed.sample_count}×{transposed.trace_count}"
        )
        report(100, 100, "B-scan 方向修正完成")
        return quality

    def save_trajectory(self, line_id: str, trajectory: TrajectoryModel) -> Path:
        line_id = self._safe_line_id(line_id)
        path = trajectory.to_csv(self.trajectory_path(line_id))
        try:
            line = self.get_line(line_id)
        except KeyError:
            line = FieldLineRecord(line_id=line_id, name=line_id)
        line.trajectory_path = path.relative_to(self.root).as_posix()
        line.updated_at = local_now()
        self.upsert_line(line)
        self.append_log(f"保存 RTK/IMU 轨迹 {line_id}: {path.name}, points={len(trajectory.points)}")
        return path

    def load_trajectory(self, line_id: str) -> TrajectoryModel:
        line_id = self._safe_line_id(line_id)
        try:
            line = self.get_line(line_id)
            path = self.root / line.trajectory_path if line.trajectory_path else self.trajectory_path(line_id)
        except KeyError:
            path = self.trajectory_path(line_id)
        if path.exists():
            return TrajectoryModel.from_csv(path)
        raise FileNotFoundError(path)


__all__ = ["FieldLineStoreMixin"]
