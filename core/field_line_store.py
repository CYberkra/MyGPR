#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Line, raw-data and trajectory persistence mixin for field projects."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from core.field_project_models import FieldLineRecord, count_csv_rows, local_now, validate_line_id
from core.coordinate_projection import ProjectionError, project_lonlat_to_xy
from core.gpr_data_model import GPRDataSet, load_gpr_dataset
from core.trajectory_model import TrajectoryModel, TrajectoryPoint
from core.field_data_quality import LineDataQualityReport, evaluate_line_data_quality


class FieldLineStoreMixin:
    """Manage measurement-line records and normalized raw artifacts."""

    @staticmethod
    def _safe_line_id(line_id: str) -> str:
        return validate_line_id(line_id)

    def _default_lines(self) -> list[FieldLineRecord]:
        now = local_now()
        return [
            FieldLineRecord("L01", "经度道路主线", 212.35, "★★★★★", "固定解", "已完成", now, target_count=6),
            FieldLineRecord("L02", "纬向道路辅线", 184.62, "★★★★☆", "固定解", "已完成", now, target_count=4),
            FieldLineRecord("L03", "过路口测线", 121.40, "★★★★☆", "浮动解", "处理中", now, target_count=5),
            FieldLineRecord("L04", "人行道测线", 96.83, "★★★☆☆", "固定解", "未处理", now, target_count=2),
            FieldLineRecord("L05", "雨水管线疑似A", 156.22, "★★★★☆", "固定解", "已完成", now, target_count=4),
            FieldLineRecord("L06", "雨水管线疑似B", 143.78, "★★★☆☆", "浮动解", "未处理", now, target_count=3),
            FieldLineRecord("L07", "检查井区域", 88.91, "★★★★☆", "固定解", "已完成", now, target_count=2),
            FieldLineRecord("L08", "横穿支路测线", 73.54, "★★★☆☆", "浮动解", "未处理", now, target_count=1),
        ]

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
    ) -> FieldLineRecord:
        line_id = self._safe_line_id(line_id)
        src = Path(source).resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        dest = self.root / "raw" / line_id / src.name if copy_into_project else src
        if copy_into_project:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src != dest:
                shutil.copy2(src, dest)
        try:
            line = self.get_line(line_id)
        except KeyError:
            line = FieldLineRecord(line_id=line_id, name=name or line_id)
        size_mb = dest.stat().st_size / (1024 * 1024)
        line.name = name or line.name
        line.raw_path = dest.relative_to(self.root).as_posix() if dest.is_relative_to(self.root) else str(dest)
        line.raw_rows = count_csv_rows(dest) if dest.suffix.lower() == ".csv" else 0
        line.raw_size_mb = round(size_mb, 3)
        line.data_quality = "★★★★☆" if line.raw_rows else line.data_quality
        line.processing_status = "已导入"
        line.updated_at = local_now()
        self.upsert_line(line)
        self.append_log(f"导入测线 {line_id}: {dest.name}, rows={line.raw_rows}, size={line.raw_size_mb:.3f}MB")
        # Supported matrix-like files are normalized into the project GPR data
        # contract. Sidecar/non-matrix files remain accepted as raw evidence.
        try:
            if src.suffix.lower() in {".csv", ".txt", ".npy", ".npz", ".h5", ".hdf5"}:
                dataset = load_gpr_dataset(dest, line_id=line_id, length_m=line.length_m or None)
                self.save_gpr_dataset(line_id, dataset)
                trajectory_rows = dataset.metadata.get("trajectory_rows") if isinstance(dataset.metadata, dict) else None
                projection_payload = {"status": "not_applicable"}
                if trajectory_rows:
                    lon = [float(row.get("longitude", 0.0)) for row in trajectory_rows]
                    lat = [float(row.get("latitude", 0.0)) for row in trajectory_rows]
                    try:
                        x_values, y_values, projection = project_lonlat_to_xy(
                            lon,
                            lat,
                            coordinate_system=getattr(self.manifest, "coordinate_system", ""),
                        )
                        projection_payload = {
                            "status": "ok",
                            "coordinate_system": projection.description,
                            "epsg": projection.epsg,
                            "zone": projection.zone,
                            "source_epsg": projection.source_epsg,
                            "is_auto": projection.is_auto,
                        }
                    except ProjectionError as exc:
                        x_values = lon
                        y_values = lat
                        projection_payload = {
                            "status": "failed",
                            "coordinate_system": getattr(self.manifest, "coordinate_system", ""),
                            "error": str(exc),
                        }
                        self.append_log(f"测线 {line_id} 坐标投影失败，空间成果将标记为未投影：{exc}")
                    points = []
                    coord_text = str(projection_payload.get("coordinate_system") or "未投影")
                    for idx, row in enumerate(trajectory_rows):
                        points.append(
                            TrajectoryPoint(
                                distance_m=float(row.get("distance_m", 0.0)),
                                x=float(x_values[idx]),
                                y=float(y_values[idx]),
                                z=float(row.get("elevation", 0.0)),
                                quality="已投影" if projection_payload.get("status") == "ok" else "未投影",
                                longitude=float(row.get("longitude", 0.0)),
                                latitude=float(row.get("latitude", 0.0)),
                                coordinate_system=coord_text,
                            )
                        )
                    self.save_trajectory(line_id, TrajectoryModel(points))
                    try:
                        projected_line = self.get_line(line_id)
                        projected_line.rtk_status = "已投影" if projection_payload.get("status") == "ok" else "未投影"
                        self.upsert_line(projected_line)
                    except KeyError:
                        pass
                self.write_json(
                    self.root / "raw" / line_id / "import_manifest.json",
                    {
                        "schema": "mygpr.import_manifest.v1",
                        "line_id": line_id,
                        "source_path": str(src),
                        "copied_path": dest.relative_to(self.root).as_posix() if dest.is_relative_to(self.root) else str(dest),
                        "format_name": dataset.format_name,
                        "sample_count": dataset.sample_count,
                        "trace_count": dataset.trace_count,
                        "length_m": dataset.length_m,
                        "time_window_ns": dataset.time_window_ns,
                        "has_trajectory": bool(trajectory_rows),
                        "projection": projection_payload,
                        "columns": dataset.metadata.get("columns", []) if isinstance(dataset.metadata, dict) else [],
                        "imported_at": local_now(),
                    },
                )
                try:
                    self.run_line_quality_check(line_id)
                except Exception as qc_exc:
                    self.append_log(f"测线 {line_id} 数据质检失败: {qc_exc}")
        except Exception as exc:
            self.append_log(f"测线 {line_id} 未归一化为矩阵数据: {exc}")
        try:
            return self.get_line(line_id)
        except KeyError:
            return line

    def gpr_dataset_path(self, line_id: str) -> Path:
        line_id = self._safe_line_id(line_id)
        return self.root / "raw" / line_id / f"{line_id}_gpr_dataset.npz"

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

    def run_project_quality_check(self) -> list[LineDataQualityReport]:
        reports: list[LineDataQualityReport] = []
        for line in self.list_lines():
            if not line.gpr_dataset_path:
                continue
            try:
                reports.append(self.run_line_quality_check(line.line_id))
            except Exception as exc:
                self.append_log(f"数据质检 {line.line_id} 失败: {exc}")
        return reports

    def save_gpr_dataset(self, line_id: str, dataset: GPRDataSet) -> Path:
        line_id = self._safe_line_id(line_id)
        path = self.gpr_dataset_path(line_id)
        dataset.save_npz(path)
        self.write_json(self.gpr_metadata_path(line_id), dataset.to_metadata())
        try:
            line = self.get_line(line_id)
        except KeyError:
            line = FieldLineRecord(line_id=line_id, name=line_id)
        line.length_m = round(float(dataset.length_m), 2) if dataset.length_m else line.length_m
        line.raw_rows = int(dataset.sample_count)
        line.raw_size_mb = round(path.stat().st_size / (1024 * 1024), 3)
        line.gpr_dataset_path = path.relative_to(self.root).as_posix()
        line.data_format = dataset.format_name
        line.processing_status = "已导入" if line.processing_status in {"未处理", "未定位", ""} else line.processing_status
        line.updated_at = local_now()
        self.upsert_line(line)
        self.append_log(f"保存归一化 GPR 数据 {line_id}: {dataset.sample_count}×{dataset.trace_count}, length={dataset.length_m:.2f}m")
        return path

    def load_gpr_dataset(self, line_id: str) -> GPRDataSet:
        line_id = self._safe_line_id(line_id)
        line = self.get_line(line_id)
        path = self.root / line.gpr_dataset_path if line.gpr_dataset_path else self.gpr_dataset_path(line_id)
        if not path.exists():
            raise FileNotFoundError(path)
        return load_gpr_dataset(path, line_id=line_id, length_m=line.length_m or None)

    def transpose_gpr_dataset(self, line_id: str) -> LineDataQualityReport:
        """Transpose the normalized B-scan matrix for a line and rerun quality checks.

        This operation is intentionally explicit and manifest-backed.  It is not
        called automatically by the quality checker because a transpose changes
        the interpretation of rows/columns.  The UI exposes it only as a user
        action when the quality report reports ``transpose_risk``.
        """
        line_id = self._safe_line_id(line_id)
        dataset = self.load_gpr_dataset(line_id)
        old_shape = [int(dataset.sample_count), int(dataset.trace_count)]
        fix_dir = self.root / "raw" / line_id / "orientation_fixes"
        fix_dir.mkdir(parents=True, exist_ok=True)
        timestamp = local_now().replace(":", "-").replace(" ", "_")
        backup_path = fix_dir / f"{line_id}_before_transpose_{timestamp}.npz"
        dataset.save_npz(backup_path)
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
        self.save_gpr_dataset(line_id, transposed)
        manifest_path = self.root / "raw" / line_id / "orientation_fix_manifest.json"
        payload = {
            "schema": "mygpr.orientation_fix.v1",
            "line_id": line_id,
            "operation": "transpose",
            "applied_at": local_now(),
            "old_shape": old_shape,
            "new_shape": [int(transposed.sample_count), int(transposed.trace_count)],
            "backup_path": backup_path.relative_to(self.root).as_posix(),
            "dataset_path": self.gpr_dataset_path(line_id).relative_to(self.root).as_posix(),
            "axis_rebuild_policy": "matrix_transposed_axes_rebuilt_from_previous_length_and_time_window",
            "axis_warning": "Distance/time/depth axes were rebuilt from the previous normalized dataset metadata; verify against source CSV before using as final engineering deliverable.",
        }
        self.write_json(manifest_path, payload)
        report = self.run_line_quality_check(line_id)
        self.append_log(
            f"B-scan 方向修正 {line_id}: {old_shape[0]}×{old_shape[1]} -> {transposed.sample_count}×{transposed.trace_count}"
        )
        return report

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

    def ensure_demo_gpr_artifacts(self, line_id: str = "L03") -> None:
        line_id = self._safe_line_id(line_id)
        try:
            line = self.get_line(line_id)
            length = line.length_m or 212.35
        except KeyError:
            length = 212.35
        needs_demo = not self.gpr_dataset_path(line_id).exists()
        if not needs_demo:
            try:
                existing = self.load_gpr_dataset(line_id)
                needs_demo = existing.sample_count < 64 or existing.trace_count < 96
            except Exception:
                needs_demo = True
        if needs_demo:
            self.save_gpr_dataset(line_id, GPRDataSet.synthetic(line_id=line_id, length_m=length))
        if not self.trajectory_path(line_id).exists():
            self.save_trajectory(line_id, TrajectoryModel.demo(length_m=length))


__all__ = ["FieldLineStoreMixin"]
