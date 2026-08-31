#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence/export helpers for continuous basal-interface annotations."""

from __future__ import annotations

import csv
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from core.basal_interface_annotations import (
    BASAL_LABEL_SCHEMA,
    BasalInterfaceAnnotation,
)
from core.chunked_gpr_io import LARGE_DATASET_THRESHOLD_BYTES, check_cancel
from core.field_project_models import local_now, validate_line_id
from core.storage_primitives import atomic_output_path, atomic_write_json


class FieldInterfaceStoreMixin:
    """Store interface JSON, training labels and spatial curve products."""

    def interface_annotation_path(self, line_id: str) -> Path:
        safe = validate_line_id(line_id)
        return self.root / "targets" / f"{safe}_basal_interface.json"

    def interface_history_dir(self, line_id: str) -> Path:
        safe = validate_line_id(line_id)
        return self.root / "targets" / "history" / safe

    def interface_labels_npz_path(self, line_id: str) -> Path:
        safe = validate_line_id(line_id)
        return self.root / "targets" / f"{safe}_basal_labels.npz"

    def interface_labels_dir(self, line_id: str) -> Path:
        safe = validate_line_id(line_id)
        return self.root / "targets" / f"{safe}_basal_labels"

    def interface_spatial_path(self, line_id: str) -> Path:
        safe = validate_line_id(line_id)
        return self.root / "spatial" / f"{safe}_basal_interface_xy.csv"

    def load_basal_interface_annotation(self, line_id: str, *, create: bool = False) -> BasalInterfaceAnnotation | None:
        safe = validate_line_id(line_id)
        path = self.interface_annotation_path(safe)
        if path.exists():
            return BasalInterfaceAnnotation.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if not create:
            return None
        dataset = self.load_gpr_dataset(safe)
        return BasalInterfaceAnnotation(
            line_id=safe,
            trace_count=int(dataset.trace_count),
            sample_count=int(dataset.sample_count),
            source_result_id=f"{safe}_raw",
            source_mode="raw",
        )

    def save_basal_interface_annotation(
        self,
        line_id: str,
        annotation: BasalInterfaceAnnotation,
        *,
        export_labels: bool = True,
        cancel_requested=None,
        cancel_checker=None,
        progress_callback=None,
    ) -> Path:
        safe = validate_line_id(line_id)
        cancel_requested = cancel_requested or cancel_checker
        if progress_callback is not None:
            progress_callback(0, 4, "保存标注 JSON")
        annotation.line_id = safe
        annotation.normalize()
        annotation.updated_at = local_now()
        current = self.interface_annotation_path(safe)
        if current.exists():
            history = self.interface_history_dir(safe)
            history.mkdir(parents=True, exist_ok=True)
            stamp = local_now().replace(":", "-").replace(" ", "_")
            shutil.copy2(current, history / f"{safe}_v{annotation.version:03d}_{stamp}.json")
            annotation.version = max(int(annotation.version) + 1, 2)
        atomic_write_json(current, annotation.to_dict())
        if export_labels:
            self.export_basal_interface_labels(
                safe, annotation, cancel_requested=cancel_requested, progress_callback=progress_callback
            )
        if progress_callback is not None:
            progress_callback(3, 4, "更新测线清单与空间曲线")
        try:
            line = self.get_line(safe)
            stats = annotation.statistics()
            line.target_count = 1 if int(stats["keypoint_count"]) > 0 else 0
            line.interface_status = "已确认" if annotation.status == "confirmed" else "草稿"
            line.interface_coverage = round(float(stats["coverage_ratio"]), 6)
            line.interface_keypoint_count = int(stats["keypoint_count"])
            line.updated_at = local_now()
            self.upsert_line(line)
        except KeyError:
            pass
        try:
            self.export_spatial_interface_curve(safe, annotation)
        except Exception as exc:
            self.append_log(f"基覆界面空间曲线暂未生成 {safe}: {exc}")
        self.append_log(
            f"保存基覆界面标注 {safe}: keypoints={len(annotation.keypoints)}, "
            f"coverage={annotation.statistics()['coverage_ratio']:.1%}, status={annotation.status}"
        )
        if progress_callback is not None:
            progress_callback(4, 4, "标注保存完成")
        return current

    def export_basal_interface_labels(
        self,
        line_id: str,
        annotation: BasalInterfaceAnnotation | None = None,
        *,
        cancel_requested=None,
        cancel_checker=None,
        progress_callback=None,
    ) -> Path:
        safe = validate_line_id(line_id)
        cancel_requested = cancel_requested or cancel_checker
        annotation = annotation or self.load_basal_interface_annotation(safe)
        if annotation is None:
            raise FileNotFoundError(self.interface_annotation_path(safe))
        dataset = self.load_gpr_dataset(safe)
        labels = annotation.build_1d_labels(
            time_axis_ns=dataset.time_axis_ns,
            depth_axis_m=dataset.depth_axis_m,
        )
        estimated_2d = int(annotation.sample_count * annotation.trace_count * (4 + 1))
        meta = {
            "schema": BASAL_LABEL_SCHEMA,
            "line_id": safe,
            "source_result_id": annotation.source_result_id,
            "source_mode": annotation.source_mode,
            "trace_count": annotation.trace_count,
            "sample_count": annotation.sample_count,
            "soft_sigma_samples": annotation.soft_sigma_samples,
            "annotation_version": annotation.version,
            "annotation_status": annotation.status,
            "generated_at": local_now(),
            "statistics": annotation.statistics(),
        }
        npz_path = self.interface_labels_npz_path(safe)
        directory = self.interface_labels_dir(safe)
        if estimated_2d < LARGE_DATASET_THRESHOLD_BYTES:
            soft = np.zeros((annotation.sample_count, annotation.trace_count), dtype=np.float32)
            ignore = np.zeros((annotation.sample_count, annotation.trace_count), dtype=bool)
            chunk_total = max(1, int(np.ceil(annotation.trace_count / 512)))
            for chunk_index, (start, end, soft_chunk, ignore_chunk) in enumerate(annotation.iter_soft_mask_chunks(), start=1):
                check_cancel(cancel_requested)
                soft[:, start:end] = soft_chunk
                ignore[:, start:end] = ignore_chunk
                if progress_callback is not None:
                    progress_callback(chunk_index, chunk_total, f"写入训练标签 {end}/{annotation.trace_count} 道")
            with atomic_output_path(npz_path, suffix=".tmp.npz") as temporary:
                np.savez_compressed(
                    temporary,
                    **labels,
                    soft_mask_gt=soft,
                    ignore_mask=ignore,
                    metadata=np.array(json.dumps(meta, ensure_ascii=False)),
                )
            shutil.rmtree(directory, ignore_errors=True)
            return npz_path

        staging = directory.with_name(f".{directory.name}.staging_{uuid.uuid4().hex}")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            for key, value in labels.items():
                np.save(staging / f"{key}.npy", value, allow_pickle=False)
            soft = np.lib.format.open_memmap(
                staging / "soft_mask_gt.npy", mode="w+", dtype=np.float32,
                shape=(annotation.sample_count, annotation.trace_count),
            )
            ignore = np.lib.format.open_memmap(
                staging / "ignore_mask.npy", mode="w+", dtype=np.bool_,
                shape=(annotation.sample_count, annotation.trace_count),
            )
            chunk_total = max(1, int(np.ceil(annotation.trace_count / 512)))
            for chunk_index, (start, end, soft_chunk, ignore_chunk) in enumerate(annotation.iter_soft_mask_chunks(), start=1):
                check_cancel(cancel_requested)
                soft[:, start:end] = soft_chunk
                ignore[:, start:end] = ignore_chunk
                if progress_callback is not None:
                    progress_callback(chunk_index, chunk_total, f"分块写入训练标签 {end}/{annotation.trace_count} 道")
            soft.flush(); ignore.flush()
            del soft, ignore
            atomic_write_json(staging / "metadata.json", meta)
            check_cancel(cancel_requested)
            shutil.rmtree(directory, ignore_errors=True)
            staging.replace(directory)
            npz_path.unlink(missing_ok=True)
            return directory
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def load_basal_interface_labels(self, line_id: str) -> dict[str, np.ndarray | dict[str, Any]]:
        safe = validate_line_id(line_id)
        directory = self.interface_labels_dir(safe)
        if directory.exists():
            payload: dict[str, np.ndarray | dict[str, Any]] = {}
            for path in directory.glob("*.npy"):
                payload[path.stem] = np.load(path, mmap_mode="r", allow_pickle=False)
            meta_path = directory / "metadata.json"
            payload["metadata"] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            return payload
        path = self.interface_labels_npz_path(safe)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as npz:
            payload = {key: np.asarray(npz[key]) for key in npz.files if key != "metadata"}
            if "metadata" in npz:
                payload["metadata"] = json.loads(str(npz["metadata"].item()))
        return payload

    def export_spatial_interface_curve(
        self,
        line_id: str,
        annotation: BasalInterfaceAnnotation | None = None,
    ) -> Path:
        safe = validate_line_id(line_id)
        annotation = annotation or self.load_basal_interface_annotation(safe)
        if annotation is None:
            raise FileNotFoundError(self.interface_annotation_path(safe))
        dataset = self.load_gpr_dataset(safe)
        labels = annotation.build_1d_labels(time_axis_ns=dataset.time_axis_ns, depth_axis_m=dataset.depth_axis_m)
        curve = labels["curve_gt"]
        semantics = annotation.trace_semantics()
        try:
            trajectory = self.load_trajectory(safe)
        except Exception:
            trajectory = None
        path = self.interface_spatial_path(safe)
        fields = [
            "line_id", "trace_index", "distance_m", "sample_index", "time_ns", "depth_m",
            "x", "y", "surface_z", "interface_z", "visibility", "is_no_interface", "is_ignored",
        ]
        with atomic_output_path(path, suffix=".csv.tmp") as temporary:
            with temporary.open("w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                for trace in range(annotation.trace_count):
                    if not np.isfinite(curve[trace]) and not semantics["no_interface_mask"][trace]:
                        continue
                    distance = float(dataset.distance_axis_m[min(trace, len(dataset.distance_axis_m) - 1)])
                    x = y = surface_z = interface_z = ""
                    if trajectory is not None:
                        point = trajectory.interpolate(distance)
                        x, y, surface_z = float(point.x), float(point.y), float(point.z)
                        if np.isfinite(labels["depth_gt"][trace]):
                            interface_z = surface_z - float(labels["depth_gt"][trace])
                    visibility_code = int(semantics["visibility_gt"][trace])
                    visibility = {0: "unknown", 1: "clear", 2: "weak", 3: "no_interface"}.get(visibility_code, "unknown")
                    writer.writerow({
                        "line_id": safe,
                        "trace_index": trace,
                        "distance_m": f"{distance:.6f}",
                        "sample_index": f"{float(curve[trace]):.6f}" if np.isfinite(curve[trace]) else "",
                        "time_ns": f"{float(labels['time_gt_ns'][trace]):.6f}" if np.isfinite(labels["time_gt_ns"][trace]) else "",
                        "depth_m": f"{float(labels['depth_gt'][trace]):.6f}" if np.isfinite(labels["depth_gt"][trace]) else "",
                        "x": f"{x:.6f}" if isinstance(x, float) else "",
                        "y": f"{y:.6f}" if isinstance(y, float) else "",
                        "surface_z": f"{surface_z:.6f}" if isinstance(surface_z, float) else "",
                        "interface_z": f"{interface_z:.6f}" if isinstance(interface_z, float) else "",
                        "visibility": visibility,
                        "is_no_interface": int(semantics["no_interface_mask"][trace]),
                        "is_ignored": int(semantics["ignore_trace_mask"][trace]),
                    })
        return path

    def basal_interface_summary(self, line_id: str) -> dict[str, Any]:
        annotation = self.load_basal_interface_annotation(line_id)
        if annotation is None:
            return {
                "line_id": validate_line_id(line_id), "status": "not_started", "keypoint_count": 0,
                "coverage_ratio": 0.0, "judged_ratio": 0.0, "weak_ratio": 0.0,
                "ignore_ratio": 0.0, "no_interface_ratio": 0.0,
            }
        return {"line_id": annotation.line_id, **annotation.statistics(), "version": annotation.version}


__all__ = ["FieldInterfaceStoreMixin"]
