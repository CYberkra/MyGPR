"""Interactive basal-interface editing use cases.

This service owns reversible interpretation operations and signal-assisted tracing.
It is intentionally presentation-agnostic: no Qt objects, widget state or legacy UI
classes cross this boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from mygpr.domain.interpretation.tracing import decimate_trace_path, trace_interface
from mygpr.application.interpretation.service import InterpretationService
from mygpr.application.project.service import ProjectService
from mygpr.domain.interpretation.models import (
    InterfaceAnnotation,
    InterfaceEditSnapshot,
    InterfaceTraceConfig,
    InterpretationLabelPackage,
    InterpretationPoint,
    InterpretationZone,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _moving_median(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or values.size < 3:
        return values.copy()
    output = values.copy()
    for index in range(values.size):
        lo = max(0, index - radius)
        hi = min(values.size, index + radius + 1)
        output[index] = float(np.median(values[lo:hi]))
    return output


@dataclass(slots=True)
class _EditSession:
    session_id: str
    project_id: str
    line_id: str
    input_artifact_id: str
    source_shape: tuple[int, int]
    initial: InterfaceAnnotation
    current: InterfaceAnnotation
    undo_stack: list[InterfaceAnnotation]
    redo_stack: list[InterfaceAnnotation]
    audit: dict[str, Any]


class InterpretationEditService:
    """Reversible, project-scoped interface editing and label export."""

    def __init__(self, projects: ProjectService, interpretation: InterpretationService) -> None:
        self._projects = projects
        self._interpretation = interpretation
        self._sessions: dict[str, _EditSession] = {}

    def _session(self, session_id: str) -> _EditSession:
        try:
            return self._sessions[str(session_id)]
        except KeyError as exc:
            raise KeyError(f"未知界面编辑会话: {session_id}") from exc

    def _snapshot(self, session: _EditSession) -> InterfaceEditSnapshot:
        return InterfaceEditSnapshot(
            session_id=session.session_id,
            project_id=session.project_id,
            line_id=session.line_id,
            annotation=session.current,
            input_artifact_id=session.input_artifact_id,
            source_shape=session.source_shape,
            undo_depth=len(session.undo_stack),
            redo_depth=len(session.redo_stack),
            dirty=session.current != session.initial,
            audit=dict(session.audit),
        )

    def _record(self, session: _EditSession, operation: str, annotation: InterfaceAnnotation, **audit: Any) -> InterfaceEditSnapshot:
        if annotation == session.current:
            return self._snapshot(session)
        session.undo_stack.append(session.current)
        session.current = annotation
        session.redo_stack.clear()
        session.audit = {
            **session.audit,
            "last_operation": operation,
            "last_operation_at": _utc_now(),
            **audit,
        }
        return self._snapshot(session)

    def _data(self, session: _EditSession) -> np.ndarray:
        if session.input_artifact_id:
            return np.asarray(
                self._projects.read_artifact_dataset(
                    session.project_id, session.line_id, session.input_artifact_id
                ).data,
                dtype=np.float64,
            )
        return np.asarray(
            self._projects.read_dataset(session.project_id, session.line_id).data,
            dtype=np.float64,
        )

    def open_session(
        self,
        project_id: str,
        line_id: str,
        *,
        input_artifact_id: str = "",
    ) -> InterfaceEditSnapshot:
        annotation = self._interpretation.load_interface(project_id, line_id, create=True)
        if annotation is None:
            raise RuntimeError("无法创建界面标注会话")
        if input_artifact_id:
            info = self._projects.get_artifact_dataset_info(project_id, line_id, input_artifact_id)
        else:
            info = self._projects.get_dataset_info(project_id, line_id)
        current = replace(
            annotation,
            processing_result=input_artifact_id or annotation.processing_result,
        )
        session = _EditSession(
            session_id=f"IES-{uuid.uuid4().hex[:14]}",
            project_id=project_id,
            line_id=line_id,
            input_artifact_id=input_artifact_id,
            source_shape=tuple(int(value) for value in info.shape),
            initial=current,
            current=current,
            undo_stack=[],
            redo_stack=[],
            audit={"opened_at": _utc_now()},
        )
        self._sessions[session.session_id] = session
        return self._snapshot(session)

    def close_session(self, session_id: str) -> bool:
        return self._sessions.pop(str(session_id), None) is not None

    def get_session(self, session_id: str) -> InterfaceEditSnapshot:
        return self._snapshot(self._session(session_id))

    def replace_points(
        self,
        session_id: str,
        points: Sequence[tuple[float, float]],
        *,
        confidence: float | None = None,
    ) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        max_sample, max_trace = session.source_shape[0] - 1, session.source_shape[1] - 1
        value = session.current.confidence if confidence is None else float(confidence)
        normalized = tuple(
            InterpretationPoint(
                float(np.clip(trace, 0, max_trace)),
                float(np.clip(sample, 0, max_sample)),
                value,
            )
            for trace, sample in sorted(points, key=lambda item: float(item[0]))
        )
        return self._record(
            session,
            "replace_points",
            replace(session.current, points=normalized, confidence=value),
            point_count=len(normalized),
        )

    def auto_trace(
        self,
        session_id: str,
        *,
        start_trace: int | None = None,
        end_trace: int | None = None,
        config: InterfaceTraceConfig | Mapping[str, Any] | None = None,
    ) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        anchors = tuple((int(round(point.trace_index)), point.sample_index) for point in session.current.points)
        if not anchors:
            raise ValueError("自动追踪至少需要一个人工锚点")
        if isinstance(config, InterfaceTraceConfig):
            cfg = config
        elif config is not None and is_dataclass(config):
            cfg = InterfaceTraceConfig(**asdict(config))
        else:
            cfg = InterfaceTraceConfig(**dict(config or {}))
        data = self._data(session)
        if start_trace is None and end_trace is None and len(anchors) == 1:
            start_trace, end_trace = 0, data.shape[1] - 1
        traces, samples = trace_interface(
            data,
            anchors,
            start_trace=start_trace,
            end_trace=end_trace,
            config=cfg,
        )
        editable = decimate_trace_path(
            traces,
            samples,
            max_points=cfg.max_points,
            mandatory_traces=(trace for trace, _sample in anchors),
        )
        points = tuple(
            InterpretationPoint(trace, sample, session.current.confidence, "自动追踪")
            for trace, sample in editable
        )
        metadata = {
            **dict(session.current.edit_metadata),
            "auto_trace": {
                "generated_at": _utc_now(),
                "dense_point_count": int(traces.size),
                "editable_point_count": len(points),
                "config": {
                    "search_half_window": cfg.search_half_window,
                    "max_step_samples": cfg.max_step_samples,
                    "smooth_radius": cfg.smooth_radius,
                    "anchor_weight": cfg.anchor_weight,
                    "continuity_weight": cfg.continuity_weight,
                },
            },
        }
        return self._record(
            session,
            "auto_trace",
            replace(session.current, points=points, edit_metadata=metadata),
            traced_interval=[int(traces[0]), int(traces[-1])],
            dense_point_count=int(traces.size),
        )

    def snap_to_signal(self, session_id: str, *, radius_samples: int = 8) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        radius = max(1, int(radius_samples))
        data = np.abs(self._data(session))
        max_sample, max_trace = data.shape[0] - 1, data.shape[1] - 1
        points: list[InterpretationPoint] = []
        shifts: list[float] = []
        for point in session.current.points:
            trace = int(np.clip(round(point.trace_index), 0, max_trace))
            center = int(np.clip(round(point.sample_index), 0, max_sample))
            lo, hi = max(0, center - radius), min(max_sample, center + radius)
            sample = lo + int(np.argmax(data[lo : hi + 1, trace]))
            shifts.append(float(sample - point.sample_index))
            points.append(replace(point, sample_index=float(sample), note="信号吸附"))
        return self._record(
            session,
            "snap_to_signal",
            replace(session.current, points=tuple(points)),
            radius_samples=radius,
            mean_shift_samples=float(np.mean(shifts)) if shifts else 0.0,
        )

    def smooth(self, session_id: str, *, radius: int = 2) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        points = tuple(sorted(session.current.points, key=lambda item: item.trace_index))
        if len(points) < 3:
            return self._snapshot(session)
        samples = np.asarray([point.sample_index for point in points], dtype=float)
        smoothed = _moving_median(samples, max(1, int(radius)))
        smoothed[0], smoothed[-1] = samples[0], samples[-1]
        updated = tuple(
            replace(point, sample_index=float(value), note="平滑")
            for point, value in zip(points, smoothed)
        )
        return self._record(
            session,
            "smooth",
            replace(session.current, points=updated),
            radius=int(radius),
        )

    def shift(self, session_id: str, *, offset_samples: float) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        max_sample = session.source_shape[0] - 1
        offset = float(offset_samples)
        points = tuple(
            replace(
                point,
                sample_index=float(np.clip(point.sample_index + offset, 0, max_sample)),
                note="整体移动",
            )
            for point in session.current.points
        )
        return self._record(
            session,
            "shift",
            replace(session.current, points=points),
            offset_samples=offset,
        )

    def update_properties(
        self,
        session_id: str,
        *,
        name: str | None = None,
        note: str | None = None,
        interface_type: str | None = None,
        confidence: float | None = None,
    ) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        metadata = dict(session.current.edit_metadata)
        if interface_type is not None:
            metadata["interface_type"] = str(interface_type)
        value = session.current.confidence if confidence is None else float(confidence)
        if not 0.0 <= value <= 1.0:
            raise ValueError("置信度必须位于 0 到 1")
        annotation = replace(
            session.current,
            name=str(name if name is not None else session.current.name),
            note=str(note if note is not None else session.current.note),
            confidence=value,
            edit_metadata=metadata,
        )
        return self._record(session, "update_properties", annotation, interface_type=metadata.get("interface_type", ""))

    def set_uncertainty(self, session_id: str, *, width_samples: float) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        width = max(0.0, float(width_samples))
        return self._record(
            session,
            "set_uncertainty",
            replace(session.current, uncertainty_samples=width),
            uncertainty_samples=width,
        )

    def add_zone(self, session_id: str, zone: InterpretationZone) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        return self._record(
            session,
            "add_zone",
            replace(session.current, zones=session.current.zones + (zone,)),
            zone_count=len(session.current.zones) + 1,
        )

    def update_zone(self, session_id: str, index: int, zone: InterpretationZone) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        zones = list(session.current.zones)
        if not 0 <= int(index) < len(zones):
            raise IndexError("语义区段索引越界")
        zones[int(index)] = zone
        return self._record(session, "update_zone", replace(session.current, zones=tuple(zones)), zone_index=int(index))

    def remove_zone(self, session_id: str, index: int) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        zones = list(session.current.zones)
        if not 0 <= int(index) < len(zones):
            raise IndexError("语义区段索引越界")
        zones.pop(int(index))
        return self._record(session, "remove_zone", replace(session.current, zones=tuple(zones)), zone_index=int(index))

    def split_zone(self, session_id: str, index: int, *, trace_index: float) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        zones = list(session.current.zones)
        if not 0 <= int(index) < len(zones):
            raise IndexError("语义区段索引越界")
        zone = zones[int(index)]
        trace = float(trace_index)
        if not zone.start_trace < trace < zone.end_trace:
            raise ValueError("拆分位置必须位于区段内部")
        zones[int(index) : int(index) + 1] = [
            replace(zone, end_trace=trace),
            replace(zone, start_trace=trace),
        ]
        return self._record(
            session,
            "split_zone",
            replace(session.current, zones=tuple(zones)),
            zone_index=int(index),
            split_trace=trace,
        )

    def undo(self, session_id: str) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        if not session.undo_stack:
            return self._snapshot(session)
        session.redo_stack.append(session.current)
        session.current = session.undo_stack.pop()
        session.audit = {**session.audit, "last_operation": "undo", "last_operation_at": _utc_now()}
        return self._snapshot(session)

    def redo(self, session_id: str) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        if not session.redo_stack:
            return self._snapshot(session)
        session.undo_stack.append(session.current)
        session.current = session.redo_stack.pop()
        session.audit = {**session.audit, "last_operation": "redo", "last_operation_at": _utc_now()}
        return self._snapshot(session)

    def reset(self, session_id: str) -> InterfaceEditSnapshot:
        session = self._session(session_id)
        return self._record(session, "reset", session.initial)

    def save_session(self, session_id: str, *, status: str = "draft") -> InterfaceAnnotation:
        session = self._session(session_id)
        saved = self._interpretation.save_interface(
            session.project_id,
            replace(session.current, status=str(status)),
        )
        session.initial = saved
        session.current = saved
        session.undo_stack.clear()
        session.redo_stack.clear()
        session.audit = {**session.audit, "last_operation": "save", "saved_at": _utc_now(), "saved_status": status}
        return saved

    def export_labels(
        self,
        session_id: str,
        destination_dir: str | None = None,
    ) -> InterpretationLabelPackage:
        session = self._session(session_id)
        saved = self.save_session(session_id, status="draft")
        summary = self._projects.get_summary(session.project_id)
        project_root = Path(summary.root_path).resolve()
        package_id = f"ILP-{uuid.uuid4().hex[:12]}"
        destination = (
            Path(destination_dir).expanduser().resolve()
            if destination_dir
            else project_root / "exports" / "interpretation" / f"{session.line_id}_{package_id}"
        )
        destination.mkdir(parents=True, exist_ok=True)

        candidates = {
            "annotation": project_root / "metadata" / "interpretations" / "interfaces" / f"{session.line_id}.json",
            "legacy_annotation": project_root / "targets" / f"{session.line_id}_basal_interface.json",
            "labels_npz": project_root / "targets" / f"{session.line_id}_basal_labels.npz",
            "labels_dir": project_root / "targets" / f"{session.line_id}_basal_labels",
            "spatial_curve": project_root / "spatial" / f"{session.line_id}_basal_interface_xy.csv",
            "features": project_root / "metadata" / "interpretations" / "features" / f"{session.line_id}.geojson",
        }
        files: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for role, source in candidates.items():
            if not source.exists():
                continue
            target = destination / source.name
            if source.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
                for child in sorted(target.rglob("*")):
                    if child.is_file():
                        key = f"{role}/{child.relative_to(target).as_posix()}"
                        files[key] = str(child)
                        hashes[key] = _sha256(child)
            else:
                shutil.copy2(source, target)
                files[role] = str(target)
                hashes[role] = _sha256(target)

        manifest = {
            "schema": "mygpr.interpretation_labels.v1",
            "package_id": package_id,
            "project_id": session.project_id,
            "line_id": session.line_id,
            "generated_at": _utc_now(),
            "annotation_version": saved.version,
            "annotation_status": saved.status,
            "point_count": len(saved.points),
            "zone_count": len(saved.zones),
            "uncertainty_samples": saved.uncertainty_samples,
            "files": files,
            "sha256": hashes,
        }
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        files["manifest"] = str(manifest_path)
        hashes["manifest"] = _sha256(manifest_path)
        return InterpretationLabelPackage(
            package_id=package_id,
            project_id=session.project_id,
            line_id=session.line_id,
            root_path=str(destination),
            manifest_path=str(manifest_path),
            files=files,
            sha256=hashes,
            summary={
                "annotation_version": saved.version,
                "point_count": len(saved.points),
                "zone_count": len(saved.zones),
                "uncertainty_samples": saved.uncertainty_samples,
            },
        )


__all__ = ["InterpretationEditService"]
