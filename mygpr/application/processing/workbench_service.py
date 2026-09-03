#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive processing-session use cases for the Studio workbench."""
from __future__ import annotations

import copy
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Sequence

import numpy as np

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.processing.analysis_service import ProcessingAnalysisService
from mygpr.application.processing.service import ProcessingService
from mygpr.application.project.service import ProjectService
from mygpr.domain.processing.models import ProcessingRequest
from mygpr.domain.processing.workbench import (
    DatasetComparison,
    ProcessingTemplate,
    ProcessingEvidencePackage,
    ProcessingStepDiagnostic,
    SignalAnalysis,
    WorkbenchPreview,
    WorkbenchSessionSnapshot,
    WorkbenchStep,
)
from mygpr.domain.project.models import ProjectArtifact


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Any) -> None:
    """镜像 core.storage_primitives.atomic_write_json 的持久化纪律。

    application 层受架构政策限制不能 import core，故在本地复刻同一套
    约定：隐藏的唯一临时名（并发写互不踩踏）+ fsync + 原子替换 +
    目录 fsync。待 application 引入持久化端口后收敛到单一实现。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2))
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except OSError:
                pass
        temporary.replace(target)
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _bounded_indices(size: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or size <= maximum:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, maximum).astype(np.int64)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)[:240]


def _clone_steps(steps: Sequence[WorkbenchStep]) -> list[WorkbenchStep]:
    return [
        WorkbenchStep(
            item.step_id,
            item.method_id,
            item.label,
            dict(item.params),
            item.enabled,
            tuple(item.output_shape),
            item.output_dtype,
            dict(item.metadata),
        )
        for item in steps
    ]


@dataclass(slots=True)
class _Session:
    session_id: str
    project_id: str
    line_id: str
    input_artifact_id: str
    branch_id: str
    name: str
    original_data: np.ndarray
    current_data: np.ndarray
    original_header: dict[str, Any]
    current_header: dict[str, Any]
    original_trace_metadata: dict[str, np.ndarray]
    current_trace_metadata: dict[str, np.ndarray]
    steps: list[WorkbenchStep] = field(default_factory=list)
    undo_stack: list[list[WorkbenchStep]] = field(default_factory=list)
    redo_stack: list[list[WorkbenchStep]] = field(default_factory=list)
    revision: int = 0


class ProcessingWorkbenchService:
    """Own interactive processing sessions independently from the UI toolkit."""

    def __init__(self, projects: ProjectService, processing: ProcessingService) -> None:
        self._projects = projects
        self._processing = processing
        self._analysis = ProcessingAnalysisService(projects, processing)
        self._sessions: dict[str, _Session] = {}
        self._lock = RLock()

    def open_session(
        self,
        project_id: str,
        line_id: str,
        *,
        input_artifact_id: str = "",
        branch_id: str = "",
        name: str = "",
        restore_draft: bool = True,
    ) -> WorkbenchSessionSnapshot:
        payload = (
            self._projects.read_artifact_dataset(project_id, line_id, input_artifact_id)
            if input_artifact_id
            else self._projects.read_dataset(project_id, line_id)
        )
        matrix = np.asarray(payload.data, dtype=np.float32)
        session_id = f"PS-{uuid.uuid4().hex[:16]}"
        session = _Session(
            session_id=session_id,
            project_id=project_id,
            line_id=line_id,
            input_artifact_id=input_artifact_id,
            branch_id=branch_id or f"{line_id}:main",
            name=name or f"{line_id} 交互处理",
            original_data=np.array(matrix, copy=True),
            current_data=np.array(matrix, copy=True),
            original_header=copy.deepcopy(payload.header_info),
            current_header=copy.deepcopy(payload.header_info),
            original_trace_metadata={key: np.array(value, copy=True) for key, value in payload.trace_metadata.items()},
            current_trace_metadata={key: np.array(value, copy=True) for key, value in payload.trace_metadata.items()},
        )
        if restore_draft:
            draft_steps = self._load_draft_steps(session)
            if draft_steps:
                session.steps = draft_steps
                self._replay(session)
                session.revision = 1
        with self._lock:
            self._sessions[session_id] = session
        return self._snapshot(session)

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(str(session_id), None) is not None

    def get_session(self, session_id: str) -> WorkbenchSessionSnapshot:
        return self._snapshot(self._session(session_id))

    def fork_session(
        self, session_id: str, *, branch_id: str, name: str = ""
    ) -> WorkbenchSessionSnapshot:
        source = self._session(session_id)
        fork = _Session(
            session_id=f"PS-{uuid.uuid4().hex[:16]}",
            project_id=source.project_id,
            line_id=source.line_id,
            input_artifact_id=source.input_artifact_id,
            branch_id=str(branch_id).strip() or f"{source.line_id}:branch",
            name=str(name).strip() or str(branch_id).strip() or f"{source.line_id} 分支",
            original_data=np.array(source.original_data, copy=True),
            current_data=np.array(source.current_data, copy=True),
            original_header=copy.deepcopy(source.original_header),
            current_header=copy.deepcopy(source.current_header),
            original_trace_metadata={key: np.array(value, copy=True) for key, value in source.original_trace_metadata.items()},
            current_trace_metadata={key: np.array(value, copy=True) for key, value in source.current_trace_metadata.items()},
            steps=_clone_steps(source.steps),
        )
        with self._lock:
            self._sessions[fork.session_id] = fork
        return self._snapshot(fork)

    def preview_method(
        self,
        session_id: str,
        method_id: str,
        params: dict[str, Any] | None = None,
        *,
        max_samples: int = 900,
        max_traces: int = 1800,
    ) -> WorkbenchPreview:
        session = self._session(session_id)
        result = self._execute(session, method_id, params or {})
        return self._preview(
            session.session_id,
            method_id,
            dict(result.params),
            result.data,
            result.metadata,
            max_samples=max_samples,
            max_traces=max_traces,
        )

    def session_window(
        self, session_id: str, *, max_samples: int = 900, max_traces: int = 1800
    ) -> WorkbenchPreview:
        session = self._session(session_id)
        return self._preview(
            session.session_id,
            "session",
            {},
            session.current_data,
            {"step_count": len(session.steps), "revision": session.revision},
            max_samples=max_samples,
            max_traces=max_traces,
        )

    def append_step(
        self, session_id: str, method_id: str, params: dict[str, Any] | None = None, *, label: str = ""
    ) -> WorkbenchSessionSnapshot:
        session = self._session(session_id)
        self._checkpoint(session)
        result = self._execute(session, method_id, params or {})
        descriptor = self._processing.get_method(method_id)
        step = WorkbenchStep(
            step_id=f"STEP-{uuid.uuid4().hex[:12]}",
            method_id=method_id,
            label=label or descriptor.name,
            params=dict(result.params),
            enabled=True,
            output_shape=tuple(int(value) for value in result.data.shape),
            output_dtype=str(result.data.dtype),
            metadata=_json_safe(result.metadata),
        )
        session.steps.append(step)
        self._accept_result(session, result)
        return self._touch(session)

    def update_step(
        self,
        session_id: str,
        index: int,
        *,
        params: dict[str, Any] | None = None,
        enabled: bool | None = None,
        label: str | None = None,
    ) -> WorkbenchSessionSnapshot:
        session = self._session(session_id)
        self._validate_index(session, index)
        self._checkpoint(session)
        current = session.steps[index]
        session.steps[index] = WorkbenchStep(
            step_id=current.step_id,
            method_id=current.method_id,
            label=current.label if label is None else label,
            params=dict(current.params) if params is None else dict(params),
            enabled=current.enabled if enabled is None else bool(enabled),
        )
        self._replay(session)
        return self._touch(session)

    def remove_step(self, session_id: str, index: int) -> WorkbenchSessionSnapshot:
        session = self._session(session_id)
        self._validate_index(session, index)
        self._checkpoint(session)
        session.steps.pop(index)
        self._replay(session)
        return self._touch(session)

    def move_step(self, session_id: str, source: int, target: int) -> WorkbenchSessionSnapshot:
        session = self._session(session_id)
        self._validate_index(session, source)
        self._validate_index(session, target)
        if source == target:
            return self._snapshot(session)
        self._checkpoint(session)
        session.steps.insert(target, session.steps.pop(source))
        self._replay(session)
        return self._touch(session)

    def undo(self, session_id: str) -> WorkbenchSessionSnapshot:
        session = self._session(session_id)
        if not session.undo_stack:
            return self._snapshot(session)
        session.redo_stack.append(_clone_steps(session.steps))
        session.steps = session.undo_stack.pop()
        self._replay(session)
        return self._touch(session)

    def redo(self, session_id: str) -> WorkbenchSessionSnapshot:
        session = self._session(session_id)
        if not session.redo_stack:
            return self._snapshot(session)
        session.undo_stack.append(_clone_steps(session.steps))
        session.steps = session.redo_stack.pop()
        self._replay(session)
        return self._touch(session)

    def reset(self, session_id: str) -> WorkbenchSessionSnapshot:
        session = self._session(session_id)
        if session.steps:
            self._checkpoint(session)
        session.steps = []
        self._restore_original(session)
        snapshot = self._touch(session)
        self._clear_draft(session)
        return snapshot

    def analyze_trace(self, session_id: str, trace_index: int) -> SignalAnalysis:
        session = self._session(session_id)
        index = min(max(0, int(trace_index)), session.current_data.shape[1] - 1)
        trace = np.asarray(session.current_data[:, index], dtype=np.float64)
        centered = trace - float(np.mean(trace))
        window = np.hanning(trace.size) if trace.size > 1 else np.ones(trace.size)
        spectrum = np.abs(np.fft.rfft(centered * window))
        time_window_ns = float(session.current_header.get("time_window_ns") or session.current_header.get("total_time_ns") or trace.size)
        dt_seconds = max(time_window_ns / max(trace.size - 1, 1), 1e-12) * 1e-9
        frequency_mhz = np.fft.rfftfreq(trace.size, d=dt_seconds) / 1e6
        statistics = {
            "minimum": float(np.min(trace)),
            "maximum": float(np.max(trace)),
            "mean": float(np.mean(trace)),
            "rms": float(np.sqrt(np.mean(trace * trace))),
            "peak_to_peak": float(np.ptp(trace)),
            "dominant_frequency_mhz": float(frequency_mhz[int(np.argmax(spectrum))]) if spectrum.size else 0.0,
        }
        return SignalAnalysis(
            line_id=session.line_id,
            trace_index=index,
            sample_axis=np.linspace(0.0, time_window_ns, trace.size),
            amplitude=trace,
            frequency_axis=frequency_mhz,
            magnitude=spectrum,
            statistics=statistics,
        )

    def compare_original(
        self,
        session_id: str,
        *,
        max_samples: int = 900,
        max_traces: int = 1800,
    ) -> DatasetComparison:
        return self._analysis.compare_original(
            self._session(session_id),
            max_samples=max_samples,
            max_traces=max_traces,
        )

    def compare_candidate(
        self,
        session_id: str,
        method_id: str,
        params: dict[str, Any] | None = None,
        *,
        max_samples: int = 900,
        max_traces: int = 1800,
    ) -> DatasetComparison:
        return self._analysis.compare_candidate(
            self._session(session_id),
            method_id,
            params,
            max_samples=max_samples,
            max_traces=max_traces,
        )

    def diagnose_steps(
        self,
        session_id: str,
    ) -> tuple[ProcessingStepDiagnostic, ...]:
        return self._analysis.diagnose_steps(self._session(session_id))

    def export_evidence(
        self,
        session_id: str,
        destination_dir: str | None = None,
        *,
        include_data: bool = False,
    ) -> ProcessingEvidencePackage:
        return self._analysis.export_evidence(
            self._session(session_id),
            destination_dir,
            include_data=include_data,
        )

    def save_session(self, session_id: str, *, result_name: str = "", branch_id: str = "") -> ProjectArtifact:
        session = self._session(session_id)
        if not session.steps:
            raise ValueError("处理会话没有可保存的步骤")
        pipeline = [
            {"method_id": item.method_id, "label": item.label, "params": dict(item.params), "enabled": item.enabled, "metadata": dict(item.metadata)}
            for item in session.steps
        ]
        final = next((item for item in reversed(session.steps) if item.enabled), session.steps[-1])
        artifact = self._projects.save_processing_artifact(
            session.project_id,
            session.line_id,
            session.current_data,
            name=result_name or session.name,
            method_id=final.method_id,
            method_name=session.name,
            params={"pipeline": pipeline, "parent_artifact_id": session.input_artifact_id, "interactive_session": True},
            pipeline=pipeline,
            branch_id=branch_id or session.branch_id,
            input_dataset={**_json_safe(session.original_header), "parent_artifact_id": session.input_artifact_id, "execution_mode": "interactive_workbench"},
            context=ExecutionContext.null(),
        )
        session.undo_stack.clear()
        session.redo_stack.clear()
        session.revision += 1
        self._clear_draft(session)
        return artifact

    def list_templates(self, project_id: str) -> tuple[ProcessingTemplate, ...]:
        records = self._read_template_records(project_id)
        return tuple(self._template_from_record(item) for item in records)

    def save_template(
        self, project_id: str, *, name: str, description: str = "", session_id: str
    ) -> ProcessingTemplate:
        session = self._session(session_id)
        if session.project_id != project_id:
            raise ValueError("processing session belongs to another project")
        records = self._read_template_records(project_id)
        now = _utc_now()
        existing = next((item for item in records if item.get("name") == name), None)
        template_id = str(existing.get("template_id")) if existing else f"TPL-{uuid.uuid4().hex[:12]}"
        created_at = str(existing.get("created_at")) if existing else now
        record = {
            "template_id": template_id,
            "name": str(name).strip(),
            "description": str(description or ""),
            "created_at": created_at,
            "updated_at": now,
            "steps": [
                {"step_id": item.step_id, "method_id": item.method_id, "label": item.label, "params": dict(item.params), "enabled": item.enabled}
                for item in session.steps
            ],
        }
        records = [item for item in records if item.get("template_id") != template_id]
        records.append(record)
        self._write_template_records(project_id, records)
        return self._template_from_record(record)

    def delete_template(self, project_id: str, template_id: str) -> bool:
        records = self._read_template_records(project_id)
        remaining = [item for item in records if item.get("template_id") != template_id]
        if len(remaining) == len(records):
            return False
        self._write_template_records(project_id, remaining)
        return True

    def apply_template(self, session_id: str, template_id: str) -> WorkbenchSessionSnapshot:
        session = self._session(session_id)
        template = next((item for item in self.list_templates(session.project_id) if item.template_id == template_id), None)
        if template is None:
            raise KeyError(template_id)
        self._checkpoint(session)
        session.steps = [
            WorkbenchStep(f"STEP-{uuid.uuid4().hex[:12]}", item.method_id, item.label, dict(item.params), item.enabled)
            for item in template.steps
        ]
        self._replay(session)
        return self._touch(session)

    def _execute(self, session: _Session, method_id: str, params: dict[str, Any]):
        request = ProcessingRequest(
            data=session.current_data,
            method_id=method_id,
            params=dict(params),
            header_info=session.current_header,
            trace_metadata=session.current_trace_metadata,
        )
        return self._processing.execute_method(request, ExecutionContext.null())

    def _replay(self, session: _Session) -> None:
        specs = _clone_steps(session.steps)
        self._restore_original(session)
        rebuilt: list[WorkbenchStep] = []
        for spec in specs:
            if not spec.enabled:
                rebuilt.append(spec)
                continue
            result = self._execute(session, spec.method_id, dict(spec.params))
            rebuilt.append(WorkbenchStep(
                spec.step_id, spec.method_id, spec.label, dict(result.params), True,
                tuple(int(value) for value in result.data.shape), str(result.data.dtype), _json_safe(result.metadata),
            ))
            self._accept_result(session, result)
        session.steps = rebuilt

    @staticmethod
    def _accept_result(session: _Session, result: Any) -> None:
        session.current_data = np.asarray(result.data, dtype=np.float32).copy()
        session.current_header = copy.deepcopy(result.header_info)
        session.current_trace_metadata = {key: np.array(value, copy=True) for key, value in result.trace_metadata.items()}

    @staticmethod
    def _restore_original(session: _Session) -> None:
        session.current_data = np.array(session.original_data, copy=True)
        session.current_header = copy.deepcopy(session.original_header)
        session.current_trace_metadata = {key: np.array(value, copy=True) for key, value in session.original_trace_metadata.items()}

    @staticmethod
    def _checkpoint(session: _Session) -> None:
        session.undo_stack.append(_clone_steps(session.steps))
        session.redo_stack.clear()

    @staticmethod
    def _validate_index(session: _Session, index: int) -> None:
        if not 0 <= int(index) < len(session.steps):
            raise IndexError(index)

    def _touch(self, session: _Session) -> WorkbenchSessionSnapshot:
        session.revision += 1
        self._save_draft(session)
        return self._snapshot(session)

    @staticmethod
    def _snapshot(session: _Session) -> WorkbenchSessionSnapshot:
        return WorkbenchSessionSnapshot(
            session.session_id, session.project_id, session.line_id,
            session.input_artifact_id, session.branch_id, session.name,
            tuple(session.steps), tuple(int(value) for value in session.current_data.shape),
            str(session.current_data.dtype), bool(session.undo_stack), bool(session.redo_stack),
            bool(session.steps), session.revision,
        )

    def _session(self, session_id: str) -> _Session:
        with self._lock:
            session = self._sessions.get(str(session_id))
        if session is None:
            raise KeyError(f"processing session not found: {session_id}")
        return session

    @staticmethod
    def _preview(
        session_id: str,
        method_id: str,
        params: dict[str, Any],
        data: np.ndarray,
        metadata: dict[str, Any],
        *,
        max_samples: int,
        max_traces: int,
    ) -> WorkbenchPreview:
        matrix = np.asarray(data)
        sample_indices = _bounded_indices(matrix.shape[0], max_samples)
        trace_indices = _bounded_indices(matrix.shape[1], max_traces)
        window = matrix[np.ix_(sample_indices, trace_indices)]
        return WorkbenchPreview(session_id, method_id, params, np.array(window, copy=True), sample_indices, trace_indices, _json_safe(metadata))

    def _template_path(self, project_id: str) -> Path:
        root = Path(self._projects.get_summary(project_id).root_path)
        return root / ".mygpr" / "processing_templates.json"

    def _draft_path(self, session: _Session) -> Path:
        root = Path(self._projects.get_summary(session.project_id).root_path)
        directory = root / ".mygpr" / "processing_drafts"
        safe_line = "".join(char if char.isalnum() or char in "-_." else "_" for char in session.line_id)
        suffix = session.input_artifact_id or "raw"
        safe_suffix = "".join(char if char.isalnum() or char in "-_." else "_" for char in suffix)
        branch = session.branch_id or f"{session.line_id}:main"
        safe_branch = "".join(char if char.isalnum() or char in "-_." else "_" for char in branch)
        return directory / f"{safe_line}__{safe_suffix}__{safe_branch}.json"

    def _save_draft(self, session: _Session) -> None:
        if self._projects.get_summary(session.project_id).read_only:
            return
        if not session.steps:
            self._clear_draft(session)
            return
        payload = {
            "schema": "mygpr.processing_draft.v1",
            "project_id": session.project_id,
            "line_id": session.line_id,
            "input_artifact_id": session.input_artifact_id,
            "branch_id": session.branch_id,
            "name": session.name,
            "updated_at": _utc_now(),
            "steps": [
                {
                    "step_id": item.step_id,
                    "method_id": item.method_id,
                    "label": item.label,
                    "params": dict(item.params),
                    "enabled": item.enabled,
                }
                for item in session.steps
            ],
        }
        path = self._draft_path(session)
        _atomic_write_json(path, payload)

    def _load_draft_steps(self, session: _Session) -> list[WorkbenchStep]:
        path = self._draft_path(session)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if payload.get("schema") != "mygpr.processing_draft.v1":
            return []
        if payload.get("project_id") != session.project_id or payload.get("line_id") != session.line_id:
            return []
        session.branch_id = str(payload.get("branch_id") or session.branch_id)
        session.name = str(payload.get("name") or session.name)
        return [
            WorkbenchStep(
                str(item.get("step_id") or f"STEP-{uuid.uuid4().hex[:12]}"),
                str(item["method_id"]),
                str(item.get("label") or item["method_id"]),
                dict(item.get("params") or {}),
                bool(item.get("enabled", True)),
            )
            for item in payload.get("steps", [])
            if isinstance(item, dict) and item.get("method_id")
        ]

    def _clear_draft(self, session: _Session) -> None:
        if self._projects.get_summary(session.project_id).read_only:
            return
        path = self._draft_path(session)
        if path.exists():
            path.unlink()

    def _read_template_records(self, project_id: str) -> list[dict[str, Any]]:
        path = self._template_path(project_id)
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != "mygpr.processing_templates.v1":
            raise ValueError("invalid processing template store")
        return [dict(item) for item in payload.get("templates", []) if isinstance(item, dict)]

    def _write_template_records(self, project_id: str, records: Sequence[dict[str, Any]]) -> None:
        if self._projects.get_summary(project_id).read_only:
            raise PermissionError("read-only project cannot modify processing templates")
        path = self._template_path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": "mygpr.processing_templates.v1", "updated_at": _utc_now(), "templates": list(records)}
        _atomic_write_json(path, payload)

    @staticmethod
    def _template_from_record(record: dict[str, Any]) -> ProcessingTemplate:
        steps = tuple(
            WorkbenchStep(
                str(item.get("step_id") or f"STEP-{uuid.uuid4().hex[:12]}"),
                str(item["method_id"]),
                str(item.get("label") or item["method_id"]),
                dict(item.get("params") or {}),
                bool(item.get("enabled", True)),
            )
            for item in record.get("steps", [])
        )
        return ProcessingTemplate(
            str(record["template_id"]), str(record["name"]), str(record.get("description") or ""),
            steps, str(record.get("created_at") or ""), str(record.get("updated_at") or ""),
        )


__all__ = ["ProcessingWorkbenchService"]
