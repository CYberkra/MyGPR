#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless processing session used by the project-first processing laboratory."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.auto_tune_comparison import (
    AutoTuneComparisonRun,
    run_auto_tune_comparison,
)
from core.auto_tune_comparison_export import export_auto_tune_comparison_artifacts
from core.auto_tune import auto_tune_method
from core.gpr_io import auto_load_data, extract_airborne_csv_payload
from core.methods_registry import (
    PROCESSING_METHODS,
    get_method_display_name,
    get_public_method_keys,
)
from core.processing_engine import (
    clone_header_info,
    clone_trace_metadata,
    merge_result_header_info,
    merge_result_trace_metadata,
    prepare_runtime_params,
    run_processing_method,
)
from core.project_models import LineRecordV1, ProcessingResultV1
from core.project_service import ProjectService
from core.qc_service import QcService


@dataclass
class ProcessingStep:
    method_id: str
    params: dict[str, Any]
    display_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingPreview:
    data: np.ndarray
    header_info: dict[str, Any]
    trace_metadata: dict[str, np.ndarray]
    metadata: dict[str, Any]
    method_id: str
    params: dict[str, Any]


@dataclass
class _SessionState:
    data: np.ndarray
    header_info: dict[str, Any]
    trace_metadata: dict[str, np.ndarray]
    steps: list[ProcessingStep]


class ProcessingSessionService:
    """Mutable in-memory processing state with explicit project-version writes."""

    def __init__(
        self,
        project: ProjectService,
        line: LineRecordV1,
        *,
        data: np.ndarray,
        header_info: dict[str, Any] | None = None,
        trace_metadata: dict[str, np.ndarray] | None = None,
    ):
        self.project = project
        self.line = line
        self.line_id = line.line_id
        self.original_data = np.array(data, dtype=np.float32, copy=True)
        self.current_data = np.array(data, dtype=np.float32, copy=True)
        self.original_header_info = clone_header_info(header_info)
        self.header_info = clone_header_info(header_info)
        self.original_trace_metadata = clone_trace_metadata(trace_metadata)
        self.trace_metadata = clone_trace_metadata(trace_metadata)
        self.steps: list[ProcessingStep] = []
        self._undo_stack: list[_SessionState] = []
        self._redo_stack: list[_SessionState] = []
        self.last_preview: ProcessingPreview | None = None
        self.last_recommendation: dict[str, Any] | None = None
        self.last_manual_auto_comparison: AutoTuneComparisonRun | None = None

    @classmethod
    def open_line(
        cls,
        project: ProjectService,
        line_id: str,
        *,
        enforce_processing_gate: bool = True,
    ) -> "ProcessingSessionService":
        if enforce_processing_gate:
            _assert_project_line_ready_for_processing(project, line_id)
        line = project.get_line(line_id)
        if not line.raw_files:
            raise ValueError(f"测线没有主数据: {line_id}")
        path = Path(line.raw_files[0].path)
        if not path.is_absolute():
            path = project.resolve_relative_path(path)
        payload = _load_project_line(project, line, path)
        data = np.asarray(payload.get("data"), dtype=np.float32)
        if data.ndim != 2 or data.size == 0:
            raise ValueError("测线处理需要二维非空 B-scan 数据")
        return cls(
            project,
            line,
            data=data,
            header_info=payload.get("header_info") or {},
            trace_metadata=payload.get("trace_metadata") or {},
        )

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    @staticmethod
    def public_method_ids() -> list[str]:
        return get_public_method_keys()

    @staticmethod
    def default_params(method_id: str) -> dict[str, Any]:
        method = PROCESSING_METHODS.get(method_id)
        if method is None:
            raise KeyError(method_id)
        return {
            str(item["name"]): copy.deepcopy(item.get("default"))
            for item in method.get("params", [])
            if isinstance(item, dict) and item.get("name")
        }

    def preview_method(
        self,
        method_id: str,
        params: dict[str, Any] | None = None,
    ) -> ProcessingPreview:
        resolved = self._resolve_params(method_id, params)
        runtime = prepare_runtime_params(
            method_id,
            resolved,
            self.header_info,
            self.trace_metadata,
            self.current_data.shape,
        )
        result, metadata = run_processing_method(self.current_data, method_id, runtime)
        preview = ProcessingPreview(
            data=np.array(result, copy=True),
            header_info=merge_result_header_info(
                self.header_info, metadata, result.shape
            ),
            trace_metadata=merge_result_trace_metadata(self.trace_metadata, metadata),
            metadata=dict(metadata),
            method_id=method_id,
            params=resolved,
        )
        self.last_preview = preview
        return preview

    def apply_method(
        self,
        method_id: str,
        params: dict[str, Any] | None = None,
    ) -> ProcessingStep:
        preview = self.preview_method(method_id, params)
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()
        self.current_data = np.array(preview.data, copy=True)
        self.header_info = clone_header_info(preview.header_info)
        self.trace_metadata = clone_trace_metadata(preview.trace_metadata)
        step = ProcessingStep(
            method_id=method_id,
            params=dict(preview.params),
            display_name=get_method_display_name(method_id),
            metadata=dict(preview.metadata),
        )
        self.steps.append(step)
        self.last_preview = None
        return step

    def apply_pipeline(self, steps: list[dict[str, Any]]) -> list[ProcessingStep]:
        applied: list[ProcessingStep] = []
        for item in steps:
            if not bool(item.get("enabled", True)):
                continue
            applied.append(
                self.apply_method(
                    str(item["method_id"]),
                    dict(item.get("params") or {}),
                )
            )
        return applied

    def remove_step(self, index: int) -> ProcessingStep:
        self._validate_step_index(index)
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()
        removed = self.steps.pop(index)
        self._replay_steps()
        return removed

    def move_step(self, source_index: int, target_index: int) -> None:
        self._validate_step_index(source_index)
        self._validate_step_index(target_index)
        if source_index == target_index:
            return
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()
        step = self.steps.pop(source_index)
        self.steps.insert(target_index, step)
        self._replay_steps()

    def set_step_enabled(self, index: int, enabled: bool) -> None:
        self._validate_step_index(index)
        if self.steps[index].enabled == bool(enabled):
            return
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()
        self.steps[index].enabled = bool(enabled)
        self._replay_steps()

    def update_step_params(self, index: int, params: dict[str, Any]) -> None:
        self._validate_step_index(index)
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()
        self.steps[index].params = self._resolve_params(
            self.steps[index].method_id, params
        )
        self._replay_steps()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        self.last_preview = None
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        self.last_preview = None
        return True

    def reset(self) -> None:
        if self.steps or not np.array_equal(self.current_data, self.original_data):
            self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()
        self.current_data = np.array(self.original_data, copy=True)
        self.header_info = clone_header_info(self.original_header_info)
        self.trace_metadata = clone_trace_metadata(self.original_trace_metadata)
        self.steps = []
        self.last_preview = None

    def compare_snapshots(self) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = [
            {"label": "原始", "data": np.array(self.original_data, copy=True)}
        ]
        for state in self._undo_stack:
            if snapshots and np.array_equal(snapshots[-1]["data"], state.data):
                continue
            label = state.steps[-1].display_name if state.steps else "原始"
            snapshots.append({"label": label, "data": np.array(state.data, copy=True)})
        if not np.array_equal(snapshots[-1]["data"], self.current_data):
            label = self.steps[-1].display_name if self.steps else "当前"
            snapshots.append({"label": label, "data": np.array(self.current_data, copy=True)})
        return snapshots

    def recommend_method(
        self,
        method_id: str,
        *,
        candidate_params: list[dict[str, Any]] | None = None,
        search_mode: str = "fast",
        roi_spec: dict[str, Any] | None = None,
        progress_callback=None,
        cancel_checker=None,
    ) -> dict[str, Any]:
        recommendation = auto_tune_method(
            self.current_data,
            method_id,
            candidate_params=candidate_params,
            header_info=self.header_info,
            trace_metadata=self.trace_metadata,
            base_params=self.default_params(method_id),
            roi_spec=roi_spec,
            search_mode=search_mode,
            progress_callback=progress_callback,
            cancel_checker=cancel_checker,
        )
        self.last_recommendation = recommendation
        return recommendation

    def apply_recommendation(self) -> ProcessingStep:
        if not self.last_recommendation:
            raise RuntimeError("没有可应用的 AutoTune 推荐")
        return self.apply_method(
            str(self.last_recommendation["method_key"]),
            dict(self.last_recommendation.get("recommended_params") or {}),
        )

    def run_manual_auto_comparison(
        self,
        *,
        pipeline: list[str] | None = None,
        manual_params_by_method: dict[str, dict[str, Any]] | None = None,
        baseline_profile_key: str | None = None,
        roi_spec: dict[str, Any] | None = None,
        search_mode: str = "fast",
        progress_callback=None,
        cancel_checker=None,
    ) -> AutoTuneComparisonRun:
        """Compare a manual baseline with AutoTune without mutating session data."""
        resolved_pipeline = pipeline
        resolved_params = manual_params_by_method
        if resolved_pipeline is None and manual_params_by_method is None and self.steps:
            active_steps = [step for step in self.steps if step.enabled]
            if active_steps:
                resolved_pipeline = [step.method_id for step in active_steps]
                resolved_params = {
                    step.method_id: dict(step.params) for step in active_steps
                }
        comparison = run_auto_tune_comparison(
            self.current_data,
            header_info=self.header_info,
            trace_metadata=self.trace_metadata,
            pipeline=resolved_pipeline,
            manual_params_by_method=resolved_params,
            baseline_profile_key=baseline_profile_key,
            roi_spec=roi_spec,
            search_mode=search_mode,
            progress_callback=progress_callback,
            cancel_checker=cancel_checker,
        )
        self.last_manual_auto_comparison = comparison
        return comparison

    def export_last_manual_auto_comparison(
        self,
        *,
        out_dir: str | Path | None = None,
        bundle_name: str | None = None,
        notes: list[str] | None = None,
        cmap: str = "gray",
    ) -> dict[str, Any]:
        if self.last_manual_auto_comparison is None:
            raise RuntimeError("没有可导出的人工/自动对比结果")
        if self.project.manifest.temporary:
            raise PermissionError("临时项目不能导出正式对比报告")
        output_dir = (
            Path(out_dir)
            if out_dir is not None
            else self.project.resolve_relative_path("exports/auto_tune_comparisons")
        )
        return export_auto_tune_comparison_artifacts(
            self.last_manual_auto_comparison,
            out_dir=output_dir,
            bundle_name=bundle_name,
            input_ref=self.line.raw_files[0].path if self.line.raw_files else None,
            notes=notes
            or [
                "测线处理导出的手动参数与推荐参数对比记录。",
                "该对比不修改当前处理链，仅用于参数选择与成果审计。",
            ],
            cmap=cmap,
        )

    def save_version(self, name: str) -> ProcessingResultV1:
        if self.project.manifest.temporary:
            raise PermissionError("临时项目不能保存正式处理结果")
        return self.project.save_processing_result(
            self.line_id,
            self.current_data,
            name=str(name).strip() or "处理结果",
            processing_chain=[step.to_dict() for step in self.steps],
            header_info=self.header_info,
            trace_metadata=self.trace_metadata,
        )

    def _resolve_params(
        self,
        method_id: str,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        resolved = self.default_params(method_id)
        resolved.update(dict(params or {}))
        return resolved

    def _replay_steps(self) -> None:
        data = np.array(self.original_data, copy=True)
        header_info = clone_header_info(self.original_header_info)
        trace_metadata = clone_trace_metadata(self.original_trace_metadata)
        for step in self.steps:
            if not step.enabled:
                continue
            runtime = prepare_runtime_params(
                step.method_id,
                step.params,
                header_info,
                trace_metadata,
                data.shape,
            )
            data, metadata = run_processing_method(data, step.method_id, runtime)
            header_info = merge_result_header_info(header_info, metadata, data.shape)
            trace_metadata = merge_result_trace_metadata(trace_metadata, metadata)
            step.metadata = dict(metadata)
        self.current_data = np.array(data, copy=True)
        self.header_info = clone_header_info(header_info)
        self.trace_metadata = clone_trace_metadata(trace_metadata)
        self.last_preview = None

    def _validate_step_index(self, index: int) -> None:
        if index < 0 or index >= len(self.steps):
            raise IndexError(index)

    def _snapshot(self) -> _SessionState:
        return _SessionState(
            data=np.array(self.current_data, copy=True),
            header_info=clone_header_info(self.header_info),
            trace_metadata=clone_trace_metadata(self.trace_metadata),
            steps=copy.deepcopy(self.steps),
        )

    def _restore(self, state: _SessionState) -> None:
        self.current_data = np.array(state.data, copy=True)
        self.header_info = clone_header_info(state.header_info)
        self.trace_metadata = clone_trace_metadata(state.trace_metadata)
        self.steps = copy.deepcopy(state.steps)


def _load_project_line(
    project: ProjectService,
    line: LineRecordV1,
    path: Path,
) -> dict[str, Any]:
    payload = auto_load_data(str(path))
    if path.suffix.lower() not in {".csv", ".txt"}:
        return payload
    header = _detect_csv_header(path)
    if not header:
        return payload
    raw = np.genfromtxt(path, delimiter=",", skip_header=4)
    sidecars: dict[str, str] = {}
    for kind in ("rtk", "imu", "altimeter"):
        value = line.sidecars.get(kind)
        if not value:
            continue
        sidecar = Path(value)
        if not sidecar.is_absolute():
            sidecar = project.resolve_relative_path(sidecar)
        sidecars[f"{kind}_path"] = str(sidecar)
    data, trace_metadata, updated_header = extract_airborne_csv_payload(
        raw,
        header,
        **sidecars,
    )
    return {
        "data": data,
        "header_info": updated_header or header,
        "trace_metadata": trace_metadata or {},
        "path": str(path),
    }


def _detect_csv_header(path: Path) -> dict[str, Any] | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:4]
    except OSError:
        return None
    if len(lines) < 4 or any("=" not in line for line in lines):
        return None
    values: dict[str, float] = {}
    for line in lines:
        left, right = line.split("=", 1)
        try:
            values[left.strip()] = float(right.strip().split()[0])
        except (ValueError, IndexError):
            return None
    samples = values.get("Number of Samples")
    traces = values.get("Number of Traces")
    if samples is None or traces is None:
        return None
    return {
        "a_scan_length": int(samples),
        "num_traces": int(traces),
        "total_time_ns": float(
            values.get("Time windows (ns)", values.get("Time windows", 0.0))
        ),
        "trace_interval_m": float(
            values.get("Trace interval (m)", values.get("Trace interval", 0.01))
        ),
    }


def _assert_project_line_ready_for_processing(
    project: ProjectService,
    line_id: str,
) -> None:
    """Keep processing gates enforceable outside the visible workbench UI."""
    if project.manifest.temporary:
        raise PermissionError("临时项目仅允许浏览与质控；归档为正式项目后才能进入测线处理。")
    report = QcService(project).run_line_qc(line_id)
    if not report.can_process:
        codes = ", ".join(item.code for item in report.items if item.severity == "error")
        raise PermissionError(f"存在阻断质控错误，不能进入测线处理：{codes or 'unknown'}")
    if report.requires_review:
        codes = ", ".join(
            item.code
            for item in report.items
            if item.severity == "warning" and not item.acknowledged
        )
        raise PermissionError(f"存在未确认质控警告，记录说明后才能进入测线处理：{codes or 'unknown'}")


__all__ = [
    "ProcessingPreview",
    "ProcessingSessionService",
    "ProcessingStep",
]
