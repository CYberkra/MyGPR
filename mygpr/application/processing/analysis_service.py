#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Processing diagnostics, comparison and evidence use cases."""
from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.processing.service import ProcessingService
from mygpr.application.project.service import ProjectService
from mygpr.domain.processing.models import ProcessingRequest
from mygpr.domain.processing.workbench import (
    DatasetComparison,
    ProcessingEvidencePackage,
    ProcessingStepDiagnostic,
    WorkbenchStep,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


@dataclass(slots=True)
class _AnalysisSession:
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
    steps: list[WorkbenchStep]
    revision: int


class ProcessingAnalysisService:
    """Own analysis and evidence operations outside the mutable session service."""

    def __init__(self, projects: ProjectService, processing: ProcessingService) -> None:
        self._projects = projects
        self._processing = processing

    def compare_original(
        self,
        session: Any,
        *,
        max_samples: int = 900,
        max_traces: int = 1800,
    ) -> DatasetComparison:
        rows = min(session.original_data.shape[0], session.current_data.shape[0])
        cols = min(session.original_data.shape[1], session.current_data.shape[1])
        sample_indices = _bounded_indices(rows, max_samples)
        trace_indices = _bounded_indices(cols, max_traces)
        left = session.original_data[np.ix_(sample_indices, trace_indices)]
        right = session.current_data[np.ix_(sample_indices, trace_indices)]
        difference = right - left
        return DatasetComparison(
            session.line_id,
            "原始数据",
            "当前处理",
            left,
            right,
            difference,
            self._comparison_metrics(left, right),
        )

    def compare_candidate(
        self,
        session: Any,
        method_id: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_samples: int = 900,
        max_traces: int = 1800,
    ) -> DatasetComparison:
        result = self._execute(session, method_id, dict(params or {}))
        rows = min(session.current_data.shape[0], result.data.shape[0])
        cols = min(session.current_data.shape[1], result.data.shape[1])
        sample_indices = _bounded_indices(rows, max_samples)
        trace_indices = _bounded_indices(cols, max_traces)
        left = session.current_data[np.ix_(sample_indices, trace_indices)]
        right = np.asarray(result.data)[np.ix_(sample_indices, trace_indices)]
        difference = right - left
        return DatasetComparison(
            session.line_id,
            "当前处理",
            f"推荐参数 · {method_id}",
            left,
            right,
            difference,
            self._comparison_metrics(left, right),
        )

    def diagnose_steps(self, session: Any) -> tuple[ProcessingStepDiagnostic, ...]:
        working = self._clone_session(session)
        diagnostics: list[ProcessingStepDiagnostic] = []
        for index, spec in enumerate(session.steps):
            if not spec.enabled:
                diagnostics.append(
                    ProcessingStepDiagnostic(
                        index,
                        spec.step_id,
                        spec.method_id,
                        spec.label,
                        False,
                        "disabled",
                        100.0,
                        {},
                        ("步骤已禁用，未执行质量诊断。",),
                        tuple(working.current_data.shape),
                        str(working.current_data.dtype),
                    )
                )
                continue
            before = np.asarray(working.current_data, dtype=np.float64)
            try:
                result = self._execute(working, spec.method_id, dict(spec.params))
            except (
                ArithmeticError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                diagnostics.append(
                    ProcessingStepDiagnostic(
                        index,
                        spec.step_id,
                        spec.method_id,
                        spec.label,
                        True,
                        "error",
                        0.0,
                        {},
                        (f"执行失败：{type(exc).__name__}: {exc}",),
                        (0, 0),
                        "",
                    )
                )
                break
            after = np.asarray(result.data, dtype=np.float64)
            metrics, warnings, score, status = self._step_quality(
                before,
                after,
                np.asarray(session.original_data, dtype=np.float64),
            )
            diagnostics.append(
                ProcessingStepDiagnostic(
                    index,
                    spec.step_id,
                    spec.method_id,
                    spec.label,
                    True,
                    status,
                    score,
                    metrics,
                    warnings,
                    tuple(int(value) for value in after.shape),
                    str(after.dtype),
                )
            )
            self._accept_result(working, result)
        return tuple(diagnostics)

    def export_evidence(
        self,
        session: Any,
        destination_dir: str | None = None,
        *,
        include_data: bool = False,
    ) -> ProcessingEvidencePackage:
        summary = self._projects.get_summary(session.project_id)
        if summary.read_only:
            raise PermissionError("只读项目不能导出 Processing Evidence")
        package_id = f"PEV-{uuid.uuid4().hex[:12]}"
        created_at = _utc_now()
        base = (
            Path(destination_dir).expanduser().resolve()
            if destination_dir
            else Path(summary.root_path) / "exports" / "processing_evidence"
        )
        package_dir = base / package_id
        package_dir.mkdir(parents=True, exist_ok=False)
        diagnostics = self.diagnose_steps(session)
        comparison = self.compare_original(session, max_samples=512, max_traces=1024)
        manifest = self._build_evidence_manifest(
            session,
            package_id=package_id,
            created_at=created_at,
            diagnostics=diagnostics,
            comparison=comparison,
            include_data=include_data,
        )
        manifest_path, manifest_sha, files = self._write_evidence_files(
            session,
            package_dir,
            manifest,
            include_data=include_data,
        )
        return ProcessingEvidencePackage(
            package_id,
            session.project_id,
            session.line_id,
            session.session_id,
            str(manifest_path),
            manifest_sha,
            created_at,
            tuple(files),
            manifest,
        )

    @staticmethod
    def _build_evidence_manifest(
        session: Any,
        *,
        package_id: str,
        created_at: str,
        diagnostics: Sequence[ProcessingStepDiagnostic],
        comparison: DatasetComparison,
        include_data: bool,
    ) -> dict[str, Any]:
        return {
            "schema": "mygpr.processing_evidence.v1",
            "package_id": package_id,
            "created_at": created_at,
            "project_id": session.project_id,
            "line_id": session.line_id,
            "session_id": session.session_id,
            "input_artifact_id": session.input_artifact_id,
            "branch_id": session.branch_id,
            "session_name": session.name,
            "revision": session.revision,
            "current_shape": list(session.current_data.shape),
            "current_dtype": str(session.current_data.dtype),
            "pipeline": [
                {
                    "step_index": index,
                    "step_id": item.step_id,
                    "method_id": item.method_id,
                    "label": item.label,
                    "params": _json_safe(dict(item.params)),
                    "enabled": item.enabled,
                    "metadata": _json_safe(dict(item.metadata)),
                }
                for index, item in enumerate(session.steps)
            ],
            "step_diagnostics": [
                {
                    "step_index": item.step_index,
                    "step_id": item.step_id,
                    "method_id": item.method_id,
                    "label": item.label,
                    "enabled": item.enabled,
                    "status": item.status,
                    "score": item.score,
                    "metrics": dict(item.metrics),
                    "warnings": list(item.warnings),
                    "output_shape": list(item.output_shape),
                    "output_dtype": item.output_dtype,
                }
                for item in diagnostics
            ],
            "original_comparison": dict(comparison.metrics),
            "data_included": bool(include_data),
            "files": [],
        }

    @staticmethod
    def _write_evidence_files(
        session: Any,
        package_dir: Path,
        manifest: dict[str, Any],
        *,
        include_data: bool,
    ) -> tuple[Path, str, list[str]]:
        files: list[str] = []
        if include_data:
            data_path = package_dir / "processed_data.npy"
            np.save(
                data_path,
                np.asarray(session.current_data, dtype=np.float32),
                allow_pickle=False,
            )
            manifest["files"].append(
                {
                    "name": data_path.name,
                    "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    "size_bytes": data_path.stat().st_size,
                }
            )
            files.append(str(data_path))
        canonical = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        manifest_sha = hashlib.sha256(canonical).hexdigest()
        manifest["manifest_sha256"] = manifest_sha
        manifest_path = package_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files.insert(0, str(manifest_path))
        return manifest_path, manifest_sha, files

    def _execute(self, session: Any, method_id: str, params: dict[str, Any]) -> Any:
        request = ProcessingRequest(
            data=session.current_data,
            method_id=method_id,
            params=params,
            header_info=session.current_header,
            trace_metadata=session.current_trace_metadata,
        )
        return self._processing.execute_method(request, ExecutionContext.null())

    @staticmethod
    def _comparison_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
        left_arr = np.asarray(left, dtype=np.float64)
        right_arr = np.asarray(right, dtype=np.float64)
        difference = right_arr - left_arr
        left_flat = left_arr.ravel()
        right_flat = right_arr.ravel()
        correlation = 0.0
        if left_flat.size > 1 and np.std(left_flat) > 0 and np.std(right_flat) > 0:
            correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])
        return {
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "mae": float(np.mean(np.abs(difference))),
            "correlation": correlation,
            "energy_ratio": float(
                np.sum(right_flat * right_flat)
                / max(np.sum(left_flat * left_flat), 1e-12)
            ),
            "peak_ratio": float(
                np.max(np.abs(right_flat)) / max(np.max(np.abs(left_flat)), 1e-12)
            ),
        }

    @classmethod
    def _step_quality(
        cls,
        before: np.ndarray,
        after: np.ndarray,
        original: np.ndarray,
    ) -> tuple[dict[str, float], tuple[str, ...], float, str]:
        rows = min(before.shape[0], after.shape[0])
        cols = min(before.shape[1], after.shape[1])
        before_view = np.asarray(before[:rows, :cols], dtype=np.float64)
        after_view = np.asarray(after[:rows, :cols], dtype=np.float64)
        original_view = np.asarray(original[:rows, :cols], dtype=np.float64)
        finite_ratio = float(np.isfinite(after_view).mean())
        safe_after = np.nan_to_num(after_view, nan=0.0, posinf=0.0, neginf=0.0)
        metrics = cls._comparison_metrics(before_view, safe_after)
        original_metrics = cls._comparison_metrics(original_view, safe_after)
        metrics.update(
            {
                "finite_ratio": finite_ratio,
                "rms_before": float(np.sqrt(np.mean(before_view * before_view))),
                "rms_after": float(np.sqrt(np.mean(safe_after * safe_after))),
                "correlation_original": original_metrics["correlation"],
                "shape_changed": float(before.shape != after.shape),
            }
        )
        warnings: list[str] = []
        score = 100.0
        if finite_ratio < 1.0:
            warnings.append(f"输出包含非有限值，有限比例 {finite_ratio:.6f}。")
            score -= 100.0 * min(1.0, max(0.0, 1.0 - finite_ratio) * 100.0)
        energy_ratio = metrics["energy_ratio"]
        if energy_ratio < 0.01:
            warnings.append("输出能量低于输入的 1%，可能过度抑制。")
            score -= 35.0
        elif energy_ratio > 25.0:
            warnings.append("输出能量超过输入的 25 倍，可能发生增益失控。")
            score -= 35.0
        peak_ratio = metrics["peak_ratio"]
        if peak_ratio < 0.03:
            warnings.append("峰值保留低于 3%，需复核目标响应损失。")
            score -= 25.0
        elif peak_ratio > 12.0:
            warnings.append("峰值放大超过 12 倍，需复核削波和数值稳定性。")
            score -= 25.0
        if before.shape != after.shape:
            warnings.append(f"数据形状由 {before.shape} 变为 {after.shape}。")
            score -= 5.0
        status = "error" if finite_ratio < 0.999999 else "warning" if warnings else "pass"
        return metrics, tuple(warnings), max(0.0, score), status

    @staticmethod
    def _clone_session(source: Any) -> _AnalysisSession:
        return _AnalysisSession(
            session_id=source.session_id,
            project_id=source.project_id,
            line_id=source.line_id,
            input_artifact_id=source.input_artifact_id,
            branch_id=source.branch_id,
            name=source.name,
            original_data=np.array(source.original_data, copy=True),
            current_data=np.array(source.original_data, copy=True),
            original_header=copy.deepcopy(source.original_header),
            current_header=copy.deepcopy(source.original_header),
            original_trace_metadata={
                key: np.array(value, copy=True)
                for key, value in source.original_trace_metadata.items()
            },
            current_trace_metadata={
                key: np.array(value, copy=True)
                for key, value in source.original_trace_metadata.items()
            },
            steps=list(source.steps),
            revision=int(source.revision),
        )

    @staticmethod
    def _accept_result(session: _AnalysisSession, result: Any) -> None:
        session.current_data = np.asarray(result.data, dtype=np.float32).copy()
        session.current_header = copy.deepcopy(result.header_info)
        session.current_trace_metadata = {
            key: np.array(value, copy=True)
            for key, value in result.trace_metadata.items()
        }


__all__ = ["ProcessingAnalysisService"]
