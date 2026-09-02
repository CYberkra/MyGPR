#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project-bound processing use cases."""
from __future__ import annotations

import uuid

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.processing.service import ProcessingService
from mygpr.application.project.service import (
    ProjectApplicationError,
    ProjectService,
)
from mygpr.domain.processing.models import BlockPipelineSummary, PipelineDefinition, ResourceEstimate
from mygpr.domain.project.models import LineDatasetInfo, ProjectArtifact


class ProjectReadOnlyError(ProjectApplicationError):
    """只读会话上请求写入类操作（错误码 MYGPR_PROJECT_READ_ONLY）。"""

    error_code = "MYGPR_PROJECT_READ_ONLY"

# 输出 header 中需要持久化的轴标量（P1-1：形状变更方法如 time_cut/set_zero_time
# 会改变时间窗与零点，若不保存，二次处理/成像的物理轴会按原始时窗错误计算）
_OUTPUT_HEADER_KEYS = (
    "total_time_ns", "time_window_ns", "time_cut_offset_ns",
    "a_scan_length", "num_traces", "dielectric_constant", "axis_transform",
)


def _output_header_summary(header: dict[str, Any] | None) -> dict[str, Any]:
    """从处理结果 header 提取可 JSON 序列化的输出轴标量。"""
    header = dict(header or {})
    out: dict[str, Any] = {}
    for key in _OUTPUT_HEADER_KEYS:
        if key not in header:
            continue
        value = header[key]
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        out[key] = value
    return out


@dataclass(slots=True)
class ProjectLineBlockSource:
    """Adapter exposing a persisted line through bounded row blocks."""

    projects: ProjectService
    project_id: str
    line_id: str
    info: LineDatasetInfo
    artifact_id: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.info.shape

    @property
    def dtype(self) -> str:
        return self.info.dtype

    def iter_blocks(self, *, block_rows: int) -> Iterator[tuple[int, int, Any]]:
        if self.artifact_id:
            return self.projects.iter_artifact_blocks(
                self.project_id, self.line_id, self.artifact_id, block_rows=block_rows
            )
        return self.projects.iter_dataset_blocks(
            self.project_id, self.line_id, block_rows=block_rows
        )


class ProjectProcessingService:
    """Run a processing pipeline against a persisted line and commit the result."""

    def __init__(self, projects: ProjectService, processing: ProcessingService) -> None:
        self._projects = projects
        self._processing = processing

    def estimate_pipeline(
        self,
        project_id: str,
        line_id: str,
        pipeline: PipelineDefinition,
        *,
        input_artifact_id: str = "",
    ) -> ResourceEstimate:
        info = (
            self._projects.get_artifact_dataset_info(project_id, line_id, input_artifact_id)
            if input_artifact_id else self._projects.get_dataset_info(project_id, line_id)
        )
        return self._processing.estimate_pipeline_shape(info.shape, info.dtype, pipeline)

    def execute_pipeline(
        self,
        project_id: str,
        line_id: str,
        pipeline: PipelineDefinition,
        *,
        result_name: str = "",
        branch_id: str = "",
        input_artifact_id: str = "",
        save_intermediates: bool = True,
        context: ExecutionContext | None = None,
    ) -> ProjectArtifact:
        """跑链并提交最终成果。

        ``save_intermediates=True``（默认，需求 B7）时每步输出落盘为
        ``intermediate`` 成果并归入同一 run_group；此时分块流式执行不可用
        （中间矩阵必须物化），自动改走 loaded 路径——多 GB 数据请传 False。
        """
        execution_context = context or ExecutionContext.null()
        summary = self._projects.get_summary(project_id)
        if summary.read_only:
            # 只读会话（陈旧锁未恢复/另一实例占用）整条链算完才在 catalog
            # 写入时失败，白费算力且报错不指向根因——入口即拦截并给
            # 可操作提示（P1 评审：processing_service 缺 read_only 前置检查）。
            raise ProjectReadOnlyError(
                f"项目以只读方式打开（可能存在残留锁或另一实例占用），"
                f"无法保存处理成果：{summary.root_path}。"
                f"请关闭其他实例后重新打开项目。")
        use_block = (
            not save_intermediates
            and self._processing.supports_block_pipeline(pipeline)
        )
        if use_block:
            artifact = self._execute_block_pipeline(
                project_id, line_id, pipeline, result_name, branch_id, input_artifact_id, execution_context
            )
        else:
            artifact = self._execute_loaded_pipeline(
                project_id, line_id, pipeline, result_name, branch_id, input_artifact_id, execution_context
            )
        self._emit_artifact(artifact, execution_context)
        return artifact

    def _execute_block_pipeline(
        self,
        project_id: str,
        line_id: str,
        pipeline: PipelineDefinition,
        result_name: str,
        branch_id: str,
        input_artifact_id: str,
        context: ExecutionContext,
    ) -> ProjectArtifact:
        context.report_progress(0, 1, "准备分块处理")
        info = (
            self._projects.get_artifact_dataset_info(project_id, line_id, input_artifact_id)
            if input_artifact_id else self._projects.get_dataset_info(project_id, line_id)
        )
        source = ProjectLineBlockSource(
            self._projects, project_id, line_id, info, artifact_id=input_artifact_id
        )
        header = self._header_from_info(info)

        def save_result(matrix: Any, summary: BlockPipelineSummary) -> ProjectArtifact:
            return self._save_block_result(
                project_id,
                line_id,
                pipeline,
                matrix,
                summary,
                result_name,
                branch_id,
                input_artifact_id,
                context.child(1, 2),
            )

        return self._processing.execute_pipeline_blocks(
            source,
            pipeline,
            header_info=header,
            trace_metadata={},
            consumer=save_result,
            context=context.child(0, 2),
        )

    def _execute_loaded_pipeline(
        self,
        project_id: str,
        line_id: str,
        pipeline: PipelineDefinition,
        result_name: str,
        branch_id: str,
        input_artifact_id: str,
        context: ExecutionContext,
    ) -> ProjectArtifact:
        context.report_progress(0, 1, "执行资源预检")
        info = (
            self._projects.get_artifact_dataset_info(project_id, line_id, input_artifact_id)
            if input_artifact_id else self._projects.get_dataset_info(project_id, line_id)
        )
        estimate = self._processing.estimate_pipeline_shape(info.shape, info.dtype, pipeline)
        self._processing.validate_estimate(estimate, context=context, operation=pipeline.name)
        context.report_progress(0, 1, "读取项目测线")
        source = (
            self._projects.read_artifact_dataset(project_id, line_id, input_artifact_id)
            if input_artifact_id else self._projects.read_dataset(project_id, line_id)
        )
        run_group_id = uuid.uuid4().hex
        result = self._processing.execute_pipeline(
            source.data,
            pipeline,
            header_info=source.header_info,
            trace_metadata=source.trace_metadata,
            context=context.child(0, 2),
        )
        steps = pipeline_steps(pipeline)
        lineage = loaded_pipeline_lineage(result.step_results, self._processing)
        final_method = result.step_results[-1].method_id if result.step_results else "pipeline"

        # B7：逐步中间成果落盘（final 之外每步一个 intermediate 成果，
        # run_group_id 归组；数据可重跑再生，清单哈希保留审计）
        step_context = context.child(1, 2)
        for index, step_result in enumerate(result.step_results[:-1], start=1):
            step_name = f"{result_name or pipeline.name} 步骤{index}_{step_result.method_id}"
            self._projects.save_processing_artifact(
                project_id,
                line_id,
                step_result.data,
                name=step_name,
                method_id=step_result.method_id,
                method_name=step_result.method_id,
                params={
                    "pipeline": steps[:index],
                    "lineage": lineage[:index],
                    "execution_mode": "loaded",
                    "parent_artifact_id": input_artifact_id,
                    "output_header": _output_header_summary(step_result.header_info),
                    "artifact_kind": "intermediate",
                    "run_group_id": run_group_id,
                    "run_step_index": index,
                },
                pipeline=[{**step, "lineage": lineage[i]} for i, step in enumerate(steps[:index])],
                branch_id=branch_id or f"{line_id}:main",
                input_dataset={
                    **source.header_info, "execution_mode": "loaded",
                    "parent_artifact_id": input_artifact_id,
                },
                context=step_context,
            )

        return self._projects.save_processing_artifact(
            project_id,
            line_id,
            result.data,
            name=result_name or pipeline.name,
            method_id=final_method,
            method_name=pipeline.name,
            params={
                "pipeline": steps, "lineage": lineage, "execution_mode": "loaded",
                "parent_artifact_id": input_artifact_id,
                "output_header": _output_header_summary(result.header_info),
                "artifact_kind": "processing",
                "run_group_id": run_group_id,
            },
            pipeline=[{**step, "lineage": lineage[index]} for index, step in enumerate(steps)],
            branch_id=branch_id or f"{line_id}:main",
            input_dataset={
                **source.header_info, "execution_mode": "loaded",
                "parent_artifact_id": input_artifact_id,
            },
            context=step_context,
        )

    def _save_block_result(
        self,
        project_id: str,
        line_id: str,
        pipeline: PipelineDefinition,
        matrix: Any,
        summary: BlockPipelineSummary,
        result_name: str,
        branch_id: str,
        input_artifact_id: str,
        context: ExecutionContext,
    ) -> ProjectArtifact:
        steps = pipeline_steps(pipeline)
        lineage = [
            {
                "method_id": item.method_id,
                "params": dict(item.params),
                "metadata": dict(item.metadata),
                "implementation_version": item.implementation_version,
                "output_shape": list(item.output_shape),
                "output_dtype": item.output_dtype,
                "output_sha256": item.output_sha256,
            }
            for item in summary.step_records
        ]
        final_method = summary.step_records[-1].method_id if summary.step_records else "pipeline"
        input_dataset = {
            **summary.header_info,
            "input_data_sha256": summary.input_sha256,
            "output_data_sha256": summary.output_sha256,
            "execution_mode": "file_backed_blocks",
            "parent_artifact_id": input_artifact_id,
        }
        return self._projects.save_processing_artifact(
            project_id,
            line_id,
            matrix,
            name=result_name or pipeline.name,
            method_id=final_method,
            method_name=pipeline.name,
            params={
                "pipeline": steps, "lineage": lineage,
                "execution_mode": "file_backed_blocks",
                "parent_artifact_id": input_artifact_id,
                "output_header": _output_header_summary(summary.header_info),
            },
            pipeline=[{**step, "lineage": lineage[index]} for index, step in enumerate(steps)],
            branch_id=branch_id or f"{line_id}:main",
            input_dataset=input_dataset,
            context=context,
        )

    @staticmethod
    def _header_from_info(info: LineDatasetInfo) -> dict[str, Any]:
        return {
            **dict(info.metadata),
            "line_id": info.line_id,
            "a_scan_length": info.shape[0],
            "num_traces": info.shape[1],
            "length_m": info.length_m,
            "track_length_m": info.length_m,
            "time_window_ns": info.time_window_ns,
            "total_time_ns": info.time_window_ns,
            "dielectric_constant": info.dielectric_constant,
            "source_path": info.source_path,
            "format_name": info.format_name,
        }

    @staticmethod
    def _emit_artifact(artifact: ProjectArtifact, context: ExecutionContext) -> None:
        context.emit_artifact(
            {
                "artifact_id": artifact.artifact_id,
                "line_id": artifact.line_id,
                "data_reference": artifact.data_reference,
            }
        )
        context.report_progress(1, 1, "项目处理成果已保存")


def pipeline_steps(pipeline: PipelineDefinition) -> list[dict[str, Any]]:
    return [
        {
            "method_id": step.method_id,
            "label": step.label,
            "params": dict(step.params),
            "enabled": bool(step.enabled),
        }
        for step in pipeline.steps
        if step.enabled
    ]


def loaded_pipeline_lineage(step_results: list[Any], processing: ProcessingService) -> list[dict[str, Any]]:
    """Build bounded, serializable lineage for loaded/global execution."""
    lineage: list[dict[str, Any]] = []
    for result in step_results:
        descriptor = processing.get_method(result.method_id)
        metadata = summarize_metadata(result.metadata)
        lineage.append(
            {
                "method_id": result.method_id,
                "params": summarize_metadata(result.params),
                "metadata": metadata,
                "implementation_version": str(
                    result.metadata.get("implementation_version") or descriptor.implementation_version
                ),
                "output_shape": [int(value) for value in result.data.shape],
                "output_dtype": str(result.data.dtype),
                "runtime_warnings": [dict(item) for item in result.runtime_warnings],
            }
        )
    return lineage


def summarize_metadata(value: Any) -> Any:
    """Replace large arrays/opaque values with deterministic structural summaries."""
    if isinstance(value, np.ndarray):
        return {"shape": [int(item) for item in value.shape], "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): summarize_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "repr": repr(value)[:240]}


__all__ = ["ProjectLineBlockSource", "ProjectProcessingService", "loaded_pipeline_lineage"]
