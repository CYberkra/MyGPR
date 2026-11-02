"""Read-only, full-resolution input preparation for manual gain editing."""
from __future__ import annotations

from typing import Any

import numpy as np

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.processing.service import ProcessingService
from mygpr.application.project.service import ProjectService
from mygpr.domain.processing.models import ProcessingRequest

MAX_PREVIEW_CELLS = 8_000_000
PREFIX_REQUIRING_SAVED_INPUT = frozenset({
    "time_to_depth", "motion_compensation_speed", "motion_compensation_v2",
    "stolt_migration", "kirchhoff_migration", "rtm_migration",
})


def prepare_gain_preview(projects: ProjectService, processing: ProcessingService, project_id: str, line_id: str,
                         artifact_id: str, prefix: list[dict[str, Any]],
                         context: ExecutionContext | None = None) -> dict[str, Any]:
    """Replay enabled preceding steps in memory, never save artifacts or drafts."""
    context = context or ExecutionContext.null()
    context.raise_if_cancelled()
    enabled = [step for step in prefix if step.get("enabled", True)]
    for step in enabled:
        if step.get("method_id") in PREFIX_REQUIRING_SAVED_INPUT:
            name = processing.get_method(step["method_id"]).name
            raise ValueError(f"前置步骤「{name}」可能扩展网格，请先运行该步骤，再从其成果编辑增益；预览不会降采样")
    info = (projects.get_artifact_dataset_info(project_id, line_id, artifact_id)
            if artifact_id else projects.get_dataset_info(project_id, line_id))
    if int(np.prod(info.shape, dtype=np.int64)) > MAX_PREVIEW_CELLS:
        raise ValueError("当前输入超过完整预览上限，请先裁剪为较小范围；不会降采样预览")
    matrix = np.empty(info.shape, dtype=np.float32)
    blocks = (projects.iter_artifact_blocks(project_id, line_id, artifact_id, block_rows=64)
              if artifact_id else projects.iter_dataset_blocks(project_id, line_id, block_rows=64))
    expected = 0
    for start, end, block in blocks:
        context.raise_if_cancelled()
        if start != expected or end <= start or end > info.shape[0] or block.shape != (end-start, info.shape[1]):
            raise ValueError("增益预览输入数据块不连续或形状不一致")
        matrix[start:end] = block
        expected = end
    if expected != info.shape[0]:
        raise ValueError("增益预览输入不完整")
    header = dict(info.metadata)
    # Dataset-info JSON serializes stored float32 coordinates as Python lists.
    # Restore the same precision as ProjectLineData before validating a grid;
    # interpreting those rounded values as float64 invents nonuniform sampling.
    for key in ('time_axis_ns', 'distance_axis_m', 'depth_axis_m'):
        if header.get(key) is not None:
            header[key] = np.asarray(header[key], dtype=np.float32)
    header.setdefault("time_window_ns", info.time_window_ns)
    header.setdefault("length_m", info.length_m)
    trace_metadata = (projects.read_artifact_trace_metadata(project_id, line_id, artifact_id)
                      if artifact_id else projects.read_trace_metadata(project_id, line_id))
    for index, step in enumerate(enabled):
        context.raise_if_cancelled()
        result = processing.execute_method(ProcessingRequest(
            matrix, str(step["method_id"]), dict(step.get("params") or {}), header, trace_metadata,
        ), context.child(index, len(enabled)))
        matrix, header, trace_metadata = result.data, result.header_info, result.trace_metadata
        del result
        if matrix.size > MAX_PREVIEW_CELLS:
            raise ValueError("前置步骤的结果超过完整预览上限；不会截断或降采样")
    context.raise_if_cancelled()
    return {"matrix": matrix, "header": header, "trace_metadata": trace_metadata,
            "prefix_count": len(enabled), "nonfinite_count": int(matrix.size - np.isfinite(matrix).sum())}
