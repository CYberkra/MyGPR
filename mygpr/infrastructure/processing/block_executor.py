"""File-backed processing pipeline for project-scale GPR matrices."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.processing.ports import (
    BlockPipelineConsumer,
    BlockPipelineExecutorPort,
    MatrixBlockSourcePort,
)
from mygpr.domain.processing.models import (
    BlockPipelineSummary,
    PipelineDefinition,
    ProcessingStepRecord,
    ResourceEstimate,
)
from mygpr.infrastructure.processing.algorithms.methods import NATIVE_ALGORITHMS, NativeAlgorithm

DEFAULT_BLOCK_BYTES = 64 * 1024 * 1024
MIN_FREE_RESERVE = 128 * 1024 * 1024


class InsufficientProcessingStorage(RuntimeError):
    """Raised before execution when the workspace cannot hold bounded intermediates."""


class FileBackedBlockPipelineExecutor(BlockPipelineExecutorPort):
    """Run migrated methods through rotating NumPy memmaps."""

    def __init__(self, workspace_root: str | Path | None = None, *, block_bytes: int = DEFAULT_BLOCK_BYTES) -> None:
        configured = workspace_root or os.environ.get("MYGPR_PROCESSING_TEMP")
        self._workspace_root = Path(configured).expanduser().resolve() if configured else None
        self._block_bytes = max(64 * 1024, int(block_bytes))

    def supports(self, pipeline: PipelineDefinition) -> bool:
        enabled = [step for step in pipeline.steps if step.enabled]
        return bool(enabled) and all(
            (algorithm := NATIVE_ALGORITHMS.get(step.method_id)) is not None
            and algorithm.supports_file_backed
            and algorithm.supports_block_params(dict(step.params))
            for step in enabled
        )

    def estimate(self, shape: tuple[int, int], dtype: str, pipeline: PipelineDefinition) -> ResourceEstimate:
        if not self.supports(pipeline):
            return ResourceEstimate(
                relative_cost="unsupported",
                supports_cancellation=True,
                supports_chunking=False,
                notes=("pipeline contains a legacy-only or unsupported method",),
            )
        source_bytes = int(np.prod(shape, dtype=np.int64)) * max(1, np.dtype(dtype).itemsize)
        normalized_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
        enabled = [NATIVE_ALGORITHMS[step.method_id] for step in pipeline.steps if step.enabled]
        minimum_block = max(
            shape[0] * np.dtype(np.float64).itemsize if item.block_axis == "columns"
            else shape[1] * np.dtype(np.float64).itemsize
            if item.block_axis == "rows"
            else normalized_bytes
            for item in enabled
        )
        global_peak = max(
            (int(normalized_bytes * item.memory_multiplier) for item in enabled if item.block_axis == "global"),
            default=0,
        )
        block_peak = max(min(normalized_bytes, self._block_bytes), minimum_block) * 5
        peak_memory = max(global_peak, block_peak)
        temp_multiplier = max((float(item.temporary_multiplier) for item in enabled), default=1.0)
        relative_cost = "very_high" if any(item.relative_cost == "very_high" for item in enabled) else (
            "high" if any(item.relative_cost == "high" for item in enabled) else "medium"
        )
        return ResourceEstimate(
            memory_bytes=peak_memory,
            temporary_disk_bytes=source_bytes + int(normalized_bytes * max(2.0, temp_multiplier)),
            relative_cost=relative_cost,
            supports_cancellation=True,
            supports_chunking=True,
            notes=(
                "file-backed rotating workspace",
                f"target block bytes: {self._block_bytes}",
                "global transforms avoid HDF5 materialisation but still require bounded spectral/factor workspaces",
            ),
        )

    def execute(
        self,
        source: MatrixBlockSourcePort,
        pipeline: PipelineDefinition,
        *,
        header_info: dict[str, Any],
        trace_metadata: dict[str, Any],
        consumer: BlockPipelineConsumer[Any],
        context: ExecutionContext | None = None,
    ) -> Any:
        if not self.supports(pipeline):
            raise ValueError("pipeline is not eligible for file-backed block execution")
        execution_context = context or ExecutionContext.null()
        estimate = self.estimate(source.shape, source.dtype, pipeline)
        workspace_parent = self._prepare_workspace_parent(estimate.temporary_disk_bytes)
        with tempfile.TemporaryDirectory(prefix="mygpr-block-", dir=workspace_parent) as temporary:
            return self._execute_in_workspace(
                Path(temporary), source, pipeline, header_info, trace_metadata, consumer, execution_context
            )

    def _prepare_workspace_parent(self, required_bytes: int) -> str | None:
        root = self._workspace_root
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        probe = root or Path(tempfile.gettempdir())
        free = shutil.disk_usage(probe).free
        required = max(0, int(required_bytes)) + MIN_FREE_RESERVE
        if free < required:
            raise InsufficientProcessingStorage(
                f"processing workspace requires {required} bytes but only {free} bytes are free"
            )
        return str(root) if root is not None else None

    def _execute_in_workspace(
        self,
        workspace: Path,
        source: MatrixBlockSourcePort,
        pipeline: PipelineDefinition,
        header_info: dict[str, Any],
        trace_metadata: dict[str, Any],
        consumer: BlockPipelineConsumer[Any],
        context: ExecutionContext,
    ) -> Any:
        enabled = [step for step in pipeline.steps if step.enabled]
        current_path = workspace / "input.f32"
        current, input_hash = self._stage_source(source, current_path, context.child(0, len(enabled) + 1))
        records: list[ProcessingStepRecord] = []
        current_header = dict(header_info or {})
        for index, step in enumerate(enabled, start=1):
            algorithm = NATIVE_ALGORITHMS[step.method_id]
            next_path = workspace / f"step-{index:03d}.f32"
            next_matrix, metadata = self._execute_step(
                current, next_path, algorithm, dict(step.params), current_header,
                context.child(index, len(enabled) + 1),
            )
            output_hash = hash_matrix(
                next_matrix, self._row_block_size(next_matrix.shape), context=context.child(index, len(enabled) + 1)
            )
            records.append(
                ProcessingStepRecord(
                    method_id=step.method_id,
                    params=dict(step.params),
                    metadata=json_safe_metadata(metadata),
                    implementation_version=algorithm.implementation_version,
                    output_shape=next_matrix.shape,
                    output_dtype=str(next_matrix.dtype),
                    output_sha256=output_hash,
                )
            )
            old_path = current_path
            del current
            old_path.unlink(missing_ok=True)
            current, current_path = next_matrix, next_path
            current_header.update(a_scan_length=current.shape[0], num_traces=current.shape[1])
        summary = BlockPipelineSummary(
            shape=current.shape,
            dtype=str(current.dtype),
            header_info=current_header,
            trace_metadata={key: np.asarray(value) for key, value in (trace_metadata or {}).items()},
            step_records=records,
            input_sha256=input_hash,
            output_sha256=records[-1].output_sha256,
        )
        context.raise_if_cancelled()
        try:
            return consumer(current, summary)
        finally:
            close_memmap(current)

    def _stage_source(
        self,
        source: MatrixBlockSourcePort,
        path: Path,
        context: ExecutionContext,
    ) -> tuple[np.memmap, str]:
        create_memmap_file(path, source.shape)
        digest = new_matrix_digest(source.shape, np.dtype(np.float32))
        block_rows = self._row_block_size(source.shape)
        total = max(1, int(np.ceil(source.shape[0] / block_rows)))
        expected_start = 0
        for index, (start, end, block) in enumerate(source.iter_blocks(block_rows=block_rows), start=1):
            context.raise_if_cancelled()
            normalized = np.asarray(block, dtype=np.float32)
            if int(start) != expected_start or int(end) <= int(start):
                raise ValueError(f"matrix block sequence is not contiguous at {start}:{end}")
            if normalized.shape != (int(end) - int(start), source.shape[1]):
                raise ValueError(f"matrix block shape mismatch at {start}:{end}: {normalized.shape}")
            matrix = np.memmap(path, mode="r+", dtype=np.float32, shape=source.shape)
            try:
                matrix[start:end] = normalized
                matrix.flush()
            finally:
                close_memmap(matrix)
            digest.update(np.ascontiguousarray(normalized).tobytes())
            expected_start = int(end)
            context.report_progress(index, total, f"暂存输入数据块 {index}/{total}")
        if expected_start != source.shape[0]:
            raise ValueError(f"matrix block source ended at row {expected_start}, expected {source.shape[0]}")
        return np.memmap(path, mode="r+", dtype=np.float32, shape=source.shape), digest.hexdigest()

    def _execute_step(
        self,
        source: np.memmap,
        output_path: Path,
        algorithm: NativeAlgorithm,
        params: dict[str, Any],
        header_info: dict[str, Any],
        context: ExecutionContext,
    ) -> tuple[np.memmap, dict[str, Any]]:
        shape = tuple(int(value) for value in source.shape)
        source_path = Path(str(source.filename))
        close_memmap(source)
        create_memmap_file(output_path, shape)
        runtime_params = prepare_block_params(params, header_info, shape, algorithm.method_id)
        metadata: dict[str, Any] = {}
        runtime_params["_execution_context"] = context
        runtime_params.setdefault("cancel_checker", context.is_cancelled)
        if algorithm.block_axis == "global":
            source_view = np.memmap(source_path, mode="r", dtype=np.float32, shape=shape)
            output_view = np.memmap(output_path, mode="r+", dtype=np.float32, shape=shape)
            try:
                context.report_progress(0, 3, f"{algorithm.name}: 全局算子准备")
                if algorithm.file_function is not None:
                    metadata = algorithm.file_function(
                        source_view,
                        output_view,
                        runtime_params,
                        context,
                        self._row_block_size(shape),
                    )
                else:
                    processed, metadata = algorithm.function(source_view, runtime_params)
                    context.raise_if_cancelled()
                    if tuple(processed.shape) != shape:
                        raise ValueError(
                            f"global method {algorithm.method_id} changed matrix shape from {shape} to {processed.shape}"
                        )
                    spans = self._spans(shape, "rows")
                    for index, (start, end) in enumerate(spans, start=1):
                        context.raise_if_cancelled()
                        output_view[start:end] = np.asarray(processed[start:end], dtype=np.float32)
                        output_view.flush()
                        context.report_progress(index, len(spans), f"{algorithm.name}: 写入结果块 {index}/{len(spans)}")
            finally:
                close_memmap(output_view)
                close_memmap(source_view)
        else:
            spans = self._spans(shape, algorithm.block_axis)
            for index, (start, end) in enumerate(spans, start=1):
                context.raise_if_cancelled()
                source_view = np.memmap(source_path, mode="r", dtype=np.float32, shape=shape)
                try:
                    block_view = (
                        source_view[:, start:end]
                        if algorithm.block_axis == "columns"
                        else source_view[start:end, :]
                    )
                    block = np.array(block_view, dtype=np.float32, copy=True, order="C")
                finally:
                    close_memmap(source_view)
                processed, block_meta = algorithm.function(block, runtime_params)
                output_view = np.memmap(output_path, mode="r+", dtype=np.float32, shape=shape)
                try:
                    if algorithm.block_axis == "columns":
                        output_view[:, start:end] = processed
                    else:
                        output_view[start:end, :] = processed
                    output_view.flush()
                finally:
                    close_memmap(output_view)
                metadata = merge_metadata(metadata, block_meta)
                context.report_progress(index, len(spans), f"{algorithm.name}: 数据块 {index}/{len(spans)}")
        metadata["implementation_version"] = algorithm.implementation_version
        for item in metadata.get("runtime_warnings", []):
            if isinstance(item, dict):
                context.emit_warning(item)
        return np.memmap(output_path, mode="r+", dtype=np.float32, shape=shape), metadata

    def _spans(self, shape: tuple[int, int], axis: str) -> list[tuple[int, int]]:
        samples, traces = shape
        if axis == "columns":
            width = max(1, self._block_bytes // max(samples * 8 * 5, 1))
            return [(start, min(start + width, traces)) for start in range(0, traces, width)]
        height = max(1, self._block_bytes // max(traces * 8 * 5, 1))
        return [(start, min(start + height, samples)) for start in range(0, samples, height)]

    def _row_block_size(self, shape: tuple[int, int]) -> int:
        return max(1, self._block_bytes // max(int(shape[1]) * np.dtype(np.float32).itemsize, 1))


def prepare_block_params(
    params: dict[str, Any], header: dict[str, Any], shape: tuple[int, int], method_id: str
) -> dict[str, Any]:
    runtime = dict(params)
    total_ns = float(header.get("total_time_ns") or header.get("time_window_ns") or 0.0)
    if total_ns > 0.0:
        step_s = total_ns * 1.0e-9 / max(1, shape[0])
        runtime.setdefault("time_step_s", step_s)
        runtime.setdefault("time_window_ns", total_ns)
        if method_id == "frequency_filter_1d":
            runtime.setdefault("sample_rate_hz", 1.0 / step_s)
    return runtime


def new_matrix_digest(shape: tuple[int, int], dtype: np.dtype) -> hashlib._Hash:
    digest = hashlib.sha256()
    digest.update(str(np.dtype(dtype)).encode("utf-8"))
    digest.update(str(tuple(int(value) for value in shape)).encode("utf-8"))
    return digest


def hash_matrix(
    matrix: Any,
    block_rows: int,
    *,
    context: ExecutionContext | None = None,
) -> str:
    shape = tuple(int(value) for value in matrix.shape)
    dtype = np.dtype(matrix.dtype)
    path = Path(str(matrix.filename)) if isinstance(matrix, np.memmap) else None
    digest = new_matrix_digest(shape, dtype)
    spans = list(range(0, shape[0], max(1, block_rows)))
    for index, start in enumerate(spans, start=1):
        if context is not None:
            context.raise_if_cancelled()
        end = min(start + max(1, block_rows), shape[0])
        if path is None:
            block = np.ascontiguousarray(matrix[start:end])
        else:
            view = np.memmap(path, mode="r", dtype=dtype, shape=shape)
            try:
                block = np.array(view[start:end], copy=True, order="C")
            finally:
                close_memmap(view)
        digest.update(block.tobytes())
        if context is not None:
            context.report_progress(index, len(spans), f"计算处理结果哈希 {index}/{len(spans)}")
    return digest.hexdigest()


def create_memmap_file(path: Path, shape: tuple[int, int]) -> None:
    matrix = np.memmap(path, mode="w+", dtype=np.float32, shape=shape)
    try:
        matrix.flush()
    finally:
        close_memmap(matrix)


def close_memmap(matrix: Any) -> None:
    mapping = getattr(matrix, "_mmap", None)
    if mapping is None:
        return
    try:
        mapping.close()
    except (BufferError, OSError, ValueError):
        return


def merge_metadata(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in incoming.items():
        if key != "runtime_warnings" and key not in merged:
            merged[key] = value
    warnings = [dict(item) for item in merged.get("runtime_warnings", []) if isinstance(item, dict)]
    seen = {(item.get("code"), item.get("message")) for item in warnings}
    for item in incoming.get("runtime_warnings", []) or []:
        if isinstance(item, dict) and (item.get("code"), item.get("message")) not in seen:
            warnings.append(dict(item))
            seen.add((item.get("code"), item.get("message")))
    if warnings:
        merged["runtime_warnings"] = warnings
    return merged


def json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            safe[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, np.generic):
            safe[key] = value.item()
        elif isinstance(value, dict):
            safe[key] = json_safe_metadata(value)
        elif isinstance(value, (list, tuple)):
            safe[key] = [json_safe_metadata(item) if isinstance(item, dict) else item for item in value]
        else:
            safe[key] = value
    return safe


__all__ = ["FileBackedBlockPipelineExecutor", "InsufficientProcessingStorage"]
