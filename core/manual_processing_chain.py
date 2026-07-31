#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual step-by-step processing chain for the field workbench.

The field workbench explicitly avoids heavy preset pipelines in the default UI,
while users still need a practical way to apply several processing methods in a
controlled order.  This module keeps the transient session state outside the UI
layer so the processing page can offer a simple workflow:

original -> execute current step -> optionally execute another step -> undo/reset.

Each executed step becomes part of the current engineering chain immediately, so
the UI can expose it in the project tree without requiring a separate save click.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import copy
import shutil
import tempfile

import numpy as np
from typing import Any

from core.field_processing_bridge import display_name, field_category, run_registered_method
from core.gpr_data_model import GPRDataSet
from core.trajectory_model import TrajectoryModel


@dataclass
class ManualProcessingStep:
    """One executed manual step in the transient processing chain."""

    index: int
    method_id: str
    method_name: str
    category: str
    params: dict[str, Any] = field(default_factory=dict)
    input_shape: tuple[int, int] = (0, 0)
    output_shape: tuple[int, int] = (0, 0)
    elapsed_s: float = 0.0
    warnings: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "method_id": self.method_id,
            "method_name": self.method_name,
            "category": self.category,
            "params": copy.deepcopy(self.params),
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "elapsed_s": float(self.elapsed_s),
            "warnings": copy.deepcopy(self.warnings),
            "manifest": copy.deepcopy(self.manifest),
            "enabled": bool(self.enabled),
        }

    @property
    def status_text(self) -> str:
        if not self.enabled:
            return "已停用"
        if self.input_shape != self.output_shape:
            return "需复核"
        if self.warnings:
            return "有警告"
        return "完成"

    @property
    def time_text(self) -> str:
        return str(self.manifest.get("created_at", "--")).split(" ")[-1]

    def to_history_row(self) -> tuple[str, str, str, str, str]:
        params_text = ", ".join(f"{k}={v}" for k, v in self.params.items())
        summary = self.method_name if not params_text else f"{self.method_name}｜{params_text}"
        return (str(self.index), summary, self.status_text, self.time_text, "…")


class ManualProcessingSession:
    """Transient chain session for one selected line."""

    def __init__(
        self,
        original_dataset: GPRDataSet,
        *,
        spill_threshold_bytes: int = 64 * 1024 * 1024,
        resident_budget_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self._original_dataset = original_dataset
        self._current_dataset = original_dataset
        self._steps: list[ManualProcessingStep] = []
        self._datasets: list[GPRDataSet] = [original_dataset]
        self._current_step_index = 0
        self.spill_threshold_bytes = max(int(spill_threshold_bytes), 1)
        self.resident_budget_bytes = max(int(resident_budget_bytes), self.spill_threshold_bytes)
        self._spill_root = Path(tempfile.mkdtemp(prefix=f"mygpr_chain_{original_dataset.line_id}_"))
        self._spill_paths: dict[int, Path] = {}

    @property
    def line_id(self) -> str:
        return self._original_dataset.line_id

    @property
    def step_count(self) -> int:
        return len(self._steps)

    @property
    def dirty(self) -> bool:
        return bool(self._steps)

    @property
    def last_manifest(self) -> dict[str, Any] | None:
        return self._steps[-1].manifest if self._steps else None

    @property
    def steps(self) -> list[ManualProcessingStep]:
        return list(self._steps)

    @property
    def datasets(self) -> list[GPRDataSet]:
        return list(self._datasets)

    @property
    def current_step_index(self) -> int:
        return self._current_step_index

    @property
    def current_dataset(self) -> GPRDataSet | None:
        return self._current_dataset

    @property
    def original_dataset(self) -> GPRDataSet:
        return self._original_dataset

    def to_draft_dict(self) -> dict[str, Any]:
        """Return a JSON-safe editable-chain recovery document."""
        return {
            "schema": "mygpr.processing_draft.v1",
            "line_id": self.line_id,
            "source_path": str(self._original_dataset.source_path or ""),
            "source_shape": list(self._original_dataset.matrix.shape),
            "current_step_index": int(self._current_step_index),
            "steps": [step.to_dict() for step in self._steps],
        }

    @classmethod
    def from_draft(
        cls,
        original_dataset: GPRDataSet,
        payload: dict[str, Any],
        *,
        trajectory: TrajectoryModel | None = None,
        cancel_checker=None,
        progress_callback=None,
    ) -> "ManualProcessingSession":
        """Rebuild a session from its raw source and persisted chain.

        No formal result is mutated.  Replaying from raw also validates that the
        current processing kernels can still reproduce the draft.
        """
        if str(payload.get("line_id") or original_dataset.line_id) != original_dataset.line_id:
            raise ValueError("处理草稿与当前测线不一致")
        expected_shape = tuple(int(v) for v in payload.get("source_shape") or ())
        if expected_shape and expected_shape != tuple(original_dataset.matrix.shape):
            raise ValueError(
                f"原始数据尺寸已变化：草稿 {expected_shape}，当前 {tuple(original_dataset.matrix.shape)}"
            )
        session = cls(original_dataset)
        specs = []
        for item in payload.get("steps") or []:
            if not isinstance(item, dict) or not item.get("method_id"):
                continue
            specs.append(
                {
                    "method_id": str(item["method_id"]),
                    "params": dict(item.get("params") or {}),
                    "enabled": bool(item.get("enabled", True)),
                }
            )
        session._replace_chain_from_specs(
            specs,
            trajectory=trajectory,
            cancel_checker=cancel_checker,
            progress_callback=progress_callback,
        )
        selected = int(payload.get("current_step_index") or len(session._steps))
        session.select_step(max(0, min(selected, len(session._steps))))
        return session

    @staticmethod
    def _is_memmap_dataset(dataset: GPRDataSet) -> bool:
        return isinstance(dataset.matrix, np.memmap)

    def _dataset_from_memmap(self, dataset: GPRDataSet, path: Path) -> GPRDataSet:
        matrix = np.load(path, mmap_mode="r+", allow_pickle=False)
        metadata = dict(dataset.metadata or {})
        metadata["manual_chain_spill_path"] = str(path)
        metadata["manual_chain_storage"] = "memmap"
        return GPRDataSet(
            line_id=dataset.line_id, matrix=matrix,
            distance_axis_m=dataset.distance_axis_m, time_axis_ns=dataset.time_axis_ns,
            depth_axis_m=dataset.depth_axis_m, source_path=dataset.source_path,
            time_window_ns=dataset.time_window_ns, dielectric_constant=dataset.dielectric_constant,
            format_name=dataset.format_name, metadata=metadata,
        )

    def _spill_dataset(self, index: int) -> None:
        if index <= 0 or index >= len(self._datasets):
            return
        dataset = self._datasets[index]
        if self._is_memmap_dataset(dataset):
            return
        path = self._spill_root / f"step_{index:03d}.npy"
        output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=dataset.matrix.shape)
        rows = int(dataset.matrix.shape[0])
        for start in range(0, rows, 256):
            end = min(rows, start + 256)
            output[start:end] = np.asarray(dataset.matrix[start:end], dtype=np.float32)
        output.flush()
        del output
        self._datasets[index] = self._dataset_from_memmap(dataset, path)
        self._spill_paths[index] = path
        if self._current_step_index == index:
            self._current_dataset = self._datasets[index]

    def _enforce_memory_budget(self) -> None:
        # Spill the newest output immediately when one step alone exceeds the
        # threshold, then spill older resident intermediates until the budget is
        # respected.  The original dataset is owned by the line store and is not
        # duplicated here.
        last = len(self._datasets) - 1
        if last > 0 and self._datasets[last].matrix.nbytes >= self.spill_threshold_bytes:
            self._spill_dataset(last)
        def resident_bytes() -> int:
            return sum(
                int(ds.matrix.nbytes) for ds in self._datasets[1:]
                if not self._is_memmap_dataset(ds)
            )
        for index in range(1, max(last, 1)):
            if resident_bytes() <= self.resident_budget_bytes:
                break
            self._spill_dataset(index)

    def _remove_spills_after(self, step_index: int) -> None:
        for index, path in list(self._spill_paths.items()):
            if index > int(step_index):
                try:
                    path.unlink(missing_ok=True)
                finally:
                    self._spill_paths.pop(index, None)

    def close(self) -> None:
        self._datasets = [self._original_dataset]
        self._steps.clear()
        self._current_step_index = 0
        self._current_dataset = self._original_dataset
        self._spill_paths.clear()
        shutil.rmtree(self._spill_root, ignore_errors=True)

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown
        try:
            shutil.rmtree(getattr(self, "_spill_root", Path()), ignore_errors=True)
        except OSError:
            pass

    def append_step(
        self,
        method_id: str,
        params: dict[str, Any] | None = None,
        *,
        trajectory: TrajectoryModel | None = None,
        cancel_checker=None,
        progress_callback=None,
    ) -> tuple[GPRDataSet, dict[str, Any]]:
        if self._current_step_index < len(self._steps):
            # The user selected an earlier project-tree step and then executed a
            # new operation.  Keep a simple linear engineering chain by dropping
            # downstream steps, matching the undo/reset mental model.
            self._steps = self._steps[: self._current_step_index]
            self._datasets = self._datasets[: self._current_step_index + 1]
            self._remove_spills_after(self._current_step_index)
        base = self._current_dataset
        output, manifest = run_registered_method(
            base,
            method_id,
            params or {},
            trajectory=trajectory,
            cancel_checker=cancel_checker,
            progress_callback=progress_callback,
        )
        step = ManualProcessingStep(
            index=len(self._steps) + 1,
            method_id=method_id,
            method_name=str(manifest.get("method_name") or display_name(method_id)),
            category=str(manifest.get("category") or field_category(method_id)),
            params=dict(params or {}),
            input_shape=tuple(int(v) for v in manifest.get("input_shape", list(base.matrix.shape))),
            output_shape=tuple(int(v) for v in manifest.get("output_shape", list(output.matrix.shape))),
            elapsed_s=float(manifest.get("elapsed_s", 0.0)),
            warnings=list(manifest.get("warnings") or []),
            manifest=dict(manifest),
        )
        self._steps.append(step)
        self._datasets.append(output)
        self._current_step_index = len(self._steps)
        self._current_dataset = output
        self._enforce_memory_budget()
        return self._current_dataset, manifest

    def recompute_from_step(
        self,
        step_index: int,
        *,
        method_id: str | None = None,
        params: dict[str, Any] | None = None,
        trajectory: TrajectoryModel | None = None,
        cancel_checker=None,
        progress_callback=None,
    ) -> tuple[GPRDataSet, dict[str, Any]]:
        """Replace one step and atomically rebuild the complete downstream chain.
        The UI uses this operation when an operator edits parameters on an
        existing step.  The currently valid chain remains untouched until every
        recomputed step succeeds.  This preserves the last complete B-scan when
        a later algorithm fails or the task is cancelled.
        """
        index = int(step_index)
        if index < 1 or index > len(self._steps):
            raise IndexError(f"processing step out of range: {step_index}")
        specs: list[dict[str, Any]] = [
            {"method_id": step.method_id, "params": dict(step.params), "enabled": step.enabled}
            for step in self._steps
        ]
        old_method = str(specs[index - 1]["method_id"])
        old_params = dict(specs[index - 1]["params"])
        specs[index - 1]["method_id"] = str(method_id or old_method)
        specs[index - 1]["params"] = dict(old_params if params is None else params)
        trial = ManualProcessingSession(
            self._original_dataset,
            spill_threshold_bytes=self.spill_threshold_bytes,
            resident_budget_bytes=self.resident_budget_bytes,
        )
        try:
            total = len(specs)
            last_manifest: dict[str, Any] = {}
            for offset, item in enumerate(specs, start=1):
                candidate_method = str(item["method_id"])
                candidate_params = dict(item.get("params") or {})
                candidate_enabled = bool(item.get("enabled", True))
                if cancel_checker is not None and cancel_checker():
                    from core.job_manager import JobCancelled
                    raise JobCancelled("处理链重算已取消")
                def relay(done, stage_total, message, *, current=offset):
                    if progress_callback is None:
                        return
                    # Report step-level progress while retaining the kernel
                    # message for the task detail panel.
                    progress_callback(
                        current - 1 + (float(done) / max(float(stage_total), 1.0)),
                        total,
                        f"重算 Step {current:02d}/{total:02d} · {message}",
                    )
                if candidate_enabled:
                    _dataset, last_manifest = trial.append_step(
                        candidate_method,
                        candidate_params,
                        trajectory=trajectory,
                        cancel_checker=cancel_checker,
                        progress_callback=relay,
                    )
                else:
                    base = trial._current_dataset
                    disabled_step = ManualProcessingStep(
                        index=len(trial._steps) + 1,
                        method_id=candidate_method,
                        method_name=display_name(candidate_method),
                        category=field_category(candidate_method),
                        params=candidate_params,
                        input_shape=tuple(base.matrix.shape),
                        output_shape=tuple(base.matrix.shape),
                        manifest={"method_id": candidate_method, "disabled": True},
                        enabled=False,
                    )
                    trial._steps.append(disabled_step)
                    trial._datasets.append(base)
                    trial._current_step_index = len(trial._steps)
                    trial._current_dataset = base
                    last_manifest = dict(disabled_step.manifest)
            if progress_callback is not None:
                progress_callback(total, total, "处理链重算完成")
            old_root = self._spill_root
            self._steps = trial._steps
            self._datasets = trial._datasets
            self._current_step_index = trial._current_step_index
            self._current_dataset = trial._current_dataset
            self._spill_root = trial._spill_root
            self._spill_paths = trial._spill_paths
            # Detach the transferred state so ``trial.__del__`` cannot remove
            # files now owned by this session.
            trial._steps = []
            trial._datasets = [trial._original_dataset]
            trial._current_step_index = 0
            trial._current_dataset = trial._original_dataset
            trial._spill_paths = {}
            trial._spill_root = Path(tempfile.gettempdir()) / "__mygpr_detached_processing_session__"
            shutil.rmtree(old_root, ignore_errors=True)
            return self._current_dataset, last_manifest
        except Exception:
            trial.close()
            raise

    def remove_step(
        self,
        step_index: int,
        *,
        trajectory: TrajectoryModel | None = None,
        cancel_checker=None,
        progress_callback=None,
    ) -> bool:
        index = int(step_index)
        if index < 1 or index > len(self._steps):
            return False
        specs = [
            {"method_id": step.method_id, "params": dict(step.params), "enabled": step.enabled}
            for step in self._steps
        ]
        specs.pop(index - 1)
        self._replace_chain_from_specs(
            specs, trajectory=trajectory, cancel_checker=cancel_checker, progress_callback=progress_callback
        )
        return True

    def move_step(
        self,
        source_index: int,
        target_index: int,
        *,
        trajectory: TrajectoryModel | None = None,
        cancel_checker=None,
        progress_callback=None,
    ) -> bool:
        source = int(source_index)
        target = int(target_index)
        if source < 1 or source > len(self._steps) or target < 1 or target > len(self._steps):
            return False
        if source == target:
            return True
        specs = [
            {"method_id": step.method_id, "params": dict(step.params), "enabled": step.enabled}
            for step in self._steps
        ]
        item = specs.pop(source - 1)
        specs.insert(target - 1, item)
        self._replace_chain_from_specs(
            specs, trajectory=trajectory, cancel_checker=cancel_checker, progress_callback=progress_callback
        )
        return True

    def set_step_enabled(
        self,
        step_index: int,
        enabled: bool,
        *,
        trajectory: TrajectoryModel | None = None,
        cancel_checker=None,
        progress_callback=None,
    ) -> bool:
        index = int(step_index)
        if index < 1 or index > len(self._steps):
            return False
        desired = bool(enabled)
        if self._steps[index - 1].enabled == desired:
            return True
        specs = [
            {"method_id": step.method_id, "params": dict(step.params), "enabled": step.enabled}
            for step in self._steps
        ]
        specs[index - 1]["enabled"] = desired
        self._replace_chain_from_specs(
            specs, trajectory=trajectory, cancel_checker=cancel_checker, progress_callback=progress_callback
        )
        return True

    def _replace_chain_from_specs(
        self,
        specs: list[dict[str, Any]],
        *,
        trajectory: TrajectoryModel | None = None,
        cancel_checker=None,
        progress_callback=None,
    ) -> None:
        """Atomically rebuild a chain, preserving disabled steps as no-op nodes."""
        trial = ManualProcessingSession(
            self._original_dataset,
            spill_threshold_bytes=self.spill_threshold_bytes,
            resident_budget_bytes=self.resident_budget_bytes,
        )
        try:
            total = max(len(specs), 1)
            for offset, item in enumerate(specs, start=1):
                if cancel_checker is not None and cancel_checker():
                    from core.job_manager import JobCancelled
                    raise JobCancelled("处理链重算已取消")
                method_id = str(item["method_id"])
                params = dict(item.get("params") or {})
                enabled = bool(item.get("enabled", True))
                if enabled:
                    def relay(done, stage_total, message, *, current=offset):
                        if progress_callback is not None:
                            progress_callback(
                                current - 1 + float(done) / max(float(stage_total), 1.0),
                                total,
                                f"重算 Step {current:02d}/{len(specs):02d} · {message}",
                            )
                    trial.append_step(
                        method_id, params, trajectory=trajectory,
                        cancel_checker=cancel_checker, progress_callback=relay,
                    )
                else:
                    base = trial._current_dataset
                    step = ManualProcessingStep(
                        index=len(trial._steps) + 1,
                        method_id=method_id,
                        method_name=display_name(method_id),
                        category=field_category(method_id),
                        params=params,
                        input_shape=tuple(base.matrix.shape),
                        output_shape=tuple(base.matrix.shape),
                        manifest={"method_id": method_id, "disabled": True},
                        enabled=False,
                    )
                    trial._steps.append(step)
                    trial._datasets.append(base)
                    trial._current_step_index = len(trial._steps)
                    trial._current_dataset = base
            if progress_callback is not None:
                progress_callback(len(specs), total, "处理链重算完成")
            old_root = self._spill_root
            self._steps = trial._steps
            self._datasets = trial._datasets
            self._current_step_index = trial._current_step_index
            self._current_dataset = trial._current_dataset
            self._spill_root = trial._spill_root
            self._spill_paths = trial._spill_paths
            trial._steps = []
            trial._datasets = [trial._original_dataset]
            trial._current_step_index = 0
            trial._current_dataset = trial._original_dataset
            trial._spill_paths = {}
            trial._spill_root = Path(tempfile.gettempdir()) / "__mygpr_detached_processing_session__"
            shutil.rmtree(old_root, ignore_errors=True)
        except Exception:
            trial.close()
            raise

    def undo_last_step(self) -> bool:
        if not self._steps:
            return False
        removed_index = len(self._steps)
        self._steps.pop()
        self._datasets.pop()
        self._remove_spills_after(removed_index - 1)
        self._current_step_index = len(self._steps)
        self._current_dataset = self._datasets[self._current_step_index]
        return True

    def reset_to_original(self) -> bool:
        if not self._steps:
            return False
        self._steps.clear()
        self._remove_spills_after(0)
        self._datasets = [self._original_dataset]
        self._current_step_index = 0
        self._current_dataset = self._original_dataset
        return True

    def truncate_after_step(self, step_index: int) -> bool:
        """Keep the selected step and delete all downstream steps."""
        index = int(step_index)
        if index < 0 or index > len(self._steps):
            return False
        if len(self._steps) == index:
            self._current_step_index = index
            self._current_dataset = self._datasets[index]
            return False
        self._steps = self._steps[:index]
        self._datasets = self._datasets[: index + 1]
        self._remove_spills_after(index)
        self._current_step_index = index
        self._current_dataset = self._datasets[index]
        return True

    def select_step(self, step_index: int) -> bool:
        index = int(step_index)
        if index < 0 or index > len(self._steps):
            return False
        self._current_step_index = index
        self._current_dataset = self._datasets[index]
        return True

    def current_step_label(self) -> str:
        if self._current_step_index <= 0:
            return "Step 00 原始 B-scan"
        step = self._steps[self._current_step_index - 1]
        return f"Step {step.index:02d} {step.method_name}"

    def build_save_payload(self, final_method_id: str, final_params: dict[str, Any]) -> dict[str, Any]:
        final_manifest = dict(self.last_manifest or {})
        final_manifest.update(
            {
                "processing_mode": "manual_step_chain",
                "chain_step_count": len(self._steps),
                "chain_steps": [
                    {
                        "index": step.index,
                        "method_id": step.method_id,
                        "method_name": step.method_name,
                        "category": step.category,
                        "params": dict(step.params),
                        "input_shape": list(step.input_shape),
                        "output_shape": list(step.output_shape),
                        "elapsed_s": round(float(step.elapsed_s), 4),
                        "warning_count": len(step.warnings),
                        "created_at": step.manifest.get("created_at", ""),
                        "enabled": bool(step.enabled),
                    }
                    for step in self._steps
                ],
            }
        )
        return {
            "method": final_method_id,
            "method_name": str(final_manifest.get("method_name") or display_name(final_method_id)),
            "params": dict(final_params),
            "manifest": final_manifest,
            "input_dataset": self._original_dataset.to_metadata(),
        }

    def summary_text(self) -> str:
        if not self._steps:
            return "当前状态：原始 B-scan"
        return f"当前状态：已生成 {len(self._steps)} 个处理步骤，当前 Step {self._current_step_index:02d}"


__all__ = ["ManualProcessingSession", "ManualProcessingStep"]
