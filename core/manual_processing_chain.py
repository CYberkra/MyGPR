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

    @property
    def status_text(self) -> str:
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

    def __init__(self, original_dataset: GPRDataSet) -> None:
        self._original_dataset = original_dataset
        self._current_dataset = original_dataset
        self._steps: list[ManualProcessingStep] = []
        self._datasets: list[GPRDataSet] = [original_dataset]
        self._current_step_index = 0

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

    def append_step(
        self,
        method_id: str,
        params: dict[str, Any] | None = None,
        *,
        trajectory: TrajectoryModel | None = None,
    ) -> tuple[GPRDataSet, dict[str, Any]]:
        if self._current_step_index < len(self._steps):
            # The user selected an earlier project-tree step and then executed a
            # new operation.  Keep a simple linear engineering chain by dropping
            # downstream steps, matching the undo/reset mental model.
            self._steps = self._steps[: self._current_step_index]
            self._datasets = self._datasets[: self._current_step_index + 1]
        base = self._current_dataset
        output, manifest = run_registered_method(base, method_id, params or {}, trajectory=trajectory)
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
        return output, manifest

    def undo_last_step(self) -> bool:
        if not self._steps:
            return False
        self._steps.pop()
        self._datasets.pop()
        self._current_step_index = len(self._steps)
        self._current_dataset = self._datasets[self._current_step_index]
        return True

    def reset_to_original(self) -> bool:
        if not self._steps:
            return False
        self._steps.clear()
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
