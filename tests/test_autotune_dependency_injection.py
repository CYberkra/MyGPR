#!/usr/bin/env python3
"""AutoTune can run with injected application ports and no legacy engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from mygpr.application.autotune.ports import AutoTuneDependencies
from mygpr.application.autotune.service import auto_tune_method_with_dependencies
from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.autotune.constraints import ParameterConstraintResult
from mygpr.domain.processing.models import (
    ProcessingMethodDescriptor,
    ProcessingRequest,
    ProcessingResult,
    ResourceEstimate,
)


class FakeCatalog:
    def get(self, method_id: str) -> ProcessingMethodDescriptor | None:
        if method_id != "fake_dewow":
            return None
        return ProcessingMethodDescriptor(
            method_id=method_id,
            name="Fake Dewow",
            category="preprocessing",
            auto_tune_enabled=True,
            auto_tune_family="drift",
            auto_tune_stage="drift",
        )

    def list(self, *, public_only: bool = False) -> Sequence[ProcessingMethodDescriptor]:
        del public_only
        descriptor = self.get("fake_dewow")
        return (descriptor,) if descriptor is not None else ()

    def auto_tune_stage(self, method_id: str) -> str:
        return "drift" if method_id == "fake_dewow" else ""

    def raw_metadata(self, method_id: str) -> dict[str, Any]:
        if method_id != "fake_dewow":
            return {}
        return {
            "name": "Fake Dewow",
            "auto_tune_enabled": True,
            "auto_tune_family": "drift",
            "auto_tune_candidates": {"window": [3, 5, 7]},
            "params": [{"name": "window", "type": "int"}],
        }


@dataclass
class FakeExecutor:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def execute(
        self,
        request: ProcessingRequest,
        context: ExecutionContext | None = None,
    ) -> ProcessingResult:
        if context is not None:
            context.raise_if_cancelled()
        self.calls.append(dict(request.params))
        window = max(1, int(request.params.get("window", 3)))
        # A deterministic, cheap baseline correction whose strength varies by
        # candidate, sufficient to exercise the full AutoTune orchestration.
        kernel = np.ones(window, dtype=float) / window
        output = np.empty_like(request.data, dtype=float)
        for trace in range(request.data.shape[1]):
            baseline = np.convolve(request.data[:, trace], kernel, mode="same")
            output[:, trace] = request.data[:, trace] - baseline
        return ProcessingResult(
            data=output,
            method_id=request.method_id,
            params=request.params,
            header_info=request.header_info,
            trace_metadata=request.trace_metadata,
        )

    def estimate(self, request: ProcessingRequest) -> ResourceEstimate:
        return ResourceEstimate(memory_bytes=request.data.nbytes * 2, relative_cost="low")


class FakeConstraints:
    def constrain(self, method_id, params, data_shape, header_info=None):
        del method_id, data_shape, header_info
        return ParameterConstraintResult(
            requested_params=dict(params),
            effective_params=dict(params),
            warnings=[],
        )


def test_autotune_uses_injected_catalog_executor_and_constraints() -> None:
    samples, traces = 64, 20
    t = np.linspace(0.0, 1.0, samples)[:, None]
    x = np.linspace(0.0, 1.0, traces)[None, :]
    raw = (0.3 * t + np.sin(7.0 * t) + 0.03 * np.cos(3.0 * x)).astype(np.float32)
    executor = FakeExecutor()
    dependencies = AutoTuneDependencies(
        catalog=FakeCatalog(),
        executor=executor,
        constraints=FakeConstraints(),
    )

    result = auto_tune_method_with_dependencies(
        dependencies,
        raw,
        "fake_dewow",
        candidate_params=[{"window": 3}, {"window": 5}, {"window": 7}],
        search_mode="fast",
    )

    assert result["method_key"] == "fake_dewow"
    assert result["method_name"] == "Fake Dewow"
    assert result["best_params"]["window"] in {3, 5, 7}
    assert result["execution_stats"]["valid_trial_count"] >= 3
    assert len(executor.calls) >= 3
