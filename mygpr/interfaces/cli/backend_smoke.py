#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless end-to-end smoke test for the backend façade."""
from __future__ import annotations

import argparse
import json

import numpy as np

from mygpr.domain.processing.models import (
    PipelineDefinition,
    PipelineStep,
    ProcessingRequest,
)
from mygpr.interfaces.backend import MyGPRBackend


def _synthetic_bscan(samples: int = 96, traces: int = 48) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, traces, dtype=np.float64)[None, :]
    layer = 0.4 * np.exp(-((t - 0.58 - 0.04 * np.sin(2.0 * np.pi * x)) ** 2) / 0.002)
    drift = 0.08 * t + 0.03 * np.cos(4.0 * np.pi * x)
    return (layer + drift).astype(np.float32)


def run_smoke(*, include_autotune: bool = True) -> dict[str, object]:
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        raw = _synthetic_bscan()
        single = backend.processing.execute_method(
            ProcessingRequest(data=raw, method_id="dewow", params={"window": 23})
        )
        pipeline = backend.processing.execute_pipeline(
            raw,
            PipelineDefinition(
                name="Headless smoke pipeline",
                steps=(
                    PipelineStep("dewow", {"window": 23}),
                    PipelineStep("agcGain", {"window": 11}),
                ),
            ),
        )
        payload: dict[str, object] = {
            "backend_api_version": backend.api_version,
            "input_shape": list(raw.shape),
            "single_shape": list(single.data.shape),
            "pipeline_shape": list(pipeline.data.shape),
            "pipeline_steps": [result.method_id for result in pipeline.step_results],
            "finite": bool(np.isfinite(pipeline.data).all()),
        }
        if include_autotune:
            tune = backend.autotune.tune_method(
                raw,
                "dewow",
                candidate_params=[{"window": 11}, {"window": 23}, {"window": 35}],
                search_mode="fast",
            )
            payload["autotune_method"] = tune["method_key"]
            payload["autotune_best_params"] = tune["best_params"]
            payload["autotune_valid_trials"] = tune["execution_stats"]["valid_trial_count"]
        return payload
    finally:
        backend.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-autotune", action="store_true")
    args = parser.parse_args()
    result = run_smoke(include_autotune=not args.skip_autotune)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("finite") else 1


if __name__ == "__main__":
    raise SystemExit(main())
