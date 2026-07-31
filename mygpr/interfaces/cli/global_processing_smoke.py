"""Headless project smoke workflow for native global transforms."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


def _data(samples: int = 128, traces: int = 96) -> np.ndarray:
    rng = np.random.default_rng(73)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, traces, dtype=np.float32)[None, :]
    background = 0.35 * np.sin(5.0 * np.pi * t)
    reflector = 0.25 * np.exp(-((t - (0.48 + 0.09 * x**2)) ** 2) / 0.0015)
    return np.asarray(background + reflector + rng.normal(0.0, 0.025, (samples, traces)), dtype=np.float32)


def run() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="mygpr_global_smoke_") as temporary:
        backend = MyGPRBackend.create_default(max_workers=1)
        try:
            project = backend.projects.create_project(Path(temporary) / "project", name="Global smoke")
            matrix = _data()
            backend.projects.save_line_dataset(project.project_id, "L01", matrix, time_window_ns=600.0)
            pipeline = PipelineDefinition(
                name="Native global smoke",
                steps=(
                    PipelineStep("svd_bg", {"rank": 1, "solver": "exact"}),
                    PipelineStep("fk_filter", {"angle_low": 12, "angle_high": 58, "taper_width": 4}),
                ),
            )
            estimate = backend.project_processing.estimate_pipeline(project.project_id, "L01", pipeline)
            artifact = backend.project_processing.execute_pipeline(project.project_id, "L01", pipeline)
            return {
                "backend_api": backend.api_version,
                "artifact_id": artifact.artifact_id,
                "shape": list(artifact.shape),
                "sha256": artifact.sha256,
                "resource_estimate": {
                    "memory_bytes": estimate.memory_bytes,
                    "temporary_disk_bytes": estimate.temporary_disk_bytes,
                    "relative_cost": estimate.relative_cost,
                    "file_backed": estimate.supports_chunking,
                },
                "qt_loaded": any(name.startswith("PyQt") for name in sys.modules),
            }
        finally:
            backend.shutdown()


def main() -> int:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["sha256"] and not result["qt_loaded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
