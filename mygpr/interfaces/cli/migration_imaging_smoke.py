"""Headless smoke workflow for native Kirchhoff and experimental RTM imaging."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


def _bscan(samples: int = 56, traces: int = 20) -> np.ndarray:
    rng = np.random.default_rng(2088)
    t = np.arange(samples, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, traces, dtype=np.float32)[None, :]
    event = np.exp(-0.5 * ((t - (16.0 + 5.0 * x**2)) / 1.4) ** 2)
    return np.asarray(event + rng.normal(0.0, 0.01, (samples, traces)), dtype=np.float32)


def run() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="mygpr_migration_smoke_") as temporary:
        backend = MyGPRBackend.create_default(max_workers=1)
        try:
            project = backend.projects.create_project(Path(temporary) / "project", name="Migration smoke")
            matrix = _bscan()
            backend.projects.save_line_dataset(
                project.project_id,
                "L01",
                matrix,
                length_m=1.9,
                time_window_ns=14.0,
            )
            kirchhoff = PipelineDefinition(
                name="Native Kirchhoff smoke",
                steps=(
                    PipelineStep(
                        "kirchhoff_migration",
                        {
                            "freq": 5.0e7,
                            "depth": 0.5,
                            "v": 0.10,
                            "weight": 0.0,
                            "length_m": 1.9,
                            "time_window_ns": 14.0,
                            "backend": "cpu",
                        },
                    ),
                ),
            )
            rtm = PipelineDefinition(
                name="Experimental RTM smoke",
                steps=(
                    PipelineStep(
                        "rtm_migration",
                        {
                            "v": 0.10,
                            "depth_m": 0.8,
                            "dx_m": 0.1,
                            "dz_m": 0.1,
                            "boundary_width": 2,
                            "max_cell_updates": 20_000_000,
                        },
                    ),
                ),
            )
            kir_artifact = backend.project_processing.execute_pipeline(
                project.project_id, "L01", kirchhoff, result_name="Kirchhoff smoke"
            )
            rtm_artifact = backend.project_processing.execute_pipeline(
                project.project_id, "L01", rtm, result_name="RTM smoke"
            )
            return {
                "backend_api": backend.api_version,
                "kirchhoff": {
                    "artifact_id": kir_artifact.artifact_id,
                    "shape": list(kir_artifact.shape),
                    "sha256": kir_artifact.sha256,
                    "implementation": backend.processing.get_method("kirchhoff_migration").implementation_version,
                },
                "rtm": {
                    "artifact_id": rtm_artifact.artifact_id,
                    "shape": list(rtm_artifact.shape),
                    "sha256": rtm_artifact.sha256,
                    "implementation": backend.processing.get_method("rtm_migration").implementation_version,
                    "contract": "zero_offset_exploding_reflector_scalar_2d",
                },
                "qt_loaded": any(name.startswith("PyQt") for name in sys.modules),
            }
        finally:
            backend.shutdown()


def main() -> int:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    hashes = result["kirchhoff"]["sha256"] and result["rtm"]["sha256"]
    return 0 if hashes and not result["qt_loaded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
