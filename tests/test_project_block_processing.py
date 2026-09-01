from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


def test_project_native_pipeline_does_not_materialize_full_dataset(tmp_path: Path) -> None:
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        summary = backend.projects.create_project(tmp_path / "project", name="Block Project")
        data = np.random.default_rng(7).normal(size=(220, 170)).astype(np.float32)
        backend.projects.save_line_dataset(
            summary.project_id,
            "L01",
            data,
            time_window_ns=520.0,
        )

        def fail_read_dataset(*args, **kwargs):
            raise AssertionError("full dataset materialization must not be used")

        backend.projects.read_dataset = fail_read_dataset  # type: ignore[method-assign]
        pipeline = PipelineDefinition(
            name="Project block pipeline",
            steps=(
                PipelineStep("dewow", {"window": 19}),
                PipelineStep("subtracting_average_2D", {"ntraces": 31}),
                PipelineStep("agcGain", {"window": 13}),
            ),
        )
        artifact = backend.project_processing.execute_pipeline(
            summary.project_id,
            "L01",
            pipeline,
            save_intermediates=False,
        )
        params_path = Path(summary.root_path) / str(artifact.manifest["params_path"])
        payload = json.loads(params_path.read_text(encoding="utf-8"))
        assert payload["params"]["execution_mode"] == "file_backed_blocks"
        assert len(payload["params"]["lineage"]) == 3
        assert payload["input_dataset"]["input_data_sha256"]
        assert payload["input_dataset"]["output_data_sha256"] == artifact.sha256
        assert artifact.shape == data.shape
    finally:
        backend.shutdown()


def test_project_pipeline_resource_estimate_reports_chunking(tmp_path: Path) -> None:
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        summary = backend.projects.create_project(tmp_path / "estimate-project", name="Estimate Project")
        backend.projects.save_line_dataset(
            summary.project_id,
            "L01",
            np.zeros((160, 120), dtype=np.float32),
        )
        pipeline = PipelineDefinition(
            steps=(PipelineStep("dewow", {"window": 15}), PipelineStep("agcGain", {"window": 9}))
        )
        estimate = backend.project_processing.estimate_pipeline(summary.project_id, "L01", pipeline)
        assert estimate.supports_chunking
        assert estimate.memory_bytes > 0
        assert estimate.temporary_disk_bytes > 0
    finally:
        backend.shutdown()


def test_global_pipeline_memory_guard_runs_before_full_dataset_read(tmp_path: Path) -> None:
    from mygpr.application.jobs.context import ExecutionContext
    from mygpr.infrastructure.system.resource_policy import InsufficientProcessingMemory

    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        summary = backend.projects.create_project(tmp_path / "guard-project", name="Guard Project")
        backend.projects.save_line_dataset(
            summary.project_id,
            "L01",
            np.zeros((80, 64), dtype=np.float32),
        )

        def fail_read_dataset(*args, **kwargs):
            raise AssertionError("resource guard must execute before full dataset read")

        backend.projects.read_dataset = fail_read_dataset  # type: ignore[method-assign]
        context = ExecutionContext(metadata={"max_memory_bytes": 1})
        pipeline = PipelineDefinition(steps=(PipelineStep("svd_bg", {"rank": 1}),))
        with pytest.raises(InsufficientProcessingMemory):
            backend.project_processing.execute_pipeline(
                summary.project_id,
                "L01",
                pipeline,
                context=context,
            )
    finally:
        backend.shutdown()


def test_project_global_native_pipeline_avoids_full_hdf5_materialization(tmp_path: Path) -> None:
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        summary = backend.projects.create_project(tmp_path / "global-project", name="Global Project")
        data = np.random.default_rng(81).normal(size=(120, 90)).astype(np.float32)
        backend.projects.save_line_dataset(summary.project_id, "L01", data)

        def fail_read_dataset(*args, **kwargs):
            raise AssertionError("native global pipeline must use project block staging")

        backend.projects.read_dataset = fail_read_dataset  # type: ignore[method-assign]
        pipeline = PipelineDefinition(
            name="Native global project pipeline",
            steps=(PipelineStep("svd_bg", {"rank": 1, "solver": "exact"}),),
        )
        artifact = backend.project_processing.execute_pipeline(
            summary.project_id,
            "L01",
            pipeline,
            save_intermediates=False,
        )
        params_path = Path(summary.root_path) / str(artifact.manifest["params_path"])
        payload = json.loads(params_path.read_text(encoding="utf-8"))
        assert payload["params"]["execution_mode"] == "file_backed_blocks"
        assert payload["params"]["lineage"][0]["implementation_version"] == "native-global-1.0"
        assert artifact.shape == data.shape
    finally:
        backend.shutdown()
