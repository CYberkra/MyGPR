from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.hdf5_array_proxy import HDF5ArrayProxy
from mygpr.application.jobs.models import JobStatus
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend

pytestmark = [
    pytest.mark.industrial,
    pytest.mark.performance,
    pytest.mark.large_data,
    pytest.mark.slow,
    pytest.mark.requirement("REQ-PERF-001"),
    pytest.mark.risk("RISK-LARGE-DATA-OOM"),
    pytest.mark.level("system"),
]


def test_file_backed_project_pipeline_never_materializes_hdf5_proxy(tmp_path: Path, monkeypatch) -> None:
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "block-project", name="block")
        rows, cols = 1024, 2048
        x = np.linspace(0, 1, cols, dtype=np.float32)[None, :]
        y = np.linspace(0, 1, rows, dtype=np.float32)[:, None]
        matrix = (0.02 * y + np.sin(12 * x) * np.exp(-((y - 0.55) ** 2) / 0.02)).astype(np.float32)
        backend.projects.save_line_dataset(project.project_id, "L01", matrix, length_m=200.0, time_window_ns=700.0)

        def forbid(self, dtype=None, copy=None):
            raise AssertionError("industrial block pipeline attempted full HDF5 materialization")

        monkeypatch.setattr(HDF5ArrayProxy, "__array__", forbid)
        pipeline = PipelineDefinition(
            name="file-backed acceptance",
            steps=(PipelineStep("dewow", {"window": 23}), PipelineStep("sec_gain", {"gain_min": 1.0, "gain_max": 4.0, "power": 1.0})),
        )
        estimate = backend.project_processing.estimate_pipeline(project.project_id, "L01", pipeline)
        assert estimate.supports_chunking
        assert estimate.memory_bytes < 64 * 1024 * 1024
        assert estimate.temporary_disk_bytes < 64 * 1024 * 1024
        job_id = backend.submit_project_pipeline(
            project.project_id, "L01", pipeline, save_intermediates=False)
        snapshot = backend.jobs.wait(job_id, timeout=120)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        artifact = snapshot.result
        assert artifact.shape == matrix.shape
        assert artifact.sha256
        # Successful completion while HDF5ArrayProxy.__array__ is forbidden proves
        # the persisted source stayed on the bounded block path.
        assert backend.processing.supports_block_pipeline(pipeline)
    finally:
        backend.shutdown()
