from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mygpr.domain.processing.models import PipelineDefinition, PipelineStep, ProcessingRequest
from mygpr.infrastructure.processing.algorithms.rtm import RTMResourceLimitError
from mygpr.infrastructure.processing.block_executor import FileBackedBlockPipelineExecutor
from mygpr.infrastructure.processing.native_adapter import NativeProcessingCatalog, NativeProcessingExecutor


def _small_bscan(samples: int = 48, traces: int = 16) -> np.ndarray:
    rng = np.random.default_rng(2088)
    time = np.arange(samples, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, traces, dtype=np.float32)[None, :]
    reflector = np.exp(-0.5 * ((time - (15.0 + 4.0 * x**2)) / 1.5) ** 2)
    return np.asarray(reflector + rng.normal(0.0, 0.01, (samples, traces)), dtype=np.float32)


def test_native_kirchhoff_matches_historical_cpu_kernel_bitwise() -> None:
    params = {
        "freq": 5.0e7,
        "depth": 0.4,
        "v": 0.10,
        "alpha": 1.0,
        "weight": 0.0,
        "num_cal": 1,
        "topo_cor": 0,
        "hei_cor": 0,
        "length_m": 1.5,
        "time_window_ns": 12.0,
        "backend": "cpu",
    }
    request = ProcessingRequest(
        data=_small_bscan(),
        method_id="kirchhoff_migration",
        params=params,
        header_info={"total_time_ns": 12.0, "trace_interval_m": 0.1},
    )
    native = NativeProcessingExecutor().execute(request)
    digest = hashlib.sha256(np.ascontiguousarray(native.data).tobytes()).hexdigest()
    assert digest == "b9a5013a9fa455cefa0333acc5a87725f6a126e0236d1fdeab09ec4bfbf6af88"
    assert native.metadata["implementation_version"] == "native-kirchhoff-2.0"
    assert native.metadata["mapped_params"]["execution_backend"] == "cpu"
    assert native.header_info["is_depth"] is True
    assert native.header_info["total_time_ns"] == 0.0


def test_kirchhoff_and_rtm_are_loaded_global_not_block_pipeline(tmp_path) -> None:
    catalog = NativeProcessingCatalog()
    for method_id in ("kirchhoff_migration", "rtm_migration"):
        descriptor = catalog.get(method_id)
        assert descriptor is not None
        assert "global_transform" in descriptor.capabilities
        assert "loaded_global" in descriptor.capabilities
        assert "file_backed_staging" not in descriptor.capabilities
        pipeline = PipelineDefinition(steps=(PipelineStep(method_id, {}),))
        assert not FileBackedBlockPipelineExecutor(tmp_path).supports(pipeline)


def test_experimental_rtm_is_deterministic_and_updates_depth_axis() -> None:
    request = ProcessingRequest(
        data=_small_bscan(56, 20),
        method_id="rtm_migration",
        params={
            "v": 0.10,
            "depth_m": 0.8,
            "dx_m": 0.1,
            "dz_m": 0.1,
            "boundary_width": 2,
            "max_cell_updates": 20_000_000,
        },
        header_info={"total_time_ns": 14.0, "trace_interval_m": 0.1},
    )
    executor = NativeProcessingExecutor()
    first = executor.execute(request)
    second = executor.execute(request)
    np.testing.assert_array_equal(first.data, second.data)
    assert first.data.shape == (9, 20)
    assert np.isfinite(first.data).all()
    assert first.header_info["is_depth"] is True
    assert first.header_info["depth_step_m"] == pytest.approx(0.1)
    assert first.metadata["migration_mode"] == "zero_offset_exploding_reflector_scalar_2d"
    assert {item["code"] for item in first.runtime_warnings} == {"experimental_rtm_baseline"}


def test_rtm_rejects_work_above_explicit_budget() -> None:
    request = ProcessingRequest(
        data=_small_bscan(64, 24),
        method_id="rtm_migration",
        params={"depth_m": 2.0, "dx_m": 0.05, "dz_m": 0.05, "max_cell_updates": 100},
        header_info={"total_time_ns": 20.0},
    )
    with pytest.raises(RTMResourceLimitError):
        NativeProcessingExecutor().execute(request)


def test_loaded_migration_project_artifact_records_native_lineage(tmp_path) -> None:
    import json
    from pathlib import Path

    from mygpr.interfaces.backend import MyGPRBackend

    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "project", name="Migration lineage")
        backend.projects.save_line_dataset(
            project.project_id,
            "L01",
            _small_bscan(),
            length_m=1.5,
            time_window_ns=12.0,
        )
        pipeline = PipelineDefinition(
            name="Kirchhoff lineage",
            steps=(
                PipelineStep(
                    "kirchhoff_migration",
                    {
                        "freq": 5.0e7,
                        "depth": 0.4,
                        "v": 0.10,
                        "weight": 0.0,
                        "length_m": 1.5,
                        "time_window_ns": 12.0,
                        "backend": "cpu",
                    },
                ),
            ),
        )
        artifact = backend.project_processing.execute_pipeline(project.project_id, "L01", pipeline)
        params_path = Path(project.root_path) / str(artifact.manifest["params_path"])
        payload = json.loads(params_path.read_text(encoding="utf-8"))
        assert payload["params"]["execution_mode"] == "loaded"
        lineage = payload["params"]["lineage"]
        assert lineage[0]["implementation_version"] == "native-kirchhoff-2.0"
        assert lineage[0]["output_shape"] == [48, 16]
        assert lineage[0]["metadata"]["display_data"] == {"shape": [4, 15], "dtype": "float32"}
    finally:
        backend.shutdown()
