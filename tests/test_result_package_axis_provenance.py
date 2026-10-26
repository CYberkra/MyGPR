"""Replacing raw data must not silently change a historical result's axes."""

from pathlib import Path
import json

import h5py
import numpy as np
import pytest

from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


@pytest.mark.parametrize("new_length", [7.0, 70.0])
@pytest.mark.parametrize("inline", [False, True])
def test_replaced_raw_without_saved_axes_refuses_historical_read_and_export(tmp_path, new_length, inline):
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "project", name="Axis provenance")
        pid = project.project_id
        matrix = np.arange(128, dtype=np.float32).reshape(16, 8)
        backend.projects.save_line_dataset(pid, "L01", matrix, time_window_ns=100.0, length_m=7.0)
        artifact = backend.project_processing.execute_pipeline(
            pid, "L01", PipelineDefinition(steps=(PipelineStep("dewow", {"window": 5}),))
        )
        # Reproduce already-persisted results from the descriptor-only writer.
        from core.hdf5_line_container import locate_processing_artifact

        container = Path(project.root_path) / "data/lines/L01.h5"
        sidecar, _ = locate_processing_artifact(container, artifact.artifact_id)
        with h5py.File(sidecar, "r+") as saved:
            group = saved[f"processing/artifacts/{artifact.artifact_id}"]
            manifest = json.loads(group.attrs["manifest_json"])
            for key in ("distance_axis_m", "time_axis_ns", "depth_axis_m"):
                values = manifest["input_dataset"][key]
                manifest["input_dataset"][key] = {"shape": [len(values)], "dtype": "float32"}
            group.attrs["manifest_json"] = json.dumps(manifest)
        if inline:
            with h5py.File(sidecar, "r") as saved, h5py.File(container, "r+") as current:
                saved.copy(
                    saved[f"processing/artifacts/{artifact.artifact_id}"],
                    current["processing/artifacts"],
                    name=artifact.artifact_id,
                )
            sidecar.unlink()
        expected = backend.projects.read_artifact_dataset(pid, "L01", artifact.artifact_id)
        np.testing.assert_array_equal(expected.header_info["distance_axis_m"], np.arange(8))
        backend.projects.save_line_dataset(pid, "L01", matrix, time_window_ns=100.0, length_m=new_length)
        with pytest.raises(ValueError, match="坐标轴|原始测线"):
            backend.projects.read_artifact_dataset(pid, "L01", artifact.artifact_id)
        target = tmp_path / "result.zip"
        job = backend.submit_processing_result_package(pid, "L01", str(target))
        done = backend.jobs.wait(job, timeout=30)
        assert done.status.value == "failed" and "坐标轴" in done.error_message
        assert not target.exists() and not list(tmp_path.glob(".mygpr-package-*"))
        np.testing.assert_array_equal(backend.projects.read_dataset(pid, "L01").data, matrix)
    finally:
        backend.shutdown()


@pytest.mark.parametrize("save_intermediates", [True, False, "workbench"])
def test_explicit_saved_axes_survive_raw_replacement_and_continuation(tmp_path, save_intermediates):
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "project", name="Axis snapshot")
        pid = project.project_id
        matrix = np.arange(128, dtype=np.float32).reshape(16, 8)
        backend.projects.save_line_dataset(pid, "L01", matrix, time_window_ns=100.0, length_m=7.0)
        source = backend.projects.read_dataset(pid, "L01")
        header = {
            key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in source.header_info.items()
        }
        artifact = backend.projects.save_processing_artifact(
            pid,
            "L01",
            matrix,
            name="Saved axes",
            method_id="dewow",
            method_name="Dewow",
            params={},
            input_dataset=header,
            pipeline=[],
            branch_id="L01:main",
        )
        backend.projects.save_line_dataset(
            pid, "L01", matrix, time_window_ns=1000.0, length_m=70.0, dielectric_constant=4.0
        )
        saved = backend.projects.read_artifact_dataset(pid, "L01", artifact.artifact_id)
        for axis in ("distance_axis_m", "time_axis_ns", "depth_axis_m"):
            np.testing.assert_array_equal(saved.header_info[axis], source.header_info[axis])
        assert saved.header_info["dielectric_constant"] == source.header_info["dielectric_constant"]
        if save_intermediates == "workbench":
            service = backend.processing_workbench
            session = service.open_session(pid, "L01", input_artifact_id=artifact.artifact_id, restore_draft=False)
            service.append_step(session.session_id, "dewow", {"window": 5})
            continued = service.save_session(session.session_id)
        else:
            continued = backend.project_processing.execute_pipeline(
                pid,
                "L01",
                PipelineDefinition(steps=(PipelineStep("dewow", {"window": 5}),)),
                input_artifact_id=artifact.artifact_id,
                save_intermediates=save_intermediates,
            )
        reloaded = backend.projects.read_artifact_dataset(pid, "L01", continued.artifact_id)
        for axis in ("distance_axis_m", "time_axis_ns", "depth_axis_m"):
            np.testing.assert_array_equal(reloaded.header_info[axis], source.header_info[axis])
        # A legacy continuation with descriptors must validate its input's raw
        # revision too, rather than treating its own newer commit as proof.
        from core.hdf5_line_container import locate_processing_artifact

        path, _ = locate_processing_artifact(Path(project.root_path) / "data/lines/L01.h5", continued.artifact_id)
        with h5py.File(path, "r+") as handle:
            group = handle[f"processing/artifacts/{continued.artifact_id}"]
            manifest = json.loads(group.attrs["manifest_json"])
            for key in ("distance_axis_m", "time_axis_ns", "depth_axis_m"):
                manifest["input_dataset"][key] = {"shape": [len(source.header_info[key])], "dtype": "float32"}
            group.attrs["manifest_json"] = json.dumps(manifest)
        with pytest.raises(ValueError, match="坐标轴"):
            backend.projects.read_artifact_dataset(pid, "L01", continued.artifact_id)
        assert Path(project.root_path).is_dir()
    finally:
        backend.shutdown()
