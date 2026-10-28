import numpy as np
import pytest

pytest.importorskip('PyQt6')
from ui.controllers.project_controller import ProjectController
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


def test_depth_window_preview_uses_saved_depth_axis(tmp_path):
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        pid = backend.projects.create_project(tmp_path / 'project', name='depth').project_id
        backend.projects.save_line_dataset(pid, 'L01', np.ones((40, 5), np.float32), length_m=10, time_window_ns=400)
        result = backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(PipelineStep('time_to_depth', {'dt': 10., 'v': .1, 'dz': 1.}),)))
        info = backend.projects.get_artifact_dataset_info(pid, 'L01', result.artifact_id)
        matrix, si, ti = backend.projects.read_artifact_window(pid, 'L01', result.artifact_id)
        bundle = ProjectController._bundle_from_window(matrix, 'L01', sample_indices=si, trace_indices=ti, total_samples=info.shape[0], total_traces=info.shape[1], metadata=info.metadata)
        assert bundle.sample_axis_label == '深度 (m)'
        np.testing.assert_allclose(bundle.sample_axis, np.linspace(0, 19.5, 20))
    finally:
        backend.shutdown()
