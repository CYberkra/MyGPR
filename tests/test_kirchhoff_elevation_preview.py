import numpy as np
import pytest
pytest.importorskip('PyQt6')
from ui.controllers.project_controller import ProjectController
from ui.widgets.bscan_axes import physical_time_ticks


def test_shared_elevation_grid_is_used_without_warp():
    axis = np.array([100.4, 100.3, 100.2])
    bundle = ProjectController._bundle_from_window(np.ones((3, 2)), 'L', sample_indices=np.arange(3), trace_indices=np.arange(2), total_samples=3, total_traces=2, trace_elevation_m=[100.3,100.2], metadata={'is_elevation': True, 'elevation_axis_m': axis, 'time_axis_status': 'unknown', 'time_axis_ns': [0,0,0], 'depth_axis_m':[1,2,3]})
    assert bundle.sample_axis_label == '海拔 (m)'
    np.testing.assert_array_equal(bundle.sample_axis, axis)
    assert bundle.depth_axis_m is None
    assert physical_time_ticks(axis, [(1, [0, 1, 2])], 0, 3)


def test_public_artifact_window_elevation_axis(tmp_path):
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
    from mygpr.interfaces.backend import MyGPRBackend
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        pid = backend.projects.create_project(tmp_path / 'p', name='p').project_id
        backend.projects.save_line_dataset(pid, 'L01', np.ones((48,16), np.float32), length_m=1.5, time_window_ns=12.)
        session = backend.projects._session(pid)
        folder = session._store.root / 'sensors'
        folder.mkdir(exist_ok=True)
        np.savez(folder / 'ground.npz', ground_elevation_m=np.linspace(100,100.3,16))
        record = session._store.get_line('L01')
        record.trace_metadata_path = 'sensors/ground.npz'
        session._store.upsert_line(record)
        params = {'freq':5e7,'depth':.4,'v':.1,'weight':0.,'length_m':1.5,'time_window_ns':12.,'backend':'cpu','topo_cor':1}
        art = backend.project_processing.execute_pipeline(pid,'L01',PipelineDefinition(steps=(PipelineStep('kirchhoff_migration',params),)))
        info = backend.projects.get_artifact_dataset_info(pid,'L01',art.artifact_id)
        matrix, si, ti = backend.projects.read_artifact_window(pid,'L01',art.artifact_id)
        bundle = ProjectController._bundle_from_window(matrix, 'L01', sample_indices=si, trace_indices=ti, total_samples=info.shape[0], total_traces=info.shape[1], metadata=info.metadata)
        assert bundle.sample_axis_label == '海拔 (m)'
        np.testing.assert_allclose(bundle.sample_axis, np.asarray(info.metadata['elevation_axis_m'])[si])
        assert bundle.depth_axis_m is None
    finally:
        backend.shutdown()
