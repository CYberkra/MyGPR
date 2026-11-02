import numpy as np
import pytest
import csv
from pathlib import Path
from core.gui_rendering import bundle_from_dataset
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


def test_processed_annotation_exports_its_own_axes(tmp_path):
    from mygpr.domain.interpretation.models import InterfaceAnnotation, InterpretationPoint
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / 'project', name='annotation axes')
        pid = project.project_id
        backend.projects.save_line_dataset(pid, 'L01', np.zeros((40, 5), np.float32), length_m=10, time_window_ns=400)
        cut = backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(
            PipelineStep('time_cut', {'mode': 'keep_range', 'time_start_ns': 100, 'time_end_ns': 300}),)))
        artifact = backend.projects.read_artifact_dataset(pid, 'L01', cut.artifact_id)
        annotation = InterfaceAnnotation(annotation_id='B-L01', line_id='L01', name='interface', version=1,
            status='confirmed', processing_result=cut.artifact_id,
            points=(InterpretationPoint(0, 0), InterpretationPoint(4, 0)))
        backend.projects.save_interface_annotation(pid, annotation)
        with (Path(project.root_path) / 'spatial/L01_basal_interface_xy.csv').open(encoding='utf-8-sig', newline='') as fh:
            row = next(csv.DictReader(fh))
        assert float(row['time_ns']) == pytest.approx(float(artifact.header_info['time_axis_ns'][0]), abs=1e-5)
        labels = backend.projects._session(pid)._store.load_basal_interface_labels('L01')
        assert float(labels['time_gt_ns'][0]) == pytest.approx(float(artifact.header_info['time_axis_ns'][0]), abs=1e-5)
        assert labels['soft_mask_gt'].shape == artifact.data.shape
        raw_session = backend.interpretation_edit.open_session(pid, 'L01')
        assert raw_session.annotation.processing_result == ''
        assert not raw_session.annotation.points
        assert raw_session.audit['source_changed'] is True
    finally:
        backend.shutdown()


@pytest.mark.parametrize('crop', [False, True])
def test_depth_result_roundtrip_and_preview(tmp_path, crop):
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        pid = backend.projects.create_project(tmp_path / 'project', name='depth').project_id
        raw = np.arange(200, dtype=np.float32).reshape(40, 5)
        backend.projects.save_line_dataset(pid, 'L01', raw, length_m=10, time_window_ns=400)
        parent = ''
        if crop:
            cut = backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(PipelineStep('time_cut', {'mode': 'keep_range', 'time_start_ns': 100, 'time_end_ns': 300}),)))
            parent = cut.artifact_id
        result = backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(PipelineStep('time_to_depth', {'dt': 10., 'v': .1, 'dz': 1.}),)), input_artifact_id=parent)
        data = backend.projects.read_artifact_dataset(pid, 'L01', result.artifact_id)
        n = 20 if not crop else 10
        expected_time = np.linspace(100 if crop else 0, 290 if crop else 390, n, dtype=np.float32)
        expected_depth = expected_time * .05
        np.testing.assert_allclose(data.header_info['depth_axis_m'], expected_depth, rtol=1e-6)
        np.testing.assert_allclose(data.header_info['time_axis_ns'], expected_time, rtol=1e-6)
        assert data.header_info['is_depth'] is True
        dataset = backend.projects._session(pid)._artifact_dataset('L01', result.artifact_id)[2]
        bundle = bundle_from_dataset(dataset)
        assert bundle.sample_axis_label == '深度 (m)'
        np.testing.assert_allclose(bundle.sample_axis, expected_depth, rtol=1e-6)
        continued = backend.project_processing.execute_pipeline(pid, 'L01', PipelineDefinition(steps=(PipelineStep('amplitude_scale', {'scale': 1.}),)), input_artifact_id=result.artifact_id)
        after = backend.projects.read_artifact_dataset(pid, 'L01', continued.artifact_id)
        np.testing.assert_allclose(after.header_info['depth_axis_m'], expected_depth, rtol=1e-6)
        assert after.header_info['is_depth'] is True
        np.testing.assert_array_equal(backend.projects.read_dataset(pid, 'L01').data, raw)
    finally:
        backend.shutdown()

@pytest.mark.parametrize('key', ['dt', 'v', 'dz'])
@pytest.mark.parametrize('value', [0, -1, float('nan')])
def test_depth_conversion_rejects_invalid_parameters(key, value):
    from mygpr.infrastructure.processing.algorithms.extended import native_time_to_depth
    with pytest.raises(ValueError, match=key):
        native_time_to_depth(np.ones((4, 2), np.float32), {key: value})


def test_unknown_origin_cannot_be_saved_but_other_migrations_remain_compatible():
    from mygpr.infrastructure.processing.algorithms.extended import native_time_to_depth
    from mygpr.infrastructure.persistence.field_project_adapter import _validated_output_header
    _, metadata = native_time_to_depth(np.ones((4, 2), np.float32), {'dt': 1, 'v': .1, 'dz': .1})
    header = metadata['header_info_updates']
    with pytest.raises(ValueError, match='时间起点'):
        _validated_output_header(header, (2, 2), method_id='time_to_depth')
    assert _validated_output_header({'is_depth': True}, (2, 2), method_id='rtm_migration')['is_depth']
