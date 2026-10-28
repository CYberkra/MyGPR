"""Batch import and application quality-center contracts."""
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('PyQt6')


def test_project_page_emits_all_selected_import_sources(qapp):
    from ui.pages.project_page import ProjectPage

    page = ProjectPage()
    page.set_project_info(NS(read_only=False))
    page._import_sources = ['D:/data/L01.h5', 'D:/data/L02.h5']
    page.file_edit.setText('2 个文件')
    requests = []
    page.import_requested.connect(requests.append)
    page._emit_import_request(preflight=False)
    assert requests[0]['sources'] == page._import_sources
    page.close()
    page.deleteLater()


def test_quality_center_displays_issues_and_filters_failed_lines(qapp):
    from ui.pages.project_quality import QualityCenterDialog

    reports = [
        NS(line_id='L01', status='passed', status_label='通过', sample_count=501, trace_count=1000,
           time_window_ns=900., length_m=120., finite_ratio=1., trajectory_points=1000,
           sampled=False, issues=()),
        NS(line_id='L02', status='warning', status_label='警告', sample_count=501, trace_count=900,
           time_window_ns=900., length_m=100., finite_ratio=.98, trajectory_points=800,
           sampled=True, issues=(NS(severity='warning', code='nan_inf',
                                  message='存在坏值', suggestion='检查原始文件'),)),
    ]
    dialog = QualityCenterDialog()
    dialog.set_reports(reports)
    assert dialog._table.rowCount() == 2
    dialog._table.selectRow(1)
    assert '存在坏值' in dialog._details.toPlainText()
    assert '检查原始文件' in dialog._details.toPlainText()
    dialog._status_filter.setCurrentIndex(dialog._status_filter.findData('warning'))
    assert dialog._table.rowCount() == 1
    assert dialog._table.item(0, 0).text() == 'L02'
    dialog.close()
    dialog.deleteLater()


def test_trace_issue_locations_use_real_sensor_and_qc_indices():
    import numpy as np
    from core.quality_trace_locations import locate_trace_issues
    issues = locate_trace_issues({'trace_qc_bad_mask': np.array([0, 1, 0, 1, 0]),
        'roll_deg': np.array([0., np.nan, 0., 0., 0.]),
        'trace_timestamp_s': np.array([0., 1., 2., 1., 20.])}, 5)
    ranges = {(issue.code, issue.trace_start, issue.trace_end) for issue in issues}
    assert ('trace_qc_bad', 1, 2) in ranges
    assert ('sensor_missing', 1, 2) in ranges
    assert ('timestamp_nonmonotonic', 2, 4) in ranges
    assert ('timestamp_gap', 3, 5) in ranges
    assert not locate_trace_issues({}, 5)


def test_quality_dialog_only_locates_evidenced_ranges(qapp):
    from ui.pages.project_quality import QualityCenterDialog
    dialog = QualityCenterDialog()
    issue = NS(severity='warning', code='sensor_missing', message='缺失', suggestion='', trace_start=2, trace_end=4)
    unknown = NS(severity='warning', code='missing_trajectory', message='无轨迹', suggestion='')
    report = NS(line_id='L01', status='warning', sample_count=10, trace_count=5,
                time_window_ns=1, length_m=0, finite_ratio=1, trajectory_points=0, issues=(unknown, issue))
    dialog.set_reports([report])
    dialog._table.selectRow(0)
    assert dialog._locations.count() == 1
    emitted = []
    dialog.locate_requested.connect(lambda *args: emitted.append(args))
    dialog._locations.setCurrentRow(0)
    dialog._locate_selected()
    assert emitted == [('L01', 2, 4)]
    dialog.close()


def test_quality_report_exposes_acquisition_sensor_locations(tmp_path):
    from tests.test_cdut_streaming_hdf5_import import write_stream
    from mygpr.interfaces.backend import MyGPRBackend
    source = tmp_path / 'quality.h5'
    write_stream(source)
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / 'project', name='quality')
        backend.projects.import_line_source(project.project_id, 'L01', source)
        report = backend.maintenance.run_line_quality_check(project.project_id, 'L01')
        assert any(issue.code == 'sensor_missing' and issue.trace_start == 2 and issue.trace_end == 3
                   for issue in report.issues)
        reread = backend.maintenance.list_quality_reports(project.project_id)[0]
        assert any(issue.code == 'sensor_missing' and issue.trace_start == 2 for issue in reread.issues)
    finally:
        backend.shutdown()


def test_old_quality_locations_do_not_apply_after_raw_replacement(tmp_path):
    import numpy as np
    from mygpr.interfaces.backend import MyGPRBackend
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / 'project', name='quality')
        pid = project.project_id
        matrix = np.ones((20, 10), dtype=np.float32)
        matrix[5, 3] = np.nan
        backend.projects.save_line_dataset(pid, 'L01', matrix)
        report = backend.maintenance.run_line_quality_check(pid, 'L01')
        assert any(i.code == 'nonfinite_trace' and i.trace_start == 3 for i in report.issues)
        backend.projects.save_line_dataset(pid, 'L01', np.zeros((20, 10), dtype=np.float32))
        report = backend.maintenance.list_quality_reports(pid)[0]
        assert not any(i.code == 'nonfinite_trace' and i.trace_start is not None for i in report.issues)
        assert any(i.code == 'quality_revision_unknown' for i in report.issues)
    finally:
        backend.shutdown()


def test_sampled_trace_indices_are_actual_source_columns():
    import numpy as np
    from core.field_data_quality import _representative_matrix_sample
    matrix = np.arange(10000, dtype=np.float32).reshape(100, 100)
    sample, sampled, indices = _representative_matrix_sample(matrix, max_values=12)
    assert sampled
    np.testing.assert_array_equal(sample[0], matrix[0, indices])
    np.testing.assert_array_equal(indices, [0, 29, 58, 87])


def test_quality_sampling_reads_proxy_by_bounded_slice():
    import numpy as np
    from core.field_data_quality import evaluate_line_data_quality
    from core.gpr_data_model import GPRDataSet
    class Proxy:
        shape = (2200, 2200)
        ndim = 2
        dtype = np.dtype('float32')
        size = 2200 * 2200
        def __array__(self, *args, **kwargs):
            raise AssertionError('full matrix materialization')
        def __getitem__(self, key):
            rows, cols = key
            return np.ones((len(range(*rows.indices(2200))), len(range(*cols.indices(2200)))), dtype=np.float32)
    dataset = GPRDataSet('L01', Proxy(), np.arange(2200), np.arange(2200), np.arange(2200))
    report = evaluate_line_data_quality(dataset)
    assert report.sampled
    assert report.evaluated_value_count <= 4_000_000
