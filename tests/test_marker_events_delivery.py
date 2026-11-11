"""External snapshots keep coordinates, identity and interval provenance explicit."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from mygpr.application.project.marker_service import MarkerView, project_rows
from mygpr.domain.markers.models import MarkerAnchor, MarkerDocument, MarkerRecord, EventInterval
from mygpr.domain.spatial.models import SpatialTrack, SpatialTrackPoint
from mygpr.domain.markers.geographic import geographic_rows
from mygpr.infrastructure.reporting.marker_export import export_views
from mygpr.interfaces.backend import MyGPRBackend
from mygpr.interfaces.cli.markers import main


def geometry_view():
    doc = MarkerDocument('project', 'L01', 'raw', markers=(MarkerRecord(anchor=MarkerAnchor('raw', 0)),),
        intervals=(EventInterval(start=MarkerAnchor('raw', 0), end=MarkerAnchor('raw', 3)),))
    # Deliberately missing trace 2: geometry must stay split.
    track = SpatialTrack('L01', 'line', tuple(SpatialTrackPoint(i, 100 + i, 30.) for i in (0, 1, 3)),
                         'EPSG:4326', 'synchronized')
    view = MarkerView(doc, 'raw', '', 4, {'_marker_track': track}, {}, (), False, False)
    return replace(view, rows=project_rows(view))


def test_geojson_keeps_actual_trajectory_fragments_and_complete_list(tmp_path):
    view = geometry_view()
    path = tmp_path / 'markers.geojson'
    export_views((view,), path, 'geojson')
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert len(payload['features']) == 2 and payload['unlocated_records'] == 0
    assert payload['features'][1]['geometry']['type'] == 'GeometryCollection'
    assert payload['features'][1]['properties']['meaning'] == '沿测线参考范围'
    assert view.rows[1]['geometry']['parts'] == [[[100., 30.], [101., 30.]], [[103., 30.]]]


def test_projected_coordinates_are_transformed_not_relabelled(tmp_path):
    view = geometry_view()
    rows = ({**view.rows[0], 'geometry': {**view.rows[0]['geometry'], 'crs': 'EPSG:3857',
                                       'parts': [[[111319.49079327357, 0.]]]}},)
    view = replace(view, rows=rows)
    path = tmp_path / 'projected.geojson'
    export_views((view,), path, 'geojson')
    xy = json.loads(path.read_text(encoding='utf-8'))['features'][0]['geometry']['coordinates'][0]
    np.testing.assert_allclose(xy, [1., 0.], atol=1e-8)


def test_failed_conversion_omits_invalid_geometry_without_connecting_gaps():
    view = geometry_view()
    def partial(crs, xs, ys):
        return [xs[0], float('nan'), xs[2]], ys
    converted = geographic_rows((view.rows[1],), partial)
    assert converted[0]['geometry']['parts'] == [[[100., 30.]], [[103., 30.]]]
    def unavailable(crs, xs, ys):
        raise ValueError('unknown CRS')
    assert geographic_rows(view.rows, unavailable)[0]['geometry'] is None


def test_cli_save_preview_execute_and_stale_snapshot(tmp_path, capsys):
    backend = MyGPRBackend.create_default()
    project = backend.projects.create_project(tmp_path / 'project', name='CLI')
    backend.projects.save_line_dataset(project.project_id, 'L01', np.ones((8, 6), dtype=np.float32))
    source = backend.projects.marker_source_id(project.project_id, 'L01')
    backend.shutdown()
    record = tmp_path / 'event.json'
    record.write_text(json.dumps({'name': '等待', 'start': {'source_id': source, 'trace': 1},
                                 'end': {'source_id': source, 'trace': 2}}), encoding='utf-8')
    prefix = [str(tmp_path / 'project'), '--line', 'L01']
    main([*prefix, 'save', str(record), '--interval', '--revision', '0'])
    doc = json.loads(capsys.readouterr().out)
    uid = doc['intervals'][0]['uid']
    main([*prefix, 'preview', '--id', uid, '--mode', 'exclude'])
    preview = json.loads(capsys.readouterr().out)
    assert preview['output_count'] == 4
    path = tmp_path / 'preview.json'
    path.write_text(json.dumps(preview), encoding='utf-8')
    main([*prefix, 'execute', str(path)])
    assert json.loads(capsys.readouterr().out)['artifact_id']
    with pytest.raises(ValueError, match='变化'):
        main([*prefix, 'save', str(record), '--interval', '--revision', '0'])


def test_report_strip_and_snapshot_include_every_record(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from core.gpr_data_model import GPRDataSet
    from core.report_marker_figures import draw_marker_strip
    view = geometry_view()
    dataset = GPRDataSet.from_matrix('L01', np.ones((8, 4), dtype=np.float32), length_m=3.)
    fig, ax = plt.subplots()
    ax.set_xlim(0, 3)
    draw_marker_strip(ax, view.document.to_dict(), dataset)
    assert len(ax.child_axes) == 1
    destination = Path(tmp_path / 'marked.png')
    fig.savefig(destination)
    assert destination.stat().st_size > 0
    plt.close(fig)


def test_selected_result_geometry_follows_disjoint_output_segments():
    from mygpr.domain.markers.mapping import add_source_mapping, SEGMENT_KEY
    view = geometry_view()
    header, metadata = dict(view.header), {}
    add_source_mapping(header, metadata, 'raw', 4)
    key = header['marker_maps']['raw']['key']
    header['marker_maps']['raw']['kind'] = 'selection'
    metadata[key] = np.array([0., 3.])
    metadata[SEGMENT_KEY] = np.array([0, 1])
    artifact = replace(view, target_id='artifact:A', artifact_id='A', count=2, header=header, trace_metadata=metadata)
    rows = project_rows(artifact)
    assert rows[1]['projection'].spans == ((0, 1), (1, 2))
    assert rows[1]['geometry']['parts'] == [[[100., 30.]], [[103., 30.]]]
