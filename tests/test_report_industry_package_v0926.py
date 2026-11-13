from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from openpyxl import load_workbook

from core.field_project_operations import import_line_data
from core.field_project_store import FieldLineRecord, FieldProjectStore
from core.field_report_export import _select_report_scope, generate_project_report_package
from core.report_line_content import processing_chain
from core.report_profiles import project_ranges
from core.trajectory_model import TrajectoryModel, TrajectoryPoint
from mygpr.domain.markers.mapping import TraceMapping
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


def test_explicit_empty_report_scope_does_not_expand_to_everything():
    line = SimpleNamespace(line_id="L01")
    artifact = SimpleNamespace(line_id="L01", artifact_id="A1")
    store = SimpleNamespace(root=Path("."), list_lines=lambda: [line])
    catalog = {'all_artifacts': [artifact]}

    assert _select_report_scope(store, selected_line_ids=None, selected_artifact_ids=None, **catalog) == ([line], [artifact])
    assert _select_report_scope(store, selected_line_ids=("L01",), selected_artifact_ids=(), **catalog) == ([line], [])
    with pytest.raises(ValueError, match="请至少选择一条"):
        _select_report_scope(store, selected_line_ids=(), selected_artifact_ids=None, **catalog)


def test_report_exposes_missing_processing_ancestor():
    result = SimpleNamespace(artifact_id='B', line_id='L01', parent_artifact_id='A',
                             method_id='dewow', method_name='去漂移', warnings=[], params={})
    other_line = SimpleNamespace(artifact_id='A', line_id='L02')
    for index in ({}, {'A': other_line}):
        chain, incomplete = processing_chain(result, index)
        assert incomplete and [entry['artifact_id'] for entry in chain] == ['B']


def test_report_selected_result_retains_its_unselected_ancestor(tmp_path):
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / 'project', name='两次处理')
        project_id = project.project_id
        data = np.random.default_rng(83).normal(size=(48, 32)).astype(np.float32)
        backend.projects.save_line_dataset(project_id, 'L01', data, time_window_ns=400.0)
        pipeline = PipelineDefinition(steps=(PipelineStep('dewow', {'window': 7}),))
        first = backend.project_processing.execute_pipeline(project_id, 'L01', pipeline)
        final = backend.project_processing.execute_pipeline(
            project_id, 'L01', pipeline, input_artifact_id=first.artifact_id)
        package = backend.reporting.generate_package(project_id, package_name='final_only', report_profile={
            'selected_line_ids': ['L01'], 'selected_artifact_ids': [final.artifact_id],
            'include_spatial_results': False,
        })
        root = Path(project.root_path) / package.package_dir
        evidence = json.loads((root / 'json/processing_parameters.json').read_text(encoding='utf-8'))
        assert [entry['artifact_id'] for entry in evidence['selected_versions']] == [final.artifact_id]
        assert [entry['artifact_id'] for entry in evidence['primary_chains']['L01']] == [first.artifact_id, final.artifact_id]
        assert (root / 'figures/L01_processed.png').is_file()
    finally:
        backend.shutdown()


def test_report_range_projection_keeps_only_retained_fragments():
    segments = np.array([0, 0, 1, 1, 1])
    mapping = TraceMapping('raw:1', 'A1', np.array([0, 1, 6, 7, 8]), segments, 'selection')
    ranges = [('L01-F01', 1, 8), ('L01-A01', 2, 6)]
    assert project_ranges(ranges, mapping, segments) == [('L01-F01', 1, 2), ('L01-F01', 2, 4)]
    assert project_ranges(ranges, None, segments) == []


def test_formal_report_contains_excel_audit_figures_approvals_and_checksums(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(
        tmp_path / "project",
        name="野外基覆界面勘探",
        project_no="MYGPR-2026-001",
        coordinate_system="EPSG:32648",
        vertical_datum="1985国家高程基准",
    )
    import_line_data(
        store,
        Path("sample_data/gui_sidecar_all_data_main.csv"),
        line_id="L01",
        name="一号测线",
    )
    store.upsert_line(FieldLineRecord('L02', '未选线'))
    for line_id, east in [('L01', 500000), ('L02', 500200)]:
        store.save_trajectory(line_id, TrajectoryModel([
            TrajectoryPoint(distance_m=i, x=east + i, y=3300000 + i / 2, z=100, trace_index=i)
            for i in (0, 1, 4, 5)
        ]))
    result = generate_project_report_package(
        store,
        package_name="formal_delivery",
        report_profile={
            "compiler": "编制人",
            "reviewer": "复核人",
            "approver": "批准人",
            "revision": "A",
            'selected_line_ids': ['L01'],
            'primary_artifacts': {'L01': ''},
            'project_analysis': {'objective': '确认交付范围'},
            'line_analysis': {'L01': {'first_trace': 2, 'last_trace': 5, 'observation': '重点区段'}},
            'acquisition_overrides': {'L01': {'power': '5 dBm（设定）', 'source': '采集记录'}},
        },
    )

    package = store.root / result.package_dir
    xlsx = store.root / result.xlsx_path
    audit = store.root / result.audit_csv_path
    checksums = store.root / result.checksums_path
    figures = store.root / result.figures_dir
    pdf = store.root / result.pdf_path
    html = store.root / result.html_path

    assert xlsx.exists() and xlsx.stat().st_size > 0
    assert audit.exists() and checksums.exists()
    assert pdf.exists() and pdf.stat().st_size > 0
    assert html.exists() and "批准人" in html.read_text(encoding="utf-8")
    assert figures.is_dir() and any(figures.glob("*.png"))
    dossiers = json.loads((package / 'json/line_dossiers.json').read_text(encoding='utf-8'))
    assert dossiers['plan_map'] == 'project_plan_map.png'
    assert [item['line_id'] for item in dossiers['lines']] == ['L01']
    line = dossiers['lines'][0]
    assert line['acquisition']['功率'] == '5 dBm（设定）'
    assert line['analysis']['code'] == 'L01-F01'
    assert line['raw_figure'] == 'L01_raw.png'
    assert line['route_figure'] == dossiers['plan_map']
    assert line['processed_figure'] == ''
    assert line['detail_figures'] == ['L01-F01_raw.png']
    assert all((figures / name).is_file() for name in [dossiers['plan_map'], line['raw_figure'], *line['detail_figures']])
    assert (package / 'json/processing_parameters.json').is_file()
    assert '重点区段' in html.read_text(encoding='utf-8')

    workbook = load_workbook(xlsx, read_only=True)
    assert {"测线清单", "数据质检", "传感器同步", "基覆界面", "处理记录", "空间成果", "GIS图层"}.issubset(workbook.sheetnames)
    workbook.close()

    with audit.open("r", encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    assert any(row["path"] == "project_report.pdf" for row in audit_rows)
    assert all(len(row["sha256"]) == 64 for row in audit_rows)

    checksum_entries = {}
    for line in checksums.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        checksum_entries[rel] = digest
    assert "project_report.pdf" in checksum_entries
    actual_pdf_hash = hashlib.sha256((package / "project_report.pdf").read_bytes()).hexdigest()
    assert checksum_entries["project_report.pdf"] == actual_pdf_hash

    manifest = json.loads((store.root / result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["summary"]["report_revision"] == "A"
    assert manifest["summary"]["approval"]["compiler"] == "编制人"
