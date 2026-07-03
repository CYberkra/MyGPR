from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from core.field_project_operations import import_line_data
from core.field_project_store import FieldProjectStore
from core.field_report_export import REPORT_PACKAGE_SCHEMA, generate_project_report_package


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_generate_report_package_writes_auditable_files(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="report-test", coordinate_system="CGCS2000 / 3-degree GK")
    line = import_line_data(store, Path("sample_data/gui_sidecar_all_data_main.csv"), line_id="L01", name="测试测线")
    store.run_line_quality_check(line.line_id)
    dataset = store.load_gpr_dataset(line.line_id)
    store.save_processed_line(
        line.line_id,
        dataset.matrix * 0.5,
        {
            "method": "unit_test_filter",
            "params": {"gain": 0.5},
            "manifest": {
                "schema": "mygpr.processing_manifest.v2",
                "line_id": line.line_id,
                "method_id": "unit_test_filter",
                "method_name": "单元测试滤波",
                "status": "ok",
                "input_shape": list(dataset.matrix.shape),
                "output_shape": list(dataset.matrix.shape),
            },
        },
    )
    store.save_targets(
        line.line_id,
        [
            {
                "target_id": "T-01",
                "distance_m": 1.0,
                "depth_m": 0.5,
                "type": "疑似管线",
                "status": "待复核",
                "confidence": "★★★☆☆",
                "source_method_id": "unit_test_filter",
            }
        ],
    )

    result = generate_project_report_package(store, package_name="report_test")

    package_dir = store.root / result.package_dir
    assert package_dir.exists()
    assert (store.root / result.manifest_path).exists()
    assert (store.root / result.html_path).exists()
    assert (store.root / result.summary_json_path).exists()
    assert (store.root / result.line_manifest_csv_path).exists()
    assert (store.root / result.quality_csv_path).exists()
    assert (store.root / result.targets_csv_path).exists()
    assert (store.root / result.processing_csv_path).exists()
    assert (store.root / result.spatial_csv_path).exists()
    assert (store.root / result.pdf_path).exists()

    manifest = json.loads((store.root / result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["schema"] == REPORT_PACKAGE_SCHEMA
    assert manifest["summary"]["metrics"]["line_count"] == 1
    assert manifest["summary"]["metrics"]["processing_artifact_count"] == 1
    assert manifest["summary"]["metrics"]["target_count"] == 1

    line_rows = _read_csv(store.root / result.line_manifest_csv_path)
    quality_rows = _read_csv(store.root / result.quality_csv_path)
    target_rows = _read_csv(store.root / result.targets_csv_path)
    processing_rows = _read_csv(store.root / result.processing_csv_path)
    assert line_rows[0]["line_id"] == "L01"
    assert quality_rows[0]["status"] in {"通过", "警告", "失败"}
    assert target_rows[0]["target_id"] == "T-01"
    assert processing_rows[0]["method_id"] == "unit_test_filter"

    reopened = FieldProjectStore.open(store.root)
    assert reopened.manifest.reports["status"] == "已生成"
    assert reopened.manifest.reports["latest_manifest_path"] == result.manifest_path
    assert (store.root / "reports" / "latest_report_manifest.json").exists()


def test_report_package_marks_unchecked_line_without_fake_quality(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="unchecked-test")
    import_line_data(store, Path("sample_data/gui_sidecar_all_data_main.csv"), line_id="L01", name="未质检测线")
    # Remove the auto-generated quality report to simulate an incomplete project.
    quality_path = store.quality_report_path("L01")
    if quality_path.exists():
        quality_path.unlink()

    result = generate_project_report_package(store, package_name="report_unchecked")
    quality_rows = _read_csv(store.root / result.quality_csv_path)
    assert quality_rows[0]["status"] == "未质检"
    assert "未生成质检报告" in quality_rows[0]["orientation_message"]


def test_delivery_page_exposes_report_package_actions() -> None:
    source = Path("ui/field_panels/delivery_page.py").read_text(encoding="utf-8")
    assert "生成报告包" in source
    assert "generate_project_report_package" in source
    assert "生成/打开 PDF" in source
    assert "_action_generate_or_open_pdf_report" in source
    assert "_action_generate_report_package" in source
