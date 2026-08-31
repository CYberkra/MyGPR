from __future__ import annotations

from pathlib import Path

from core.field_import_preview import build_import_preflight
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import detect_mygpr_airborne_sidecar_csv, load_gpr_dataset


def test_detect_and_load_legacy_mygpr_sidecar_csv() -> None:
    path = Path("sample_data/gui_sidecar_all_data_main.csv")
    header = detect_mygpr_airborne_sidecar_csv(path)
    assert header is not None
    assert int(header["sample_count"]) == 10
    assert int(header["trace_count"]) == 12

    dataset = load_gpr_dataset(path, line_id="L99")
    assert dataset.format_name == "mygpr-airborne-sidecar-csv"
    assert dataset.matrix.shape == (10, 12)
    assert dataset.metadata["columns"][3] == "amplitude"
    assert len(dataset.metadata["trajectory_rows"]) == 12


def test_preflight_reports_sidecar_csv_details() -> None:
    result = build_import_preflight("sample_data/gui_sidecar_all_data_main.csv", line_id="L98")
    assert result.can_import is True
    assert result.format_name == "mygpr-airborne-sidecar-csv"
    assert result.source_kind == "MyGPR 航空 GPR 主数据 CSV"
    assert result.has_trajectory is True
    assert result.sample_count == 10
    assert result.trace_count == 12


def test_project_import_normalizes_sidecar_csv(tmp_path: Path) -> None:
    project = FieldProjectStore.create_empty(tmp_path / "project", name="sidecar-import-test")
    src = Path("sample_data/uav_gpr_motion_demo_v1/main.csv")
    line = project.import_line_file("L01", src, name="uav-demo-line", copy_into_project=True)

    assert line.data_format == "mygpr-airborne-sidecar-csv"
    dataset = project.load_gpr_dataset("L01")
    assert dataset.matrix.shape == (160, 180)
    trajectory = project.load_trajectory("L01")
    assert len(trajectory.points) == 180
    assert (project.root / "raw" / "L01" / "import_manifest.json").exists()
