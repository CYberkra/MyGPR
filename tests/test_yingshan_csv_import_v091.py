from __future__ import annotations

from pathlib import Path

from core.field_import_preview import build_import_preflight
from core.field_project_store import FieldProjectStore
from core.gpr_data_model import detect_mygpr_airborne_sidecar_csv, load_gpr_dataset


def _write_yingshan_like_csv(path: Path, *, samples: int = 5, traces: int = 4) -> None:
    lines = [
        f"Number of Samples = {samples},,,\n",
        "Time windows (ns) = 700,,\n",
        f"Number of Traces = {traces},,\n",
        "Trace interval (m) = 0.09093,,\n",
    ]
    for trace in range(traces):
        for sample in range(samples):
            amp = trace * 0.1 + sample * 0.01
            lines.append(f"106.8068912,31.26075037,441.7954,{amp:.8f},8.9436\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_yingshan_header_with_trailing_commas_is_detected(tmp_path: Path) -> None:
    csv_path = tmp_path / "Line9origin(30).csv"
    _write_yingshan_like_csv(csv_path)

    header = detect_mygpr_airborne_sidecar_csv(csv_path)
    assert header is not None
    assert int(header["sample_count"]) == 5
    assert int(header["trace_count"]) == 4

    dataset = load_gpr_dataset(csv_path, line_id="L09")
    assert dataset.format_name == "mygpr-airborne-sidecar-csv"
    assert dataset.matrix.shape == (5, 4)
    assert dataset.metadata["columns"] == ["longitude", "latitude", "elevation_m", "amplitude", "height_m"]
    assert len(dataset.metadata["trajectory_rows"]) == 4


def test_yingshan_preflight_enables_import_button(tmp_path: Path) -> None:
    csv_path = tmp_path / "Line9origin(30).csv"
    _write_yingshan_like_csv(csv_path)

    result = build_import_preflight(csv_path, line_id="L09")
    assert result.can_import is True
    assert result.support == "direct"
    assert result.format_name == "mygpr-airborne-sidecar-csv"
    assert result.shape_text == "5 × 4"
    assert result.has_trajectory is True
    assert "height_m" in result.column_summary


def test_yingshan_project_import_writes_standard_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "Line9origin(30).csv"
    _write_yingshan_like_csv(csv_path, samples=6, traces=3)
    project = FieldProjectStore.create_empty(tmp_path / "project", name="yingshan-test")

    line = project.import_line_file("L09", csv_path, name="Line9", copy_into_project=True)
    assert line.data_format == "mygpr-airborne-sidecar-csv"
    assert line.gpr_dataset_path
    assert line.trajectory_path
    assert (project.root / "raw" / "L09" / "import_manifest.json").exists()
    assert project.load_gpr_dataset("L09").matrix.shape == (6, 3)
    assert len(project.load_trajectory("L09").points) == 3
