from __future__ import annotations

from pathlib import Path

import numpy as np

from core.ingest_service import IngestService
from core.interpretation_service import InterpretationService
from core.project_models import LineRecordV1
from core.project_service import ProjectService
from core.processing_session import ProcessingSessionService
from core.spatial_synthesis_service import SpatialSynthesisService


def _stacked_airborne_csv(path: Path) -> None:
    samples = 4
    traces = 3
    lines = [
        f"Number of Samples = {samples}",
        "Time windows (ns) = 80",
        f"Number of Traces = {traces}",
        "Trace interval (m) = 1",
    ]
    rows = []
    for trace in range(traces):
        for sample in range(samples):
            rows.append(
                [
                    104.0 + trace * 0.00001,
                    30.0 + trace * 0.00001,
                    500.0 + trace,
                    float(trace * 10 + sample),
                    10.0 + trace,
                    float(trace),
                ]
            )
    path.write_text(
        "\n".join(lines)
        + "\n"
        + "\n".join(",".join(str(value) for value in row) for row in rows),
        encoding="utf-8",
    )


def test_spatial_synthesis_builds_tracks_terrain_and_interpretation_points(
    tmp_path: Path,
) -> None:
    source = tmp_path / "airborne.csv"
    _stacked_airborne_csv(source)
    temporary = IngestService.open_temporary(source)
    project = IngestService.formalize(temporary, tmp_path / "formal", name="Spatial")
    temporary.close()
    try:
        line_id = project.list_lines()[0].line_id
        InterpretationService(project).add_point(
            line_id,
            trace=1,
            sample=2,
            confidence=0.9,
            label="界面点",
        )
        synthesis = SpatialSynthesisService(project).build()

        assert synthesis["summary"]["located_line_count"] == 1
        assert synthesis["summary"]["track_point_count"] == 3
        assert synthesis["summary"]["interpretation_feature_count"] == 1
        assert synthesis["tracks"][0]["longitude"][1] > synthesis["tracks"][0]["longitude"][0]
        assert synthesis["terrain_points"][0]["ground_elevation_m"] == 500.0
        point = synthesis["interpretation_features"][0]
        assert point["longitude"] == synthesis["tracks"][0]["longitude"][1]
        assert point["properties"]["confidence"] == 0.9
    finally:
        project.close()


def test_spatial_synthesis_reports_unlocated_lines_without_faking_coordinates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "matrix.csv"
    np.savetxt(source, np.arange(20, dtype=np.float32).reshape(5, 4), delimiter=",")
    temporary = IngestService.open_temporary(source)
    project = IngestService.formalize(temporary, tmp_path / "formal", name="Spatial")
    temporary.close()
    try:
        synthesis = SpatialSynthesisService(project).build()
        assert synthesis["summary"]["located_line_count"] == 0
        assert synthesis["summary"]["unlocated_line_count"] == 1
        assert synthesis["tracks"] == []
    finally:
        project.close()


def test_spatial_synthesis_handles_metadata_only_line_records(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "project", name="Spatial")
    try:
        project.add_line(LineRecordV1(line_id="L001", name="planned line"))
        synthesis = SpatialSynthesisService(project).build()
        assert synthesis["summary"]["located_line_count"] == 0
        assert synthesis["summary"]["unlocated_line_count"] == 1
        assert synthesis["unlocated_lines"][0]["reason"] == "缺少主雷达数据"
    finally:
        project.close()
