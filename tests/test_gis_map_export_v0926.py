from __future__ import annotations

from pathlib import Path

import pytest

from core.field_project_store import FieldLineRecord, FieldProjectStore
from core.gis_map_export import export_project_plan_map
from core.job_manager import JobCancelled
from core.trajectory_model import TrajectoryModel, TrajectoryPoint


def _project(tmp_path: Path) -> FieldProjectStore:
    store = FieldProjectStore.create_empty(
        tmp_path / "project",
        name="Field GIS",
        coordinate_system="EPSG:32648",
    )
    store.upsert_line(FieldLineRecord("L01", "Line 1"))
    store.save_trajectory(
        "L01",
        TrajectoryModel(
            [
                TrajectoryPoint(distance_m=0.0, x=500000.0, y=3300000.0, z=100.0),
                TrajectoryPoint(distance_m=10.0, x=500010.0, y=3300004.0, z=101.0),
            ]
        ),
    )
    return store


def test_headless_plan_map_export_writes_real_project_coordinates(tmp_path: Path) -> None:
    store = _project(tmp_path)
    output = store.root / "spatial" / "plan.png"
    progress = []

    result = export_project_plan_map(
        store,
        output,
        selected_line="L01",
        progress_callback=lambda current, total, message: progress.append((current, total, message)),
    )

    assert result == output
    assert output.exists() and output.stat().st_size > 1000
    assert progress[-1][0] == progress[-1][1]


def test_plan_map_cancellation_does_not_replace_previous_file(tmp_path: Path) -> None:
    store = _project(tmp_path)
    output = store.root / "spatial" / "plan.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"previous")

    with pytest.raises(JobCancelled):
        export_project_plan_map(store, output, cancel_requested=lambda: True)

    assert output.read_bytes() == b"previous"
    assert not list(output.parent.glob(".plan.*.tmp.png"))
