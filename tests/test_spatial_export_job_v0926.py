from __future__ import annotations

from pathlib import Path

import pytest

from core.field_project_store import FieldLineRecord, FieldProjectStore
from core.job_manager import JobCancelled
from core.trajectory_model import TrajectoryModel, TrajectoryPoint


def test_spatial_export_cancellation_keeps_previous_valid_file(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="同步空间成果")
    store.upsert_line(FieldLineRecord("L01", "一号测线"))
    store.save_trajectory(
        "L01",
        TrajectoryModel(
            [
                TrajectoryPoint(distance_m=float(i), x=float(i), y=0.0, z=100.0)
                for i in range(3000)
            ]
        ),
    )
    destination = store.root / "spatial" / "project_spatial_coordinates.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("previous-valid\n", encoding="utf-8")

    with pytest.raises(JobCancelled):
        store.export_project_spatial_coordinates(cancel_requested=lambda: True)

    assert destination.read_text(encoding="utf-8") == "previous-valid\n"
    assert not list(destination.parent.glob(".project_spatial_coordinates.csv.*.tmp"))
