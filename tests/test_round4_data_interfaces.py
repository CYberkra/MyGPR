#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np

from core.field_project_store import FieldProjectStore
from tests.field_project_test_utils import create_test_project
from core.gpr_data_model import GPRDataSet, load_gpr_dataset
from core.trajectory_model import TrajectoryModel


def test_gpr_dataset_csv_npy(tmp_path: Path) -> None:
    matrix = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    csv_path = tmp_path / "line.csv"
    np.savetxt(csv_path, matrix, delimiter=",")
    ds = load_gpr_dataset(csv_path, line_id="L99", length_m=12.5)
    assert ds.matrix.shape == (64, 32)
    assert round(ds.length_m, 2) == 12.5


def test_project_store_gpr_and_trajectory_exports(tmp_path: Path) -> None:
    store = create_test_project(tmp_path / "project")
    seeded = GPRDataSet.synthetic_basal_interface("L03", rows=96, cols=128, length_m=90.0)
    store.save_gpr_dataset("L03", seeded)
    store.save_trajectory("L03", TrajectoryModel.demo(length_m=90.0))
    ds = store.load_gpr_dataset("L03")
    assert ds.matrix.ndim == 2
    trajectory = store.load_trajectory("L03")
    point = trajectory.interpolate(50.0)
    assert point.x > 451000
    targets = store.default_targets("L03")[:1]
    targets[0].pop("x", None)
    targets[0].pop("y", None)
    store.save_targets("L03", targets)
    spatial = (store.root / "spatial" / "L03_targets_xy.csv").read_text(encoding="utf-8-sig")
    assert "451" in spatial
