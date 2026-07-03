#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np

from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet, load_gpr_dataset
from core.gpr_processing_pipeline import ProcessingParams, process_gpr_dataset
from core.target_detection import detect_targets
from core.trajectory_model import TrajectoryModel


def test_gpr_dataset_csv_npy_and_processing(tmp_path: Path) -> None:
    matrix = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    csv_path = tmp_path / "line.csv"
    np.savetxt(csv_path, matrix, delimiter=",")
    ds = load_gpr_dataset(csv_path, line_id="L99", length_m=12.5)
    assert ds.matrix.shape == (64, 32)
    assert round(ds.length_m, 2) == 12.5
    processed = process_gpr_dataset(ds, ProcessingParams(background_window=3, gain_factor=1.2))
    assert processed.matrix.shape == ds.matrix.shape
    assert processed.format_name == "processed-pipeline-v1"


def test_project_store_gpr_and_trajectory_exports(tmp_path: Path) -> None:
    store = FieldProjectStore.create(tmp_path / "project")
    store.ensure_demo_gpr_artifacts("L03")
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


def test_target_detection_contract() -> None:
    ds = GPRDataSet.synthetic("L03", rows=120, cols=160, length_m=80.0)
    processed = process_gpr_dataset(ds)
    candidates = detect_targets(processed, max_targets=3)
    assert 1 <= len(candidates) <= 3
    payload = candidates[0].to_target_dict()
    assert payload["line_id"] == "L03"
    assert payload["mileage"] >= 0
