from __future__ import annotations

from pathlib import Path

import numpy as np

from core.gpr_io import extract_airborne_csv_payload
from core.trajectory_model import TrajectoryModel, TrajectoryPoint
from PythonModule.compensatingGain import GainPreview


def test_airborne_four_column_csv_uses_declared_amplitude_column() -> None:
    # Two traces, three samples per trace: lon, lat, ground_z, amplitude.
    rows = np.asarray(
        [
            [100.0, 30.0, 10.0, 1.0],
            [100.0, 30.0, 10.0, 2.0],
            [100.0, 30.0, 10.0, 3.0],
            [100.1, 30.1, 11.0, 4.0],
            [100.1, 30.1, 11.0, 5.0],
            [100.1, 30.1, 11.0, 6.0],
        ],
        dtype=np.float64,
    )
    matrix, metadata, header = extract_airborne_csv_payload(
        rows,
        {"a_scan_length": 3, "num_traces": 2, "total_time_ns": 60.0},
    )
    assert metadata is None
    assert header is not None
    np.testing.assert_array_equal(
        matrix,
        np.asarray([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32),
    )


def test_trajectory_diagnostics_uses_sorted_distance_axis() -> None:
    model = TrajectoryModel(
        [
            TrajectoryPoint(distance_m=5.0, x=0.0, y=0.0),
            TrajectoryPoint(distance_m=1.0, x=1.0, y=0.0),
            TrajectoryPoint(distance_m=9.0, x=2.0, y=0.0),
        ]
    )
    diagnostics = model.diagnostics()
    assert diagnostics["length_m"] == 8.0
    assert diagnostics["point_count"] == 3


def test_gain_preview_imports_reader_in_all_entry_points(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    np.savetxt(source, np.arange(12, dtype=float).reshape(4, 3), delimiter=",")
    result = GainPreview(
        infilename=str(source),
        Gainfunction=[0.0, 0.0, 0.0, 0.0],
        trace=1,
    )
    assert result["error_sign"] == 0
    assert result["Gaintrace"] == [1.0, 4.0, 7.0, 10.0]
