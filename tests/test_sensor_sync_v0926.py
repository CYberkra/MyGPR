from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.sensor_sync import SensorSyncConfig, save_sensor_sync_result, synchronize_sensor_streams


def _rtk_payload() -> dict[str, np.ndarray | str]:
    return {
        "source_kind": "rtk",
        "timestamp_s": np.array([10.0, 11.0, 12.0], dtype=np.float64),
        "longitude": np.array([104.0, 104.00001, 104.00002]),
        "latitude": np.array([30.0, 30.0, 30.0]),
        "local_x_m": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "local_y_m": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "local_z_m": np.array([100.0, 100.0, 100.0], dtype=np.float32),
        "flight_height_m": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "rtk_fix_type": np.array([4, 4, 5], dtype=np.int32),
        "satellites": np.array([18, 18, 16], dtype=np.int32),
        "hdop": np.array([0.6, 0.6, 0.9], dtype=np.float32),
    }


def _imu_payload() -> dict[str, np.ndarray | str]:
    return {
        "source_kind": "imu",
        "timestamp_s": np.array([10.0, 11.0, 12.0], dtype=np.float64),
        "roll_deg": np.zeros(3, dtype=np.float32),
        "pitch_deg": np.zeros(3, dtype=np.float32),
        "yaw_deg": np.full(3, 90.0, dtype=np.float32),
    }


def test_timestamp_sync_applies_offsets_lever_arm_and_records_quality() -> None:
    result = synchronize_sensor_streams(
        trace_timestamps_s=np.array([9.9, 10.9, 11.9], dtype=np.float64),
        rtk_payload=_rtk_payload(),
        imu_payload=_imu_payload(),
        config=SensorSyncConfig(
            rtk_time_offset_s=-0.1,
            imu_time_offset_s=-0.1,
            lever_arm_x_m=1.0,
            project_crs="EPSG:32648",
        ),
        line_id="L01",
    )

    metadata = result.trace_metadata
    assert result.diagnostics.rtk.coverage_ratio == 1.0
    assert result.diagnostics.imu.coverage_ratio == 1.0
    assert np.allclose(metadata["local_x_m"], [0.0, 1.0, 2.0], atol=1e-6)
    assert np.allclose(metadata["local_y_m"], [1.0, 1.0, 1.0], atol=1e-6)
    assert metadata["rtk_fix_type"].tolist() == ["固定解", "固定解", "浮动解"]
    assert metadata["alignment_status"].tolist() == ["aligned", "aligned", "aligned"]
    assert result.trajectory.points[1].trace_index == 1
    assert result.trajectory.points[1].yaw_deg == 90.0


def test_sync_does_not_silently_extrapolate_outside_sensor_coverage() -> None:
    result = synchronize_sensor_streams(
        trace_timestamps_s=np.array([9.0, 10.0, 11.0, 12.0, 13.0]),
        rtk_payload=_rtk_payload(),
        config=SensorSyncConfig(allow_extrapolation=False),
        line_id="L02",
    )

    metadata = result.trace_metadata
    assert np.isnan(metadata["local_x_m"][[0, -1]]).all()
    assert metadata["alignment_status"].tolist()[0] == "rtk_out_of_range"
    assert metadata["alignment_status"].tolist()[-1] == "rtk_out_of_range"
    assert result.diagnostics.rtk.coverage_ratio == 0.6
    assert any("未静默夹取" in warning for warning in result.diagnostics.warnings)


def test_sync_result_persists_manifest_trajectory_and_trace_metadata(tmp_path: Path) -> None:
    result = synchronize_sensor_streams(
        trace_timestamps_s=np.array([10.0, 11.0, 12.0]),
        rtk_payload=_rtk_payload(),
        imu_payload=_imu_payload(),
        line_id="L03",
    )
    paths = save_sensor_sync_result(result, tmp_path, basename="L03_sync")

    assert set(paths) == {"trajectory", "manifest", "trace_metadata"}
    assert all(path.exists() for path in paths.values())
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["schema"] == "mygpr.sensor_sync.v2"
    assert manifest["diagnostics"]["trace_count"] == 3
    with np.load(paths["trace_metadata"], allow_pickle=False) as payload:
        assert payload["trace_index"].tolist() == [0, 1, 2]


def test_sync_rejects_in_range_samples_when_nearest_time_residual_exceeds_limit() -> None:
    sparse_rtk = _rtk_payload()
    sparse_rtk["timestamp_s"] = np.array([10.0, 12.0, 14.0], dtype=np.float64)
    result = synchronize_sensor_streams(
        trace_timestamps_s=np.array([10.0, 11.0, 12.0]),
        rtk_payload=sparse_rtk,
        config=SensorSyncConfig(maximum_nearest_residual_s=0.25),
    )
    assert result.trace_metadata["alignment_status"].tolist() == ["aligned", "rtk_time_residual", "aligned"]
    assert np.isnan(result.trace_metadata["local_x_m"][1])
    assert result.diagnostics.rtk.accepted_ratio == pytest.approx(2 / 3)
