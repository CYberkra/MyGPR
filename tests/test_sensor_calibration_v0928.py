import numpy as np
from core.sensor_sync import SensorCalibrationProfile, SensorSyncConfig, synchronize_sensor_streams


def test_calibration_profile_applies_and_exports_uncertainty():
    result = synchronize_sensor_streams(
        trace_timestamps_s=np.array([0.0, 1.0, 2.0]),
        rtk_payload={
            "timestamp_s": [0.0, 1.0, 2.0],
            "local_x_m": [0.0, 1.0, 2.0],
            "local_y_m": [0.0, 0.0, 0.0],
            "local_z_m": [10.0, 10.0, 10.0],
            "speed_mps": [1.0, 1.0, 1.0],
            "rtk_fix_type": [4, 4, 4],
        },
        config=SensorSyncConfig(calibration_profile=SensorCalibrationProfile(profile_id="CAL-1", position_sigma_m=0.02)),
    )
    assert result.diagnostics.calibration_profile_id == "CAL-1"
    assert "position_sigma_m" in result.trace_metadata
    assert np.all(np.isfinite(result.trace_metadata["position_sigma_m"]))
