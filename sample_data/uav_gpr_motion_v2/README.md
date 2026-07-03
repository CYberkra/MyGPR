# UAV-GPR Motion V2 Synthetic Package

This package is synthetic. It is for MyGPR software-contract testing before real field data is available.

## Files

- `main.csv`: airborne stacked GPR CSV. Columns are `longitude, latitude, ground_elevation_m, amplitude, flight_height_m, trace_timestamp_s`.
- `rtk.csv`: RTK sidecar with `timestamp_s, longitude, latitude, ground_elevation_m, flight_height_m, rtk_fix_type, satellites, hdop`.
- `imu.csv`: IMU sidecar with `timestamp_s, roll_deg, pitch_deg, yaw_deg, angular_rate_x, angular_rate_y, angular_rate_z`.
- `altimeter.csv`: NAR15-style height sidecar with `timestamp_s, height_agl_m, height_source, snr, target_count, valid`.
- `batch_motion_v2.json`: CLI batch config that runs `motion_compensation_v2`.

## CLI Check

```bash
python cli_batch.py validate --config config/uav_gpr_motion_v2_synthetic.json
python cli_batch.py run --config config/uav_gpr_motion_v2_synthetic.json --force
```

## Field Trip Acceptance Checklist

- The main GPR CSV must preserve one timestamp per trace, preferably as `trace_timestamp_s`.
- RTK records must include timestamp, longitude, latitude, fix type, satellites, and HDOP.
- IMU records must include timestamp, roll, pitch, and yaw in degrees.
- NAR15/altimeter records must include timestamp and AGL height. Keep `valid`, `target_count`, and SNR when available.
- Record lever arms between RTK antenna, IMU, radar antenna phase center, and altimeter beam center.
- Confirm all devices share a consistent time base or store enough information to align timestamps after acquisition.
