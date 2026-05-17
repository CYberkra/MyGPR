# UAV-GPR Motion Demo v1

This is a synthetic UAV-GPR motion compensation demonstration dataset for MyGPR.
It is not field evidence and must not be used as an external geological
conclusion.

## Expected Visual Result

- `main.csv` is an airborne stacked CSV that reshapes to a 160 x 180 B-scan.
- The B-scan contains shallow layered reflections, a clear pipe/cylinder-like
  hyperbola, mild noise, and weak striping/background components.
- `rtk.csv` contains a lightly curved and non-equidistant UAV trajectory.
- `imu.csv` contains small roll/pitch/yaw variations.
- `altimeter.csv` contains NAR15-style AGL height variations around 0.08-0.16 m.
- After `motion_compensation_v2`, the current 3D curtain should show a slightly
  more consistent top interface/target position than the raw curtain.

## Files

- `main.csv`: stacked UAV-GPR CSV with columns
  `longitude, latitude, ground_elevation_m, amplitude, flight_height_m, trace_timestamp_s`.
- `trace_timestamps.csv`: one timestamp per trace for sidecar synchronization checks.
- `rtk.csv`: RTK sidecar with longitude/latitude and local xyz fields.
- `imu.csv`: IMU sidecar with roll/pitch/yaw.
- `altimeter.csv`: height sidecar with `height_agl_m`, SNR, target count and validity.
- `manifest.json`: dataset contract and expected target notes.
- `metadata.json`: compact generation parameters and target ROI.
- `batch_motion_v2.json`: CLI config for a smoke run.

Recommended workflow: `motion_compensation_v2`.
