# gprMax Motion Compensation Validation

## Source

- Scenario: `uav_pipe_gain_workflow_bscan_smoke`
- gprMax manifest: `D:\CDUT-UavGPR-Controller\MyGPR\output\gprmax_native_longline\uav_pipe_gain_workflow_bscan_smoke_20260519_012145\uav_pipe_gain_workflow_bscan_smoke_manifest.json`
- Primary .out: `D:\CDUT-UavGPR-Controller\MyGPR\output\gprmax_native_longline\uav_pipe_gain_workflow_bscan_smoke_20260519_012145\uav_pipe_gain_workflow_bscan_smoke_merged.out`
- Source ground truth: `D:\CDUT-UavGPR-Controller\MyGPR\output\gprmax_native_longline\uav_pipe_gain_workflow_bscan_smoke_20260519_012145\ground_truth.yaml`
- Shape: `[2037, 90]`
- Original gprMax shape: `[2037, 90]`
- Derived long-line scaffold: `False`
- Trace interval: `0.01` m
- Time window: `24.02289441979553` ns
- This run uses an exaggerated demo/stress UAV motion profile for visibility and is not a field-flight baseline.

## Shapes

| data | shape |
| --- | --- |
| gprMax source | `[2037, 90]` |
| motion-injected raw | `[2037, 90]` |
| four atomic steps | `[2037, 120]` |
| motion_compensation_v2 | `[2037, 123]` |

## Target / ROI

- Target geometry: `{'id': 'target_0', 'type': 'pipe', 'material': 'pec', 'depth_m': 0.33, 'center_x_m': 0.62, 'center_y_m': 0.22, 'radius_m': 0.035, 'roi': {'time_start_idx': 760, 'time_end_idx': 883, 'dist_start_idx': 40, 'dist_end_idx': 53}}`
- Target ROI: `{'time_start_idx': 760, 'time_end_idx': 883, 'dist_start_idx': 40, 'dist_end_idx': 53}`

## Workflow

- Atomic: `trajectory_smoothing -> motion_compensation_attitude -> motion_compensation_speed -> motion_compensation_height`
- Unified: `motion_compensation_v2`

## Metrics

| metric | value |
| --- | ---: |
| `spacing_std_before_m` | 0.00286412 |
| `spacing_std_atomic_m` | 0.000354912 |
| `spacing_std_atomic_after_speed_m` | 0.000354912 |
| `spacing_std_v2_m` | 0.000359328 |
| `trace_spacing_cv_before` | 0.251014 |
| `trace_spacing_cv_atomic` | 0.0353756 |
| `trace_spacing_cv_v2` | 0.0358158 |
| `max_gap_ratio_before` | 1.48138 |
| `max_gap_ratio_atomic` | 1.38428 |
| `max_gap_ratio_v2` | 1.39397 |
| `raw_vs_source_rms` | 0.23467 |
| `atomic_vs_source_rms` | 0.0483498 |
| `v2_vs_source_rms` | 0.0110029 |
| `atomic_rms_delta_from_raw` | 0.23739 |
| `v2_rms_delta_from_raw` | 0.232881 |
| `target_ratio_raw` | 0.0869308 |
| `target_ratio_atomic` | 0.0572553 |
| `target_ratio_v2` | 0.0550667 |
| `ridge_rmse_samples_raw` | 44.6801 |
| `ridge_rmse_samples_atomic` | 5.07718 |
| `ridge_rmse_samples_v2` | 3.15394 |
| `reflector_flatness_metric_raw` | 8.1698 |
| `reflector_flatness_metric_atomic` | 3.22988 |
| `reflector_flatness_metric_v2` | 3.04991 |
| `target_apex_error_samples_raw` | 48 |
| `target_apex_error_samples_atomic` | 5 |
| `target_apex_error_samples_v2` | 0 |
| `target_roi_energy_preservation_raw` | 1.74697 |
| `target_roi_energy_preservation_atomic` | 1.11315 |
| `target_roi_energy_preservation_v2` | 1.01829 |
| `resample_spacing_m` | 0.01 |
| `target_traces` | 123 |

## V2 Resampling Explanation

- V2 performed equal-distance resampling: `True`
- Source/raw trace count: `90`
- V2 target_traces: `123`
- V2 resample_spacing_m: `0.01`
- RMS and ROI metrics are computed against the gprMax source B-scan resampled to the processed trace axis when trace counts differ.
- Therefore a V2 shape mismatch is expected when equal-distance resampling changes the trace count; it is not treated as a processing error.

## Quality Flags / Runtime Warnings

- Atomic quality_flags: `[]`
- V2 quality_flags: `[]`
- Atomic runtime_warnings: `[]`
- V2 runtime_warnings: `[]`

## Validation Notes

- `info` `atomic_spacing_improved_after_reorder`: The final atomic trace spacing CV is lower than raw after running attitude/APC before speed compensation.

## Artifacts

- Main CSV: `D:\CDUT-UavGPR-Controller\MyGPR\output\gprmax_motion_validation\native_90trace_uav_pipe_visible_motion\main.csv`
- Comparison image: `D:\CDUT-UavGPR-Controller\MyGPR\output\gprmax_motion_validation\native_90trace_uav_pipe_visible_motion\bscan_motion_validation_comparison.png`
- Paper comparison image: `D:\CDUT-UavGPR-Controller\MyGPR\output\gprmax_motion_validation\native_90trace_uav_pipe_visible_motion\paper_motion_validation_comparison.png`
- Raw 3D preview: `D:\CDUT-UavGPR-Controller\MyGPR\output\gprmax_motion_validation\native_90trace_uav_pipe_visible_motion\raw_3d_preview.png`
- Motion V2 3D preview: `D:\CDUT-UavGPR-Controller\MyGPR\output\gprmax_motion_validation\native_90trace_uav_pipe_visible_motion\motion_v2_3d_preview.png`
- Copied source artifacts: `{'source_manifest': 'D:\\CDUT-UavGPR-Controller\\MyGPR\\output\\gprmax_motion_validation\\native_90trace_uav_pipe_visible_motion\\source_manifest.json', 'source_ground_truth': 'D:\\CDUT-UavGPR-Controller\\MyGPR\\output\\gprmax_motion_validation\\native_90trace_uav_pipe_visible_motion\\source_ground_truth.yaml', 'source_model_in': 'D:\\CDUT-UavGPR-Controller\\MyGPR\\output\\gprmax_motion_validation\\native_90trace_uav_pipe_visible_motion\\source_model_in.in'}`

## Current Limitation

- 这里的地下波场来自 gprMax，但 UAV 运动扰动是 MyGPR 侧可控注入；它用于验证运动补偿链路，不代表实测外业结论。
- 短测线 gprMax 数据只能做 smoke；论文展示建议后续使用更长测线和更完整双曲线。
