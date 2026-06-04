# GX-008-HYPERBOLA-RUN-001 Audit

## Date
- 2026-05-25

## Branch
- main

## Base commit
- 0b8122fd823ca3de03f15ed48cf9578f92fec4c5

## Remote verification
- `git rev-parse HEAD` = `0b8122fd823ca3de03f15ed48cf9578f92fec4c5`
- `git rev-parse origin/main` = `0b8122fd823ca3de03f15ed48cf9578f92fec4c5`
- `git ls-remote origin main` = `0b8122fd823ca3de03f15ed48cf9578f92fec4c5`

## Observed issue in scene_001~003
- Existing scene_001/002/003 target-response previews are mainly horizontal band-like; no clear local hyperbola apex/limbs.

## Root cause finding
- Scan direction is x-axis (`#src_steps: 0.005 0 0`, `#rx_steps: 0.005 0 0`).
- `#rx` start is x=0.25 for scene_001/002/003, with expected runs 41.
- Aperture end x = `0.25 + 0.005*(41-1) = 0.45`.
- Targets are centered at x=1.00, so scan aperture does not cover target center.
- Conclusion: scene_001/002/003 are valid paired-pipeline diagnostics, but not suitable as hyperbola benchmark scenes.

## Scene_001~003 scan aperture table
| Scene | rx_start_x | rx_step_x | expected_num_runs | rx_end_x | target center (x,y,z) | scan covers target center |
|---|---:|---:|---:|---:|---|---|
| scene_001_flat_dry_sand_pec_shallow | 0.25 | 0.005 | 41 | 0.45 | (1.00, 0.50, 0.16) | false |
| scene_002_flat_damp_sand_pec_shallow | 0.25 | 0.005 | 41 | 0.45 | (1.00, 0.50, 0.17) | false |
| scene_003_flat_dry_sand_pvc_shallow | 0.25 | 0.005 | 41 | 0.45 | (1.00, 0.50, 0.16) | false |

## Scene_007 design
- Scene: `scene_007_flat_dry_sand_pec_sphere_shallow`
- Role: `hyperbola_oriented_local_target_diagnostic`
- Domain/grid/time_window kept aligned with GX-008 baseline:
  - `#domain: 2.0 1.0 0.5`
  - `#dx_dy_dz: 0.01 0.01 0.01`
  - `#time_window: 18e-9`
- Scan design:
  - `#rx: 0.55 0.50 0.10`
  - `#src_steps/#rx_steps: 0.005 0 0`
  - expected runs: 201
  - aperture x range: 0.55 -> 1.55 (covers target center x=1.00)
- Target:
  - `#sphere: 1.00 0.50 0.20 0.04 pec`

## Why scene_007 should produce hyperbola
- The receiver/source aperture now crosses the local target center x=1.00.
- Local compact target geometry (`sphere`, radius 0.04 m) is chosen to avoid laterally continuous response and increase chance of hyperbola-like response.

## Sphere syntax decision / fallback
- Decision: keep `#sphere` (no fallback to box needed in this run).
- Evidence: dry-run passed and real GPU run completed for both raw/background without syntax failure.

## Campaign dry-run result
- Command: `python scripts\gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --dry-run`
- Result:
  - campaign_status: ready
  - total_scenes: 7
  - ready_count: 7
  - invalid_count: 0
  - scene_007 included and ready

## Pair contract result
- scene_007 raw/background static validation: `pairable`.
- Checked items:
  - domain/grid/time_window/waveform/source/rx/src_steps/rx_steps/material region all consistent.
  - target object only in raw; background has no target object.
  - expected pairing formula: `target_response = raw - background`.

## GPU wrapper check result
- `scripts\run_gprmax_gpu_env.bat --check`: passed

## GPU smoke result
- `scripts\run_gprmax_gpu_env.bat --smoke`: passed

## Raw run status
- success
- return_code: 0
- requested_num_runs: 201
- runtime_seconds: 1102.2335991
- manifest:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_007_flat_dry_sand_pec_sphere_shallow\raw_with_target\run_manifest.json`

## Background run status
- success
- return_code: 0
- requested_num_runs: 201
- runtime_seconds: 1138.3528851999945
- manifest:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_007_flat_dry_sand_pec_sphere_shallow\background_only\run_manifest.json`

## Requested num-runs
- 201

## Actual raw/background output count
- raw numbered outputs: 201
- background numbered outputs: 201

## Runtime summary
- raw + background total runtime: ~2240.59 s (~37.34 min)

## Stdout/stderr paths
- raw stdout/stderr:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_007_flat_dry_sand_pec_sphere_shallow\raw_with_target\stdout.log`
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_007_flat_dry_sand_pec_sphere_shallow\raw_with_target\stderr.log`
- background stdout/stderr:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_007_flat_dry_sand_pec_sphere_shallow\background_only\stdout.log`
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_007_flat_dry_sand_pec_sphere_shallow\background_only\stderr.log`

## Conversion status
- success
- summary:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_007_flat_dry_sand_pec_sphere_shallow\convert_summary_201.json`
- converted shape:
  - raw: [936, 201]
  - background: [936, 201]

## Pairing status
- success
- output root:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_007_flat_dry_sand_pec_sphere_shallow\paired_outputs`
- target_response shape: [936, 201]

## Metrics status
- success (`paired_metrics.json` generated)
- standardized metrics included in pairing output.
- warning present:
  - `roi_missing_ranges` (current ROI draft uses `trace_window/depth_window`; pairing ROI metric expects `trace_range/sample_range`).

## Preview status
- success
- generated:
  - `raw_preview.png`
  - `background_preview.png`
  - `target_response_preview.png`
  - `paired_preview_panel.png`
  - `paired_target_response_report.md`
  - `paired_report_summary.json`

## Visual hyperbola check result
- Result: **no clear典型双曲线** observed in current `target_response_preview.png` / `paired_preview_panel.png`.
- Current scene_007 therefore cannot be claimed as achieved hyperbola benchmark scene yet.
- Next tuning likely needed on one or more of:
  - target depth / size
  - Tx/Rx height and offset
  - waveform center frequency
  - aperture length / step
  - object geometry/material contrast

## Generated local artifact paths
- Local run and conversion/pairing outputs are under:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_007_flat_dry_sand_pec_sphere_shallow\`

## Files changed (source repo)
- `experiments/gprmax/GX-008/campaign_draft.yaml`
- `experiments/gprmax/GX-008/models/scene_007_flat_dry_sand_pec_sphere_shallow/raw_with_target.in`
- `experiments/gprmax/GX-008/models/scene_007_flat_dry_sand_pec_sphere_shallow/background_only.in`
- `experiments/gprmax/GX-008/models/scene_007_flat_dry_sand_pec_sphere_shallow/materials.txt`
- `experiments/gprmax/GX-008/models/scene_007_flat_dry_sand_pec_sphere_shallow/roi_draft.json`
- `experiments/gprmax/GX-008/models/scene_007_flat_dry_sand_pec_sphere_shallow/scene_manifest_draft.json`
- `experiments/gprmax/GX-008/gx008_hyperbola_run_001_audit.md`

## Files deliberately excluded
- Not committed:
  - `*.out`, `*.h5`, `*.vti`, `*.vtk`, `*.vtu`
  - generated `*.csv`, `*.npy`, `*.png`
  - MyGPR-Evidence git operations
  - scratch/temporary outputs

## Repository hygiene
- Only source model/campaign/audit files are intended for commit in this task.
- Existing unrelated untracked `gprmax/` source-repo folder is left untouched.

## Claim boundary
- scene_007 synthetic hyperbola-oriented diagnostic only
- not Evidence artifact yet
- not AutoTune evaluation
- not field validation
- not paper-candidate benchmark
- not CR-Net/CLT-GPR replication

## Recommended next task
- `GX-008-HYPERBOLA-TUNE-001`: iterate scene_007 geometry/scan settings to produce a clear hyperbola before any benchmark claim.
