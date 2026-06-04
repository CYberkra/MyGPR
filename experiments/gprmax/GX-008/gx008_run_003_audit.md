#!/usr/bin/env markdown
# GX-008-RUN-003 Audit

Date: 2026-05-23  
Branch: `main`  
Base commit: `f08c134dd3caf1bc1d97726586533f518807cb01`

## Remote Verification

Commands:

- `git rev-parse HEAD`
- `git rev-parse origin/main`
- `git ls-remote origin main`

Result:

- all three matched `f08c134dd3caf1bc1d97726586533f518807cb01` before execution.

## Scene Run

- Scene: `scene_003_flat_dry_sand_pvc_shallow`
- Scope: raw/background paired synthetic run only

## Reason for Prioritizing scene_003

- `scene_001` baseline is dry sand + PEC shallow.
- `scene_003` is dry sand + PVC shallow.
- Main variable is target material (`PEC -> PVC`) while keeping soil/surface/depth class comparable.
- PVC target is weaker and useful for later over-suppression checks in background suppression diagnostics.
- Medium-depth scenes were intentionally deferred to avoid mixing material and depth factors.

## Commands Run

```bat
python scripts\gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --dry-run
scripts\run_gprmax_gpu_env.bat --check
scripts\run_gprmax_gpu_env.bat --smoke

scripts\run_gprmax_gpu_env.bat -- python scripts\gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --run-scene scene_003_flat_dry_sand_pvc_shallow --variant raw_with_target --num-runs 41 --gpu-device 0 --gprmax-python E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe --timeout-seconds 1800
scripts\run_gprmax_gpu_env.bat -- python scripts\gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --run-scene scene_003_flat_dry_sand_pvc_shallow --variant background_only --num-runs 41 --gpu-device 0 --gprmax-python E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe --timeout-seconds 1800

python scripts\gprmax_campaign_convert_scene001.py --raw-out D:\CDUT-UavGPR-Controller\MyGPR\experiments\gprmax\GX-008\models\scene_003_flat_dry_sand_pvc_shallow\raw_with_target.out --background-out D:\CDUT-UavGPR-Controller\MyGPR\experiments\gprmax\GX-008\models\scene_003_flat_dry_sand_pvc_shallow\background_only.out --raw-converted-dir D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\raw_with_target\converted --background-converted-dir D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\background_only\converted --raw-run-count 41 --background-run-count 41 --json D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\convert_summary_41.json

python scripts\gprmax_campaign_pair_converted.py --scene-root D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow --output-dir D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\paired_outputs --campaign-id GX-008_paired_synthetic_mini_benchmark --scene-id scene_003_flat_dry_sand_pvc_shallow --roi-json D:\CDUT-UavGPR-Controller\MyGPR\experiments\gprmax\GX-008\models\scene_003_flat_dry_sand_pvc_shallow\roi_draft.json --json D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\pair_preview_generic_41.json
```

## Dry-run Result

- `campaign_status: ready`
- `total_scenes: 6`
- `ready_count: 6`
- `invalid_count: 0`

## GPU Wrapper Check Result

- `--check`: passed (`cl`, `nvcc`, `nvidia-smi`, gprMax runtime python help OK)
- `--smoke`: passed (`gprmax_runtime_gpu_ready: true`, minimal GPU smoke success)

Note:
- `--smoke` output included host-side `UnicodeDecodeError` reader thread noise in diagnostic subprocess logs, but readiness JSON and smoke return status were successful.

## Raw Run Status

- status: `success`
- return_code: `0`
- runtime_seconds: `258.763`
- requested_num_runs: `41`
- actual raw output count (`raw_with_target*.out`): `41`
- run_manifest: `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\raw_with_target\run_manifest.json`
- stdout: `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\raw_with_target\stdout.log`
- stderr: `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\raw_with_target\stderr.log`

## Background Run Status

- status: `success`
- return_code: `0`
- runtime_seconds: `259.355`
- requested_num_runs: `41`
- actual background output count (`background_only*.out`): `41`
- run_manifest: `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\background_only\run_manifest.json`
- stdout: `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\background_only\stdout.log`
- stderr: `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\background_only\stderr.log`

## Runtime Summary

- Raw + background total runtime: ~`518.12s` (~8.64 minutes)

## Conversion Status

- conversion status: `success`
- output summary: `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\convert_summary_41.json`
- raw shape: `[936, 41]`
- background shape: `[936, 41]`
- shape match: `true`

## Pairing Status

- pairing status: `success`
- target_response generated:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\paired_outputs\target_response.npy`
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\paired_outputs\target_response.csv`
- validation summary:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\paired_outputs\paired_validation_summary.json`

## Metrics Status

- metrics status: `success`
- metrics file:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\paired_outputs\paired_metrics.json`
- key metric highlights:
  - `target_response_shape: [936, 41]`
  - `raw_background_rmse: 0.004104715243901884`
  - `raw_background_psnr: 89.5029388371074`
  - `target_to_background_energy_ratio: 6.091546651631798e-08`
- warning:
  - `roi_missing_ranges` (current ROI draft uses depth_window/trace_window fields, not sample_range/trace_range)

## Preview Status

- preview status: `success`
- generated:
  - `raw_preview.png`
  - `background_preview.png`
  - `target_response_preview.png`
  - `paired_preview_panel.png`
  - `paired_target_response_report.md`
  - `paired_report_summary.json`

## Generated Local Artifact Paths

- Scene root:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_003_flat_dry_sand_pvc_shallow\`
- Converted arrays:
  - `raw_with_target\converted\raw_bscan.csv/.npy`
  - `background_only\converted\background_bscan.csv/.npy`
- Pair summary:
  - `pair_preview_generic_41.json`
- Pair outputs:
  - `paired_outputs\*` (target_response, metrics, validation, previews, report)

## Files Deliberately Excluded

- No generated `.out/.h5/.vti/.vtk/.vtu` committed to MyGPR.
- No generated `.csv/.npy/.png` committed to MyGPR.
- No MyGPR-Evidence git operations in this task.

## Repository Hygiene

- Source repository commit scope limited to this audit document only.
- Generated artifacts remain in local evidence workspace paths.

## Claim Boundary

- GX-008 `scene_003` only
- synthetic paired run
- target material variation diagnostic (PVC shallow)
- not Evidence artifact yet
- not AutoTune evaluation
- not field validation
- not paper-candidate benchmark

## Recommended Next Task

- `GX-008-EVIDENCE-003` (curate and commit scene_003 paired diagnostic artifact into MyGPR-Evidence), then
- `AT-BG-004C-GX008-SCENE003-DIAGNOSTIC` (run background suppression diagnostic on scene_003 for weak-target behavior comparison).
