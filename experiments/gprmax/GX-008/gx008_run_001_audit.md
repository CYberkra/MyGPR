#!/usr/bin/env markdown
# GX-008-RUN-001 Audit

Date: 2026-05-23  
Branch: `main`  
Base commit: `e323e4ab209552ab070419db44db28098a146d39`

## Remote verification

Commands:

- `git rev-parse HEAD`
- `git rev-parse origin/main`
- `git ls-remote origin main`

Result:

- all matched `e323e4ab209552ab070419db44db28098a146d39` before running.

## Scene run

Target scene:

- `scene_001_flat_dry_sand_pec_shallow`

Variants executed:

1. `raw_with_target`
2. `background_only`

## Commands run

Pre-check:

- `python scripts\preflight_check.py`
- `scripts\run_gprmax_gpu_env.bat --check`
- `scripts\run_gprmax_gpu_env.bat --smoke`

Raw run:

```bat
scripts\run_gprmax_gpu_env.bat -- python scripts\gprmax_campaign_runner.py ^
  --campaign experiments/gprmax/GX-008/campaign_draft.yaml ^
  --run-scene scene_001_flat_dry_sand_pec_shallow ^
  --variant raw_with_target ^
  --num-runs 41 ^
  --gpu-device 0 ^
  --gprmax-python E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe ^
  --timeout-seconds 1800
```

Background run:

```bat
scripts\run_gprmax_gpu_env.bat -- python scripts\gprmax_campaign_runner.py ^
  --campaign experiments/gprmax/GX-008/campaign_draft.yaml ^
  --run-scene scene_001_flat_dry_sand_pec_shallow ^
  --variant background_only ^
  --num-runs 41 ^
  --gpu-device 0 ^
  --gprmax-python E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe ^
  --timeout-seconds 1800
```

Note:

- one typo command was attempted and rejected (`scene_001_flat_damp_sand_pec_shallow` not found); then rerun with correct scene id succeeded.

## GPU wrapper check result

- `--check`: success
- `--smoke`: success
- diagnostic readiness in smoke output: `gprmax_runtime_gpu_ready = true`

## Raw run status

- status: `success`
- return_code: `0`
- runtime_seconds: `222.46100640000077`
- requested_num_runs: `41`
- run_manifest:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\raw_with_target\run_manifest.json`
- stdout:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\raw_with_target\stdout.log`
- stderr:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\raw_with_target\stderr.log`

## Background run status

- status: `success`
- return_code: `0`
- runtime_seconds: `238.244924300001`
- requested_num_runs: `41`
- run_manifest:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\background_only\run_manifest.json`
- stdout:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\background_only\stdout.log`
- stderr:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\background_only\stderr.log`

## Requested num-runs

- `41`

## Actual numbered output count

- raw output series count: `41` (`raw_with_target1.out` ... `raw_with_target41.out`)
- background output series count: `41` (`background_only1.out` ... `background_only41.out`)

## Runtime summary

- wrapper + GPU environment remained stable for both runs.
- both run manifests recorded `gpu_requested=true`, `gpu_device_ids=[0]`.
- host MyGPR Python lacked pycuda import, but external runtime python handled GPU run successfully.

## Conversion status

- conversion script availability checked:
  - `python scripts\gprmax_campaign_convert_scene001.py --help`
- script is GX-007-named but parameterized; used for GX-008 scene_001 conversion.
- conversion command succeeded and produced:
  - raw shape `[936, 41]`
  - background shape `[936, 41]`
  - summary JSON:
    - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\convert_summary_41.json`

## Pairing status

- pairing command executed but returned `invalid`.
- error:
  - `raw_missing`
  - `background_missing`
- this occurred even though converted files exist and `Test-Path` returns true.
- likely root cause: pair CLI path resolution/normalization bug in current implementation for these absolute paths.

## Preview status

- preview command executed but returned `invalid` for same missing-file reason (`raw_missing/background_missing`).

## Generated local artifact paths

- raw/background run outputs under:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\raw_with_target\`
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_001_flat_dry_sand_pec_shallow\background_only\`
- conversion outputs under:
  - `...\raw_with_target\converted\`
  - `...\background_only\converted\`
- pairing/preview JSON attempt outputs under:
  - `...\paired_outputs\`

## Files deliberately excluded

Not committed to MyGPR source repo:

- `.out`
- `.h5`
- `.vti`
- `.vtk`
- `.vtu`
- generated `.csv/.npy/.png`
- runtime logs in MyGPR-Evidence path

## Repository hygiene

- MyGPR source commit in this task includes audit doc only.
- No MyGPR-Evidence git operations were performed.

## Claim boundary

- GX-008 scene_001 only
- synthetic paired run
- not Evidence artifact yet
- not AutoTune evaluation
- not field validation
- not paper-candidate benchmark

## Recommended next task

- `GX-RUN-CONVERT-GENERIC-001` (or `GX-008-RUN-FIX-001`) to fix pair/preview CLI file-path resolution so converted GX-008 arrays can complete target_response + preview/report generation.
