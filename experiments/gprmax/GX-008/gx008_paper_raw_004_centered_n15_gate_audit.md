#!/usr/bin/env markdown
# GX-008-PAPER-RAW-004 Centered N15 Gate Audit

## Date
2026-05-25 20:37:08

## Branch
main

## Base commit
ca95799528b6c7ebe11d0f7a5a66861f188f9d52

## Remote verification
- `git rev-parse HEAD` = `ca95799528b6c7ebe11d0f7a5a66861f188f9d52`
- `git rev-parse origin/main` = `ca95799528b6c7ebe11d0f7a5a66861f188f9d52`
- `git ls-remote origin main` = `ca95799528b6c7ebe11d0f7a5a66861f188f9d52`

## scene_029 recap
- Scene: `scene_029_gssi_ey_depth03_raw_gate_n31`
- depth03 raw-only gate at n15 completed in prior task.
- Target center: `x=0.50`.
- scene_029 scan start used `antenna_like_GSSI_1500(0.35 + (run-1)*0.01, ...)`.

## Why scene_029 n15 may be invalid as depth-failure evidence
- Prior GSSI n15 trajectory evidence showed early n15 aperture can remain left-biased (example from previous GSSI gate: rx up to about `x=0.458`).
- Therefore `scene_029` n15 visual weakness cannot be treated as depth03 failure unless centered coverage is explicitly verified.

## scene_031 design
- Added scene: `scene_031_gssi_ey_depth03_centered_n15_raw_gate`
- Based on: `scene_029_gssi_ey_depth03_raw_gate_n31`
- Single change: antenna scan start shifted to center n15 aperture over target center.
- Raw model command:
  - `antenna_like_GSSI_1500(0.462 + (current_model_run - 1) * 0.01, 0.075, 0.05, resolution=0.002)`
- Kept unchanged: depth03, radius 0.03, soil/material, antenna type, component Ey, domain, dx, scan step.

## Dry-run result
- `campaign_status: ready`
- `scene_031_gssi_ey_depth03_centered_n15_raw_gate: ready`
- `invalid_count: 0`

## GPU wrapper result
- `scripts/run_gprmax_gpu_env.bat --check`: pass
- `scripts/run_gprmax_gpu_env.bat --smoke`: pass (known non-blocking decode-thread noise after output)

## Raw n15 result
- Command scope: raw-only, no background
- Scene: `scene_031_gssi_ey_depth03_centered_n15_raw_gate`
- status: success
- return_code: 0
- runtime_seconds: `420.471`
- actual output count: `15`
- manifest:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008/scene_031_gssi_ey_depth03_centered_n15_raw_gate/raw_with_target/run_manifest.json`

## Actual n1/n8/n15 positions
From `.out` metadata (rx1/src1 Position):
- n1:  rx `[0.43, 0.072, 0.054]`, src `[0.49, 0.072, 0.054]`
- n8:  rx `[0.50, 0.072, 0.054]`, src `[0.56, 0.072, 0.054]`
- n15: rx `[0.57, 0.072, 0.054]`, src `[0.63, 0.072, 0.054]`
- target center x: `0.50`

## Target center coverage check
- rx coverage range: `0.43 -> 0.57`
- middle trace rx x: `0.50`
- `scan_covers_target_center = true`
- Conclusion: scene_031 n15 is a valid centered gate for depth03 visual judgment.

## Ey conversion status
- Raw-only Ey conversion completed from existing `.out` files.
- Shape: `[3636, 15]`
- Component: `Ey`
- No background conversion in this task.

## Raw visual output paths
Scratch root:
- `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/scratch/GX-008_centered_n15_raw_visual/scene_031_gssi_ey_depth03_centered_n15_raw_gate/`

Generated:
- `raw_full_percentile_1_99.png`
- `raw_surface_muted_display_only.png`
- `raw_trace_normalized_display_only.png`
- `raw_crop_best_candidate.png`
- `raw_centered_gate_summary.md`
- `raw_bscan_Ey.npy`
- `raw_bscan_Ey.csv`

## Comparison with scene_029
- scene_031 explicitly centers n15 aperture over target x=0.50.
- scene_029 did not include this explicit centered gate evidence.
- Visual result in scene_031 remains background-band dominant; target cue is still weak/limited.
- Therefore depth03 weakness is less likely to be only aperture-centering artifact.

## Visual assessment
- Top surface/background band: clearly visible.
- Single-target response: weak / inconclusive.
- Hyperbola-like trend: weak, not clearly separable.
- Centering improved geometric validity of the gate, but did not produce strong raw target visibility at n15.

## n31 decision
- Deferred in this task.
- Suggested only after deciding next single-variable gate order (depth03 centered n31 vs radius gate).

## Background decision
- Not run by design (raw-only task boundary).

## Files deliberately excluded
- No MyGPR-Evidence git operations.
- No generated `.out/.h5/.vti/.vtk/.vtu/.csv/.npy/.png` committed.
- No background/clutter-free outputs generated.

## Claim boundary
- centered raw-only scan gate only
- no background
- no clutter-free
- not exact replication
- not paper benchmark
- not AutoTune evaluation
- not field validation

## Recommended next task
- `GX-008-PAPER-RAW-005-CENTERED-DEPTH03-N31-RAW` (keep all variables fixed and extend centered aperture to n31), or if compute budget is tight, run `radius` single-variable gate next at centered n15.
