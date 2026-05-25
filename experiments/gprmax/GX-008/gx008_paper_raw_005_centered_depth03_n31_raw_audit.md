#!/usr/bin/env markdown
# GX-008-PAPER-RAW-005 Centered Depth03 N31 Raw Audit

## Date
2026-05-25 21:04:01

## Branch
main

## Base commit
515fc5215330cd0034982455651726ee3beaa172

## Remote verification
- `git rev-parse HEAD` = `515fc5215330cd0034982455651726ee3beaa172`
- `git rev-parse origin/main` = `515fc5215330cd0034982455651726ee3beaa172`
- `git ls-remote origin main` = `515fc5215330cd0034982455651726ee3beaa172`

## scene_031 recap
- `scene_031_gssi_ey_depth03_centered_n15_raw_gate` was centered and valid for target coverage:
  - n1 rx x = 0.43
  - n8 rx x = 0.50
  - n15 rx x = 0.57
- Raw result remained weak/inconclusive under n15 aperture.

## Why centered n31 was needed
- n15 centered gate removed scan-bias ambiguity but remained visually weak.
- Next conservative step is aperture-only expansion to n31 while keeping all physical variables fixed.

## scene_032 design
- Added scene: `scene_032_gssi_ey_depth03_centered_n31_raw_gate`
- Based on: `scene_031_gssi_ey_depth03_centered_n15_raw_gate`
- Changes from scene_031:
  - `expected_num_runs: 15 -> 31`
  - scan start adjusted to keep n31 centered: `antenna_like_GSSI_1500(0.382 + (current_model_run - 1) * 0.01, ...)`
- Unchanged from scene_031:
  - depth03, radius 0.03, dry sand Table II, PEC target, GSSI-like antenna, Ey component, domain, dx/dy/dz, scan step 0.01.

## Dry-run result
- `campaign_status: ready`
- `scene_032_gssi_ey_depth03_centered_n31_raw_gate: ready`
- `invalid_count: 0`

## GPU wrapper result
- `scripts/run_gprmax_gpu_env.bat --check`: pass
- `scripts/run_gprmax_gpu_env.bat --smoke`: pass (known non-blocking decode-thread noise after output)

## Raw n31 result
- Scene: `scene_032_gssi_ey_depth03_centered_n31_raw_gate`
- Variant: `raw_with_target`
- status: success
- return_code: 0
- runtime_seconds: `878.164`
- actual output count: `31`
- manifest:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008/scene_032_gssi_ey_depth03_centered_n31_raw_gate/raw_with_target/run_manifest.json`

## Actual n1/n16/n31 positions
From `.out` metadata (rx1/src1 Position):
- n1:  rx `[0.35, 0.072, 0.054]`, src `[0.41, 0.072, 0.054]`
- n16: rx `[0.50, 0.072, 0.054]`, src `[0.56, 0.072, 0.054]`
- n31: rx `[0.65, 0.072, 0.054]`, src `[0.71, 0.072, 0.054]`
- target center x: `0.50`

## Target center coverage check
- rx coverage range: `0.35 -> 0.65`
- middle trace rx x: `0.50`
- `scan_covers_target_center = true`
- scene_032 is geometrically valid for centered n31 aperture judgment.

## Ey conversion status
- Raw-only Ey conversion completed from scene_032 `.out` files.
- Shape: `[3636, 31]`
- Component: `Ey`
- No background conversion in this task.

## Raw visual output paths
Scratch root:
- `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/scratch/GX-008_centered_n31_raw_visual/scene_032_gssi_ey_depth03_centered_n31_raw_gate/`

Generated:
- `raw_full_percentile_1_99.png`
- `raw_surface_muted_display_only.png`
- `raw_trace_normalized_display_only.png`
- `raw_crop_best_candidate.png`
- `raw_centered_n31_summary.md`
- `raw_bscan_Ey.npy`
- `raw_bscan_Ey.csv`

## Comparison with scene_031
- scene_032 expands centered aperture from n15 to n31 with all physical variables fixed.
- n31 provides broader lateral context and better interpretability than n15.
- Top band remains dominant; target cue remains weak but more trackable than n15 in cropped/normalized views.

## Visual assessment
- top surface/background band: visible and strong.
- single target response: weak-to-moderate cue (improved interpretability vs n15, still not a clean strong arch).
- hyperbola-like trend: preliminary/limited, not clearly definitive.

## Background decision
- Not run by design (raw-only task boundary).
- Recommendation: if paired diagnostic is needed next, run background on scene_032 with same n31 and Ey conversion path.

## Files deliberately excluded
- No MyGPR-Evidence git operations.
- No generated `.out/.h5/.vti/.vtk/.vtu/.csv/.npy/.png` committed.
- No background/clutter-free outputs generated.

## Claim boundary
- centered depth03 raw-only n31 aperture gate only
- no background
- no clutter-free
- not exact replication
- not paper benchmark
- not AutoTune evaluation
- not field validation

## Recommended next task
- `GX-008-PAPER-RAW-006-CENTERED-DEPTH03-N31-BG-PAIR-GATE` (run background n31 only for scene_032 and evaluate paired visibility), or if staying raw-only, run one single-variable radius gate at centered n31.
