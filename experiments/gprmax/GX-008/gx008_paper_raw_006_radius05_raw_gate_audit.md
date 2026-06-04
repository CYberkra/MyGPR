#!/usr/bin/env markdown
# GX-008-PAPER-RAW-006 Radius05 Raw Gate Audit

## Date
2026-05-25 21:39:17

## Branch
main

## Base commit
7cf193e5529fe0501ce80f6c51bfd2d8c9521e77

## Remote verification
- `git rev-parse HEAD` = `7cf193e5529fe0501ce80f6c51bfd2d8c9521e77`
- `git rev-parse origin/main` = `7cf193e5529fe0501ce80f6c51bfd2d8c9521e77`
- `git ls-remote origin main` = `7cf193e5529fe0501ce80f6c51bfd2d8c9521e77`

## scene_032 recap
- Scene: `scene_032_gssi_ey_depth03_centered_n31_raw_gate`
- centered n31 coverage confirmed:
  - n1 rx x=0.35
  - n16 rx x=0.50
  - n31 rx x=0.65
- Component: Ey
- Shape: `[3636, 31]`
- Visual status: improved interpretability vs n15, but target response still weak.

## Why radius05 was selected
- Depth/aperture centering ambiguity has been removed by scene_032.
- Remaining weakness suggests target scattering amplitude may be insufficient in current raw view.
- Paper one-object radius range allows up to 5 cm; choose `radius=0.05 m` as single-variable gate while keeping all other variables fixed.

## scene_033 design
- Added scene: `scene_033_gssi_ey_depth03_radius05_centered_n31_raw_gate`
- Based on: `scene_032_gssi_ey_depth03_centered_n31_raw_gate`
- Changed from scene_032:
  - target radius `0.03 -> 0.05`
- Unchanged from scene_032:
  - depth, target center, soil, antenna, target material, domain, dx/dy/dz, scan step, centered n31 aperture.

## Dry-run result
- `campaign_status: ready`
- `scene_033_gssi_ey_depth03_radius05_centered_n31_raw_gate: ready`
- `invalid_count: 0`

## GPU wrapper result
- `scripts/run_gprmax_gpu_env.bat --check`: pass
- `scripts/run_gprmax_gpu_env.bat --smoke`: pass (known non-blocking decode-thread noise after output)

## Raw n31 result
- Scene: `scene_033_gssi_ey_depth03_radius05_centered_n31_raw_gate`
- Variant: `raw_with_target`
- status: success
- return_code: 0
- runtime_seconds: `827.884`
- actual output count: `31`
- manifest:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008/scene_033_gssi_ey_depth03_radius05_centered_n31_raw_gate/raw_with_target/run_manifest.json`

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

## Ey conversion status
- Raw-only Ey conversion completed from scene_033 `.out` files.
- Shape: `[3636, 31]`
- No background conversion in this task.

## Raw visual output paths
Scratch root:
- `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/scratch/GX-008_radius05_raw_visual/scene_033_gssi_ey_depth03_radius05_centered_n31_raw_gate/`

Generated:
- `raw_full_percentile_1_99.png`
- `raw_surface_muted_display_only.png`
- `raw_trace_normalized_display_only.png`
- `raw_crop_best_candidate.png`
- `raw_radius05_summary.md`
- `raw_bscan_Ey.npy`
- `raw_bscan_Ey.csv`

## Comparison with scene_032
Quantitative raw-level comparison (Ey):
- scene_032 energy: `0.3282`
- scene_033 energy: `0.3989` (higher)
- scene_032 mean_abs: `0.1436`
- scene_033 mean_abs: `0.1538` (higher)
- scene_032 p99_abs: `3.1759`
- scene_033 p99_abs: `3.3615` (higher)

Visual comparison:
- scene_033 shows stronger response contrast than scene_032 in cropped/normalized displays.
- Top surface/background band is still dominant, but localized target-related cue is more visible than radius03.
- Hyperbola-like trend remains preliminary; still not a clean strong benchmark-like arch.

## Visual assessment
- top surface/background band: visible and strong
- single target response: visible and stronger than scene_032
- curvature trend: weak-to-moderate preliminary trend, not definitive
- radius05 provides meaningful raw visibility gain under current centered n31 setup.

## Background decision
- Not run by design in this task.
- Given improved raw visibility, background/pair gate is now justified as next step if paired diagnostics are needed.

## Files deliberately excluded
- No MyGPR-Evidence git operations.
- No generated `.out/.h5/.vti/.vtk/.vtu/.csv/.npy/.png` committed.
- No background/clutter-free outputs generated.

## Claim boundary
- centered depth03 radius05 raw-only n31 gate only
- no background
- no clutter-free
- not exact replication
- not paper benchmark
- not AutoTune evaluation
- not field validation

## Recommended next task
- `GX-008-PAPER-RAW-007-RADIUS05-CENTERED-N31-BG-PAIR-GATE`:
  run background n31 for scene_033 only, then paired conversion/preview for conservative validation of target-response separability.
