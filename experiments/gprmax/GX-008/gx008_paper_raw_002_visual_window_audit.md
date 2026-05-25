#!/usr/bin/env markdown
# GX-008-PAPER-RAW-002 Raw Visual Window Audit

## Date
2026-05-25

## Branch
main

## Base commit
8ac614102f61d22c9858a5d892bae72015f896de

## Remote verification
- `git rev-parse HEAD` = `8ac614102f61d22c9858a5d892bae72015f896de`
- `git rev-parse origin/main` = `8ac614102f61d22c9858a5d892bae72015f896de`
- `git ls-remote origin main` = `8ac614102f61d22c9858a5d892bae72015f896de`

## Source scene used
- `scene_025_paper_aligned_gssi_antenna_gate_n15`

## Raw source path
- `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008/scene_025_paper_aligned_gssi_antenna_gate_n15/raw_with_target/converted_Ey_n31/raw_bscan.npy`
- conversion summary:
  - `.../convert_summary_31_Ey.json`
  - `selected_component = Ey`

## Selected component
- Ey

## Raw shape
- `[3636, 31]`

## Raw statistics
- min/max: `-3.1610538959503174 / 4.016218185424805`
- mean/std: `-0.0007390366517938673 / 0.5728259682655334`
- percentiles:
  - p0.5: `-2.8758535385131836`
  - p1: `-2.1766228675842285`
  - p2: `-1.2406021356582642`
  - p5: `-0.2852126955986023`
  - p95: `0.04424389451742172`
  - p98: `1.3323895931243896`
  - p99: `3.1759328842163086`
  - p99.5: `3.7837023735046387`
- strong surface band estimate: sample `0-450`
- estimated target-visible sample range (display heuristic): `1262-1582`

## Generated visualization files
Scratch output dir:
- `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/scratch/GX-008_raw_visual_window_audit/`

Generated files:
- `raw_full_percentile_1_99.png`
- `raw_full_percentile_2_98.png`
- `raw_crop_0_1200.png`
- `raw_crop_300_1400.png`
- `raw_crop_500_1800.png`
- `raw_surface_muted_display_only.png`
- `raw_trace_normalized_display_only.png`
- `raw_zoom_best_candidate.png`
- `raw_visual_window_audit_summary.md`
- `index.html` (quick local gallery)

## Crop/clipping methods
- full-window clipping: `1-99%` and `2-98%`
- cropped windows:
  - `0-1200`
  - `300-1400`
  - `500-1800`
- display-only transforms:
  - surface-muted top band clipping
  - per-trace normalization
- best-candidate zoom:
  - traces `0-14`
  - samples `1222-1621`
  - local clip `2-98%`

## Visual assessment
- Top surface/background reflection band is clearly visible in all full-window views.
- Crop and clipping variants improve contrast in mid/low sample regions.
- In current n31 raw-only data, a clearly separable single-target arch/hyperbola is still weak/inconclusive.
- Best-candidate zoom shows localized texture change but not a confident standalone target curve.

## Best candidate figure path
- `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/scratch/GX-008_raw_visual_window_audit/raw_zoom_best_candidate.png`

## Whether target is visible
- weak / inconclusive (not clearly separable).

## Whether display scale likely hides target
- yes, partly.
- dynamic range and strong top band suppress lower-amplitude target-like structure in full-window display.
- however, even with crop/contrast/display-only transforms, target curvature remains weak.

## Recommended next task
- `GX-008-PAPER-RAW-003-SINGLE-VARIABLE-GATE`:
  keep GSSI/Ey and scan geometry fixed, then run exactly one variable gate (target depth **or** radius) to improve raw-only target visibility before any background/pairing path.

## Files deliberately excluded
- no new gprMax simulation outputs
- no background/pairing/clutter-free outputs
- no MyGPR-Evidence git operations
- no generated `.out/.h5/.vti/.vtk/.vtu`
- no generated `.csv/.npy/.png/.html` committed

## Claim boundary
- raw display-window audit only
- no new simulation
- no background
- no clutter-free
- not exact replication
- not paper benchmark
- not AutoTune evaluation
