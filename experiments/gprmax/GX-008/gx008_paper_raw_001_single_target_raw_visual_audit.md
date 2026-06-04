#!/usr/bin/env markdown
# GX-008-PAPER-RAW-001 Single-Target Raw Visual Audit

## Date
2026-05-25

## Branch
main

## Base commit
c08487521207289bbec8dab55f909270df518261

## Remote verification
- `git rev-parse HEAD` = `c08487521207289bbec8dab55f909270df518261`
- `git rev-parse origin/main` = `c08487521207289bbec8dab55f909270df518261`
- `git ls-remote origin main` = `c08487521207289bbec8dab55f909270df518261`

## Paper raw target requirement
- Target style: Fig.1(b) left raw B-scan style (background/surface reflection + single target response).
- This task scope is raw-only visual candidate.
- No background run, no clutter-free subtraction, no pairing, no AutoTune.

## Source scene used
- `scene_025_paper_aligned_gssi_antenna_gate_n15`
- Existing n31 raw outputs and Ey conversion were reused.

## Whether new simulation was run
- No.
- Existing outputs reused from:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008/scene_025_paper_aligned_gssi_antenna_gate_n15/`

## Raw run status
- Existing n31 raw chain available and valid for reuse:
  - `convert_summary_31_Ey.json` present
  - `selected_component = Ey`
  - `shape = [3636, 31]`
  - source series includes `raw_with_target1.out ... raw_with_target31.out`

## Selected component
- Ey (explicitly confirmed by conversion summary metadata).

## Raw shape
- `[3636, 31]`

## Visualization method
- Input array: `raw_with_target/converted_Ey_n31/raw_bscan.npy`
- Colormap: grayscale (`gray`)
- Figure title: `Raw B-scan`
- Axes: `Trace` (x), `Sample` (y)
- Robust clipping:
  - primary: 1st–99th percentiles
  - enhanced: 2nd–98th percentiles

## Output local figure path
- `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/scratch/GX-008_raw_visual/raw_bscan_paper_style.png`

## Enhanced figure path
- `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/scratch/GX-008_raw_visual/raw_bscan_paper_style_enhanced.png`

## Summary path
- `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/scratch/GX-008_raw_visual/raw_bscan_paper_style_summary.md`

## Visual assessment
- Top region shows strong horizontal background/surface reflection bands, consistent with paper-like raw layout.
- Single-target localized response/hyperbolic curvature is not clearly separable at this n31 raw-only visual level.
- Enhanced contrast still keeps dominant horizontal bands; target signature remains weak/inconclusive.

## Whether single target is visible
- Weak / not clearly isolated in raw-only view.

## Whether paper-style raw B-scan was produced
- Yes, a raw-only paper-style candidate figure was produced.
- It matches the general raw layout style (surface/background-dominant), but single-target visibility remains weak.

## Files deliberately excluded
- No MyGPR-Evidence git operations.
- No generated `.out/.h5/.vti/.vtk/.vtu`.
- No generated `.csv/.npy/.png`.
- No scratch directory content committed.

## Claim boundary
- single-target raw visual candidate only
- no background run in this task
- no clutter-free output
- not exact replication
- not paper benchmark completion
- not AutoTune evaluation
- not field validation

## Recommended next task
- `GX-008-PAPER-RAW-002-SINGLE-VARIABLE-DEPTH-OR-RADIUS-GATE`:
  keep GSSI/Ey and scan geometry fixed, then perform a controlled single-variable gate (target depth or radius) to improve raw-only target visibility before any paired/evidence path.
