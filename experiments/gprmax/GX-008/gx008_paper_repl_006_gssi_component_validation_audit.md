#!/usr/bin/env markdown
# GX-008-PAPER-REPL-006 GSSI Component Validation Audit

## Date
2026-05-25

## Branch
main

## Base commit
fbcd996b3d5acbc4cd21aa115839eabbb763cd8d

## Remote verification
- `git rev-parse HEAD` = `fbcd996b3d5acbc4cd21aa115839eabbb763cd8d`
- `git rev-parse origin/main` = `fbcd996b3d5acbc4cd21aa115839eabbb763cd8d`
- `git ls-remote origin main` = `fbcd996b3d5acbc4cd21aa115839eabbb763cd8d`

## scene_025 recap
- Scene: `scene_025_paper_aligned_gssi_antenna_gate_n15`
- gprMax antenna command: `antenna_like_GSSI_1500(..., resolution=0.002)`
- Existing raw/background n=15 outputs reused; no raw rerun.

## Component ambiguity problem
- Prior variability checks used `Ey`.
- Existing converter used hardcoded `rxs/rx1/Ez` for run-series conversion.
- No explicit conversion component field was recorded in conversion summaries.

## HDF5 receiver structure summary
Representative files checked:
- `raw_with_target1/8/15.out`
- `background_only1/8/15.out`

All checked files expose:
- Receiver: `rx1`
- Components: `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, `Hz`
- Sample count per component trace: `3636`

## Available components
- Electric: `Ex`, `Ey`, `Ez`
- Magnetic: `Hx`, `Hy`, `Hz`

## Component variability table (L2 over traces 1/8/15)

### Raw
- Ex: L2(1,8)=0.0, L2(8,15)=0.0, L2(1,15)=0.0
- Ey: L2(1,8)=0.129359, L2(8,15)=0.111980, L2(1,15)=0.053505
- Ez: L2(1,8)=0.051038, L2(8,15)=0.043800, L2(1,15)=0.020741

### Background
- Ex: all zero
- Ey: non-zero but near-static, L2 scale ~1e-6
- Ez: non-zero but near-static, L2 scale ~1e-7

## Converter behavior before change
- `scripts/gprmax_campaign_convert_scene001.py` `_convert_series` hardcoded `rxs/rx1/Ez`.
- No CLI parameter for component selection.
- No explicit component metadata in summary output.

## Converter changes
- Added `--component` option (default `Ez` for backward compatibility).
- Multi-run conversion now reads requested component explicitly.
- Missing requested component raises clear error with available component list.
- Summary JSON now records:
  - `selected_component`
  - `receiver_name`
  - `available_components`
  - `component_source`

## Component-specific conversion/pairing comparison (scene_025 n=15)
- Ey:
  - `target_response_energy = 0.08120551083780915`
  - `raw_background_psnr = 70.34764021246053`
  - `roi_energy_ratio = 0.000885225598584636`
- Ez:
  - `target_response_energy = 0.012152296873563428`
  - `raw_background_psnr = 70.2012604481515`
  - `roi_energy_ratio = 0.0009864274054182684`
- Ex:
  - raw/background/target_response all-zero
  - pairing metrics include denominator-zero warnings

## Recommended component
- Recommended: `Ey`
- Confidence: `medium`
- Rationale:
  1. GSSI raw-gate variability already validated on `Ey`.
  2. `Ey` has stronger inter-trace variability and stronger target-response energy than `Ez`.
  3. `Ex` is invalid in this scene due to all-zero traces.

## Confidence and unresolved issues
- Confidence is medium (n=15 gate only).
- Unresolved:
  - Need larger aperture (`n>15`) to verify whether `Ey` remains optimal for curvature visibility.
  - Need broader scene coverage before treating component choice as global rule.

## Generated local artifacts
- Reused/produced locally under:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008/scene_025_paper_aligned_gssi_antenna_gate_n15/`
  - `convert_summary_15_{Ey,Ez,Ex}.json`
  - `component_eval_{Ey,Ez,Ex}/paired_outputs/*`
- These are local diagnostics only and were not committed.

## Files deliberately excluded
- `MyGPR-Evidence` git changes
- Generated `.out/.h5/.vti/.vtk/.vtu`
- Generated `.csv/.npy/.png`
- scratch/temp outputs

## Claim boundary
- GSSI component validation only
- not full GSSI benchmark
- not exact paper replication
- not full 80 A-scan B-scan
- not Evidence artifact
- not AutoTune evaluation

## Recommended next task
- `GX-008-PAPER-REPL-007-GSSI-N31-COMPONENT-CONFIRM`: run a centered `n=31` GSSI paired diagnostic with `Ey` and `Ez` side-by-side using explicit `--component`, then reassess curvature visibility and component stability.
