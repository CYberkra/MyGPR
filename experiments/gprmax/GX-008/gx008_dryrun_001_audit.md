#!/usr/bin/env markdown
# GX-008-DRYRUN-001 Audit

Date: 2026-05-23  
Branch: `main`  
Base commit: `44de5614833cf970912289f5a192e0eb42b7aab8`

## Remote Verification

Executed:

- `git rev-parse HEAD`
- `git rev-parse origin/main`
- `git ls-remote origin main`

Result:
- all three point to `44de5614833cf970912289f5a192e0eb42b7aab8` before this task started.

## Files Inspected

- `experiments/gprmax/GX-008/gx008_mini_benchmark_spec.md`
- `experiments/gprmax/GX-008/campaign_draft.yaml`
- `experiments/gprmax/GX-008/gx008_model_001_audit.md`
- `experiments/gprmax/GX-008/models/scene_001_flat_dry_sand_pec_shallow/*`
- `experiments/gprmax/GX-008/models/scene_002_flat_damp_sand_pec_shallow/*`

## Scenes Inspected

1. `scene_001_flat_dry_sand_pec_shallow`
2. `scene_002_flat_damp_sand_pec_shallow`

## gprMax Syntax Findings (Static Review)

Both scene `.in` files include required command set:

- `#domain`
- `#dx_dy_dz`
- `#time_window`
- `#waveform`
- `#hertzian_dipole`
- `#rx`
- `#src_steps`
- `#rx_steps`
- `#material`
- `#box`
- `#cylinder` (raw only)

Geometry sanity checks:

- Domain: `2.0 x 1.0 x 0.5`
- Cylinder scene_001: center line endpoints `(1.00,0.40,0.16)` to `(1.00,0.60,0.16)`, radius `0.04`
- Cylinder scene_002: endpoints `(1.00,0.40,0.17)` to `(1.00,0.60,0.17)`, radius `0.04`
- All target coordinates and radii are inside domain bounds.

## Pair Contract Findings

For each scene, raw/background are consistent on:

- domain
- dx_dy_dz
- time_window
- waveform
- hertzian_dipole
- rx
- src_steps / rx_steps
- material
- background box

Contract difference check:

- raw contains target object line (`#cylinder`)
- background omits target object line

Result: pair contract passes static audit.

## Campaign Draft Findings

Initial issue found and fixed:

- `campaign_draft.yaml` had paths starting with `experiments/gprmax/GX-008/...`
- loader resolves paths relative to YAML directory, so that produced invalid duplicated paths.

Fix applied:

- switched scene paths to relative `models/...` form.

Post-fix checks:

- scene ids and directory names are consistent
- raw/background/material/roi paths exist
- tags include `draft`, `not_validated`, `not_run`
- `output_root` points outside MyGPR source repo (`MyGPR-Evidence` path)

## ROI Draft Findings

Validation:

- both `roi_draft.json` files parse as valid JSON
- both include required fields:
  - `scene_id`
  - `roi_role`
  - `expected_target_region`
  - `depth_window`
  - `trace_window`
  - `notes`
- notes explicitly keep draft/claim boundary language
- ROI depth and target geometry are broadly consistent with cylinder depth settings.

## Manifest Draft Findings

Validation:

- both `scene_manifest_draft.json` files parse as valid JSON
- include:
  - `expected_pairing_formula`
  - `scan_design.expected_num_runs`
  - target type/material/radius/depth/orientation
  - shape policy and pairing consistency requirement

`expected_num_runs` consistency:

- both manifests set `expected_num_runs: 41`
- runtime note explicitly requires `--num-runs N`.

## Dry-Run Command and Result

Command:

```bat
python scripts\gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --dry-run
```

Result:

- `campaign_status: ready`
- `total_scenes: 2`
- `ready_count: 2`
- `invalid_count: 0`

## Issues Fixed

1. `campaign_draft.yaml` path resolution bug fixed (relative path normalization to `models/...`).

## Remaining Risks

- This is static + dry-run readiness only; no real gprMax execution performed.
- Material values remain draft-level and still need formal citation lock.
- ROI windows remain draft placeholders until first preview-driven refinement.

## Claim Boundary

- No real gprMax run in this task.
- No native outputs generated/committed (`.out/.h5/.vti/.vtk/.vtu`).
- No generated arrays/figures committed.
- No MyGPR-Evidence operation performed.
- No AutoTune scoring change, no motion compensation change, no UI change.
- GX-008 is not validated benchmark evidence yet.

## Next Task

- `GX-008-RUN-001` (execute first controlled run), or `GX-008-MODEL-FIX-001` if additional modeling constraints are requested before run.
