#!/usr/bin/env markdown
# GX-008-EXPAND-001 Audit

Date: 2026-05-23  
Branch: `main`  
Base commit: `12848a580aa0b8086576ecb946a7beef138349de`

## Remote Verification

Executed:

- `git rev-parse HEAD`
- `git rev-parse origin/main`
- `git ls-remote origin main`

Result:

- all three point to `12848a580aa0b8086576ecb946a7beef138349de` before this task started.

## Scenes Added

1. `scene_003_flat_dry_sand_pvc_shallow`
2. `scene_004_flat_damp_sand_pvc_shallow`
3. `scene_005_flat_dry_sand_pec_medium`
4. `scene_006_flat_damp_sand_pec_medium`

## Modeling Choices

- Kept the same stable baseline as scene_001/002:
  - `#domain: 2.0 1.0 0.5`
  - `#dx_dy_dz: 0.01 0.01 0.01`
  - `#time_window: 18e-9`
  - Ricker + Hertzian dipole
  - `#src_steps: 0.005 0 0`
  - `#rx_steps: 0.005 0 0`
  - expected run count in manifest: `41`
- No rough surface and no heterogeneous soil were introduced in this expansion pass.

## Material Choices

- PVC scenes (`scene_003`, `scene_004`) use draft dielectric target material `pvc_like`:
  - `eps_r=3.5`, `sigma=0.0`, `mu_r=1.0`, `mag_sigma=0.0`
- PEC scenes (`scene_005`, `scene_006`) keep `pec` cylinder target.
- Soil materials:
  - dry sand-like: `eps_r=3.0`, `sigma=0.001`
  - damp sand-like: `eps_r=8.0`, `sigma=0.01`
- `materials.txt` in each scene explicitly marks parameters as draft and citation lock pending.

## Target Depth Choices

- Shallow PVC scenes:
  - `scene_003`: cylinder depth `z=0.16`
  - `scene_004`: cylinder depth `z=0.17`
- Medium PEC scenes:
  - `scene_005`: cylinder depth `z=0.23`
  - `scene_006`: cylinder depth `z=0.23`
- Medium-depth targets remain inside domain and not near boundaries/PML.

## ROI Draft Notes

- Added `roi_draft.json` for each new scene with:
  - `scene_id`
  - `roi_role`
  - `expected_target_region`
  - `depth_window`
  - `trace_window`
  - `notes`
- All ROI files are explicitly marked as draft-only and not field truth claims.
- Medium scenes use deeper sample windows to track the lower target depth class.

## Pair Contract Audit

Static audit was run for all added scenes:

- raw/background share identical:
  - `#domain`
  - `#dx_dy_dz`
  - `#time_window`
  - `#waveform`
  - `#hertzian_dipole`
  - `#rx`
  - `#src_steps`
  - `#rx_steps`
  - `#box`
- raw includes `#cylinder`, background omits `#cylinder`.
- all scene manifest JSON and ROI JSON parse successfully.
- `expected_num_runs` confirmed as `41`.
- `pairing_formula` confirmed as `target_response = raw - background`.

Result: static pair contract check passed.

## Dry-run Result

Command:

```bat
python scripts\gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --dry-run
```

Result:

- `campaign_status: ready`
- `total_scenes: 6`
- `ready_count: 6`
- `invalid_count: 0`

## Files Changed

- `experiments/gprmax/GX-008/campaign_draft.yaml`
- `experiments/gprmax/GX-008/models/scene_003_flat_dry_sand_pvc_shallow/*`
- `experiments/gprmax/GX-008/models/scene_004_flat_damp_sand_pvc_shallow/*`
- `experiments/gprmax/GX-008/models/scene_005_flat_dry_sand_pec_medium/*`
- `experiments/gprmax/GX-008/models/scene_006_flat_damp_sand_pec_medium/*`
- `experiments/gprmax/GX-008/gx008_expand_001_audit.md`

## Generated Files Excluded

- No `.out/.h5/.vti/.vtk/.vtu` generated or committed.
- No generated `.csv/.npy/.png` committed.
- No MyGPR-Evidence repository operations.

## Claim Boundary

- model drafts only
- dry-run/static audit only
- not real gprMax run
- not Evidence artifact
- not AutoTune evaluation
- not field validation
- not paper-candidate benchmark

## Recommended Next Task

- `GX-008-RUN-003` (first controlled real paired run on new scenes, start with one PVC scene and one medium-depth PEC scene).
