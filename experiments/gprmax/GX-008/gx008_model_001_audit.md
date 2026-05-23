#!/usr/bin/env markdown
# GX-008-MODEL-001 Audit

Date: 2026-05-23  
Branch: `main`  
Base commit: `af7b08ec26b1d6734eb7c05101977d8d36db3a0f`

## Files Created/Changed

- `experiments/gprmax/GX-008/campaign_draft.yaml` (updated)
- `experiments/gprmax/GX-008/models/scene_001_flat_dry_sand_pec_shallow/raw_with_target.in`
- `experiments/gprmax/GX-008/models/scene_001_flat_dry_sand_pec_shallow/background_only.in`
- `experiments/gprmax/GX-008/models/scene_001_flat_dry_sand_pec_shallow/materials.txt`
- `experiments/gprmax/GX-008/models/scene_001_flat_dry_sand_pec_shallow/roi_draft.json`
- `experiments/gprmax/GX-008/models/scene_001_flat_dry_sand_pec_shallow/scene_manifest_draft.json`
- `experiments/gprmax/GX-008/models/scene_002_flat_damp_sand_pec_shallow/raw_with_target.in`
- `experiments/gprmax/GX-008/models/scene_002_flat_damp_sand_pec_shallow/background_only.in`
- `experiments/gprmax/GX-008/models/scene_002_flat_damp_sand_pec_shallow/materials.txt`
- `experiments/gprmax/GX-008/models/scene_002_flat_damp_sand_pec_shallow/roi_draft.json`
- `experiments/gprmax/GX-008/models/scene_002_flat_damp_sand_pec_shallow/scene_manifest_draft.json`
- `experiments/gprmax/GX-008/gx008_model_001_audit.md`

## Scene List

1. `scene_001_flat_dry_sand_pec_shallow`
2. `scene_002_flat_damp_sand_pec_shallow`

## Modeling Choices

- Simplified starter modeling retained (Ricker + Hertzian dipole).
- Flat surface only; no roughness, no heterogeneous soil, no water layer.
- Soil presets:
  - dry sand-like: `eps_r=3.0`, `sigma=0.001`
  - damp sand-like: `eps_r=8.0`, `sigma=0.01`
- Target type: PEC cylinder for both scenes.
- Cylinder axis chosen cross-scan (`y` direction) to support localized hyperbola behavior in B-scan.

## Raw/Background Pairing Contract

For both scenes, `raw_with_target.in` and `background_only.in` keep identical:

- `#domain`
- `#dx_dy_dz`
- `#time_window`
- `#waveform`
- `#hertzian_dipole`
- `#rx`
- `#src_steps`
- `#rx_steps`
- soil material definition
- background box

Expected difference only:
- raw contains `#cylinder ... pec`
- background omits target object

Draft scene manifest enforces:
- `expected_pairing_formula: target_response = raw - background`

## Expected Num-Runs

- `expected_num_runs` for both scenes: `41`
- scan stepping:
  - `#src_steps: 0.005 0 0`
  - `#rx_steps: 0.005 0 0`
- runtime note included: execution must pass `--num-runs N` (e.g. `41`) to avoid single-trace output.

## Expected Output Shape Policy

- policy target: 2D arrays `[samples, traces]`
- raw/background paired shape must match exactly
- mismatch should be treated as invalid in later pairing stages

## ROI Draft Notes

Each scene has `roi_draft.json` with:
- `scene_id`
- `roi_role`
- `expected_target_region`
- `depth_window`
- `trace_window`
- draft-only note

ROI is draft metadata for later synthetic scoring and visualization, not field ground truth.

## GPU Run Entry Recommendation

When running later tasks, prefer:

```bat
scripts\run_gprmax_gpu_env.bat --check
scripts\run_gprmax_gpu_env.bat -- python scripts\gprmax_campaign_runner.py ...
```

Use `--gprmax-python` and `--num-runs` explicitly to keep runtime reproducible.

## Risks

- Material values are still draft-level and need formal citation lock in follow-up.
- ROI windows are heuristic placeholders until first validated previews.
- Geometry and depth parameters are tuned for stable starter scenes, not final benchmark diversity.

## Claim Boundary

- This task creates model drafts only.
- No real gprMax run performed in this task.
- No generated simulation artifact created/committed.
- Not an Evidence artifact.
- Not AutoTune evaluation.
- Not field validation.

## Next Task

- `GX-008-DRYRUN-001`: audit campaign dry-run and pair-contract readiness before any real execution.
