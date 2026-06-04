#!/usr/bin/env markdown
# GX-008-PAPER-RAW-011-SCENE036-DEPTH05-RADIUS03-SAFE-N80-PAIR-GATE Setup Audit

## 1) Task ID
- GX-008-PAPER-RAW-011-SCENE036-DEPTH05-RADIUS03-SAFE-N80-PAIR-GATE

## 2) Branch / base / new commit
- Branch: `main`
- Base commit: `664fade8d178ed68685a82538822985441d7d3fe`
- New commit: pending

## 3) Why scene_036 is needed
- Previous scene_034/scene_035 direction emphasized a large shallow PEC target (`radius05`, shallow cover), which produced strong near-field target response bias.
- scene_036 is introduced as a conservative paper-like diagnostic gate to reduce that bias:
  - smaller target (`radius03`)
  - deeper cover (`depth05`)
  - safe n80 antenna input aperture
  - same dry sand and same GSSI-like antenna pipeline

## 4) Difference from scene_034 strong-target gate
- `radius05 -> radius03`
- `cover depth03 -> cover depth05`
- cylinder changed to elongated y-span inset geometry:
  - `#cylinder: 0.50 0.002 0.08 0.50 0.148 0.08 0.03 pec`
- safe aperture start changed to avoid right-boundary overflow:
  - `antenna_like_GSSI_1500(0.100 + (current_model_run - 1) * 0.01, ...)`

## 5) Geometry table (scene_036 draft)
| Item | Value |
|---|---|
| scene_id | `scene_036_gssi_ey_depth05_radius03_safe_n80_pair_gate` |
| domain | `1.0 x 0.15 x 0.40 m` |
| dx/dy/dz | `0.002 / 0.002 / 0.002 m` |
| time_window | `14e-9 s` |
| soil | dry_sand_tableii (`eps_r=3.0`, `sigma=0.001`) |
| target material | `pec` |
| target type | cylinder along y |
| target center x | `0.50 m` |
| target center z | `0.08 m` |
| target radius | `0.03 m` |
| target top z | `0.05 m` |
| target bottom z | `0.11 m` |
| cover depth interpretation | cover depth = `0.05 m` (surface `z=0` to top `z=0.05`) |
| antenna model | `antenna_like_GSSI_1500` |
| antenna input start x | `0.100 m` |
| scan step | `0.010 m` |
| expected_num_runs | `80` |
| antenna input x range | `0.100 -> 0.890 m` |
| expected target center trace (1-based) | `41` |
| selected component for visualization | `Ey` |

## 6) Campaign update summary
- Added new scene entry in:
  - `experiments/gprmax/GX-008/campaign_draft.yaml`
- Added model draft directory:
  - `experiments/gprmax/GX-008/models/scene_036_gssi_ey_depth05_radius03_safe_n80_pair_gate/`
- Files created:
  - `raw_with_target.in`
  - `background_only.in`
  - `materials.txt`
  - `scene_manifest_draft.json`
  - `roi_draft.json`

## 7) Dry-run result
- `python scripts/gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --dry-run`
- Expected outcome for this setup task: campaign ready and scene_036 ready.

## 8) Preflight result
- `python scripts/preflight_check.py`
- Expected outcome for this setup task: pass.

## 9) Claim boundary
- This task only creates and validates a conservative synthetic GSSI/Ey depth05 radius03 safe-aperture n80 raw/background pair scene.
- It does not run simulation.
- It does not generate target_response.
- It is not exact CLT-GPR replication.
- It is not a paper benchmark.
- It is not field validation.
- It is not AutoTune evaluation.

## 10) Recommended next task
1. Review scene_036 geometry and scan design.
2. If approved, run scene_036 raw/background n80 with the existing GPU wrapper.
3. Convert using explicit `--component Ey` and evaluate whether target_response remains visible under conservative geometry.
