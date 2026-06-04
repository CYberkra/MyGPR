#!/usr/bin/env markdown
# GX-008-PAPER-RAW-008 Radius05 Centered N80 Pair Gate Audit

## Date
2026-05-25 23:12:59

## Branch
main

## Base commit
eb0c24026b00169441d0d3f9083318f0e381f7fe

## Remote verification
- `git rev-parse HEAD` = `eb0c240639b652daf9c3f494744686d112482e7e`
- `git rev-parse origin/main` = `eb0c240639b652daf9c3f494744686d112482e7e`
- `git ls-remote origin main` = `eb0c240639b652daf9c3f494744686d112482e7e`

## Geometry/depth interpretation pre-run audit (scene_033 baseline)
| parameter | value |
|---|---|
| domain_x/y/z | 1.0 / 0.15 / 0.40 m |
| dx/dy/dz | 0.002 / 0.002 / 0.002 m |
| time_window | 14e-9 s |
| soil box z range | 0.0 -> 0.40 m |
| soil surface z | 0.0 m |
| GSSI antenna insertion z | 0.05 m |
| antenna z skid-bottom known? | unknown in current model comments |
| antenna-to-soil distance | 0.05 m (from model coordinates) |
| target material | PEC cylinder |
| target center x | 0.50 m |
| target center z | 0.08 m |
| target radius | 0.05 m |
| target top z | 0.03 m |
| target bottom z | 0.13 m |
| depth03 implementation | center-depth approximation in current chain, not strict cover depth |
| target fully below soil surface? | yes (`top_z=0.03 > surface_z=0.0`) |
| scan start x (scene_033 n31) | antenna input x=0.382 |
| scan step | 0.01 m |
| n31 rx range | ~0.35 -> 0.65 |
| proposed n80 rx range | ~0.105 -> 0.895 |
| target center coverage under n80 | yes |

### Abort rule check
- No protrusion/intersection above soil surface detected under current geometry.
- Abort rule **not** triggered.

## Scene preparation
- Added scene: `scene_034_gssi_ey_depth03_radius05_centered_n80_pair_gate`
- Preserved variables from validated scene_033:
  - GSSI-like antenna
  - component Ey
  - dry sand
  - radius05
  - depth03
  - scan step 0.01
- Expected runs: 80
- n80 centered aperture implemented with antenna input start x `0.137`.

## Dry-run validation
- campaign_status: ready
- scene_034: ready
- invalid_count: 0

## GPU wrapper checks
- `--check`: pass
- `--smoke`: pass (known non-blocking decode-thread noise in host wrapper logs)

## Run results (n80)
### Raw n80
- status: timeout
- runtime_seconds: ~3600.13
- requested_num_runs: 80
- actual completed `.out`: 62
- manifest:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008/scene_034_gssi_ey_depth03_radius05_centered_n80_pair_gate/raw_with_target/run_manifest.json`

### Background n80
- status: timeout
- runtime_seconds: ~3600.09
- requested_num_runs: 80
- actual completed `.out`: 62
- manifest:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008/scene_034_gssi_ey_depth03_radius05_centered_n80_pair_gate/background_only/run_manifest.json`

## Conversion/pairing outcome
- Not executed as final n80 gate output, because both raw and background timed out at 62/80.
- No complete n80 pair dataset is available from this run.

## Decision against task goal
- Goal was full n80 raw/background pair gate.
- Current attempt did not produce complete n80 outputs; therefore no valid full n80 paired audit can be claimed.

## Recommended next step
1. Re-run n80 sequentially (raw then background, not concurrent) with larger timeout budget and progress checkpoints, **or**
2. Run an intermediate n64 complete pair gate to validate scaling, then n80, **or**
3. If hardware window is fixed, keep n31 paired result as operational baseline and add official gprMax GSSI reference geometry gate before further scaling.

## Files deliberately excluded
- No generated `.out/.h5/.vti/.vtk/.vtu/.csv/.npy/.png` committed.
- No scratch outputs committed.
- No MyGPR-Evidence git content committed.

## Claim boundary
- This is an n80 synthetic pair gate attempt for one GSSI/Ey dry-sand PEC-cylinder scene.
- Not exact CLT-GPR replication.
- Not paper benchmark.
- Not field validation.
- Not AutoTune evaluation.
