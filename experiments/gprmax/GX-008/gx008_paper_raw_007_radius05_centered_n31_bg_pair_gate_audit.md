#!/usr/bin/env markdown
# GX-008-PAPER-RAW-007 Radius05 Centered N31 BG Pair Gate Audit

## Date
2026-05-25 22:03:08

## Branch
main

## Base commit
65b06e6920728198a070d007dc43dbd62b59efc3

## Remote verification
- `git rev-parse HEAD` = `65b06e6920728198a070d007dc43dbd62b59efc3`
- `git rev-parse origin/main` = `65b06e6920728198a070d007dc43dbd62b59efc3`
- `git ls-remote origin main` = `65b06e6920728198a070d007dc43dbd62b59efc3`

## Task scope
- Scene: `scene_033_gssi_ey_depth03_radius05_centered_n31_raw_gate`
- Component: `Ey`
- This task runs background-only n31 and produces paired raw/background/target_response audit.
- No AutoTune, no Evidence archival, no UI changes.

## Dry-run validation
- campaign dry-run: `ready`
- scene_033 status: `ready`
- invalid_count: `0`

## Background n31 run
- variant: `background_only`
- status: success
- return_code: 0
- runtime_seconds: 773.626
- output_count: 31
- manifest:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008/scene_033_gssi_ey_depth03_radius05_centered_n31_raw_gate/background_only/run_manifest.json`

## Pairing consistency
- Reused existing raw n31 outputs (no raw rerun).
- Converted raw/background with Ey (`rxs/rx1/Ey`).
- Paired formula: `target_response = raw_with_target - background_only`.

## Trajectory coverage check (n1/n16/n31)
- n1:  rx=[0.35000000000000003, 0.07200000000000001, 0.054], src=[0.41000000000000003, 0.07200000000000001, 0.054]
- n16: rx=[0.5, 0.07200000000000001, 0.054], src=[0.56, 0.07200000000000001, 0.054]
- n31: rx=[0.65, 0.07200000000000001, 0.054], src=[0.71, 0.07200000000000001, 0.054]
- target_center_x = 0.5
- coverage: {'rx_start': 0.35000000000000003, 'rx_mid': 0.5, 'rx_end': 0.65, 'covers_target': True}

## Quantitative summary
- raw shape: [3636, 31]
- background shape: [3636, 31]
- target_response shape: [3636, 31]
- component used: Ey
- receiver path: `rxs/rx1/Ey`

### Raw stats
- min: -4.397129535675049
- max: 8.071578979492188
- mean_abs: 0.15381363034248352
- p95_abs: 1.0835509300231934
- p99_abs: 3.36149525642395
- energy: 0.3989342451095581

### Background stats
- min: -3.1610536575317383
- max: 4.016215801239014
- mean_abs: 0.14343346655368805
- p95_abs: 1.0692198276519775
- p99_abs: 3.1759326457977295
- energy: 0.32812798023223877

### Target response stats
- min: -2.953603982925415
- max: 4.360881805419922
- mean_abs: 0.03300640732049942
- p95_abs: 0.10254106670618057
- p99_abs: 0.9077818989753723
- energy: 0.03836218640208244

### ROI/window energy
- target ROI energy (sample 300-1600, trace 8-24): 0.13380996882915497
- surface energy (sample 0-250): 0.02556711621582508
- target-window energy (sample 300-1600): 0.07903473824262619
- target/surface ratio: 3.0912652632530926

## Geometry summary
| item | value |
|---|---|
| soil surface z | 0.0 m |
| GSSI antenna z | 0.05 m |
| antenna-to-soil distance | 0.05 m |
| target center x | 0.5 m |
| target center z | 0.08 m |
| target radius | 0.05 m |
| target top z | 0.03 m |
| target bottom z | 0.13 m |
| depth03 interpretation | implemented as center-depth approximation relative to antenna reference chain, not strict cover depth |

## Visual outputs (local scratch, not committed)
- `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/scratch/GX-008_radius05_bg_pair_gate/scene_033_gssi_ey_depth03_radius05_centered_n31_raw_gate/raw_full_percentile_1_99.png`
- `.../background_full_percentile_1_99.png`
- `.../target_response_full_percentile_1_99.png`
- `.../target_response_crop_best_candidate.png`
- `.../target_response_trace_normalized_display_only.png`
- `.../target_response_surface_muted_display_only.png`

## Visibility decision
- Result: `improved_visibility_after_subtraction`
- Interpretation: target_response suppresses dominant common background and reveals stronger localized response than raw, but still not definitive benchmark-grade hyperbola

## Gate decision
- The radius05 centered depth03 n31 scene shows clearer target_response than raw after subtraction.
- Scene_033 can advance to the next larger-aperture/full-length pair gate (e.g., n80) **conditionally**.
- Still keep conservative boundary: this is not exact CLT-GPR replication or benchmark completion.

## Files deliberately excluded
- No `.out/.h5/.vti/.vtk/.vtu/.csv/.npy/.png` committed.
- No scratch outputs committed.
- No MyGPR-Evidence git content committed.

## Claim boundary
- centered depth03 radius05 n31 background-pair gate only
- not exact CLT-GPR replication
- not paper benchmark
- not field validation
- not AutoTune evaluation

## Recommended next task
- `GX-008-PAPER-RAW-008-RADIUS05-CENTERED-N80-PAIR-GATE` (if compute budget allows),
  else run an official gprMax GSSI reference gate and geometry/depth interpretation audit before scaling up.
