#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-008 PAPER RAW 015 audit."""

# GX-008-PAPER-RAW-015-SCENE037-AIR-SAND-N80-PAIR-RUN-ONLY

## 1) Task ID
- `GX-008-PAPER-RAW-015-SCENE037-AIR-SAND-N80-PAIR-RUN-ONLY`

## 2) Branch / Base / New commit
- Branch: `main`
- Base commit: `fa2b0b68d2c21a5fafd4301ed1cac906a9eddfff`
- New commit: pending at audit write time

## 3) Remote verification
- `git branch --show-current` -> `main`
- `git rev-parse HEAD` -> `fa2b0b68d2c21a5fafd4301ed1cac906a9eddfff`
- `git rev-parse origin/main` -> `fa2b0b68d2c21a5fafd4301ed1cac906a9eddfff`
- `git ls-remote origin main` -> `fa2b0b68d2c21a5fafd4301ed1cac906a9eddfff`

## 4) Environment summary
- OS: Windows 10
- gprMax runtime: `E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe`
- GPU: NVIDIA RTX 3060 Laptop GPU
- Scene:
  - `experiments/gprmax/GX-008/models/scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate`

## 5) Pre-run checks
- `python scripts/gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --dry-run` -> ready, scene_037 valid
- `python scripts/preflight_check.py` -> OK
- `scripts\run_gprmax_gpu_env.bat --check` -> OK
- `scripts\run_gprmax_gpu_env.bat --smoke` -> OK

## 6) Scene geometry summary
- Domain: `1.0 x 0.15 x 0.40 m`
- Grid: `0.002 x 0.002 x 0.002 m`
- Dry sand: `z=0.000..0.260`
- Air/free-space: `z=0.260..0.400`
- Antenna skid bottom z: `0.310`
- Standoff to soil surface: `0.050 m`
- Target: PEC cylinder along y
  - center: `x=0.50, z=0.180`
  - radius: `0.03`
  - top/bottom z: `0.210 / 0.150`
  - cover depth: `0.050 m`
- Scan: input `x=0.100 + (run-1)*0.01`, `n=80`

## 7) Commands run (segmented)
- Raw:
  1. `python -m gprMax raw_with_target.in -n 20 -gpu 0`
  2. `python -m gprMax raw_with_target.in -n 20 -restart 21 -gpu 0`
  3. `python -m gprMax raw_with_target.in -n 20 -restart 41 -gpu 0`
  4. `python -m gprMax raw_with_target.in -n 20 -restart 61 -gpu 0`
- Background:
  1. `python -m gprMax background_only.in -n 20 -gpu 0`
  2. `python -m gprMax background_only.in -n 20 -restart 21 -gpu 0`
  3. `python -m gprMax background_only.in -n 20 -restart 41 -gpu 0`
  4. `python -m gprMax background_only.in -n 20 -restart 61 -gpu 0`

## 8) Runtime per segment
- Raw segment runtimes (gprMax total):
  - 1-20: `0:07:18.318646`
  - 21-40: `0:07:24.147754`
  - 41-60: `0:07:24.585123`
  - 61-80: `0:07:00.180398`
- Background segment runtimes (gprMax total):
  - 1-20: `0:07:11.582507`
  - 21-40: `0:07:59.783401`
  - 41-60: `0:08:05.657387`
  - 61-80: `0:08:02.787511`

## 9) Output counts
- Raw `.out`: `80/80`
- Background `.out`: `80/80`
- No overwrite/restart conflict observed.

## 10) Failures / restarts
- No failed segment in this run.
- Segmented restart strategy executed as planned.

## 11) Conversion method
- Existing MyGPR converter:
  - `scripts/gprmax_campaign_convert_scene001.py`
- Component explicitly selected:
  - `--component Ey`
- Conversion summary:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\scratch\GX-008_scene037_n80_pair\convert_summary_ey_n80.json`

## 12) Shape summary
- Raw shape: `[3636, 80]`
- Background shape: `[3636, 80]`
- Target response shape: `[3636, 80]`

## 13) Component summary
- Selected component: `Ey`
- Component source: `rxs/rx1/Ey`
- Available components in `.out`: `Ex, Ey, Ez, Hx, Hy, Hz`

## 14) Visual output paths
- Root:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\scratch\GX-008_scene037_n80_pair\paired_outputs`
- Required:
  - `raw_full_percentile_1_99.png`
  - `background_full_percentile_1_99.png`
  - `target_response_full_percentile_1_99.png`
  - `target_response_crop_best_candidate.png`
  - `target_response_trace_normalized_display_only.png`
  - `target_response_surface_muted_display_only.png`
  - `central_trace_A_scan_raw_bg_target.png`
  - `trace_30_35_40_45_A_scan_overlay.png`
  - `paired_preview_panel.png`
- Optional generated:
  - `target_response_crop_0p5_99p5.png`
  - `target_response_crop_5_95.png`
  - `target_response_symmetric_absmax.png`
  - `trace_35_40_41_45_49_A_scan_overlay.png`

## 15) Metrics
- Raw:
  - energy: `116247.8203125`
  - mean_abs: `0.16072025895118713`
  - p95_abs: `0.8948243856430054`
  - p99_abs: `3.5450057983398438`
- Background:
  - energy: `112964.75`
  - mean_abs: `0.1506643295288086`
  - p95_abs: `0.7861790060997009`
  - p99_abs: `3.5450057983398438`
- Target response (`raw - background`):
  - energy: `3554.81640625`
  - mean_abs: `0.017268721014261246`
  - p95_abs: `0.05447008088231087`
  - p99_abs: `0.4012824594974518`
- Raw/background difference:
  - MAE: `0.017268721014261246`
  - MSE: `0.012220903299748898`
  - RMSE: `0.11054819077253342`
  - PSNR: `33.08998897357791`
- Ratios:
  - target_to_raw_energy_ratio: `0.03057964057539346`
  - target_to_background_energy_ratio: `0.03146837378283868`
  - roi_energy_ratio (pair script): `0.9979606645548237`
- Trace/position indicators:
  - trace closest to target center (1-based): `41`
  - strongest target-window trace (1-based): `41`
  - max abs target_response location `[sample, trace0]`: `[462, 40]`

## 16) Known limitations
- This audit only reports scene_037 n80 pair run itself; no cross-scene comparison in this task.
- Surface-window energy on differenced data is near-zero under current window definition and is not robust as an absolute physics indicator.
- No curated evidence packaging performed.

## 17) Claim boundary
- Synthetic `scene_037` GSSI/Ey dry-sand/air-interface depth05 radius03 n80 raw/background pair run only.
- Not exact CLT-GPR replication.
- Not finalized paper benchmark.
- Not field validation.
- Not AutoTune evaluation.

## 18) Recommended next task
- Perform focused visual interpretation on the generated n80 target_response and A-scan overlays, then decide whether to:
  1. freeze scene_037 for evidence candidate packaging, or
  2. run one constrained geometry/ROI timing refinement gate before packaging.
