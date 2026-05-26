#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GX-008-PAPER-RAW-016-SCENE038-DEPTH07-AIR-SAND-N80-PAIR-GATE Audit

Date: 2026-05-26  
Task ID: GX-008-PAPER-RAW-016-SCENE038-DEPTH07-AIR-SAND-N80-PAIR-GATE  
Branch: `main`  
Base commit: `04a7a5a4115486772b6d2d3e596a407cba09e55e`

## 1) Remote verification

- `git rev-parse HEAD`: `04a7a5a4115486772b6d2d3e596a407cba09e55e`
- `git rev-parse origin/main`: `04a7a5a4115486772b6d2d3e596a407cba09e55e`
- `git ls-remote origin main`: `04a7a5a4115486772b6d2d3e596a407cba09e55e refs/heads/main`

## 2) Why scene_038 was created

`scene_038_gssi_ey_depth07_radius03_air_sand_interface_n80_pair_gate` was created as a single-variable depth sensitivity gate from current primary `scene_037`, to test whether paired target response remains interpretable when cover depth is increased from 0.05 m to 0.07 m.

## 3) Single-variable difference from scene_037

Only target depth was changed:

- `z_center`: `0.180 -> 0.160`
- `target_upper_z`: `0.210 -> 0.190`
- `target_lower_z`: `0.150 -> 0.130`
- `cover_depth`: `0.050 -> 0.070`

Unchanged:

- Air/sand interface (`soil_surface_z=0.260`)
- GSSI-like 1.5 GHz antenna command and standoff
- Receiver component selection (`Ey`)
- Radius (`0.03 m`)
- Domain/grid/time window
- Scan start (`0.100`), step (`0.01`), count (`n80`)
- Pairing definition (`target_response = raw - background`)

## 4) Geometry summary

- Domain: `1.0 x 0.15 x 0.40 m`
- Grid: `0.002 x 0.002 x 0.002 m`
- Dry sand: `z=0.000..0.260`
- Air/free-space: `z=0.260..0.400`
- Soil surface: `z=0.260`
- Antenna skid-bottom input z: `0.310`
- Standoff: `0.050 m`
- Target: PEC cylinder along y
- Target center: `x=0.50, z=0.160`
- Target radius: `0.03`
- Target y range: `0.002..0.148`
- Target upper/lower z: `0.190 / 0.130`
- Component: `Ey` (`rxs/rx1/Ey`)

## 5) Geometry-only result

Executed from scene model directory:

- `... -m gprMax raw_with_target.in --geometry-only`
- `... -m gprMax background_only.in --geometry-only`

Result: passed.

Verified:

- 3D mode, expected domain/grid/time window.
- Raw geometry contains PEC cylinder at expected depth.
- Background geometry excludes cylinder.
- Antenna source/receiver created; Gaussian `myGaussian` frequency `1.71e9`, y-polarized source, `Ey` receiver output.
- No geometry boundary error.

## 6) Run commands and runtime per segment

Raw (`raw_with_target.in`, `-gpu 0`):

1. `-n 20` (1-20): ~`00:08:16`
2. `-n 20 -restart 21` (21-40): ~`00:08:15`
3. `-n 20 -restart 41` (41-60): `00:08:26`
4. `-n 20 -restart 61` (61-80): `00:08:10`

Background (`background_only.in`, `-gpu 0`):

1. `-n 20` (1-20): `00:08:22`
2. `-n 20 -restart 21` (21-40): `00:08:28`
3. `-n 20 -restart 41` (41-60): `00:08:25`
4. `-n 20 -restart 61` (61-80): `00:08:25`

## 7) Output counts

- Raw output count: `80/80`
- Background output count: `80/80`
- Overwrite/boundary failure: none observed

## 8) Conversion method

Used existing MyGPR chain (no new HDF5 reader):

- `scripts/gprmax_campaign_convert_scene001.py ... --component Ey --raw-run-count 80 --background-run-count 80`
- Summary: `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\scratch\GX-008_scene038_depth07_n80_pair\convert_summary_ey_n80.json`

Pairing and preview:

- `scripts/gprmax_campaign_pair_converted.py --prefer-format npy --roi-json .../roi_draft.json`
- Pair summary: `.../paired_outputs/paired_report_summary.json`

## 9) Shape summary

- Raw shape: `[3636, 80]`
- Background shape: `[3636, 80]`
- Target response shape: `[3636, 80]`

## 10) Component summary

- Selected component: `Ey`
- Receiver name: `rx1`
- Component source: `rxs/rx1/Ey`
- Available components in `.out`: `Ex Ey Ez Hx Hy Hz`

## 11) Visual output paths

Scratch root:
`D:\CDUT-UavGPR-Controller\MyGPR-Evidence\scratch\GX-008_scene038_depth07_n80_pair\paired_outputs`

Generated:

- `raw_full_percentile_1_99.png`
- `background_full_percentile_1_99.png`
- `target_response_full_percentile_1_99.png`
- `target_response_crop_best_candidate.png`
- `target_response_trace_normalized_display_only.png`
- `target_response_surface_muted_display_only.png`
- `target_response_symmetric_absmax.png`
- `central_trace_A_scan_raw_bg_target.png`
- `trace_35_40_41_45_49_A_scan_overlay.png`
- `paired_preview_panel.png`

Additional generated:

- `trace_30_35_40_45_A_scan_overlay.png`
- `raw_preview.png`
- `background_preview.png`
- `target_response_preview.png`

## 12) Metrics

From paired metrics:

- `raw_energy`: `116912.25021293758`
- `background_energy`: `112964.74818742435`
- `target_response_energy`: `2957.0023831295566`
- `raw_mean_abs` (abs mean of raw not explicitly exported): see `raw_std/raw_mean` in json
- `background_mean_abs` (same note as above)
- `target_response_mean_abs`: `0.016448358605592208`
- `raw_p95_abs` / `background_p95_abs` / `target_response_p95_abs`: not in default pair script output
- `raw_p99_abs` / `background_p99_abs` / `target_response_p99_abs`: not in default pair script output
- `raw_background_mae`: `0.016448358605592208`
- `raw_background_mse`: `0.010165712263234174`
- `raw_background_rmse`: `0.10082515689665042`
- `raw_background_psnr`: `33.88964461481177`
- `target_to_raw_energy_ratio`: `0.02529249396657608`
- `target_to_background_energy_ratio`: `0.02617632872711295`
- `roi_energy_ratio`: `0.9995248194699562`

From extra scratch diagnostics:

- `trace_closest_target_center_1based`: `41`
- `strongest_target_window_trace_1based`: `35`
- `max_target_response_location` (sample, trace 0-based): `[889, 51]`
- `surface_window_energy`: `1.1116859592695446`
- `target_window_energy`: `2954.2897563336376`
- `target_surface_energy_ratio`: `2657.033045297839`
- `target_roi_energy`: `2955.597356042619`

## 13) Known limitations

- This task is depth sensitivity only (`depth05 -> depth07`), not model replacement.
- No curated Evidence packaging in this task.
- No scene_037 visual comparison conclusion is asserted in this audit (left for user review).
- ROI metrics are diagnostic proxies only, not detection accuracy.

## 14) Claim boundary

- Single-variable synthetic depth sensitivity gate based on scene_037 only.
- Not a replacement for `scene_037` unless later approved.
- Not exact CLT-GPR replication.
- Not finalized paper benchmark.
- Not field validation.
- Not AutoTune evaluation.

## 15) Recommended next task

`GX-008-PAPER-RAW-017-SCENE038-VISUAL-REVIEW-AND-DECISION`:
perform manual review of scene_038 paired figures against scene_037 and decide whether scene_038 is suitable for curated packaging or should remain a sensitivity side-branch.
