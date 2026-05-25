#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GX-008 PAPER RAW 012 scene_036 n80 pair run audit."""

# GX-008-PAPER-RAW-012-SCENE036-N80-PAIR-RUN Audit

## 1. Task ID
- `GX-008-PAPER-RAW-012-SCENE036-N80-PAIR-RUN`

## 2. Branch / Base / New Commit
- Branch: `main`
- Base commit: `8f7f7ee4ad0f7b82c8fb74556debaa676dd2b3ad`
- New commit: pending (this audit file only)

## 3. Remote Verification
- `git branch --show-current` -> `main`
- `git rev-parse HEAD` -> `8f7f7ee4ad0f7b82c8fb74556debaa676dd2b3ad`
- `git rev-parse origin/main` -> `8f7f7ee4ad0f7b82c8fb74556debaa676dd2b3ad`
- `git ls-remote origin main` -> `8f7f7ee4ad0f7b82c8fb74556debaa676dd2b3ad refs/heads/main`

## 4. Scene Geometry Summary
- Scene: `scene_036_gssi_ey_depth05_radius03_safe_n80_pair_gate`
- Domain: `1.0 x 0.15 x 0.40 m`
- Grid: `dx=dy=dz=0.002 m`
- Time window: `14e-9 s`
- Soil: `dry_sand_tableii (eps_r=3.0, sigma=0.001)`
- Target: elongated `PEC` cylinder along `y`
  - center `(x,z)=(0.50,0.08) m`
  - radius `0.03 m`
  - top `z=0.05 m`, bottom `z=0.11 m`
- Antenna: `antenna_like_GSSI_1500(...)`
- Conversion component: `Ey` (`rxs/rx1/Ey`)

## 5. Run Commands
- Pre-checks:
  - `python scripts/gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --dry-run`
  - `python scripts/preflight_check.py`
  - `scripts\run_gprmax_gpu_env.bat --check`
  - `scripts\run_gprmax_gpu_env.bat --smoke`
- Raw segmented run:
  - `python -m gprMax raw_with_target.in -n 20 -gpu 0`
  - `python -m gprMax raw_with_target.in -n 20 -restart 21 -gpu 0`
  - `python -m gprMax raw_with_target.in -n 20 -restart 41 -gpu 0`
  - `python -m gprMax raw_with_target.in -n 20 -restart 61 -gpu 0`
- Background segmented run:
  - `python -m gprMax background_only.in -n 20 -gpu 0`
  - `python -m gprMax background_only.in -n 20 -restart 21 -gpu 0`
  - `python -m gprMax background_only.in -n 20 -restart 41 -gpu 0`
  - `python -m gprMax background_only.in -n 20 -restart 61 -gpu 0`

## 6. Runtime per Segment
- Raw:
  - 1-20: `~0:07:41`
  - 21-40: `~0:07:05`
  - 41-60: `~0:08:27`
  - 61-80: `~0:08:38`
- Background:
  - 1-20: `~0:08:13`
  - 21-40: `~0:08:32`
  - 41-60: `~0:08:13`
  - 61-80: `~0:07:47`

## 7. Output Counts
- Raw `.out` count: `80/80`
- Background `.out` count: `80/80`
- Completion criterion met for paired n80.

## 8. Conversion Method
- Existing MyGPR chain used:
  - `scripts/gprmax_campaign_convert_scene001.py`
  - `scripts/gprmax_campaign_pair_converted.py`
- Explicit component selection:
  - `--component Ey`
- No HDF5 reader changes.

## 9. Shapes
- Raw shape: `[3636, 80]`
- Background shape: `[3636, 80]`
- Target response shape: `[3636, 80]`

## 10. Component
- Selected component: `Ey`
- Receiver path: `rxs/rx1/Ey`
- Available components: `Ex, Ey, Ez, Hx, Hy, Hz`

## 11. Visual Output Paths
- Root:
  - `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_036_gssi_ey_depth05_radius03_safe_n80_pair_gate\paired_outputs_Ey_n80\`
- Generated:
  - `raw_full_percentile_1_99.png`
  - `background_full_percentile_1_99.png`
  - `target_response_full_percentile_1_99.png`
  - `target_response_crop_best_candidate.png`
  - `target_response_trace_normalized_display_only.png`
  - `target_response_surface_muted_display_only.png`
  - `central_trace_A_scan_raw_bg_target.png`
  - `trace_30_35_40_45_A_scan_overlay.png`

## 12. Metrics
- Raw:
  - energy `95346.43`
  - mean_abs `0.144041`
  - p95_abs `1.062171`
  - p99_abs `3.175933`
- Background:
  - energy `95445.85`
  - mean_abs `0.143433`
  - p95_abs `1.069220`
  - p99_abs `3.175933`
- Target response (`raw - background`):
  - energy `71.61210`
  - mean_abs `0.004388`
  - p95_abs `0.027051`
  - p99_abs `0.064403`
  - min/max `[-0.280984, 0.289925]`
- Pair quality:
  - RMSE `0.015690`
  - PSNR `48.1753`
- ROI metric from pairing script:
  - `roi_energy_ratio = 0.816595`
- Trace markers:
  - closest target-center trace (1-based): `41`
  - strongest target-window trace (1-based): `49`

## 13. Comparison with scene_033 and scene_034
- vs scene_033 n31 baseline (`raw 0.3989 / bg 0.3281 / target_response 0.03836`, normalized metric scale):
  - scene_036 n80 completed with full aperture and stable Ey conversion.
  - target_response remains clearly separable from raw/background after subtraction.
  - metric magnitude is not directly one-to-one comparable with earlier normalized n31 summary; visual interpretation is primary here.
- vs scene_034 incomplete 78/80:
  - scene_034 showed strong shallow-target apex artifact (qualitative only, incomplete).
  - scene_036 (radius03 + cover depth05) reduces aggressive shallow-target behavior; response is still visible but less over-saturated.
  - scene_036 is the stronger conservative-paper-like candidate for next curated evidence review.

## 14. Known Limitations
- This is still one synthetic scene only.
- No field validation.
- Not exact CLT-GPR replication (antenna internals and full dataset variability not replicated).
- Surface-window energy ratio computed from simple display windows can be unstable at near-zero denominators; use as qualitative helper only.

## 15. Claim Boundary
- Conservative synthetic `GSSI/Ey`, dry-sand, PEC-cylinder, depth05/radius03, n80 raw/background pair gate only.
- Not exact CLT-GPR replication.
- Not finalized paper benchmark.
- Not field validation.
- Not AutoTune evaluation.

## 16. Recommended Next Task
- Keep scene_036 geometry fixed, perform a curated evidence packaging pass:
  - lock Ey n80 paired artifacts,
  - add consistent windowed visual panel templates,
  - and run a side-by-side controlled comparison (`scene_034` strong-target vs `scene_036` conservative-target) for report-ready selection.
