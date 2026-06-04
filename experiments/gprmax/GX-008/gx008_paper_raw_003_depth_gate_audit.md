#!/usr/bin/env markdown
# GX-008-PAPER-RAW-003 Depth Gate Audit

## Date
2026-05-25

## Branch
main

## Base commit
90c3059c71eb41acc3169d3960bf03244ba437e5

## Remote verification
- `git rev-parse HEAD` = `90c3059c71eb41acc3169d3960bf03244ba437e5`
- `git rev-parse origin/main` = `90c3059c71eb41acc3169d3960bf03244ba437e5`
- `git ls-remote origin main` = `90c3059c71eb41acc3169d3960bf03244ba437e5`

## Baseline scene/depth
- Baseline scene: `scene_025_paper_aligned_gssi_antenna_gate_n15`
- Target geometry in baseline raw model:
  - cylinder axis endpoints: `(x=0.50, y=0.065->0.085, z=0.10)`
  - radius: `0.03 m`
- Antenna reference height in baseline: `z=0.05`
- Depth interpretation used for this gate:
  - baseline center depth relative to antenna reference plane: ~`5 cm` (`0.10 - 0.05`)
  - baseline target top relative to antenna reference plane: ~`2 cm` (`0.10 - 0.03 - 0.05`)
- This remains within the paper one-object depth range (1–10 cm) as an approximate alignment.

## Depth candidates selected
- Candidate added: depth03 (single candidate in this task)
  - `scene_029_gssi_ey_depth03_raw_gate_n31`
- Reason:
  - baseline was treated as ~depth05 center reference, so only depth03 candidate was added per single-variable rule.

## Scenes added
- `experiments/gprmax/GX-008/models/scene_029_gssi_ey_depth03_raw_gate_n31/`
  - `raw_with_target.in`
  - `background_only.in` (kept for contract only; not run in this task)
  - `materials.txt`
  - `roi_draft.json`
  - `scene_manifest_draft.json`
- Campaign updated:
  - `experiments/gprmax/GX-008/campaign_draft.yaml`

## Dry-run result
- `campaign_status: ready`
- `scene_029_gssi_ey_depth03_raw_gate_n31: ready`
- `invalid_count: 0`

## GPU wrapper result
- `scripts/run_gprmax_gpu_env.bat --check`: pass
- `scripts/run_gprmax_gpu_env.bat --smoke`: pass (noted non-blocking host-side gbk decode thread noise after JSON output)

## Raw n15 run result
- Command scope: raw-only, no background
- Scene: `scene_029_gssi_ey_depth03_raw_gate_n31`
- status: success
- return_code: 0
- runtime_seconds: `423.844`
- actual output count: `15`
- manifest:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008/scene_029_gssi_ey_depth03_raw_gate_n31/raw_with_target/run_manifest.json`

## Ey conversion status
- raw-only Ey conversion completed from existing `.out` series (no background required).
- output summary:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/scratch/GX-008_depth_gate_raw_visual/scene_029_gssi_ey_depth03_raw_gate_n31/convert_summary_raw_Ey.json`
- selected component metadata:
  - `selected_component: Ey`
  - `available_components: [Ex, Ey, Ez, Hx, Hy, Hz]`
  - `receiver_name: rx1`
  - `component_source: rxs/rx1/Ey`

## Raw visual output paths
Scratch root:
- `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/scratch/GX-008_depth_gate_raw_visual/scene_029_gssi_ey_depth03_raw_gate_n31/`

Generated:
- `raw_full_percentile_1_99.png`
- `raw_crop_best_candidate.png`
- `raw_surface_muted_display_only.png`
- `raw_trace_normalized_display_only.png`
- `raw_depth_gate_summary.md`
- `raw_bscan_Ey.npy`
- `raw_bscan_Ey.csv`
- `convert_summary_raw_Ey.json`

## Visual assessment
- Strong top surface/background band remains dominant, similar to baseline.
- Depth03 candidate shows localized lower-window structure in best-crop view, but still does not yield a clearly separable single-target arch/hyperbola in raw-only visualization.
- Visibility improvement versus baseline is limited/inconclusive at current `n=15` gate.

## Comparison with baseline
- Baseline (`scene_025`, Ey n31) and candidate (`scene_029`, Ey n15) both remain background-band dominated in full-window views.
- Candidate depth03 did not produce a decisive raw-only visibility breakthrough under this first n15 gate.
- Because trace count differs (baseline n31 vs candidate n15), comparison is directional, not final.

## Whether second candidate was run
- No second depth candidate run in this task.
- Reason:
  - task budget preserved for single-candidate gate first;
  - first candidate produced weak/inconclusive gain;
  - next step should remain single-variable but with controlled follow-up (depth05 candidate or n31 for current candidate).

## n31 decision
- `n31` for scene_029 deferred in this task.
- Reason:
  - n15 did not show a strong-enough target visibility gain to justify immediate higher-cost n31 run.

## Generated local artifacts
- All runtime/conversion/visualization outputs are local-only under `MyGPR-Evidence/scratch` and scene runtime folders.

## Files deliberately excluded
- No MyGPR-Evidence git operations.
- No generated `.out/.h5/.vti/.vtk/.vtu`.
- No generated `.csv/.npy/.png`.
- No background outputs for scene_029.
- No clutter-free outputs.

## Claim boundary
- raw-only single-variable depth diagnostic only
- background not run
- no clutter-free generated
- not exact replication
- not paper benchmark
- not AutoTune evaluation
- not field validation

## Recommended next task
- `GX-008-PAPER-RAW-004-DEPTH05-RAW-GATE`:
  keep all non-depth variables fixed and run one complementary depth05 raw-only candidate at n15 for direct depth03 vs depth05 gate comparison before any n31 follow-up.
