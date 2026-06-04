#!/usr/bin/env markdown
# GX-008-PAPER-REPL-007 GSSI n31 Component Confirmation Audit

## Date
2026-05-25

## Branch
main

## Base commit
406bc469fb792d5ae2defcf234a8c8f414587401

## Remote verification
- `git rev-parse HEAD` = `406bc469fb792d5ae2defcf234a8c8f414587401`
- `git rev-parse origin/main` = `406bc469fb792d5ae2defcf234a8c8f414587401`
- `git ls-remote origin main` = `406bc469fb792d5ae2defcf234a8c8f414587401`

## Prior n15 recap
- scene_025 n15 paired was complete.
- `Ey` recommended over `Ez`; confidence `medium`.
- `Ex` remained unusable (all-zero).

## n31 outputs reused or newly run
- n31 raw/background outputs were **newly run** in this task (not reused).
- Existing n15 artifacts were kept; component comparison only changed at conversion stage.

## Raw n31 result
- status: success
- return_code: 0
- runtime_seconds: 1691.429
- requested_num_runs: 31
- actual output count: 31
- manifest:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008/scene_025_paper_aligned_gssi_antenna_gate_n15/raw_with_target/run_manifest.json`

## Background n31 result
- status: success
- return_code: 0
- runtime_seconds: 1691.362
- requested_num_runs: 31
- actual output count: 31
- manifest:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008/scene_025_paper_aligned_gssi_antenna_gate_n15/background_only/run_manifest.json`

## Ey conversion status
- conversion command used explicit `--component Ey`
- summary file:
  - `.../convert_summary_31_Ey.json`
- summary records:
  - `selected_component = Ey`
  - `receiver_name = rx1`
  - `available_components = [Ex, Ey, Ez, Hx, Hy, Hz]`
  - `component_source = rxs/rx1/Ey`

## Ey shape
- raw/background shape: `[3636, 31]`
- target_response shape after pairing: `[3636, 31]`

## Ey pairing / metrics / preview status
- pairing: success
- metrics: success
- preview/report: success
- key Ey n31 metrics:
  - `target_response_energy = 0.16654497183557598`
  - `raw_background_psnr = 70.38088639553973`
  - `roi_energy_ratio = 0.0004316263418071876`

## Ez conversion/comparison status
- used the same n31 raw/background `.out` set (no re-simulation).
- conversion with explicit `--component Ez`: success
- pairing/metrics/preview: success
- key Ez n31 metrics:
  - `target_response_energy = 0.02504927487299834`
  - `raw_background_psnr = 70.21260369503368`
  - `roi_energy_ratio = 0.00047855112515785366`

## Ey vs Ez n31 comparison
- `Ey` target_response_energy is ~6.65x `Ez`.
- `Ey` PSNR is slightly higher than `Ez`.
- both components remain low-energy relative to full benchmark expectations, but `Ey` is consistently stronger.
- visual interpretation: `Ey` target_response remains weak-to-moderate, with preliminary curvature trend clearer than `Ez`.
- horizontal band/direct-wave structure still present.

## n15 vs n31 comparison
- Ey n15:
  - `target_response_energy = 0.08120551083780915`
  - `raw_background_psnr = 70.34764021246053`
- Ey n31:
  - `target_response_energy = 0.16654497183557598`
  - `raw_background_psnr = 70.38088639553973`
- observation:
  - longer aperture (31 vs 15) increased Ey target-response energy and improved interpretability.
  - result remains a gate-level diagnostic, not full paper scan replication.

## Visual check
- Ey n31:
  - interpretable target_response exists
  - preliminary curvature visibility improved vs n15
  - still not a strong/clean benchmark-grade hyperbola
- Ez n31:
  - weaker contrast and weaker target visibility than Ey

## Selected component conclusion
- selected component for scene_025 conversion remains: `Ey`
- confidence updated: `medium_high`

## Runtime/cost observation
- n31 cost is substantial:
  - raw ~1691s
  - background ~1691s
- combined paired run ~56 minutes on current hardware.
- scaling to larger trace counts should be gated carefully.

## Generated local artifacts
- local-only outputs under:
  - `D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-008/scene_025_paper_aligned_gssi_antenna_gate_n15/`
  - `convert_summary_31_Ey.json`
  - `convert_summary_31_Ez.json`
  - `component_eval_Ey_n31/paired_outputs/*`
  - `component_eval_Ez_n31/paired_outputs/*`

## Files deliberately excluded
- no MyGPR-Evidence git operations
- no generated `.out/.h5/.vti/.vtk/.vtu`
- no generated `.csv/.npy/.png`
- no scratch bulk artifacts

## Claim boundary
- GSSI n31 component confirmation only
- not exact paper replication
- not full 80 A-scan B-scan
- not Evidence artifact
- not AutoTune evaluation
- not paper-candidate benchmark

## Recommended next task
- `GX-008-PAPER-REPL-008-GSSI-DEPTH-GATE`:
  keep component fixed at `Ey`, then do a controlled target-depth gate (single variable only) to improve curvature visibility before considering wider aperture.
