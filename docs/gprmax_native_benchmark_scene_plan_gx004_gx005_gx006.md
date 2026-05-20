#!/usr/bin/env markdown
# Native gprMax Benchmark Scene Plan (GX-004 / GX-005 / GX-006)

## 1) Executive summary

This document defines the executable scene design for three new native gprMax
benchmark scenes before any new simulation run:

- `GX-004`: no-target false-positive control
- `GX-005`: multi-target / varying-depth target-preservation stress
- `GX-006`: layered / complex-background robustness stress

The purpose is to create auditable scene specifications for follow-up
implementation, not to claim new performance results.

## 2) Design principles

- Preserve native provenance: `model.in` + native `.out` / `_merged.out` +
  conversion metadata must remain traceable.
- Use one consistent reduced validation lane family for comparability:
  `background_suppression -> gain`, with validation-time zero-time/dewow policy
  unchanged from current research branch conventions.
- Keep scene geometry sufficiently distinct so that each scene tests a different
  failure mode.
- Define ROI and negative-control regions *before* running AutoTune validation.
- Keep parameter scales trace-count-aware; do not encode single-scene absolute
  defaults as universal.

## 3) Shared simulation constraints

All three scenes must follow these shared constraints when later implemented:

- Native source must be preserved:
  - exact `model.in`
  - native `.out` or `_merged.out`
  - converted CSV linked to native output
- Metadata must record:
  - source shape and converted shape
  - `trace_count`, `sample_count`, `trace_spacing_m`, `time_window_ns`
  - source/receiver configuration and trajectory
  - material definitions and target summary
- If thin 2D `z` dimension is used, the limitation must be disclosed explicitly
  in report/manifests.
- ROI coordinates must be fixed and versioned before AutoTune evaluation.
- Local background ROI and negative-control ROI must be included where
  applicable.
- Scene outputs must be suitable for replaying AT-011 relative background-window
  policy unchanged.

### Shared baseline geometry envelope (for consistency)

- Domain (initial proposal): `x=1.20 m`, `y=0.60 m`, `z=0.002 m` (thin 2D)
- Grid resolution: `dx=dy=dz=0.002 m`
- Scan direction: +x
- Trace spacing: `0.01 m` (target trace count around 100-120 depending on margin)
- Time window: `18-24 ns` (final value scene-dependent, must cover direct wave,
  ground/interface response, and deeper target response where applicable)
- Source/receiver are fixed-height near-surface setup, with scan path fully
  inside safe margins and outside assumed PML zone.

## 4) GX-004 no-target scene design

### Scientific purpose

- Detect false-positive creation after processing in a scene with no buried
  target.
- Provide no-target baseline for non-target energy and false-positive proxy.
- Validate no-prior/global QC behavior without target-preservation claims.

### Scene design

- Scenario id: `gx004_no_target_false_positive_control_v1`
- Domain: `1.20 x 0.60 x 0.002 m`
- Resolution: `0.002 m`
- Background material (example baseline):
  - relative permittivity `er=9`
  - conductivity `sigma=0.01 S/m`
- No embedded discrete target object.
- Optional mild heterogeneity:
  - low-amplitude random permittivity perturbation band-limited in `x`
  - purpose: avoid unrealistically flat/noiseless scene

### Source / receiver trajectory

- Linear scan along x with fixed y
- Tx/Rx pair offset fixed per existing benchmark pattern
- Trace spacing `0.01 m`
- Trace count target: `~100`
- Time window: `20 ns` initial proposal

### Expected B-scan signature

- Direct wave + ground/interface reflection baseline
- No isolated hyperbola-like target response expected
- Any strong localized hyperbola-like anomaly after processing is a
  false-positive risk signal

### ROI definitions

- `no_target_region` ROI:
  - broad central analysis region excluding boundary artifacts
- `negative_control` ROI(s):
  - at least two non-overlapping windows where no target-like response should
    appear
- Optional `layer_interface` ROI if surface/interface band is tracked

### Local background ROI

- Multiple local background windows across shallow/mid/deep bands to avoid
  overfitting one depth range.

### Failure modes tested

- Background/gain policy creates target-like artifacts in no-target scene.
- Candidate selection prefers visually bright but structurally false anomalies.
- False-positive proxy remains high while apparent contrast improves.

### Explicit non-claims

- No target preservation metric claim is allowed in this scene.
- Visual anomaly suppression alone is not proof of correct geology.

## 5) GX-005 multi-target / varying-depth scene design

### Scientific purpose

- Evaluate whether candidate policy preserves multiple targets at different
  depths instead of optimizing one and harming another.
- Test depth-sensitive robustness of relative background-window policy.

### Scene design

- Scenario id: `gx005_multi_target_varying_depth_v1`
- Domain: `1.20 x 0.60 x 0.002 m`
- Resolution: `0.002 m`
- Background material baseline: `er=9`, `sigma=0.01 S/m`
- Target set (minimum):
  - Target A (shallow): cylindrical/pipe-like object
  - Target B (deeper): cylindrical/pipe-like object with different depth
- Example initial placement (to be adjusted for safe margin):
  - Target A center: `x=0.42 m`, depth `~0.12 m`, radius `0.03 m`
  - Target B center: `x=0.78 m`, depth `~0.24 m`, radius `0.04 m`
- Material contrast:
  - A and B can share type for depth-only test, or use distinct contrast levels
    to test weak+strong coexistence

### Source / receiver trajectory

- Linear scan along x at fixed y
- Trace spacing `0.01 m`
- Trace count target `~100`
- Time window `22-24 ns` to capture deeper target response

### Expected B-scan signature

- Two distinct hyperbola-like signatures with different apex times/depths
- Shallow target likely high-saliency; deep target lower amplitude and easier
  to suppress

### ROI definitions

- `target` ROI per target:
  - `target_A_roi`
  - `target_B_roi`
- `local_background` ROI per target:
  - adjacent lateral/depth windows near A and B
- `negative_control` ROI:
  - region away from both targets

### Local background ROI

- Must be target-specific and depth-matched where practical.
- Avoid using only one global background ROI.

### Failure modes tested

- Policy improves A but destroys B without explicit warning.
- Candidate score dominated by one target due to depth/energy imbalance.
- Large background window suppresses deep target disproportionately.

### Explicit non-claims

- A single aggregated metric is insufficient for scene success.
- Scene success requires target-specific metric reporting, not one blended score.

## 6) GX-006 layered / complex-background scene design

### Scientific purpose

- Stress-test assumptions learned from simpler background scenes.
- Evaluate interaction between background suppression and meaningful interfaces.
- Measure false-positive behavior under structured clutter.

### Scene design

- Scenario id: `gx006_layered_complex_background_v1`
- Domain: `1.20 x 0.60 x 0.002 m`
- Resolution: `0.002 m`
- Layer structure (minimum):
  - Layer 1 (shallow): `er~6`, low conductivity
  - Layer 2 (mid): `er~9`, moderate conductivity
  - Layer 3 (deep): `er~12`, higher conductivity
- Optional structured clutter:
  - gentle undulating interface or localized permittivity anomalies
- Optional target:
  - one medium-depth target added to test target preservation under layering

### Source / receiver trajectory

- Linear scan along x with fixed y
- Trace spacing `0.01 m`
- Trace count target `~100`
- Time window `24 ns` preferred to include multiple interface responses

### Expected B-scan signature

- Strong layered/interface responses across traces
- Possible target hyperbola superimposed on layered reflections
- Increased risk of confusing layer curvature with target-like anomalies

### ROI definitions

- `layer_interface` ROIs:
  - at least one ROI per major interface band
- `target` ROI if optional target exists
- `local_background` ROIs in non-target lateral zones
- `negative_control` ROI in clutter region without true target

### Local background ROI

- Must include both interface-rich and interface-sparse areas.
- Avoid only smooth-background local windows.

### Failure modes tested

- Background suppression removes meaningful layer structure.
- Near-full-line window over-suppresses structured geology-like interfaces.
- False positives increase in clutter zones despite better contrast metrics.

### Explicit non-claims

- Interface suppression is not automatically “improvement.”
- Visual smoothness cannot replace interface/target-specific checks.

## 7) Ground-truth / ROI schema requirements

Use one common schema in future scene manifests/ground-truth JSON:

- `roi_id`
- `roi_type`: `target`, `local_background`, `negative_control`,
  `layer_interface`, `no_target_region`
- `source`: `model_ground_truth`, `expert_annotation`, `auto_candidate`
- `claim_level`: `ground_truth_metric`, `heuristic_qc`, `visual_only`,
  `proposal_only`
- sample/trace coordinates (index domain)
- optional physical coordinates (`m`, `ns`)
- optional `associated_object_id`
- `notes`

## 8) Negative-control ROI requirements

- Every scene must include at least one `negative_control` ROI.
- GX-004 must include multiple negative-control windows.
- Negative-control ROIs must avoid boundaries/PML-adjacent areas.
- Negative-control metrics must be reported alongside target/interface metrics.

## 9) Expected B-scan signatures (cross-scene summary)

- GX-004: no discrete target hyperbola expected.
- GX-005: at least two target signatures at different apex times.
- GX-006: layered/interface bands dominate; optional target signature may be
  partially masked.

These are expected qualitative signatures, not success proofs by themselves.

## 10) Failure modes each scene is designed to test

- GX-004:
  - false-positive amplification under no-target condition
- GX-005:
  - shallow/deep target imbalance and single-target overfitting
- GX-006:
  - structural interface loss and clutter-induced false positives

Cross-scene:

- edge-limited candidate selection that appears stable only in one scene
- over-reliance on visual contrast without non-target risk control

## 11) Future Evidence package contract

Each future GX package should include:

- `model/model.in`
- `native/native.out` or `native/*_merged.out`
- `converted/data.csv`
- `manifests/benchmark_manifest.json`
- `manifests/ground_truth.json`
- `reports/benchmark_report.md`
- `figures/raw_bscan_preview.png`
- `figures/roi_overlay.png`
- `tables/roi_definitions.csv`
- `tables/metadata_summary.csv`

Manifest minimum fields:

- `scenario_id`
- `source_commit`
- `evidence_commit` or `pending_self_reference`
- gprMax version (if available)
- model hash
- native output hash
- converted CSV hash
- source shape
- converted shape
- `trace_count`
- `sample_count`
- `trace_spacing_m`
- `time_window_ns`
- material summary
- target summary
- ROI summary
- limitations

## 12) Implementation checklist (for follow-up task)

1. Draft `model.in` for each scene using this plan.
2. Validate Tx/Rx and target/layer positions are away from boundaries.
3. Run native gprMax simulation and collect `.out`/`_merged.out`.
4. Convert to CSV with traceable manifest and hashes.
5. Create `ground_truth.json` with required ROI schema.
6. Generate raw preview + ROI overlay before AutoTune validation.
7. Run package audit scripts and fix contract violations.
8. Only then start AT policy replay across GX-003/004/005/006.

## 13) Explicit non-claims

- This task does not generate new benchmark evidence.
- This task does not run gprMax or create native outputs.
- This task does not prove AutoTune performance.
- This task does not promote any preset.
- GX-004/005/006 designs require implementation and audit before use.
- Even after generation, these scenes remain synthetic benchmarks, not field
  validation.
- If thin 2D `z` is used, that limitation must be disclosed in all reports.

