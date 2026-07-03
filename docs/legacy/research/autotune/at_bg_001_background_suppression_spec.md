#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AT-BG-001 background suppression AutoTune minimal experiment specification."""

# AT-BG-001 Background Suppression AutoTune Minimal Experiment Specification

## 1. Purpose

AT-BG-001 only defines the first-phase **background suppression parameter recommendation** scope for MyGPR.
This is not full AutoTune rollout, not production scoring replacement, and not workflow-wide automatic optimization.

## 2. Current AutoTune Status

- MyGPR already has AutoTune recommendation labels, no-prior warning semantics, and comparison/report foundations.
- Synthetic paired pipeline is available: raw/background paired artifacts and `target_response = raw - background`.
- Standardized synthetic paired metrics are implemented and can be applied to curated artifacts.
- A formal AT-BG harness for controlled background suppression recommendation is not finished yet.
- No current claim that AutoTune is generally better than manual/expert settings.

## 3. Why Background Suppression First

- Background suppression is the closest deterministic seam to clutter removal behavior.
- gprMax paired artifacts (`raw`, `background`, `target_response`) provide direct synthetic reference for this stage.
- Parameter impact is visually and numerically interpretable, suitable for a first narrow AutoTune slice.
- Stage scope intentionally excludes dewow/filter/gain/denoise/migration/motion compensation for v1.

## 4. Literature-Inspired Design (Non-DL in This Phase)

Borrowed ideas from clutter-removal literature (including hybrid synthetic/measured framing):

- paired raw/background/clutter-free data design;
- target preservation must be evaluated together with clutter suppression;
- synthetic data can use full-reference metrics (MAE/MSE/RMSE/PSNR and related diagnostics);
- real no-prior data must use proxy metrics and manual-review warnings only;
- traditional clutter-removal methods are hyperparameter-sensitive.

AT-BG-001 does **not** implement CR-Net and does **not** train deep learning models.

## 5. Candidate Parameter Strategy

Phase-1 uses a small, discrete, fixed candidate grid for reproducibility and auditability.

This is **v1 reproducible search space**, not final immutable policy.
Future extensions may include:

- data-adaptive candidate generation;
- coarse-to-fine refinement;
- Bayesian optimization;
- scene-type-specific presets.

## 6. Candidate Method Space v1

### A) Mean background subtraction

- `mode`: `global_mean` or `moving_window_mean`
- `window_size`: `[5, 9, 15, 21, 31, 41]` (for moving-window mode)
- `axis`: trace

Mapping to current codebase:

- global mean style maps to background trace mean subtraction behavior.
- moving-window mean maps to `subtracting_average_2D`/running-average style local background suppression.

### B) Median background subtraction (required in v1)

- `mode`: `global_median` or `moving_window_median`
- `window_size`: `[5, 9, 15, 21, 31, 41]`
- `axis`: trace

Mapping to current codebase:

- `median_background_2D` is already available and should be included as first-phase candidate.

### C) SVD background suppression

- `remove_rank`: `[1, 2, 3]`
- optional `energy_threshold`: deferred (not required in v1)
- note: rank removal may damage target response when clutter/target components overlap in singular subspace.

Mapping to current codebase:

- `svd_bg` candidate is already supported by existing background family logic.

### Available but Deferred (if already implemented)

The following may exist in code but are not required in AT-BG-001 v1 scope:

- RPCA-style background separation;
- Hankel/subspace variants used in broader denoise/background experiments;
- workflow-wide combined stage tuning.

## 7. Fixed Workflow Scope

AT-BG v1 can only alter the background suppression step method/parameters in a fixed workflow.

- No workflow topology search.
- No simultaneous tuning of dewow/filter/gain/denoise.
- Input/output and trial metadata must be logged through manifest-compatible records.

## 8. Synthetic Evaluation Inputs

Expected inputs for each synthetic artifact trial batch:

- `raw_bscan.csv`
- `background_bscan.csv`
- `target_response.csv`
- `standard_paired_metrics.json`
- optional `roi_draft.json`
- `evidence_manifest.json`

## 9. Candidate Output Definition

Each candidate trial should output:

- `processed_bscan`
- `method`
- `parameter_set`
- `runtime_seconds`
- `warnings`
- `metrics`
- preview path (if generated)
- `selected` / `rejected` status

## 10. Scoring Proposal (Diagnostic v1)

AT-BG-001 defines score components for future harness implementation:

### A. Full-reference similarity to `target_response`

- MAE
- MSE
- RMSE
- PSNR

### B. ROI target preservation

- ROI energy ratio
- target amplitude preservation proxy
- target structural continuity proxy (if feasible with low-risk implementation)

### C. Outside-ROI clutter suppression

- outside-ROI energy reduction
- horizontal banding reduction proxy

### D. Target distortion penalty

- over-suppression warning
- target energy loss warning
- discontinuity warning

### E. False enhancement penalty

- energy increase outside ROI
- ringing/artifact warning

### F. Warning penalty

- invalid ROI
- denominator zero
- shape mismatch
- NaN/Inf
- method runtime failure

## 11. Recommendation Labels (Conservative)

Proposed labels for AT-BG trial output:

- `recommended`
- `acceptable_alternative`
- `manual_review_recommended`
- `rejected_over_suppression`
- `rejected_shape_or_runtime_failure`
- `no_prior_proxy_only`

## 12. Trial Table Schema

Proposed schema fields:

- `trial_id`
- `artifact_id`
- `scene_id`
- `method`
- `parameter_set`
- `candidate_group`
- `processed_output_path`
- `metrics_schema`
- `mae`
- `mse`
- `rmse`
- `psnr`
- `roi_energy_ratio`
- `outside_roi_clutter_proxy`
- `target_distortion_warning`
- `false_enhancement_warning`
- `runtime_seconds`
- `warnings`
- `selected`
- `recommendation_label`
- `claim_boundary`

## 13. Output Artifacts for Future Harness (AT-BG-002)

Recommended AT-BG-002 diagnostic outputs:

- `trial_table.json`
- `trial_table.csv`
- `selected_parameters.json`
- `branch_comparison_report.md`
- `candidate_preview_panel.png`
- `background_suppression_autotune_manifest.json`

## 14. Real No-Prior Policy

For real no-prior field data, `target_response` is absent. Full-reference metrics must not be used as truth metrics.

Allowed only:

- SCR-like proxy metrics
- horizontal clutter reduction proxy
- trace continuity checks
- artifact/ringing warnings
- visual/manual review flags
- no-prior risk labels

Explicit boundary:

Real no-prior proxy metrics cannot prove closer-to-truth underground structure.

## 15. UI Alignment (Future Only)

Future frontend slice should focus on **background suppression AutoTune only**.

v1 panel should show:

- method candidates
- candidate grid
- trial table
- selected parameters
- preview comparison
- warning labels
- claim boundary

v1 panel should not claim:

- full-workflow automatic optimizer;
- expert replacement;
- real no-prior ground-truth correctness.

## 16. Claim Boundary

Allowed claims:

- MyGPR can compare background suppression candidates on synthetic paired scenes.
- MyGPR can use synthetic `target_response` as reference for background suppression diagnostics.
- MyGPR can output trial table, metrics, report, and explicit claim boundary.
- v1 discrete candidate grid is reproducible and auditable.

Disallowed claims:

- full AutoTune workflow is complete;
- AutoTune is universally better than experts/manual tuning;
- real no-prior outputs are closer to true underground structure;
- CR-Net is reproduced;
- CLT-GPR is fully replicated;
- v1 discrete grid is final globally optimal strategy.

## 17. Recommended Next Task

`AT-BG-002-HARNESS-DRAFT`

Scope expectation:

- implement diagnostic/report harness path only;
- do not modify production AutoTune scoring semantics in this phase.

## 18. AT-BG-002 Harness Draft Status

AT-BG-002 diagnostic harness draft is implemented as:

- `core/autotune_background_suppression.py`
- `scripts/autotune_background_suppression_diagnostic.py`

Current draft outputs:

- `trial_table.json`
- `trial_table.csv`
- `selected_parameters.json`
- `background_suppression_autotune_report.md`
- `background_suppression_autotune_manifest.json`

Status boundary:

- diagnostic/report path only;
- not wired into production AutoTune scoring;
- not UI-integrated in this phase.
