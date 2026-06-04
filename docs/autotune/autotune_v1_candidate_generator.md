# AutoTune V1 bounded candidate generator

Status: `final_candidate_backend_contract`  
Introduced: MyGPR `0.8.49`

## Purpose

`core/autotune_candidate_generator.py` converts the AutoTune V1 profile/recipe configuration contract into a bounded candidate parameter space. It does not execute algorithms, does not select a winner, and does not change existing scoring defaults.

The generator supports the V1 design principle:

```text
fixed candidate table + data-adaptive supplements + profile caps + candidate_space_hash
```

## Inputs

- Optional 2D B-scan array.
- Optional metadata: `dt_seconds`, `dt_ns`, `total_time_ns`, `center_frequency_hz`, `center_frequency`, `trace_spacing_m`, `target_lateral_scale_m`, `velocity_m_per_ns`.
- Target profile / goal label, resolved through `AutoTuneV1Config` aliases.
- Flags: `include_display_only`, `include_experimental`.

## Outputs

- `CandidateGenerationResult.profile_id`
- `recipe_ids`
- lightweight feature summary
- candidate list grouped by category
- deterministic `candidate_space_hash`
- warning tags for Evidence / manifest use

## Candidate categories

- `background_suppression`: mean, median, sliding mean/median, SVD rank sweep.
- `dewow`: fixed windows plus 1T/2T candidates when frequency metadata exists.
- `bandpass`: Nyquist and center/dominant-frequency based presets.
- `gain`: SEC/exponential metric-safe candidates plus AGC display-only candidates.
- `denoise`: light median, light Savitzky-Golay, optional Hampel spike removal.
- `migration`: disabled by default; object-like velocity sweep remains experimental.

## Profile caps

Interface-like, landslide, wet-zone and deep-weak-reflector profiles intentionally cap stronger background removal and aggressive high-pass behavior. In particular, landslide/interface profiles cap default SVD removal to rank 1 because low-rank components may contain true horizontal or interface-like reflectors.

## Claim boundary

This module only defines a candidate space. It is not evidence that a candidate is best, and it is not a no-prior geological truth estimator. Any final recommendation still requires scoring mode separation:

- `synthetic_paired`: full-reference scoring against `target_response` is allowed.
- `real_no_prior`: heuristic risk scoring only; `manual_review_required` remains required.
