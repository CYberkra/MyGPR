#!/usr/bin/env markdown
# Synthetic Paired Metrics Specification

## Scope

This specification defines standardized metrics for synthetic paired gprMax artifacts with:

- `raw_with_target`
- `background_only`
- `target_response = raw - background`

It is designed for GX-008 style synthetic evaluation and fixed-workflow parameter recommendation analysis.

## Non-Scope

These metrics are **not** ground-truth correctness proof for real no-prior field data.
They must not be used as standalone evidence for:

- real underground target correctness
- field-performance superiority
- universal AutoTune superiority

## Metrics (Current Implementation)

- `raw_shape`, `background_shape`, `target_response_shape`
- `raw_energy`, `background_energy`, `target_response_energy`
- `target_to_background_energy_ratio`
- `target_to_raw_energy_ratio`
- `mean_abs_target_response`
- `max_abs_target_response`
- `raw_background_mae`
- `raw_background_mse`
- `raw_background_rmse`
- `raw_background_psnr`
- `sparsity_or_concentration_proxy`
- `roi_energy_ratio` (optional ROI)
- `warnings`

Backward-compatible fields are retained:

- `abs_difference_mean`
- `abs_difference_max`

## Definitions

- Energy: `sum(x^2)` over all samples/traces.
- MAE: `mean(abs(raw - background))`.
- MSE: `mean((raw - background)^2)`.
- RMSE: `sqrt(MSE)`.
- PSNR: `20*log10(peak) - 10*log10(MSE)` where `peak=max(max(abs(raw)), max(abs(background)))`.
- Concentration proxy: `max_abs_target_response / mean_abs_target_response` (higher means more concentrated response).

## Zero / Denominator Handling

- If denominator is zero, ratio returns `null` and a warning entry is emitted.
- If `MSE == 0`, PSNR is mathematically infinite; current output sets `raw_background_psnr = null` with warning.
- No metric calculation may crash due to zero denominators.

## ROI Support (Optional)

ROI format:

```json
{
  "sample_range": [start, end],
  "trace_range": [start, end]
}
```

- Ranges use Python slice semantics `[start, end)` and must satisfy bounds.
- `roi_energy_ratio = roi_target_energy / total_target_energy`.
- Invalid ROI does not crash computation; it returns `roi_energy_ratio = null` with warnings.

## AutoTune Downstream Use

These metrics are suitable for synthetic fixed-workflow comparisons, for example:

- baseline branch vs manual/default branch vs auto-recommendation branch
- clutter suppression vs target preservation trade-off checks
- safety/diagnostic gating in synthetic experiments

They should be used with claim-boundary controls in reports and Evidence manifests.

## Claim Boundary

- Synthetic paired diagnostics only.
- Not real-field underground truth proof.
- Not standalone AutoTune superiority proof.
- Use together with scenario metadata, ROI assumptions, and manual review where required.

## Future Optional Metrics

Optional future additions (not implemented in this task):

- MS-SSIM / SSIM (dependency-gated)
- ROI target preservation score
- clutter suppression ratio outside ROI
- distortion penalties under smoothing/aggressive background suppression
