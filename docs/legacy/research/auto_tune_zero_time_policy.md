# AutoTune Zero-Time Policy (AT-006)

## Why zero-time is risky when implicit

`set_zero_time` has a method-level default `new_zero_time=5.0` ns.
In validation runners, if branch params are empty (`{}`), runtime preparation keeps that field missing and the method falls back to 5.0 ns.

For GX-003 native gprMax-converted data (`dt≈0.011793 ns/sample`), this maps to approximately:

- `5.0 / 0.011793 ≈ 423` samples shift.

Such a shift can remove or severely attenuate the effective signal band used by downstream sanity checks, producing branch-invalid results before meaningful comparison.

## Root cause path in prior validation

The unsafe path was:

1. validation branch used `set_zero_time` with empty params
2. `prepare_runtime_params()` did not inject `new_zero_time`
3. `method_set_zero_time()` used default `new_zero_time=5.0`
4. large shift happened implicitly on native gprMax-converted data

This is a validation-path policy issue, not a proof that `set_zero_time` is always wrong.

## AT-006 policy

Validation runners now infer dataset context and apply explicit policy:

- `zero_time_policy=explicit_only_fixed_zero` for `native_gprmax_converted` (and gprMax impulse-like validation context).
- If `set_zero_time` is present but `new_zero_time` is missing:
  - force fixed zero (`new_zero_time=0.0`)
  - disable implicit zero-time auto-tuning for that step
  - record policy notes/warnings in step metadata
- If user explicitly sets `new_zero_time`, that explicit value is respected in validation.

## When zero-time correction is valid

Zero-time can be valid when:

- first-break picking is physically justified for that acquisition mode;
- sampling/time-window semantics are confirmed for the dataset;
- ROI/reference handling is aligned with any sample-axis shift;
- stepwise evidence confirms no catastrophic early-stage signal collapse.

## When zero-time should be disabled/fixed-zero

For current native gprMax-converted validation chains (GX-003-like), zero-time should default to:

- excluded, or
- fixed zero unless explicitly requested.

This prevents hidden destructive shifts from dominating validation outcomes.

## No-zero-time validation vs normal field preprocessing

No-zero-time validation mode is a **research control lane**:

- purpose: isolate background/gain behavior and avoid implicit crop artifacts
- applies to native benchmark validation and ablation contexts
- not a blanket replacement for normal field preprocessing workflows

Normal field workflows may still use `set_zero_time` explicitly based on domain judgment and data characteristics.
