# AutoTune Research Validation Baseline

## Scope

This document defines the AT-001 AutoTune research validation baseline. The
goal is not to make AutoTune appear to win. The goal is to create reproducible,
auditable evidence comparing a reasonable manual expert baseline with
per-step auto-tuned parameters.

The current validation scope is deterministic, rule-based AutoTune for
processing parameters. It does not implement machine learning, Bayesian
optimization, surrogate models, or global workflow search.

## Supported Validation Design

The AT-001 runner uses a fixed processing chain:

1. `set_zero_time`
2. `dewow`
3. `frequency_filter_1d`
4. `subtracting_average_2D`
5. `energy_decay_gain`

The manual branch uses reasonable experience parameters for gprMax impulse-like
data. The auto branch starts from the same baseline and lets existing
`auto_tune_method()` choose per-step parameters from the current B-scan state.

Manual parameters must not be intentionally bad. Manual parameters must not be
forced to match AutoTune parameters as the main experiment. If a manual branch
fails early, the report must mark the branch invalid and avoid presenting the
comparison as a fair manual-vs-auto conclusion.

## Zero-Time Policy In Validation

Validation runners now support context-aware zero-time policy control:

- `legacy_default`: keep historical behavior.
- `explicit_only_fixed_zero`: if `set_zero_time` exists but `new_zero_time` is
  missing, force `new_zero_time=0.0` and disable implicit tuning for this step.
- `excluded`: remove `set_zero_time` from the validation lane.

For native gprMax-converted datasets (`source.kind=native_gprmax_converted`),
validation defaults to `explicit_only_fixed_zero` to prevent destructive
implicit shifts from method defaults.

This policy is validation-only and does not remove user ability to set zero-time
explicitly in normal workflows.

## Reusable No-Zero-Time Preset

AT-005A no-zero-time gain validation is treated as a reusable validation preset:

- zero-time excluded,
- lane separation for background-only and background+gain variants,
- ground-truth metrics for GX-003-like benchmark data,
- heuristic visual QC only for field-data lanes,
- AGC marked as display-oriented and non-amplitude-preserving.

## Post-Policy Rerun (AT-007)

AT-007 reruns AT-002-style ablation and AT-003-style stepwise diagnosis after
AT-006 policy hardening. It compares pre-fix historical evidence vs post-fix
results without overwriting old artifacts.

The purpose is to verify whether implicit default zero-time collapse is removed
and to identify the next bottleneck step. AT-007 still does not authorize an
overall AutoTune-superiority claim.

## No-Dewow Post-Fix Lane (AT-008A)

AT-008A is a post-AT-006/AT-007 lane that excludes both:

1. `set_zero_time`
2. `dewow`

in its primary GX-003 validation chain:

`background_suppression -> gain`

AT-008A exists because AT-007 identified `dewow` as the first post-fix
signal-loss bottleneck after the unsafe implicit zero-time shift was removed.
This does **not** mean dewow is globally invalid. In AT-008A, dewow is kept as
an optional diagnostic side lane with fixed windows (for example 256/512),
while the main conclusion is based on the no-dewow primary lanes.

AT-008A keeps the same claim boundary:

- ground-truth metrics are only for GX-003-like validated benchmark data;
- heuristic visual QC remains diagnostic only;
- AGC remains display-oriented and non-amplitude-preserving;
- no overall AutoTune-superiority claim is allowed without stronger multi-scene evidence.

## Background/Gain Policy Refinement (AT-009)

AT-009 refines the AT-008A reduced primary lane:

`background_suppression -> gain`

with both `set_zero_time` and `dewow` excluded in the primary validation path.

AT-009 focuses on:

1. converging `subtracting_average_2D` (`ntraces`) candidate domain on GX-003;
2. comparing gain policy variants under the same background setting;
3. constrained AutoTune comparison inside the bounded domain only.

AT-009 does not redesign global AutoTune scoring. It is a bounded policy
refinement artifact and keeps the same claim boundary:

- ground-truth metrics and heuristic QC are separated;
- AGC is display-oriented/non-amplitude-preserving;
- no overall AutoTune-superiority claim is allowed unless stronger evidence supports it.

## Metric Boundary

Heuristic QC metrics and ground-truth metrics are separate.

- Heuristic QC metrics measure image sanity, preservation, contrast, clipping,
  hot pixels, and related quality signals.
- Ground-truth metrics require a validated target/background manifest.

If no ground truth is available, the evidence is heuristic QC only. Such a run
must not claim that a result is closer to the real underground structure.

## Data Used In AT-001

The initial baseline uses:

```text
sample_data/gprmax_benchmarks/cylinder_single_v1/mygpr_bscan.csv
```

This fixture is small, deterministic, and includes `ground_truth.json`. It is
appropriate for a first research-validation baseline and CI smoke coverage. It
is not a field-data generalization result.

## Claims Allowed Now

Current evidence can support:

- MyGPR can generate stepwise manual-vs-auto AutoTune evidence.
- Each branch records parameters, runtime warnings, QC metrics, sanity warnings,
  branch invalid reasons, and B-scan previews.
- Ground truth, when available, is used for validation/reporting and not for
  AutoTune search.

## Claims Not Allowed Now

Current evidence cannot support:

- AutoTune is globally optimal.
- AutoTune chooses the best processing workflow.
- AutoTune generalizes to all UAV-GPR field data.
- Heuristic QC alone proves geological correctness.

## Runner

Example:

```bash
python scripts/auto_tune_validation/run_stepwise_validation.py ^
  --evidence-root D:\CDUT-UavGPR-Controller\MyGPR-Evidence\autotune\AT-001_research_validation_baseline ^
  --dataset cylinder_single_v1 ^
  --mode smoke
```

Outputs include:

- `reports/comparison_report.md`
- `manifests/evidence_manifest.json`
- `manifests/stepwise_report.json`
- `manifests/comparison_summary.json`
- `tables/trial_table.csv`
- `tables/trial_table.json`
- `figures/manual_bscan.png`
- `figures/auto_bscan.png`
- `figures/side_by_side.png`
- stepwise manual/auto preview PNGs

## Implementation Boundary

The AT-001 runner does not modify:

- `motion_compensation_v2`
- atomic motion compensation methods
- `core.processing_engine`
- AutoTune scoring logic
