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

