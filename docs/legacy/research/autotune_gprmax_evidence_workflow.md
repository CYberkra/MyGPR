# AutoTune gprMax Evidence Workflow

This note is the durable contract for MyGPR's current gprMax truth-validation
Evidence workflow. It describes the deterministic research path; it does not
introduce machine learning, Bayesian optimization, or a new UI page.

## Input Chain

The gprMax data producer should provide:

- a gprMax `.out` or `_merged.out` file;
- a dataset manifest such as `*_manifest.json`;
- `ground_truth.yaml` using the `gprmax_ground_truth_v1` schema.

The manifest should identify the primary output file and the ground-truth
sidecar. The sidecar ROI intervals are zero-based closed intervals. MyGPR
converts them to Python half-open intervals before computing metrics.

## MyGPR Reading Chain

1. `read_gprmax_out()` loads the primary gprMax `.out` data.
2. The reader searches for a nearby manifest.
3. `core.gprmax_ground_truth.load_ground_truth_from_manifest()` reads and
   converts `ground_truth.yaml`.
4. The converted truth payload is attached to `header_info["ground_truth"]`.
5. Manual baseline vs AutoTune comparison runs through
   `core.auto_tune_comparison.run_auto_tune_comparison()`.
6. Truth metrics are computed for both manual and automatic branches.
7. `core.auto_tune_comparison_export.export_auto_tune_comparison_artifacts()`
   writes the Evidence bundle.

## Truth Metrics

Truth-aware metrics are validation outputs, not the same thing as visual
sharpness. Current metrics include:

- `truth_score`: combined validation score.
- `truth_target_energy_preservation`: whether target ROI energy is preserved.
- `truth_target_saliency_gain`: whether target contrast improves relative to
  background.
- `truth_background_energy_reduction`: whether provided background ROIs are
  suppressed.
- `truth_false_positive_ratio`: whether strong background responses risk false
  positives.
- `truth_target_count`: number of preserved targets in the truth contract.

These metrics should be interpreted together. A bright target with excessive
background amplification is not automatically better.

## Evidence Bundle

The export directory is one bundle folder containing fixed filenames:

- `comparison_summary.json`
- `evidence_manifest.json`
- `converted_ground_truth.json` when ground truth is available
- `raw_ground_truth.json` when `raw_sidecar` is available
- `truth_metrics.json`
- `workflow_params.json`
- `trial_table.csv`
- `trial_table.json`
- `params_table.csv`
- `metrics_table.csv`
- `manual_bscan.png`
- `auto_bscan.png`
- `side_by_side.png`
- `comparison_report.md`
- `evidence_bundle.zip`

Unavailable files are recorded as `missing` in `evidence_manifest.json` rather
than being silently ignored.

## Reproduction Steps

1. Load the gprMax `.out` through MyGPR.
2. Confirm `header_info["ground_truth"]` is present when truth validation is
   expected.
3. Open the existing AutoTune page and use the manual/automatic comparison.
4. Export Evidence.
5. Review `comparison_report.md` for a readable meeting/paper summary.
6. Review `evidence_manifest.json`, `workflow_params.json`, and
   `trial_table.json` for audit and reproduction details.

## What Evidence Is

Evidence is an output record for review, reporting, and later audit. It captures
the input reference, ground-truth conversion summary, selected parameters,
candidate trials where available, truth metrics, locked-scale images, and the
current Git commit.

## What Evidence Is Not

Evidence is not the scoring input for the same AutoTune run. It should not be
used to retroactively change the selected parameters in the run that produced
it. It also does not prove performance on all field data; controlled gprMax ROI
validation and real UAV-GPR validation answer different questions.

ground_truth.yaml is an input to truth-aware validation. AutoTune Evidence is an
output record for reproducibility, review, and reporting. Evidence may support
future replay/import, but it must not be treated as the scoring input for the
same AutoTune run.

## Why ML Is Not Implemented Yet

Machine learning and Bayesian optimization remain future extensions. The current
priority is a deterministic, auditable validation chain:

`gprMax output -> MyGPR reader -> converted ground truth -> manual/AutoTune
comparison -> truth metrics -> Evidence bundle`

This keeps the research claim reviewable before adding learned recommendation
models.

## Future Extension

- Evidence replay/import from `evidence_manifest.json`.
- More gprMax scenarios and ROI types.
- Better multi-objective scoring and objective functions.
- Machine-learning parameter recommendation after the evidence contract is
  stable.
- Bayesian optimization after deterministic candidate domains and validation
  metrics are stable.
