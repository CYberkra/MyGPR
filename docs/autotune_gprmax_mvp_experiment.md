# AutoTune gprMax MVP Experiment

## Current Research Question

Can MyGPR AutoTune preserve known gprMax targets, suppress background, and
produce a reproducible Evidence bundle that is suitable for thesis, paper, and
group-meeting review?

This MVP answers that question with a minimal controlled gprMax dataset contract:

- one gprMax `.out` file;
- one `*_manifest.json`;
- one `ground_truth.yaml`;
- one manual baseline vs AutoTune comparison;
- one exported Evidence bundle.

## Method Boundary

The current method is **truth-validated AutoTune**, not **truth-guided
AutoTune**.

Ground truth is not passed into the AutoTune search call for the same run.
Instead:

1. AutoTune selects parameters from data, method defaults, candidate domains,
   and ordinary quality metrics.
2. Manual and automatic branches are processed.
3. gprMax ground truth is used after processing to compute validation metrics.
4. The validation metrics and artifacts are exported as Evidence.

This boundary matters because using the same ground truth to choose parameters
and then claim validation would overstate the result.

## Current Experiment

The MVP smoke uses a pipe / cylinder-style synthetic gprMax fixture in tests. The
smoke script also supports real gprMax dataset folders or manifest files.

Input:

- gprMax `.out` or `_merged.out`
- manifest with `primary_out_file`, optional `metadata_file`, and
  `ground_truth_file`
- `ground_truth.yaml` with zero-based closed ROI ranges

Baseline:

- manual / experience profile parameters

Automatic:

- AutoTune selected parameters

Output:

- `comparison_summary.json`
- `evidence_manifest.json`
- `converted_ground_truth.json`
- `raw_ground_truth.json` when available
- `truth_metrics.json`
- `workflow_params.json`
- `trial_table.csv`
- `trial_table.json`
- locked-scale B-scan PNGs
- `comparison_report.md`
- `evidence_bundle.zip`

## Metrics

The MVP records these truth-aware validation metrics:

- `truth_target_energy_preservation`
- `truth_target_saliency_gain`
- `truth_background_energy_reduction`
- `truth_false_positive_ratio`
- `truth_score`

Generic comparison metrics such as `comparison_score` are still exported. The
report should read these metrics together; a higher `truth_score` alone is not a
complete proof of field performance.

## How To Run

```bash
python scripts/gprmax_benchmark/run_autotune_evidence_smoke.py --dataset path/to/dataset_or_manifest --output output/gprmax_autotune_smoke
```

For tests, the dataset is generated as a small synthetic HDF5 `.out` fixture.
This keeps the repository lightweight and avoids committing large gprMax binary
outputs.

## Risks

- `truth_score` is a heuristic composite score.
- A single pipe / cylinder demo is not enough for broad claims.
- Multiple scenes, noise levels, media contrasts, and ROI layouts are needed.
- AGC, background suppression, migration, and gain can change amplitude or
  geometry interpretation.
- The validation should not optimize for images that are only brighter or
  cleaner.
- Real UAV-GPR field data still needs separate validation because its structures
  and sensor errors are not fully represented by one controlled gprMax scene.

## Next Phase

After the MVP smoke remains stable:

1. Add more gprMax scenarios.
2. Add noise and medium-contrast variations.
3. Compare more realistic manual baselines.
4. Add Evidence replay / preload.
5. Consider machine learning or Bayesian optimization only after the deterministic
   evidence contract and multi-scenario validation are stable.
