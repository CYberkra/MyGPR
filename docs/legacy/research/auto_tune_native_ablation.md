# AT-002 Native Benchmark AutoTune Ablation

## Purpose

AT-002 uses the GX-003 audited native gprMax benchmark to diagnose which
processing stages improve or degrade AutoTune validation metrics before any
AutoTune scoring redesign.

This is not a new scoring method. It does not change `core.auto_tune`,
`core.processing_engine`, motion compensation, or processing algorithms.

## Dataset

Primary dataset:

```text
MyGPR-Evidence/gprmax/GX-003_audited_native_gprmax_benchmark/
```

Scenario:

```text
pipe_demo_longline_v1
```

The benchmark is native gprMax converted to MyGPR CSV:

- receiver/component: `rx1/Ez`
- shape: `2037 samples x 90 traces`
- ground truth: available
- provenance: native `_merged.out` path/hash recorded by GX-003

GX-003 is a native benchmark provenance record. AT-002 is the first diagnostic
AutoTune ablation built on top of that dataset.

## Branches

AT-002 compares these branches where practical:

- `expert_manual`: fixed reasonable experience parameters.
- `safe_default`: registry/default parameters with no AutoTune.
- `auto_tuned`: existing AutoTune applied at every supported stage.

It also runs single-stage ablations:

- `only_dewow_auto_tuned`
- `only_frequency_filter_1d_auto_tuned`
- `only_background_suppression_auto_tuned`
- `only_gain_auto_tuned`

For each ablation, the selected stage is auto-tuned while other stages stay on
the expert manual baseline.

## Fixed Pipeline

The diagnostic pipeline is:

1. `set_zero_time`
2. `dewow`
3. `frequency_filter_1d`
4. `subtracting_average_2D`
5. `energy_decay_gain`

The task intentionally does not claim AutoTune selects a globally optimal
workflow. It only compares parameter choices within a fixed chain.

## Metric Boundary

Ground-truth metrics and heuristic QC are separate.

Ground-truth metrics include target preservation, saliency, background
reduction, false positive ratio, and `truth_score` when ground truth exists.

Heuristic QC metrics remain image-quality diagnostics. They cannot prove the
result is closer to the true subsurface structure.

Ground truth is not passed into AutoTune search. It is used after processing
for validation and evidence reporting.

## Claim Boundary

AT-002 may support:

- which stage appears helpful or harmful on `pipe_demo_longline_v1`;
- whether all-stage AutoTune is better, worse, or inconclusive under recorded
  metrics;
- which AutoTune-selected parameters deserve scoring or parameter-domain review.

AT-002 must not claim:

- AutoTune outperforms manual for field UAV-GPR data;
- the current score is globally optimal;
- one native gprMax scenario is enough for paper-level generalization.

## Runner

Example:

```bash
python scripts/auto_tune_validation/run_native_ablation.py ^
  --dataset D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-003_audited_native_gprmax_benchmark ^
  --evidence-root D:\CDUT-UavGPR-Controller\MyGPR-Evidence\autotune\AT-002_native_ablation ^
  --mode normal
```

Outputs include:

- `reports/ablation_report.md`
- `manifests/ablation_summary.json`
- `manifests/evidence_manifest.json`
- `tables/stage_ablation_table.csv`
- `tables/trial_table.csv`
- `figures/input_bscan.png`
- `figures/manual_vs_auto_side_by_side.png`
- per-branch and per-stage B-scan previews

## Frozen Modules

AT-002 must not modify:

- `PythonModule/motion_compensation_v2.py`
- atomic motion compensation methods
- `core.processing_engine`
- AutoTune scoring logic
