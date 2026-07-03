# gprMax AutoTune Iteration Log

This log records auditable MyGPR iterations for the gprMax -> AutoTune ->
truth metrics -> Evidence workflow. Large generated `.out` files and report
bundles stay under `output/` and are not committed.

## Iteration 1 - Contract Readiness Audit

- branch: `codex/research-gprmax-autotune`
- base commit: `37c2d94`
- new commit: `8d62c18 test: audit gprMax dataset contract readiness`
- goal: add a non-destructive audit tool for gprMax dataset folders before
  running AutoTune Evidence.
- changed files:
  - `scripts/gprmax_benchmark/audit_gprmax_dataset_contract.py`
  - `tests/test_gprmax_dataset_audit.py`
- commands run:
  - `pytest tests/test_gprmax_dataset_audit.py tests/test_gprmax_dataset_contract.py tests/test_gprmax_autotune_evidence_smoke.py -q`
  - `python scripts/preflight_check.py`
- output paths:
  - `output/gprmax_contract_audit/gprmax_v317_audit.json`
- result:
  - `E:\gprMax\gprMax-v.3.1.7\examples\gprmax\sim_pipe_001` was not ready at that time because `.out`, `.in`, and manifest were missing.
- next step:
  - wait for or generate a complete `.out + manifest + ground_truth.yaml` package.

## Iteration 2 - First Real gprMax Smoke

- branch: `codex/research-gprmax-autotune`
- base commit: `8d62c18`
- new commit: `60b06ec fix: stabilize gprMax truth contract warnings`
- goal: ingest the first real `sim_pipe_001` gprMax package and run MyGPR
  AutoTune Evidence smoke.
- changed files:
  - `core/gprmax_dataset_contract.py`
  - `core/gprmax_ground_truth.py`
  - `tests/test_gprmax_dataset_contract.py`
  - `tests/test_gprmax_ground_truth.py`
- commands run:
  - `python scripts/gprmax_benchmark/audit_gprmax_dataset_contract.py --path E:\gprMax\gprMax-v.3.1.7\examples\gprmax\sim_pipe_001 --output-json output\gprmax_contract_audit\sim_pipe_001_ready_audit.json`
  - `python scripts/gprmax_benchmark/run_autotune_evidence_smoke.py --dataset E:\gprMax\gprMax-v.3.1.7\examples\gprmax\sim_pipe_001\sim_pipe_001_manifest.json --output output\gprmax_autotune_smoke_real --bundle-name sim_pipe_001_real_smoke_after_contract_fix`
  - `python scripts/gprmax_benchmark/run_autotune_evidence_smoke.py --dataset E:\gprMax\gprMax-v.3.1.7\examples\gprmax\sim_pipe_001\sim_pipe_001_manifest.json --output output\gprmax_autotune_smoke_real --bundle-name sim_pipe_001_pipeline_smoke --pipeline dewow,subtracting_average_2D,agcGain`
  - `pytest tests/test_gprmax_ground_truth.py tests/test_gprmax_dataset_contract.py tests/test_gprmax_autotune_evidence_smoke.py tests/test_gprmax_dataset_audit.py -q`
  - `python scripts/preflight_check.py`
- output paths:
  - `output/gprmax_autotune_smoke_real/sim_pipe_001_real_smoke_after_contract_fix/comparison_report.md`
  - `output/gprmax_autotune_smoke_real/sim_pipe_001_pipeline_smoke/comparison_report.md`
- metrics:
  - `sim_pipe_001` `dewow`: manual truth_score `1.203498`, AutoTune truth_score `1.202911`
  - `sim_pipe_001` `dewow,subtracting_average_2D,agcGain`: manual truth_score `1.203347`, AutoTune truth_score `1.203481`
- limitations:
  - 12 traces are enough for smoke but not enough for a full hyperbola.
  - `ground_truth.yaml` had `target_roi` and `background_roi` completely overlapping.
- next step:
  - generate a longline dataset with non-overlapping target/background ROIs.

## Iteration 3 - Longline Pipe Scenario

- branch: `codex/research-gprmax-autotune`
- base commit: `60b06ec`
- new commit: pending
- goal: generate a longer gprMax pipe/cylinder scenario from the existing
  gprMax GUI resources, then run MyGPR Evidence with a standard pipeline.
- changed files:
  - `scripts/gprmax_benchmark/_uavgpr_gprmax_worker.py`
  - `scripts/gprmax_benchmark/run_uavgpr_gprmax_package.py`
  - `tests/test_gprmax_uavgpr_package_runner.py`
  - `docs/gprmax_autotune_iteration_log.md`
  - `docs/autotune_gprmax_pipe_demo_report.md`
- gprMax command:
  - `python scripts/gprmax_benchmark/run_uavgpr_gprmax_package.py --gprmax-root E:\gprMax\gprMax-v.3.1.7 --output-root output\gprmax_datasets --output-name pipe_demo_longline_v1 --preset uav_pipe_gain_workflow_bscan --traces 90 --run-timeout-s 1800`
- generated dataset:
  - `output/gprmax_datasets/pipe_demo_longline_v1/pipe_demo_longline_v1.in`
  - `output/gprmax_datasets/pipe_demo_longline_v1/pipe_demo_longline_v1_merged.out`
  - `output/gprmax_datasets/pipe_demo_longline_v1/pipe_demo_longline_v1_manifest.json`
  - `output/gprmax_datasets/pipe_demo_longline_v1/ground_truth.yaml`
  - `output/gprmax_datasets/pipe_demo_longline_v1/pipe_demo_longline_v1_metadata.json`
  - `output/gprmax_datasets/pipe_demo_longline_v1/README.md`
- generation result:
  - 90 traces, 2037 samples, 0.01 m trace spacing.
  - CPU runtime about 262 s.
  - target ROI: samples `[760, 882]`, traces `[38, 50]`.
  - background ROI: samples `[760, 882]`, traces `[77, 89]`.
- audit command:
  - `python scripts/gprmax_benchmark/audit_gprmax_dataset_contract.py --path output\gprmax_datasets\pipe_demo_longline_v1 --output-json output\gprmax_contract_audit\pipe_demo_longline_v1_audit.json`
- audit result:
  - READY.
- Evidence commands:
  - `python scripts/gprmax_benchmark/run_autotune_evidence_smoke.py --dataset output\gprmax_datasets\pipe_demo_longline_v1\pipe_demo_longline_v1_manifest.json --output output\gprmax_autotune_real_smoke --bundle-name pipe_demo_longline_v1_dewow --pipeline dewow`
  - `python scripts/gprmax_benchmark/run_autotune_evidence_smoke.py --dataset output\gprmax_datasets\pipe_demo_longline_v1\pipe_demo_longline_v1_manifest.json --output output\gprmax_autotune_real_smoke --bundle-name pipe_demo_longline_v1_standard --pipeline dewow,subtracting_average_2D,sec_gain`
  - `python scripts/gprmax_benchmark/run_autotune_evidence_smoke.py --dataset output\gprmax_datasets\pipe_demo_longline_v1\pipe_demo_longline_v1_manifest.json --output output\gprmax_autotune_real_smoke --bundle-name pipe_demo_longline_v1_full_standard --pipeline set_zero_time,dewow,subtracting_average_2D,sec_gain`
- Evidence paths:
  - `output/gprmax_autotune_real_smoke/pipe_demo_longline_v1_dewow/comparison_report.md`
  - `output/gprmax_autotune_real_smoke/pipe_demo_longline_v1_standard/comparison_report.md`
  - `output/gprmax_autotune_real_smoke/pipe_demo_longline_v1_full_standard/comparison_report.md`
- metrics:

| scenario_id | pipeline | manual truth_score | AutoTune truth_score | delta | manual false_positive | AutoTune false_positive | main report |
|---|---|---:|---:|---:|---:|---:|---|
| `pipe_demo_longline_v1` | `dewow` | 3.077809 | 3.075190 | -0.002619 | 0.229146 | 0.228918 | `output/gprmax_autotune_real_smoke/pipe_demo_longline_v1_dewow/comparison_report.md` |
| `pipe_demo_longline_v1` | `dewow,subtracting_average_2D,sec_gain` | -0.104267 | 1.693496 | 1.797763 | 1.936212 | 0.412069 | `output/gprmax_autotune_real_smoke/pipe_demo_longline_v1_standard/comparison_report.md` |
| `pipe_demo_longline_v1` | `set_zero_time,dewow,subtracting_average_2D,sec_gain` | 0.403869 | 0.730552 | 0.326682 | 1.107538 | 0.546713 | `output/gprmax_autotune_real_smoke/pipe_demo_longline_v1_full_standard/comparison_report.md` |

- interpretation:
  - The standard pipeline is the best current group-meeting evidence because
    AutoTune improves `truth_score` and reduces `truth_false_positive_ratio`
    while preserving a visible pipe hyperbola.
  - The `set_zero_time` pipeline runs but is less stable for this dataset and
    should not be the main paper claim yet.
- limitations:
  - This is still one controlled scene.
  - Manual baseline is the current default/experience profile, not a true expert-tuned baseline.
  - AutoTune warns that some selected parameter domains have low confidence or boundary risks.
- next step:
  - run a compact multi-scenario matrix: shallow/deep depth and one layered/weak-target case.

## Iteration 4 - Compact Multi-Scenario Matrix

- branch: `codex/research-gprmax-autotune`
- base commit: `60b06ec`
- new commit: current checkpoint (`scripts: generate gprMax AutoTune validation bridge`)
- goal: extend the longline pipe result into a compact scenario matrix that can
  support group-meeting and paper-methods discussion.
- changed files:
  - `scripts/gprmax_benchmark/_uavgpr_gprmax_worker.py`
  - `scripts/gprmax_benchmark/run_uavgpr_gprmax_package.py`
  - `tests/test_gprmax_uavgpr_package_runner.py`
  - `docs/gprmax_autotune_iteration_log.md`
  - `docs/autotune_gprmax_pipe_demo_report.md`
- generated datasets:
  - `output/gprmax_datasets/pipe_demo_longline_v1/`
  - `output/gprmax_datasets/cylinder_depth_shallow_v1/`
  - `output/gprmax_datasets/cylinder_depth_deep_v1/`
  - `output/gprmax_datasets/layered_medium_pipe_v1/`
  - `output/gprmax_datasets/weak_target_noise_v1/`
- gprMax commands:
  - `python scripts/gprmax_benchmark/run_uavgpr_gprmax_package.py --gprmax-root E:\gprMax\gprMax-v.3.1.7 --output-root output\gprmax_datasets --output-name pipe_demo_longline_v1 --preset uav_pipe_gain_workflow_bscan --traces 90 --run-timeout-s 1800`
  - `python scripts/gprmax_benchmark/run_uavgpr_gprmax_package.py --gprmax-root E:\gprMax\gprMax-v.3.1.7 --output-root output\gprmax_datasets --output-name cylinder_depth_shallow_v1 --preset uav_pipe_gain_workflow_bscan --traces 90 --target-center-y 0.35 --run-timeout-s 1800`
  - `python scripts/gprmax_benchmark/run_uavgpr_gprmax_package.py --gprmax-root E:\gprMax\gprMax-v.3.1.7 --output-root output\gprmax_datasets --output-name cylinder_depth_deep_v1 --preset uav_pipe_gain_workflow_bscan --traces 90 --target-center-y 0.14 --run-timeout-s 1800`
  - `$layers='[{"name":"moist_layer","eps_r":14.0,"sigma":0.01,"y_min":0.32,"y_max":0.36}]'; python scripts/gprmax_benchmark/run_uavgpr_gprmax_package.py --gprmax-root E:\gprMax\gprMax-v.3.1.7 --output-root output\gprmax_datasets --output-name layered_medium_pipe_v1 --preset uav_pipe_gain_workflow_bscan --traces 90 --background-layer-json $layers --run-timeout-s 1800`
  - `python scripts/gprmax_benchmark/run_uavgpr_gprmax_package.py --gprmax-root E:\gprMax\gprMax-v.3.1.7 --output-root output\gprmax_datasets --output-name weak_target_noise_v1 --preset uav_pipe_gain_workflow_bscan --traces 90 --target-name weak_pipe --target-eps-r 10.5 --target-sigma 0.004 --target-radius 0.03 --run-timeout-s 1800`
- audit result:
  - all five datasets reached `READY` with manifest, `.out`, metadata, and
    `ground_truth.yaml`.
- Evidence pipeline:
  - `dewow,subtracting_average_2D,sec_gain`
- Evidence summary:

| scenario_id | manual truth_score | AutoTune truth_score | delta | manual false_positive | AutoTune false_positive | report |
|---|---:|---:|---:|---:|---:|---|
| `pipe_demo_longline_v1` | -0.104267 | 1.693496 | 1.797763 | 1.936212 | 0.412069 | `output/gprmax_autotune_real_smoke/pipe_demo_longline_v1_standard/comparison_report.md` |
| `cylinder_depth_shallow_v1` | 0.929011 | 1.490861 | 0.561850 | 1.114596 | 0.299049 | `output/gprmax_autotune_real_smoke/cylinder_depth_shallow_v1_standard/comparison_report.md` |
| `cylinder_depth_deep_v1` | -0.648224 | 1.563585 | 2.211809 | 2.363366 | 0.514293 | `output/gprmax_autotune_real_smoke/cylinder_depth_deep_v1_standard/comparison_report.md` |
| `layered_medium_pipe_v1` | 1.876666 | 2.201006 | 0.324340 | 0.724088 | 0.395104 | `output/gprmax_autotune_real_smoke/layered_medium_pipe_v1_standard/comparison_report.md` |
| `weak_target_noise_v1` | 0.764274 | 1.709558 | 0.945284 | 0.910625 | 0.388032 | `output/gprmax_autotune_real_smoke/weak_target_noise_v1_standard/comparison_report.md` |

- interpretation:
  - In this compact matrix, AutoTune improves `truth_score` in all five
    scenarios and reduces `truth_false_positive_ratio` in all five scenarios.
  - The weak-target scene improves truth metrics but worsens generic
    `comparison_score`, which is useful evidence that truth validation and
    generic visual metrics can disagree.
  - The side-by-side B-scans for the standard pipeline show clearer hyperbola
    preservation in the AutoTune branch under locked-scale display.
- limitations:
  - `weak_target_noise_v1` reduces material contrast but does not inject random
    field noise yet.
  - The current matrix still uses one target and one main target ROI per scene.
  - Multi-target truth schema and metrics should be expanded before claiming
    multiple-object performance.
- next step:
  - add a true noise/low-SNR scene and a multi-target scene after the current
    deterministic evidence bridge is committed.

## Iteration 5 - Deterministic Additive Noise Stress Test

- branch: `codex/research-gprmax-autotune`
- base commit: `9603c0a`
- goal: convert the previous "weak target" note into an actual stochastic
  noise benchmark with deterministic random seeds and auditable metadata.
- changed files:
  - `scripts/gprmax_benchmark/add_noise_to_gprmax_dataset.py`
  - `tests/test_gprmax_noise_dataset.py`
  - `docs/gprmax_autotune_iteration_log.md`
  - `docs/autotune_gprmax_pipe_demo_report.md`
- source dataset:
  - `output/gprmax_datasets/pipe_demo_longline_v1/pipe_demo_longline_v1_manifest.json`
- generation commands:
  - `python scripts/gprmax_benchmark/add_noise_to_gprmax_dataset.py --source output\gprmax_datasets\pipe_demo_longline_v1\pipe_demo_longline_v1_manifest.json --output-root output\gprmax_datasets --output-name pipe_demo_low_snr_noise_v1 --target-snr-db 3.0 --seed 20260517 --overwrite`
  - `python scripts/gprmax_benchmark/add_noise_to_gprmax_dataset.py --source output\gprmax_datasets\pipe_demo_longline_v1\pipe_demo_longline_v1_manifest.json --output-root output\gprmax_datasets --output-name pipe_demo_noise_8db_v1 --target-snr-db 8.0 --seed 20260517 --overwrite`
  - `python scripts/gprmax_benchmark/add_noise_to_gprmax_dataset.py --source output\gprmax_datasets\pipe_demo_longline_v1\pipe_demo_longline_v1_manifest.json --output-root output\gprmax_datasets --output-name pipe_demo_noise_15db_v1 --target-snr-db 15.0 --seed 20260517 --overwrite`
- generated datasets:
  - `output/gprmax_datasets/pipe_demo_low_snr_noise_v1/` (`actual_snr_db=3.000`)
  - `output/gprmax_datasets/pipe_demo_noise_8db_v1/` (`actual_snr_db=8.000`)
  - `output/gprmax_datasets/pipe_demo_noise_15db_v1/` (`actual_snr_db=15.000`)
- audit result:
  - all three noisy datasets reached `READY`.
- Evidence pipeline:
  - `dewow,subtracting_average_2D,sec_gain`
- Evidence summary:

| scenario_id | manual truth_score | AutoTune truth_score | delta | manual false_positive | AutoTune false_positive | report |
|---|---:|---:|---:|---:|---:|---|
| `pipe_demo_low_snr_noise_v1` | -1.102353 | -1.224633 | -0.122280 | 0.978858 | 1.018470 | `output/gprmax_autotune_real_smoke/pipe_demo_low_snr_noise_v1_standard/comparison_report.md` |
| `pipe_demo_noise_8db_v1` | -1.125794 | -1.270860 | -0.145067 | 0.978855 | 1.020414 | `output/gprmax_autotune_real_smoke/pipe_demo_noise_8db_v1_standard/comparison_report.md` |
| `pipe_demo_noise_15db_v1` | -1.037360 | -1.404514 | -0.367154 | 0.978847 | 1.024179 | `output/gprmax_autotune_real_smoke/pipe_demo_noise_15db_v1_standard/comparison_report.md` |

- visual check:
  - The standard pipeline side-by-side images are dominated by additive noise
    texture. The target hyperbola is not reliably interpretable.
- interpretation:
  - This is a useful failure result. The current standard pipeline and scoring
    domain are not robust to global additive Gaussian noise.
  - These scenarios should not be used as a positive AutoTune claim yet.
  - They should drive the next algorithm work: noise-aware candidate domains,
    explicit denoise stage selection, and truth-aware evaluation of
    target-preserving denoise methods.
- next step:
  - add a noise-aware AutoTune experiment path that compares denoise methods and
    avoids treating gain/background settings as the whole solution to low-SNR
    data.
