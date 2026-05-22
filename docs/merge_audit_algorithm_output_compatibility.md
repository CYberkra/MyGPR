#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MR-ALG-001 algorithm output compatibility audit before merging research to main."""

# MR-ALG-001 Algorithm Output Compatibility Audit

## 1) Executive Summary

- `ready_for_merge_from_algorithm_perspective`: **yes (with P1 merge-note requirements)**
- P0 findings: **none**
- P1 findings: metadata-aware behavior and warning semantics are stronger in research baseline and must be documented at merge.
- P2 findings: one audit fixture path (`equidistant_trace_resample` without distance metadata) errors on both branches and should be explicitly documented as unsupported fixture mode.

This audit compares `origin/main` vs `origin/codex/research-gprmax-autotune` using deterministic synthetic fixtures only.  
No raw field CSV, no gprMax run, no AutoTune scoring changes.

## 2) Branch Metadata

- `origin/main` HEAD: `289c843d4c7a7d208070a4882d9c558bb2dfeaf1`
- `origin/codex/research-gprmax-autotune` HEAD: `30952a4932881334ee948cf334673441ae637b16`
- ahead/behind (`origin/main...origin/codex/research-gprmax-autotune`): `0 257`
- timestamp: generated during local run (2026-05-22)
- environment: Windows PowerShell, Python 3.11 runtime

`research` is not behind `main`; audit proceeded.

## 3) Changed Algorithm Areas

Major touched areas between branches include:

- `PythonModule/*` (filters/background/gain/motion and new methods)
- `core/processing_engine.py`
- `core/methods_registry.py`
- `core/gpr_io.py`
- `core/trace_metadata_utils.py`
- `core/auto_tune.py` + companion modules (`core/auto_tune_*`)

Diff scale (`git diff --stat origin/main..origin/codex/research-gprmax-autotune`): large research uplift, not a small patch merge.

## 4) Per-method Compatibility Table (audit fixtures)

Audit fixtures:
- `small_deterministic` (128x64)
- `field_like_501x2378`
- `metadata_rich_uav` (256x120 with trace metadata)
- `metadata_missing` (same size, no metadata)

Compact result summary:

| method | main status | research status | output numeric difference | shape difference | metadata/warning difference | risk |
|---|---|---|---|---|---|---|
| set_zero_time | ok | ok | identical on audited fixtures | no | warning semantics may differ with metadata | P1 |
| dewow | ok | ok | identical | no | low | P2 |
| frequency_filter_1d | ok | ok | identical on synthetic fixtures | no | metadata/time-step dependence is stronger by design | P1 |
| subtracting_average_2D | ok | ok | identical under fixed params | no | window/taper/time-range semantics documented in research | P1 |
| median_background_2D | ok | ok | identical under fixed params | no | time-window behavior support uplift | P1 |
| agcGain | ok | ok | identical on audited fixtures | no | low-energy guard/report semantics are richer in research path | P1 |
| sec_gain | ok | ok | identical | no | low | P2 |
| motion_compensation_v2 | ok | ok | identical for this fixture/params | no in this fixture set | resample semantics depend on metadata and `resample_spacing_m` policy | P1 |
| running_average_2D | ok | ok | identical | no | low | P2 |
| stolt_migration | ok | ok | identical | no | low | P2 |
| kirchhoff_migration | ok | ok | identical | no | low | P2 |
| energy_decay_gain | ok | ok | identical | no | new capability vs old baseline timeline | P1 |
| amplitude_scale | ok | ok | identical | no | new capability vs old baseline timeline | P1 |
| time_cut | ok | ok | identical on valid ranges | can change shape by design | expected shape_diff_expected behavior | P1 |
| trace_qc | ok | ok | identical | no | QC metadata/report output richer in research chain | P1 |
| equidistant_trace_resample | error on 3 fixtures without distance metadata | error on same fixtures | n/a | n/a | both branches fail similarly without required trace distance context | P2 |

Machine comparison artifact:
- [comparison_rows.csv](D:/CDUT-UavGPR-Controller/MyGPR/docs/merge_audit_algorithm_output_compatibility/comparison_rows.csv)
- [audit_main.json](D:/CDUT-UavGPR-Controller/MyGPR/docs/merge_audit_algorithm_output_compatibility/audit_main.json)
- [audit_research.json](D:/CDUT-UavGPR-Controller/MyGPR/docs/merge_audit_algorithm_output_compatibility/audit_research.json)

## 5) High-impact Differences

### motion_compensation_v2 (`resample_spacing_m=0`)
- Research baseline emphasizes metadata-aware auto spacing behavior.
- In this synthetic fixture, output shape did not diverge, but semantics are metadata- and spacing-policy-driven and must be merge-noted.

### set_zero_time (time-step/header injection)
- Research path is stricter on time metadata usage in broader workflows.
- No hard regression observed; behavior should be treated as baseline upgrade.

### frequency_filter_1d (metadata dependent behavior)
- Research integrates stronger sample-rate / time-step semantics and warning handling.
- On audit fixtures, outputs were identical; on real field imports behavior can differ by metadata completeness.

### AGC behavior
- No numeric regression in this audit.
- Research chain has stronger guard/report semantics; merge note should mention display-vs-amplitude claim boundaries.

### background suppression
- Fixed-parameter outputs were compatible in deterministic fixtures.
- AT-019/AT-020 show no-prior safety interpretation risk is primarily recommendation semantics, candidate-space/scoring interpretation, not kernel crash.

## 6) Added Methods / Capabilities

Research branch includes (merge-visible capability uplifts):
- `energy_decay_gain`
- `amplitude_scale`
- `time_cut`
- `trace_qc`
- `equidistant_trace_resample`
- no-prior recommendation labeling (AT-021)
- expanded AutoTune/gprMax validation utilities under `scripts/auto_tune_validation/` and `scripts/gprmax_benchmark/`

## 7) Risk Classification

### P0
- none found in this audit
- no accidental AutoTune scoring change in AT-021 scope
- no processing_engine semantic blocker detected by required test gates

### P1
- metadata-dependent behavior is stronger in research baseline
- merge communication must clearly state baseline output may differ from old main in metadata-aware paths
- no-prior label semantics must be explained as non-blocking caution metadata (not algorithm change)

### P2
- fixture incompatibility for `equidistant_trace_resample` without distance metadata (both branches)
- minor warning text/path differences are expected

## 8) Merge Note Wording (recommended)

“This merge promotes the current research processing baseline to main. Some processing outputs may differ from the previous main baseline due to metadata-aware runtime parameter injection, updated background suppression behavior, AGC compatibility changes, and UAV motion compensation V2 resampling semantics. These differences are documented and expected.”

## 9) Final Recommendation

- **Proceed to MR-001 merge readiness audit**.
- No algorithm P0 blocker found.
- Before merge, keep release note explicit about:
  - metadata-aware behavior upgrades
  - shape-changing methods by design (`time_cut`, resampling paths)
  - no-prior warning/label semantics are non-blocking and diagnostic.

