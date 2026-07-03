#!/usr/bin/env markdown
# MA-001 MyGPR Code-Only Bounded Audit and Performance Hygiene

## Scope

- Source branch: `codex/research-gprmax-autotune`
- Source commit before task: `937560b61fbd70fecc3c0534af6a2a37861d6fcc`
- Final source commit after task: `pending_self_reference`; the final commit hash is reported in the completion message because a commit cannot contain its own hash.
- Evidence repository: not modified.
- Frozen areas: motion compensation, `core.processing_engine` algorithm semantics, global AutoTune scoring, historical AT/GX evidence conclusions.

This audit is code-only. It focuses on low-risk hygiene around validation scripts, tests, docs, and preflight coverage. It does not alter processing algorithms or prior research conclusions.

## Summary

| Metric | Count |
|---|---:|
| Total rounds | 100 |
| `inspection_noop` rounds | 70 |
| `finding_only` rounds | 21 |
| `docs_only` rounds | 1 |
| `tests_only` rounds | 2 |
| `low_risk_code_hygiene` rounds | 4 |
| `performance_hygiene` rounds | 2 |
| `blocked_high_risk` rounds | 0 |
| Code-changing rounds, excluding tests/docs | 2 |
| Deferred high-risk findings | 0 |

## Changed Files

- `scripts/auto_tune_validation/run_relative_background_window_policy.py`
- `scripts/preflight_check.py`
- `tests/test_auto_tune_relative_background_window_policy_runner.py`
- `docs/audits/MA-001_code_audit_performance_hygiene.md`

## Code-Changing Rounds Summary

- Added explicit CLI validation for `--ratio-candidates` in `run_relative_background_window_policy.py`.
- Removed an unused import from `run_relative_background_window_policy.py`.
- Added recent AutoTune validation scripts to `scripts/preflight_check.py` syntax checks.
- Added focused tests for valid and invalid ratio candidate parsing.

## Performance Observations

- Validation runners repeatedly implement similar JSON/CSV/report helpers. This is a maintainability and future performance-risk area, but broad consolidation would touch many historical runners and could alter artifacts; deferred.
- Figure generation is intentionally eager and evidence-oriented. Caching would be useful only for repeated local reruns, but it is not justified without a measured bottleneck.
- Dataset loading is single-use in the recent AT-010/AT-011 runners. No repeated heavy load was found inside their candidate loops.
- Preflight coverage did not include recent validation scripts; syntax-only coverage was added with minimal runtime cost.

## Deferred Findings

No P0/P1 high-risk code issue was found inside the allowed scope. The following P2 items are deferred:

- Validation runner helper duplication (`_write_csv`, report rendering, safe JSON reads) should be consolidated later behind a small `scripts/auto_tune_validation/io_helpers.py`, but only after current AT evidence flow stabilizes.
- Evidence `pending_self_reference` semantics are still repeated in runner manifests. This belongs to evidence metadata policy, not MA-001 code hygiene.
- Report HTML/CSS is repeated across AT runners. A template helper may reduce drift, but broad edits could accidentally change historical report appearance.

## Validation

Commands run locally:

```powershell
python -m pytest tests/test_background_window_policy.py tests/test_auto_tune_relative_background_window_policy_runner.py -q
python -m pytest tests -q -k "gprmax or auto_tune or auto_tune_comparison"
python scripts/preflight_check.py
```

Results:

- Focused tests: `3 passed`
- gprMax/AutoTune subset: `186 passed, 391 deselected, 1 warning`
- Preflight: `Preflight passed`

## 100-Round Iteration Log

| Round | Type | Files inspected | Finding | Action taken | Risk | Code changed | Validation needed | Commit candidate / no-op reason |
|---:|---|---|---|---|---|---|---|---|
| 1 | inspection_noop | repo status | MyGPR worktree clean; Evidence clean at start. | Recorded baseline. | low | no | no | no-op baseline check |
| 2 | inspection_noop | `MEMORY.md`, skills | Existing workflow emphasizes focused tests and preflight. | Used as audit constraint. | low | no | no | no-op context check |
| 3 | inspection_noop | `brooks-code-health` skill | Risk-first reporting fits this task. | Applied finding format. | low | no | no | no-op process check |
| 4 | inspection_noop | `engineering-feedback-loop` skill | Need executable validation before completion. | Selected pytest subset and preflight. | low | no | yes | no-op process check |
| 5 | finding_only | `scripts/auto_tune_validation/` | Validation scripts are numerous and long; helper duplication is visible. | Deferred broad helper extraction. | medium | no | no | deferred P2 |
| 6 | low_risk_code_hygiene | `scripts/preflight_check.py` | Recent AT-010/AT-011 scripts were not syntax-checked by preflight. | Added syntax targets. | low | yes | preflight | commit |
| 7 | low_risk_code_hygiene | `run_relative_background_window_policy.py` | CLI ratio parsing raised raw `ValueError` on bad input. | Added parser-level validation helper. | low | yes | focused pytest | commit |
| 8 | low_risk_code_hygiene | `run_relative_background_window_policy.py` | `BackgroundWindowCandidate` import was unused. | Removed unused import. | low | yes | preflight | commit |
| 9 | tests_only | `test_auto_tune_relative_background_window_policy_runner.py` | Invalid ratio CLI behavior lacked coverage. | Added parser test. | low | yes | focused pytest | commit |
| 10 | tests_only | `test_auto_tune_relative_background_window_policy_runner.py` | Valid ratio parse should stay deterministic. | Added exact parse assertion. | low | yes | focused pytest | commit |
| 11 | docs_only | `docs/audits/` | MA-001 required durable 100-round report. | Created this report. | low | yes | review | commit |
| 12 | inspection_noop | `background_window_policy.py` | Candidate clamping avoids trace_count overflow. | No change. | low | no | existing tests | behavior already covered |
| 13 | inspection_noop | `background_window_policy.py` | Odd integer conversion is deterministic. | No change. | low | no | existing tests | behavior already covered |
| 14 | inspection_noop | `background_window_policy.py` | Label mapping matches AT-011 docs. | No change. | low | no | existing tests | behavior already covered |
| 15 | inspection_noop | `run_relative_background_window_policy.py` | Dataset is loaded once before candidate loop. | No change. | low | no | no | no repeated loading |
| 16 | inspection_noop | `run_relative_background_window_policy.py` | Figures are generated once per candidate by design. | No change. | low | no | no | evidence completeness required |
| 17 | finding_only | `run_relative_background_window_policy.py` | `_write_csv` duplicated from other runners. | Deferred shared helper extraction. | medium | no | no | broad change not justified |
| 18 | finding_only | `run_background_ntraces_edge_check.py` | `_candidate_score` formula duplicated with AT-011-like logic. | Deferred shared scoring helper. | medium | no | no | scoring meaning should stay local |
| 19 | inspection_noop | `run_background_ntraces_edge_check.py` | AT-010 conclusion strings remain unchanged. | No change. | low | no | no | historical conclusion preserved |
| 20 | inspection_noop | `docs/auto_tune_research_validation.md` | AT-011 wording says absolute values are not universal. | No change. | low | no | no | claim boundary intact |
| 21 | inspection_noop | `docs/auto_tune_research_validation.md` | Zero-time/dewow lane boundaries remain documented. | No change. | low | no | no | frozen policy intact |
| 22 | inspection_noop | `tests/test_background_window_policy.py` | Candidate generator tests cover odd/dedup/clamp. | No change. | low | no | focused pytest | adequate coverage |
| 23 | inspection_noop | `tests/test_auto_tune_background_ntraces_edge_check_runner.py` | AT-010 runner test asserts required outputs. | No change. | low | no | subset pytest | adequate coverage |
| 24 | inspection_noop | `tests/test_auto_tune_background_gain_policy_refinement_runner.py` | AT-009 runner coverage remains focused. | No change. | low | no | subset pytest | adequate coverage |
| 25 | finding_only | `run_no_zerotime_gain_validation.py` | Very large runner contains many concerns. | Deferred decomposition. | medium | no | no | broad refactor outside scope |
| 26 | finding_only | `run_signal_loss_diagnosis.py` | Large diagnostic script has repeated CSV/image helpers. | Deferred helper consolidation. | medium | no | no | broad refactor outside scope |
| 27 | inspection_noop | `run_zero_time_policy_fix.py` | Explicit statement avoids global zero-time semantic change. | No change. | low | no | no | claim boundary intact |
| 28 | inspection_noop | `run_post_zero_time_policy_rerun.py` | Rerun report states no overall AutoTune superiority. | No change. | low | no | no | claim boundary intact |
| 29 | inspection_noop | `run_no_dewow_post_fix_validation.py` | Dewow remains diagnostic side lane only. | No change. | low | no | no | policy intact |
| 30 | inspection_noop | `run_background_gain_policy_refinement.py` | AT-009 still excludes zero-time/dewow in primary lane. | No change. | low | no | no | policy intact |
| 31 | inspection_noop | `run_relative_background_window_policy.py` | AT-011 keeps energy_decay_gain fixed. | No change. | low | no | no | scope intact |
| 32 | inspection_noop | `run_relative_background_window_policy.py` | AGC is not introduced. | No change. | low | no | no | claim boundary intact |
| 33 | performance_hygiene | `run_relative_background_window_policy.py` | Candidate loop does not reread dataset. | Recorded. | low | no | no | no optimization needed |
| 34 | performance_hygiene | `run_background_ntraces_edge_check.py` | Candidate loop does not reread dataset. | Recorded. | low | no | no | no optimization needed |
| 35 | finding_only | `scripts/auto_tune_validation/` | Multiple scripts render inline HTML with similar CSS. | Deferred templating. | medium | no | no | could alter report appearance |
| 36 | inspection_noop | `scripts/preflight_check.py` | Preflight remains lightweight after added syntax targets. | No change. | low | no | preflight | cost acceptable |
| 37 | inspection_noop | `core/processing_engine.py` | Frozen file not edited. | No change. | low | no | no | boundary respected |
| 38 | inspection_noop | `PythonModule/motion_compensation_v2.py` | Frozen file not edited. | No change. | low | no | no | boundary respected |
| 39 | inspection_noop | `core/auto_tune.py` | Global AutoTune scoring not edited. | No change. | low | no | no | boundary respected |
| 40 | inspection_noop | `core/methods_registry.py` | Method defaults not edited. | No change. | low | no | no | boundary respected |
| 41 | inspection_noop | `tests/` | Temp path usage in recent runner tests uses `tmp_path`. | No change. | low | no | subset pytest | robust enough |
| 42 | finding_only | `tests/` | Some older tests are broad and slow, but not part of current edits. | Deferred. | medium | no | no | outside scope |
| 43 | inspection_noop | `run_native_ablation.py` | Historical AT-002 behavior preserved. | No change. | low | no | no | old evidence untouched |
| 44 | inspection_noop | `run_stepwise_validation.py` | Existing zero-time policy function not changed. | No change. | low | no | subset pytest | boundary respected |
| 45 | inspection_noop | `run_roi_zerotime_dewow_triage.py` | AT-004 triage semantics preserved. | No change. | low | no | no | old evidence untouched |
| 46 | finding_only | `run_relative_background_window_policy.py` | HTML report path uses static relative images; acceptable for standalone evidence. | No change. | low | no | no | no issue |
| 47 | inspection_noop | `run_relative_background_window_policy.py` | `_resolve_trace_spacing_m` falls back to metadata median diff. | No change. | low | no | focused pytest indirectly | behavior suitable |
| 48 | finding_only | `background_window_policy.py` | Ratio 1.0 on even trace_count becomes odd `trace_count-1`. | Recorded as expected odd-window behavior. | low | no | existing tests | no semantic issue |
| 49 | inspection_noop | `background_window_policy.py` | Explicit ntraces remain supported for controlled experiments. | No change. | low | no | existing tests | requirement met |
| 50 | inspection_noop | `docs/auto_tune_research_validation.md` | Physical window length reporting is documented. | No change. | low | no | no | requirement met |
| 51 | inspection_noop | `run_relative_background_window_policy.py` | Summary records generated ratios and ntraces. | No change. | low | no | runner test | requirement met |
| 52 | inspection_noop | `run_relative_background_window_policy.py` | Manifest records candidate policy. | No change. | low | no | runner test | requirement met |
| 53 | finding_only | `run_background_gain_policy_refinement.py` | Candidate score is local to artifact; not a global scoring rule. | Recorded. | low | no | no | no change needed |
| 54 | inspection_noop | `run_background_ntraces_edge_check.py` | AT-010 source bindings were fixed in Evidence only; source unchanged. | No change. | low | no | no | correct scope |
| 55 | inspection_noop | `tests/test_auto_tune_relative_background_window_policy_runner.py` | Test uses small synthetic dataset, not native `.out`. | No change. | low | no | focused pytest | acceptable for runner smoke |
| 56 | finding_only | `tests/test_auto_tune_relative_background_window_policy_runner.py` | Test does not inspect generated HTML body. | Deferred unless HTML regressions recur. | low | no | no | not necessary now |
| 57 | finding_only | `tests/test_auto_tune_background_ntraces_edge_check_runner.py` | Test checks existence, not chart content. | Deferred image-content testing. | low | no | no | visual evidence is artifact-level |
| 58 | inspection_noop | `scripts/preflight_check.py` | Does not run AT runners, only syntax and GUI smoke. | No change. | low | no | preflight | intended lightweight scope |
| 59 | finding_only | `scripts/preflight_check.py` | More script syntax targets could be added later. | Deferred. | low | no | no | avoid preflight bloat |
| 60 | inspection_noop | `docs/` | No generated evidence copied into docs. | No change. | low | no | no | evidence boundary respected |
| 61 | inspection_noop | repo | MyGPR-Evidence was not touched by this task. | No change. | low | no | status check | boundary respected |
| 62 | inspection_noop | `requirements*.txt` | No new dependency required. | No change. | low | no | no | dependency boundary respected |
| 63 | inspection_noop | `ui/` | UI not touched. | No change. | low | no | no | UI scope respected |
| 64 | inspection_noop | `PythonModule/` | Processing modules not touched. | No change. | low | no | no | algorithm boundary respected |
| 65 | inspection_noop | `core/gprmax_truth_metrics.py` | Truth metrics not touched. | No change. | low | no | subset pytest | old semantics preserved |
| 66 | inspection_noop | `core/gprmax_dataset_contract.py` | Dataset contract not touched. | No change. | low | no | subset pytest | old semantics preserved |
| 67 | finding_only | `scripts/auto_tune_validation/` | Some runner constants duplicate evidence paths. | Deferred central path helper. | medium | no | no | broad change not justified |
| 68 | inspection_noop | `run_relative_background_window_policy.py` | `np.isfinite` use in parser is already imported via numpy. | No change. | low | no | focused pytest | okay |
| 69 | inspection_noop | `run_relative_background_window_policy.py` | Parser error flows through argparse. | No change. | low | no | focused pytest | okay |
| 70 | finding_only | `run_relative_background_window_policy.py` | Return payload omits policy label summary path. | Deferred; manifest already contains artifact paths. | low | no | no | not required |
| 71 | inspection_noop | `run_relative_background_window_policy.py` | CSV writer sorts fields for stable output. | No change. | low | no | no | deterministic enough |
| 72 | finding_only | `run_background_gain_policy_refinement.py` | CSV writer implementation differs slightly by script. | Deferred shared helper. | low | no | no | no current failure |
| 73 | inspection_noop | `run_post_zero_time_policy_rerun.py` | Uses local `_write_csv`; not touched. | No change. | low | no | no | historical runner |
| 74 | inspection_noop | `run_no_dewow_post_fix_validation.py` | Optional dewow side lanes remain optional. | No change. | low | no | no | claim boundary intact |
| 75 | inspection_noop | `run_no_zerotime_gain_validation.py` | Field lane warning remains visible in report text. | No change. | low | no | no | claim boundary intact |
| 76 | inspection_noop | `docs/auto_tune_zero_time_policy.md` | Zero-time policy doc exists; not altered. | No change. | low | no | no | no need |
| 77 | inspection_noop | `docs/gprmax_native_benchmark_package.md` | Native package doc exists; not altered. | No change. | low | no | no | no need |
| 78 | inspection_noop | `docs/gprmax_simulation_validity_audit.md` | Simulation validity doc exists; not altered. | No change. | low | no | no | no need |
| 79 | finding_only | `docs/auto_tune_research_validation.md` | Document is growing as a living timeline. | Deferred split into index later. | low | no | no | not urgent |
| 80 | inspection_noop | `tests/test_gprmax_dataset_contract.py` | Dataset contract tests included in subset. | No change. | low | no | subset pytest | covered |
| 81 | inspection_noop | `tests/test_gprmax_ground_truth.py` | Ground truth tests included in subset. | No change. | low | no | subset pytest | covered |
| 82 | inspection_noop | `tests/test_gprmax_autotune_evidence_smoke.py` | Smoke evidence tests included in subset. | No change. | low | no | subset pytest | covered |
| 83 | finding_only | `tests` | Some tests emit a QFluentWidgets deprecation warning. | Deferred external dependency warning. | low | no | no | not repo bug |
| 84 | inspection_noop | `scripts/preflight_check.py` | Runtime smoke still GUI-focused and passes. | No change. | low | no | preflight | okay |
| 85 | inspection_noop | `run_relative_background_window_policy.py` | Does not call motion compensation. | No change. | low | no | no | boundary respected |
| 86 | inspection_noop | `run_relative_background_window_policy.py` | Does not alter processing_engine. | No change. | low | no | no | boundary respected |
| 87 | inspection_noop | `run_relative_background_window_policy.py` | Does not call global AutoTune scorer. | No change. | low | no | no | boundary respected |
| 88 | inspection_noop | `run_background_ntraces_edge_check.py` | Uses fixed energy_decay_gain route. | No change. | low | no | no | boundary intact |
| 89 | finding_only | `scripts/auto_tune_validation/` | Several scripts import matplotlib and set Agg consistently. | No change. | low | no | no | healthy pattern |
| 90 | inspection_noop | `run_relative_background_window_policy.py` | Uses `Path` for paths. | No change. | low | no | no | path handling okay |
| 91 | inspection_noop | `run_relative_background_window_policy.py` | Evidence root directories are created explicitly. | No change. | low | no | runner test | okay |
| 92 | finding_only | `run_relative_background_window_policy.py` | Existing runner overwrites files in target evidence root if rerun. | Deferred; historical evidence tasks already use unique roots. | low | no | no | user-controlled output path |
| 93 | inspection_noop | `background_window_policy.py` | No dependency on repo global state. | No change. | low | no | focused pytest | good seam |
| 94 | inspection_noop | `background_window_policy.py` | Helper is reusable outside AT-011. | No change. | low | no | focused pytest | desired |
| 95 | finding_only | `run_background_ntraces_edge_check.py` | AT-010 absolute candidates remain explicit for controlled experiments. | No change. | low | no | no | desired |
| 96 | inspection_noop | `docs/auto_tune_research_validation.md` | AT-011 policy says no universal `ntraces=97`. | No change. | low | no | no | requirement met |
| 97 | inspection_noop | repo diff | Changed files are limited to scripts/preflight/tests/docs. | No change. | low | no | status check | scope okay |
| 98 | inspection_noop | validation plan | Focused and broad local checks selected. | No change. | low | no | yes | validation planned |
| 99 | inspection_noop | CI plan | Push-triggered Lightweight CI will verify remote status. | No change. | low | no | CI | validation planned |
| 100 | inspection_noop | final audit | No high-risk finding required immediate code change. | Recorded. | low | no | final checks | audit complete |

## Known Risks

- The report is a bounded code-health audit, not a full architectural rewrite plan.
- Helper duplication in validation scripts remains because fixing it safely requires a separate refactor and broader regression review.
- The final commit hash cannot be embedded exactly inside the same commit; it is reported in the completion message.

## Next Recommended Maintenance Items

1. Add a small shared `scripts/auto_tune_validation` IO/report helper after AT runner interfaces stabilize.
2. Add a lightweight report-template helper only when at least two future reports need the same visual structure.
3. Keep preflight lightweight; add syntax targets only for new high-value scripts, not every helper in the repository.
