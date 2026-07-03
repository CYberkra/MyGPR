# TEST-GUI-PRESETS-001 Failure Audit

- test_command: `pytest -q tests/test_gui_presets.py -vv`
- result: `10 failed, 52 passed`
- scope: AutoTune page compatibility after `ui/autotune_tuning_page.py` redesign
- decision: **high-risk to hotfix in this round**, keep current UI behavior and record precise compatibility gap

## Failure classification

### A) Real UI contract regression (new page removed legacy interface)
These failures are not environment issues; they come from missing legacy attributes/methods expected by current tests and `app_qt` bridge contracts.

1. `test_auto_tune_defaults_live_in_auto_tune_page`
   - missing: `page_auto_tune.segmented`
2. `test_auto_tune_page_handles_state_transitions`
   - missing: `page_auto_tune.auto_tune_summary`
3. `test_auto_tune_truth_validation_page_displays_ground_truth_metadata`
   - missing method: `page_auto_tune.refresh_truth_validation`
4. `test_auto_tune_truth_validation_page_records_evidence_preview_path`
   - missing: `page_auto_tune.truth_bscan_status_label`
5. `test_auto_tune_truth_validation_open_path_handles_missing_path`
   - missing method: `page_auto_tune._open_path`
6. `test_auto_tune_truth_validation_incomplete_ground_truth_info_is_warning`
   - missing: `page_auto_tune.truth_status_label`
7. `test_auto_tune_truth_validation_uses_complete_comparison_ground_truth_info`
   - missing: `page_auto_tune.truth_status_label`
8. `test_auto_tune_tab_exposes_research_console_pages`
   - missing: `page_auto_tune.segmented`
9. `test_stage_compare_result_summary_is_human_readable`
   - missing: `page_auto_tune.stage_compare_label`

### B) Test expectation still anchored to legacy visual hierarchy
10. `test_phase2_tabs_expose_prioritized_group_hierarchy_and_bridge`
   - expected top-level group title: `实验流程`
   - actual new page title: `AutoTune Session`
   - this is a deliberate UX redesign delta, not a runtime crash.

### C) Non-root causes
- fixture/import issues: **not observed**
- environment dependency issues: **not observed**
- PyVista/PyVistaQt dependency issues: **not observed in this failure set**

## Risk assessment

Directly patching the new page to satisfy all legacy attributes can re-introduce hidden coupling and undo the redesign intent.
Current failure cluster indicates the right fix is a compatibility adapter or controlled test migration, not ad hoc per-attribute patching.

## Recommended next step

1. Add an explicit compatibility shim on `AutoTuneTuningPage` for required legacy contract points only:
   - `segmented`, `auto_tune_summary`, `truth_status_label`, `truth_bscan_status_label`, `stage_compare_label`
   - methods `refresh_truth_validation()`, `_open_path()`
2. Keep redesigned visual layout, but expose these properties/methods as aliases/delegates.
3. Update only assertions that intentionally depend on old visual text labels (e.g., `"实验流程"` vs `"AutoTune Session"`), with a migration note.
4. Re-run:
   - `pytest -q tests/test_gui_presets.py`
   - `pytest -q`
