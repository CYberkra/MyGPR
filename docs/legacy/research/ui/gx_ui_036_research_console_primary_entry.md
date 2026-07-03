# GX-UI-036 Research Console Primary Entry Completion

Base version: 0.8.60  
Output version: 0.8.61

This pass completes the unfinished research-validation UI path left by the earlier Codex UI work. The read-only research console already existed inside the hidden legacy `AutoTunePage`, but the normal `AutoTuneTuningPage` path did not expose it directly.

## Completed

- Added a visible `研究验证` action to the main AutoTune recommendation page header.
- Added `高级设置与审计明细 -> 研究验证` as a first-class tab in `AutoTuneTuningPage`.
- Kept the primary AutoTune tabs concise: `流程 / 候选 / 说明`.
- Preserved the legacy segmented research page for backward-compatible tests and callbacks.
- Expanded gprMax model draft discovery to scan local GX-008 model directories instead of using only the initial six scene ids.
- Replaced the model geometry placeholder with parsed target/background directives.
- Hardened research-console file opening for Windows, macOS, and Linux.
- Hardened Workbench close teardown after regression exposed that a missing optional page shutdown hook could interrupt layout persistence and project lock release.

## Boundaries

- No processing algorithm changes.
- No AutoTune scoring changes.
- No gprMax execution from the UI.
- No Evidence writing from the UI.
- No destructive model editing.

## Verification

- `python scripts/preflight_check.py`: passed.
- `python -m pytest -q tests/test_autotune_recipe_ui.py tests/test_research_dashboard_model.py tests/test_gui_presets.py::test_auto_tune_tab_exposes_research_console_pages`: passed.
- `python -m pytest -q tests/test_workbench_ui.py tests/test_processing_lab_ui.py tests/test_interpretation_workbench_ui.py tests/test_spatial_synthesis_ui.py tests/test_delivery_page_ui.py`: 20 passed, 1 skipped on Linux font fallback.
