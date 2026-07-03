# MyGPR V0.8.65 Patch Notes

## Audit scope

This pass re-audited the V0.8.64 UI-completion package against the visible user-facing requirements from the startup README, the recent UI completion notes, and the project-first Workbench lifecycle:

```text
项目管理 -> 测线处理 -> 目标定位 -> 空间成果 -> 成果报告
```

The audit focused on whether promised entry points exist, whether guarded UI flows actually show the result to the user, and whether safe boundaries prevent accidental long-running or invalid operations.

## Gaps found and closed in V0.8.65

1. The startup README listed Chinese Windows aliases for launch, debug launch, and environment check, but V0.8.64 only included the English scripts. Added:
   - `启动MyGPR.bat`
   - `启动MyGPR_调试日志.bat`
   - `检查MyGPR环境.bat`

2. `check_mygpr_environment.bat` checked `pywt`, but its Python discovery path was narrower than `start_mygpr.bat`. It now follows the same practical order for `.venv`, Python 3.13, 3.12, 3.11, 3.10, then PATH `python`.

3. The gprMax command preview used POSIX single-quote formatting. That is fragile for Windows users copying commands into `cmd.exe`. The preview now uses conservative double-quote formatting for paths with spaces.

4. Invalid gprMax campaign scenes disabled the copy button but could still display a runnable-looking command. Invalid scenes now produce no command preview and direct programmatic copy calls are also blocked.

5. Saved processing versions opened a document tab, but if the user selected a result while another workspace was active, the preview could be hidden behind that workspace. Selecting a processing result now switches to the data-document workspace so the read-only B-scan preview is immediately visible.

6. Preflight syntax coverage now includes the newer UI completion files `ui/simulation_validation_page.py` and `ui/matplotlib_fonts.py`.

## Requirement coverage matrix

| Area | Status after audit | Evidence |
| --- | --- | --- |
| Local Windows environment installer | Covered | `install_mygpr_environment.bat`, `安装MyGPR环境.bat`, dependency import verification |
| Environment checker/launcher dependency alignment | Covered | both check `PyQt6`, `qfluentwidgets`, scientific stack, `pywt`; checker Python search order aligned |
| Chinese user-facing launcher aliases | Covered in V0.8.65 | wrappers for launch, debug launch, environment check |
| Workbench lifecycle includes simulation validation | Covered | `WORKSPACES` includes `simulation_validation`; `SimulationValidationPage` is registered |
| gprMax dry-run validation boundary | Covered | GUI validates and previews commands only; it does not execute gprMax |
| Invalid gprMax scene guard | Covered in V0.8.65 | invalid scenes no longer generate runnable commands |
| Delivery package feedback in Evidence drawer | Covered | `DeliveryPage.package_built` populates Workbench evidence table |
| Processing version navigation | Covered and hardened in V0.8.65 | result-tree selection opens visible read-only preview |
| AutoTune stale recommendation guard | Covered | recommendation application checks method identity |
| Interpretation raw/result QC gate separation | Covered | raw mode does not reuse processing-result QC gate |
| Matplotlib CJK fallback | Covered where fonts are available | shared helper is imported by `ui/__init__.py` |

## Validation performed in this audit environment

Environment:

```text
Python 3.13.5 virtual environment at /mnt/data/mygpr_audit_venv
runtime_imports_ok: PyQt6, qfluentwidgets, numpy, pandas, scipy, matplotlib, h5py, yaml, pywt
pip check: No broken requirements found
```

Commands that passed:

```text
QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python scripts/preflight_check.py
# [OK] Preflight passed

python scripts/check_version_consistency.py --expected 0.8.65
# version_check_ok: 0.8.65

python -m compileall -q app_qt.py ui core scripts cli_batch.py
# passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q \
  tests/test_version_consistency.py \
  tests/test_environment_and_workbench_polish.py \
  tests/test_workbench_ui.py \
  tests/test_processing_lab_ui.py \
  tests/test_interpretation_workbench_ui.py \
  tests/test_simulation_validation_page_ui.py
# 25 passed, 1 skipped

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q \
  tests/test_spatial_synthesis_ui.py \
  tests/test_delivery_page_ui.py \
  tests/test_theme_polish_stylesheet.py \
  tests/test_no_prior_ui_guardrails.py \
  tests/test_app_qt_controller_boundaries.py \
  tests/test_workbench_entry_and_bridge.py
# 18 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q \
  tests/test_workbench_project_core.py \
  tests/test_processing_session_service.py \
  tests/test_delivery_service.py \
  tests/test_spatial_synthesis_service.py \
  tests/test_interpretation_service.py \
  tests/test_gpr_format_registry_and_readers.py \
  tests/test_gpr_io_airborne_contract.py \
  tests/test_gpr_io_ascans_folder.py
# 49 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q \
  tests/test_gprmax_campaign_loader.py \
  tests/test_gprmax_campaign_validator.py \
  tests/test_gprmax_campaign_preview.py
# 17 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q tests/test_gprmax_campaign_cli.py
# 3 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q tests/test_gprmax_campaign_runner_execution.py
# 16 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q tests/test_autotune_recipe_ui.py
# 7 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q tests/test_research_dashboard_model.py
# 5 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q tests/test_gui_presets.py::test_auto_tune_tab_exposes_research_console_pages
# 1 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q tests/test_autotune_tuning_page_target_response.py
# 1 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q tests/test_auto_tune_result_dialog.py
# 2 passed
```

Notes:

- The single skipped test is expected on this Linux/offscreen host because Windows CJK fonts are not available here.
- I did not claim a full 887-test suite pass. A monolithic Qt/Matplotlib/offscreen run and a combined AutoTune UI batch were unstable in this sandbox, so I split validation into targeted batches and reran the affected tests individually.
- Some long AutoTune algorithm tests were not part of this UI-completion audit because this pass intentionally did not modify processing algorithms, AutoTune scoring, gprMax execution semantics, or Evidence schemas.

## Windows usage

1. Extract the delivered ZIP.
2. Enter the project root.
3. Run `安装MyGPR环境.bat` or `install_mygpr_environment.bat` once to create `.venv` and install dependencies.
4. Run `启动MyGPR.bat` or `start_mygpr.bat` to launch.
5. If launch fails, run `检查MyGPR环境.bat` or `check_mygpr_environment.bat`, then inspect `%LOCALAPPDATA%\MyGPR\logs\launcher`.
